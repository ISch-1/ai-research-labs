"""Retrievers and IR metrics.

Spec strings:

* ``bm25``               — Okapi BM25 (k1=1.5, b=0.75), implemented with sparse matrices
* ``tfidf``              — TF-IDF cosine
* ``dense:<encoder>``    — e.g. ``dense:st:BAAI/bge-small-en-v1.5`` (FAISS inner product)
* ``hybrid:<a>+<b>``     — reciprocal-rank fusion of two retrievers (k = 60)
* ``rerank:<first>``     — cross-encoder re-ranking of the first stage's top 20
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from lab_b.encoders import Encoder

CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Retriever:
    spec: str

    def fit(self, texts: list[str]) -> Retriever:
        raise NotImplementedError

    def search(self, queries: list[str], k: int) -> tuple[np.ndarray, np.ndarray]:
        """Top-k passage indices and scores, shape (n_queries, k), best first."""
        raise NotImplementedError


def _top_k(scores: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    k = min(k, scores.shape[1])
    idx = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(scores, idx, axis=1), axis=1)
    idx = np.take_along_axis(idx, order, axis=1)
    return idx, np.take_along_axis(scores, idx, axis=1)


@dataclass
class BM25(Retriever):
    k1: float = 1.5
    b: float = 0.75
    spec: str = "bm25"
    _vec: Any = field(default=None, repr=False)
    _weights: Any = field(default=None, repr=False)

    def fit(self, texts: list[str]) -> BM25:
        self._vec = CountVectorizer(lowercase=True, token_pattern=r"(?u)\b\w+\b")
        tf = self._vec.fit_transform(texts).tocsr().astype(np.float32)
        n_docs = tf.shape[0]
        doc_len = np.asarray(tf.sum(axis=1)).ravel()
        df = np.bincount(tf.indices, minlength=tf.shape[1])
        idf = np.log(1 + (n_docs - df + 0.5) / (df + 0.5))
        # tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl)), applied to the non-zeros only
        norm = self.k1 * (1 - self.b + self.b * doc_len / doc_len.mean())
        rows = np.repeat(np.arange(n_docs), np.diff(tf.indptr))
        data = tf.data * (self.k1 + 1) / (tf.data + norm[rows])
        self._weights = sparse.csr_matrix(
            (data * idf[tf.indices], tf.indices, tf.indptr), shape=tf.shape
        )
        return self

    def search(self, queries: list[str], k: int) -> tuple[np.ndarray, np.ndarray]:
        q = self._vec.transform(queries)
        q.data[:] = 1.0  # each query term counts once
        scores = (q @ self._weights.T).toarray()
        return _top_k(scores, k)


@dataclass
class TfidfRetriever(Retriever):
    spec: str = "tfidf"
    _vec: Any = field(default=None, repr=False)
    _docs: Any = field(default=None, repr=False)

    def fit(self, texts: list[str]) -> TfidfRetriever:
        self._vec = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=1)
        self._docs = self._vec.fit_transform(texts)
        return self

    def search(self, queries: list[str], k: int) -> tuple[np.ndarray, np.ndarray]:
        scores = (self._vec.transform(queries) @ self._docs.T).toarray()
        return _top_k(scores, k)


@dataclass
class DenseRetriever(Retriever):
    encoder_spec: str = "st:sentence-transformers/all-MiniLM-L6-v2"
    query_prefix: str = ""
    cache_corpus: bool = True
    spec: str = ""
    _index: Any = field(default=None, repr=False)
    _encoder: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.spec = self.spec or f"dense:{self.encoder_spec}"
        if "bge-" in self.encoder_spec and not self.query_prefix:
            # BGE models expect an instruction on queries (not on passages).
            self.query_prefix = "Represent this sentence for searching relevant passages: "

    def fit(self, texts: list[str]) -> DenseRetriever:
        import faiss

        self._encoder = Encoder(self.encoder_spec, cache=self.cache_corpus).fit(texts)
        emb = self._encoder.transform(texts)
        self._encoder.cache = False  # queries are always encoded live, so latency is real
        if sparse.issparse(emb):  # sparse encoders (e.g. ``hashing`` in offline tests)
            emb = emb.toarray()
            emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
        emb = np.ascontiguousarray(emb, dtype=np.float32)
        self._index = faiss.IndexFlatIP(emb.shape[1])
        self._index.add(emb)
        return self

    def search(self, queries: list[str], k: int) -> tuple[np.ndarray, np.ndarray]:
        q = self._encoder.transform([self.query_prefix + x for x in queries])
        if sparse.issparse(q):
            q = q.toarray()
            q /= np.linalg.norm(q, axis=1, keepdims=True) + 1e-12
        scores, idx = self._index.search(np.ascontiguousarray(q, dtype=np.float32), k)
        return idx, scores


@dataclass
class HybridRRF(Retriever):
    """Reciprocal-rank fusion: score(d) = sum over retrievers of 1 / (60 + rank)."""

    a: Retriever = field(default_factory=BM25)
    b: Retriever = field(default_factory=TfidfRetriever)
    depth: int = 100
    rrf_k: int = 60
    spec: str = ""

    def __post_init__(self) -> None:
        self.spec = self.spec or f"hybrid:{self.a.spec}+{self.b.spec}"

    def fit(self, texts: list[str]) -> HybridRRF:
        self.a.fit(texts)
        self.b.fit(texts)
        self._n = len(texts)
        return self

    def search(self, queries: list[str], k: int) -> tuple[np.ndarray, np.ndarray]:
        depth = min(self.depth, self._n)
        fused = np.zeros((len(queries), self._n), dtype=np.float32)
        for retriever in (self.a, self.b):
            idx, _ = retriever.search(queries, depth)
            ranks = np.arange(1, idx.shape[1] + 1)
            np.add.at(fused, (np.arange(len(queries))[:, None], idx), 1.0 / (self.rrf_k + ranks))
        return _top_k(fused, k)


@dataclass
class Reranker(Retriever):
    first: Retriever = field(default_factory=BM25)
    candidates: int = 20
    model: str = CROSS_ENCODER
    spec: str = ""

    def __post_init__(self) -> None:
        self.spec = self.spec or f"rerank:{self.first.spec}"

    def fit(self, texts: list[str]) -> Reranker:
        from sentence_transformers import CrossEncoder

        self.first.fit(texts)
        self._texts = texts
        self._ce = CrossEncoder(self.model, device="cpu")
        return self

    def search(self, queries: list[str], k: int) -> tuple[np.ndarray, np.ndarray]:
        cand, _ = self.first.search(queries, self.candidates)
        pairs = [(q, self._texts[j]) for q, row in zip(queries, cand, strict=True) for j in row]
        scores = np.asarray(
            self._ce.predict(pairs, batch_size=64, show_progress_bar=False)
        ).reshape(cand.shape)
        order = np.argsort(-scores, axis=1)[:, :k]
        return np.take_along_axis(cand, order, axis=1), np.take_along_axis(scores, order, axis=1)


def build(spec: str, *, cache: bool = True) -> Retriever:
    """``cache=False`` re-encodes the corpus, so index time includes the embedding cost."""
    if spec == "bm25":
        return BM25()
    if spec == "tfidf":
        return TfidfRetriever()
    if spec.startswith("dense:"):
        return DenseRetriever(spec.removeprefix("dense:"), cache_corpus=cache)
    if spec.startswith("hybrid:"):
        left, right = spec.removeprefix("hybrid:").split("+", 1)
        return HybridRRF(build(left, cache=cache), build(right, cache=cache))
    if spec.startswith("rerank:"):
        return Reranker(build(spec.removeprefix("rerank:"), cache=cache))
    raise ValueError(f"unknown retriever spec {spec!r}")


def short_name(spec: str) -> str:
    """Readable label: 'rerank:hybrid:bm25+dense:st:BAAI/bge-small-en-v1.5' -> 'rerank(bm25+bge-small)'."""

    def one(s: str) -> str:
        if s.startswith("dense:"):
            name = s.split("/")[-1]
            return name.replace("-en-v1.5", "").replace("all-", "")
        if s.startswith("hybrid:"):
            a, b = s.removeprefix("hybrid:").split("+", 1)
            return f"{one(a)}+{one(b)}"
        if s.startswith("rerank:"):
            return f"rerank({one(s.removeprefix('rerank:'))})"
        return s

    return one(spec)


# --------------------------------------------------------------------------- metrics


def ir_metrics(
    retrieved: np.ndarray, relevant: list[dict[int, int]], ks: tuple[int, ...] = (1, 5, 10)
) -> dict[str, np.ndarray]:
    """Per-query Recall@k, MRR@10 and nDCG@10 (graded relevance)."""
    out: dict[str, list[float]] = {f"recall@{k}": [] for k in ks}
    out["mrr@10"], out["ndcg@10"] = [], []
    for row, rel in zip(retrieved, relevant, strict=True):
        hits = [rel.get(int(d), 0) for d in row]
        n_rel = sum(1 for v in rel.values() if v > 0)
        for k in ks:
            out[f"recall@{k}"].append(sum(1 for h in hits[:k] if h > 0) / max(n_rel, 1))
        first = next((i for i, h in enumerate(hits[:10]) if h > 0), None)
        out["mrr@10"].append(0.0 if first is None else 1.0 / (first + 1))
        dcg = sum((2**h - 1) / np.log2(i + 2) for i, h in enumerate(hits[:10]))
        ideal = sorted(rel.values(), reverse=True)[:10]
        idcg = sum((2**h - 1) / np.log2(i + 2) for i, h in enumerate(ideal))
        out["ndcg@10"].append(dcg / idcg if idcg > 0 else 0.0)
    return {k: np.asarray(v) for k, v in out.items()}


def timed_search(retriever: Retriever, queries: list[str], k: int) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    idx, _ = retriever.search(queries, k)
    return idx, (time.perf_counter() - start) / max(len(queries), 1)
