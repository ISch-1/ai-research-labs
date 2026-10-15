"""Text representations, from sparse counts to sentence-transformer embeddings.

Every encoder follows ``fit(texts) -> self`` and ``transform(texts) -> matrix`` so
classical and neural representations plug into the same experiments. Spec strings:

* ``bow``                 — word counts (1-grams)
* ``tfidf``               — TF-IDF over 1-2 grams, sublinear TF
* ``hashing``             — hashed TF-IDF (no vocabulary; offline tests)
* ``static:<hf model>``   — static word embeddings (model2vec), mean-pooled
* ``st:<hf model>``       — sentence-transformer, normalized embeddings

Dense embeddings are cached on disk by (model, texts) hash, so re-running an
experiment does not re-encode the corpus.
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import (
    CountVectorizer,
    HashingVectorizer,
    TfidfTransformer,
    TfidfVectorizer,
)
from sklearn.pipeline import make_pipeline

from airlab.data import data_dir

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

_MODELS: dict[str, Any] = {}


def sentence_model(spec: str) -> Any:
    """Load (once per process) a sentence-transformers model for ``st:`` / ``static:`` specs."""
    if spec not in _MODELS:
        from sentence_transformers import SentenceTransformer

        kind, name = spec.split(":", 1)
        if kind == "static":
            from sentence_transformers.models import StaticEmbedding

            _MODELS[spec] = SentenceTransformer(
                modules=[StaticEmbedding.from_model2vec(name)], device="cpu"
            )
        else:
            _MODELS[spec] = SentenceTransformer(name, device="cpu")
    return _MODELS[spec]


@dataclass
class Encoder:
    spec: str
    batch_size: int = 64
    cache: bool = True
    _sparse: Any = field(default=None, init=False, repr=False)
    encode_seconds: float = field(default=0.0, init=False)

    @property
    def is_dense(self) -> bool:
        return ":" in self.spec

    @property
    def label(self) -> str:
        return self.spec.split("/")[-1] if self.is_dense else self.spec

    def fit(self, texts: list[str]) -> Encoder:
        if self.spec == "bow":
            self._sparse = CountVectorizer(min_df=2).fit(texts)
        elif self.spec == "tfidf":
            self._sparse = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(
                texts
            )
        elif self.spec == "hashing":
            self._sparse = make_pipeline(
                HashingVectorizer(n_features=2**16, alternate_sign=False, norm=None),
                TfidfTransformer(sublinear_tf=True),
            ).fit(texts)
        elif not self.is_dense:
            raise ValueError(f"unknown encoder spec {self.spec!r}")
        return self

    def transform(self, texts: list[str]) -> np.ndarray | sparse.csr_matrix:
        start = time.perf_counter()
        out = self._dense(texts) if self.is_dense else self._sparse.transform(texts)
        self.encode_seconds += time.perf_counter() - start
        return out

    def _dense(self, texts: list[str]) -> np.ndarray:
        path = None
        if self.cache:
            digest = hashlib.sha1("\x1f".join([self.spec, *texts]).encode()).hexdigest()[:20]
            path = data_dir() / "cache" / "embeddings" / f"{digest}.npy"
            if path.exists():
                return np.load(path)
        emb = sentence_model(self.spec).encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        emb = np.asarray(emb, dtype=np.float32)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, emb)
        return emb

    def dimension(self, sample: np.ndarray | sparse.csr_matrix) -> int:
        return int(sample.shape[1])
