"""Dataset registry for Lab B.

Hugging Face datasets are cached under ``<data_dir>/hf``. Every family has an
offline synthetic twin (``synthetic_*``) that tests and CI smoke runs use.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from airlab.data import data_dir

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def _hf(path: str, *args: str):
    from datasets import load_dataset

    return load_dataset(path, *args, cache_dir=str(data_dir() / "hf"))


# ------------------------------------------------------------------ classification


@dataclass
class TextClassification:
    name: str
    train_texts: list[str]
    train_labels: np.ndarray
    test_texts: list[str]
    test_labels: np.ndarray
    label_names: list[str]
    source: str

    @property
    def n_classes(self) -> int:
        return len(self.label_names)


def _stratified_subset(labels: np.ndarray, n: int | None, seed: int) -> np.ndarray:
    """Indices of a class-stratified random subset (all rows when ``n`` is None or larger)."""
    if n is None or n >= len(labels):
        return np.arange(len(labels))
    rng = np.random.default_rng(seed)
    chosen = []
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        k = max(1, round(n * len(idx) / len(labels)))
        chosen.append(rng.choice(idx, min(k, len(idx)), replace=False))
    out = np.concatenate(chosen)
    rng.shuffle(out)
    return out[:n]


def _from_hf(
    name: str,
    path: str,
    config: str | None,
    text_col: str,
    test_split: str,
    label_names: list[str] | None,
    max_train: int | None,
    max_test: int | None,
    seed: int,
) -> TextClassification:
    ds = _hf(path, *([config] if config else []))
    train, test = ds["train"], ds[test_split]
    names = label_names or train.features["label"].names
    y_tr, y_te = np.asarray(train["label"]), np.asarray(test["label"])
    i_tr, i_te = _stratified_subset(y_tr, max_train, seed), _stratified_subset(y_te, max_test, seed)
    texts_tr, texts_te = train[text_col], test[text_col]
    return TextClassification(
        name,
        [texts_tr[i].strip() for i in i_tr],
        y_tr[i_tr],
        [texts_te[i].strip() for i in i_te],
        y_te[i_te],
        names,
        f"HF `{path}`",
    )


_TOPICS = {
    "sports": ["team", "match", "goal", "coach", "league", "season", "player", "score"],
    "business": ["market", "shares", "profit", "company", "bank", "investors", "deal", "price"],
    "science": ["research", "space", "computer", "scientists", "software", "planet", "data"],
}
_FILLER = ["the", "a", "new", "on", "after", "with", "today", "report", "said", "for", "big"]


def _synthetic_topics(n_train: int, n_test: int, seed: int) -> TextClassification:
    """Keyword-driven topic sentences with filler noise: trivially learnable, fully offline."""
    rng = np.random.default_rng(seed)
    names = list(_TOPICS)

    def sample(n: int) -> tuple[list[str], np.ndarray]:
        labels = rng.integers(0, len(names), n)
        texts = []
        for y in labels:
            words = list(rng.choice(_TOPICS[names[y]], 3)) + list(rng.choice(_FILLER, 5))
            if rng.random() < 0.15:  # label noise via a misleading keyword
                words.append(rng.choice(_TOPICS[names[(y + 1) % len(names)]]))
            rng.shuffle(words)
            texts.append(" ".join(words))
        return texts, labels

    tr, te = sample(n_train), sample(n_test)
    return TextClassification("synthetic_topics", tr[0], tr[1], te[0], te[1], names, "synthetic")


CLASSIFICATION: dict[str, Callable[[int | None, int | None, int], TextClassification]] = {
    "ag_news": lambda tr, te, s: _from_hf(
        "ag_news",
        "fancyzhx/ag_news",
        None,
        "text",
        "test",
        ["World", "Sports", "Business", "Science"],
        tr,
        te,
        s,
    ),
    "sst2": lambda tr, te, s: _from_hf(
        # GLUE's SST-2 test labels are hidden: the validation split is the test set.
        "sst2",
        "stanfordnlp/sst2",
        None,
        "sentence",
        "validation",
        ["negative", "positive"],
        tr,
        te,
        s,
    ),
    "emotion": lambda tr, te, s: _from_hf(
        "emotion", "dair-ai/emotion", "split", "text", "test", None, tr, te, s
    ),
    "synthetic_topics": lambda tr, te, s: _synthetic_topics(tr or 600, te or 300, s),
}


def load_classification(
    name: str, *, max_train: int | None = None, max_test: int | None = None, seed: int = 0
) -> TextClassification:
    return CLASSIFICATION[name](max_train, max_test, seed)


# ------------------------------------------------------------------- QA / retrieval


@dataclass
class Passage:
    id: str
    title: str
    text: str


@dataclass
class Question:
    id: str
    text: str
    answers: list[str]
    relevant: dict[str, int]  # passage id -> graded relevance
    title: str = ""


@dataclass
class QACorpus:
    name: str
    passages: list[Passage]
    questions: list[Question]
    # Questions whose answer is NOT in the corpus (abstention tests). May be empty.
    out_of_corpus: list[Question] = field(default_factory=list)
    source: str = ""
    has_answers: bool = True

    @property
    def texts(self) -> list[str]:
        return [f"{p.title}. {p.text}" if p.title else p.text for p in self.passages]


def _squad(n_questions: int | None, n_held_out_articles: int, seed: int) -> QACorpus:
    """SQuAD v1.1 dev as an open-domain corpus: every paragraph is a passage.

    A few whole articles are held out of the corpus; their questions become
    "out-of-corpus" questions that a faithful RAG system should refuse.
    """
    rows = _hf("rajpurkar/squad")["validation"]
    titles = sorted(set(rows["title"]))
    rng = np.random.default_rng(seed)
    held_out = (
        set(rng.choice(titles, n_held_out_articles, replace=False))
        if n_held_out_articles
        else set()
    )

    passage_id: dict[str, str] = {}
    passages: list[Passage] = []
    inside, outside = [], []
    for row in rows:
        context = row["context"]
        q = Question(
            row["id"],
            row["question"],
            list(dict.fromkeys(row["answers"]["text"])),
            {},
            row["title"],
        )
        if row["title"] in held_out:
            outside.append(q)
            continue
        if context not in passage_id:
            passage_id[context] = f"p{len(passages)}"
            passages.append(Passage(passage_id[context], row["title"].replace("_", " "), context))
        q.relevant = {passage_id[context]: 1}
        inside.append(q)

    def pick(items: list[Question], n: int | None) -> list[Question]:
        if n is None or n >= len(items):
            return items
        return [items[i] for i in sorted(rng.choice(len(items), n, replace=False))]

    return QACorpus(
        "squad",
        passages,
        pick(inside, n_questions),
        pick(outside, n_questions),
        f"SQuAD v1.1 dev ({len(titles) - len(held_out)} articles in corpus, {len(held_out)} held out)",
    )


def _scifact(n_questions: int | None, seed: int) -> QACorpus:
    """BEIR SciFact: scientific claims -> abstracts that support or refute them."""
    corpus = _hf("BeIR/scifact", "corpus")["corpus"]
    queries = _hf("BeIR/scifact", "queries")["queries"]
    qrels = _hf("BeIR/scifact-qrels")["test"]
    passages = [Passage(str(r["_id"]), r["title"], r["text"]) for r in corpus]
    relevant: dict[str, dict[str, int]] = {}
    for r in qrels:
        relevant.setdefault(str(r["query-id"]), {})[str(r["corpus-id"])] = int(r["score"])
    text = {str(r["_id"]): r["text"] for r in queries}
    qs = [Question(qid, text[qid], [], rel) for qid, rel in sorted(relevant.items())]
    if n_questions and n_questions < len(qs):
        rng = np.random.default_rng(seed)
        qs = [qs[i] for i in sorted(rng.choice(len(qs), n_questions, replace=False))]
    return QACorpus("scifact", passages, qs, [], "BEIR SciFact (test qrels)", has_answers=False)


_FACTS = [
    ("Velmora", "Ostrin", "copper", 1842),
    ("Kadesh Vale", "Lumen Port", "salt", 1795),
    ("Orrin", "Tallow Bay", "amber", 1901),
    ("Saphir", "Greywick", "wool", 1764),
    ("Duskmere", "Halven", "tin", 1888),
    ("Brightfen", "Marrow Cross", "glass", 1823),
    ("Quellan", "Sorrel", "silk", 1911),
    ("Ashgrove", "Pellin", "coffee", 1856),
]


def _synthetic_qa() -> QACorpus:
    """Fictional countries, so a model cannot answer from memory: only retrieval helps."""
    passages, questions = [], []
    for i, (country, capital, export, year) in enumerate(_FACTS):
        pid = f"s{i}"
        passages.append(
            Passage(
                pid,
                country,
                f"{country} is a small fictional country. The capital city of {country} is {capital}. "
                f"The main export of {country} is {export}. {country} became independent in {year}.",
            )
        )
        questions += [
            Question(
                f"{pid}-cap", f"What is the capital of {country}?", [capital], {pid: 1}, country
            ),
            Question(
                f"{pid}-exp", f"What is the main export of {country}?", [export], {pid: 1}, country
            ),
            Question(
                f"{pid}-year",
                f"When did {country} become independent?",
                [str(year)],
                {pid: 1},
                country,
            ),
        ]
    out = [Question("ooc-1", "What is the capital of Norvania?", ["Eskel"], {}, "Norvania")]
    return QACorpus("synthetic_qa", passages, questions, out, "synthetic")


QA: dict[str, Callable[[int | None, int], QACorpus]] = {
    "squad": lambda n, s: _squad(n, n_held_out_articles=6, seed=s),
    "scifact": _scifact,
    "synthetic_qa": lambda n, s: _synthetic_qa(),
}


def load_qa(name: str, *, n_questions: int | None = None, seed: int = 0) -> QACorpus:
    return QA[name](n_questions, seed)
