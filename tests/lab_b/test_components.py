"""Unit tests for Lab B building blocks (offline)."""

import numpy as np
import pytest

from lab_b import retrieval as R
from lab_b.datasets import load_classification, load_qa
from lab_b.encoders import Encoder
from lab_b.llm import FakeLLM
from lab_b.rag import GenerationCache, groundedness, messages_for
from lab_b.text import contains_answer, exact_match, is_abstention, normalize_answer, token_f1

# ----------------------------------------------------------------------------- text


def test_squad_normalization_and_metrics():
    assert normalize_answer("The  Eiffel Tower!") == "eiffel tower"
    assert exact_match("the Eiffel tower", ["Eiffel Tower"]) == 1.0
    assert token_f1("Eiffel Tower in Paris", ["Eiffel Tower"]) == pytest.approx(2 * 0.5 * 1 / 1.5)
    assert token_f1("nothing", ["Eiffel Tower"]) == 0.0
    assert token_f1("x", ["a", "x"]) == 1.0  # best over references
    assert contains_answer("It was completed in 1889.", ["1889"]) == 1.0


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("I don't know", True),
        ("The context does not mention it; I do not know.", True),
        ("The answer is not provided in the context.", True),
        ("", True),
        ("Denver Broncos", False),
        ("Knowledge is power", False),
    ],
)
def test_abstention_detection(text, expected):
    assert is_abstention(text) is expected


# ------------------------------------------------------------------------- datasets


def test_synthetic_classification_is_offline_and_stratified():
    ds = load_classification("synthetic_topics", max_train=300, max_test=90, seed=1)
    assert len(ds.train_texts) == 300 and len(ds.test_texts) == 90
    assert set(np.unique(ds.train_labels)) == {0, 1, 2}


def test_synthetic_qa_corpus():
    qa = load_qa("synthetic_qa")
    assert len(qa.passages) == 8 and len(qa.questions) == 24
    ids = {p.id for p in qa.passages}
    assert all(set(q.relevant) <= ids for q in qa.questions)
    assert qa.out_of_corpus and not qa.out_of_corpus[0].relevant


# ------------------------------------------------------------------------- encoders


@pytest.mark.parametrize("spec", ["bow", "tfidf", "hashing"])
def test_sparse_encoders(spec):
    texts = ["the cat sat", "the dog ran", "a cat and a dog", "the cat ran"]
    enc = Encoder(spec).fit(texts)
    X = enc.transform(texts)
    assert X.shape[0] == 4 and X.nnz > 0
    assert not enc.is_dense


def test_unknown_encoder():
    with pytest.raises(ValueError):
        Encoder("word2vec").fit(["x"])


# ------------------------------------------------------------------------ retrieval


def test_bm25_matches_hand_computation():
    docs = ["apple apple banana", "banana cherry", "cherry cherry cherry date"]
    bm25 = R.BM25().fit(docs)
    idx, scores = bm25.search(["apple"], 3)
    # only doc 0 contains "apple": idf = ln(1 + (3 - 1 + .5) / (1 + .5)); tf = 2; dl = 3; avgdl = 3
    idf = np.log(1 + 2.5 / 1.5)
    expected = idf * 2 * 2.5 / (2 + 1.5 * (1 - 0.75 + 0.75 * 3 / 3))
    assert idx[0, 0] == 0
    assert scores[0, 0] == pytest.approx(expected, rel=1e-5)
    assert scores[0, 1] == 0.0


def test_bm25_length_normalization_prefers_shorter_documents():
    docs = ["cherry date", "cherry " + "filler " * 30]
    idx, _ = R.BM25().fit(docs).search(["cherry"], 2)
    assert idx[0, 0] == 0


@pytest.mark.parametrize("spec", ["bm25", "tfidf", "dense:hashing", "hybrid:bm25+tfidf"])
def test_retrievers_find_the_right_passage(spec):
    qa = load_qa("synthetic_qa")
    retriever = R.build(spec).fit(qa.texts)
    idx, _ = retriever.search([q.text for q in qa.questions], 3)
    pos = {p.id: i for i, p in enumerate(qa.passages)}
    rel = [{pos[d]: 1 for d in q.relevant} for q in qa.questions]
    assert R.ir_metrics(idx, rel)["recall@1"].mean() >= 0.9


def test_rrf_fusion_rewards_agreement():
    class Fixed(R.Retriever):
        def __init__(self, order):
            self.order, self.spec = order, "fixed"

        def fit(self, texts):
            return self

        def search(self, queries, k):
            return np.array([self.order[:k]]), np.zeros((1, k))

    hybrid = R.HybridRRF(Fixed([0, 1, 2]), Fixed([1, 2, 0])).fit(["a", "b", "c"])
    idx, _ = hybrid.search(["q"], 3)
    assert idx[0, 0] == 1  # ranked 2nd and 1st beats ranked 1st and 3rd


def test_ir_metrics_known_values():
    retrieved = np.array([[3, 1, 2], [5, 6, 7]])
    relevant = [{1: 1}, {9: 1}]
    m = R.ir_metrics(retrieved, relevant, ks=(1, 3))
    assert m["recall@1"].tolist() == [0.0, 0.0]
    assert m["recall@3"].tolist() == [1.0, 0.0]
    assert m["mrr@10"].tolist() == [0.5, 0.0]
    assert m["ndcg@10"][0] == pytest.approx(1 / np.log2(3))


def test_short_names():
    assert (
        R.short_name("rerank:hybrid:bm25+dense:st:BAAI/bge-small-en-v1.5")
        == "rerank(bm25+bge-small)"
    )
    assert R.short_name("dense:st:sentence-transformers/all-MiniLM-L6-v2") == "MiniLM-L6-v2"


# ------------------------------------------------------------------------------ RAG


def test_fake_llm_reads_context_and_follows_injections():
    qa = load_qa("synthetic_qa")
    q = qa.questions[0]  # capital of Velmora -> Ostrin
    llm = FakeLLM()
    answer = llm.generate(messages_for(q, qa.passages[:2], "sys")).text
    assert contains_answer(answer, q.answers)
    from lab_b.datasets import Passage
    from lab_b.rag import INJECTION

    injected = [qa.passages[0], Passage("x", "x", INJECTION)]
    assert llm.generate(messages_for(q, injected, "sys")).text == "BANANA"


def test_groundedness():
    qa = load_qa("synthetic_qa")
    assert groundedness("Ostrin", qa.passages[:1]) == 1.0
    assert groundedness("Paris", qa.passages[:1]) == 0.0
    assert np.isnan(groundedness("I don't know", qa.passages[:1]))
    assert np.isnan(groundedness("Ostrin", None))


def test_generation_cache_resumes(tmp_path):
    class Counting(FakeLLM):
        calls = 0

        def generate(self, messages, max_new_tokens=48):
            Counting.calls += 1
            return super().generate(messages, max_new_tokens)

    msgs = [{"role": "user", "content": "Context: Ostrin is the capital. Question: capital?"}]
    first = GenerationCache("counting").get_or_generate(Counting(), msgs, 16)
    again = GenerationCache("counting").get_or_generate(
        Counting(), msgs, 16
    )  # new instance reads the file
    assert Counting.calls == 1
    assert first == again


def test_generation_cache_survives_a_truncated_last_line():
    msgs = [{"role": "user", "content": "Context: Ostrin is the capital. Question: capital?"}]
    cache = GenerationCache("truncated")
    cache.get_or_generate(FakeLLM(), msgs, 16)
    with cache.path.open("a", encoding="utf-8") as fh:
        fh.write('{"key": "abc", "text": "cut sh')  # process killed mid-write
    reloaded = GenerationCache("truncated")
    assert len(reloaded._items) == 1
