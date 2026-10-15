"""Real Hugging Face models, kept tiny so CI can download them (marker: ``hf``)."""

import numpy as np
import pytest

pytestmark = pytest.mark.hf

TINY_CHAT = "HuggingFaceTB/SmolLM2-135M-Instruct"
SENTENCE = "st:sentence-transformers/all-MiniLM-L6-v2"


@pytest.fixture(scope="module")
def llm():
    from lab_b.llm import HFChatLLM

    return HFChatLLM(TINY_CHAT)


def test_generate_reports_tokens_and_timing(llm):
    g = llm.generate([{"role": "user", "content": "Say hello."}], max_new_tokens=8)
    assert g.text
    assert g.prompt_tokens > 0 and 0 < g.completion_tokens <= 8
    assert 0 < g.ttft_s <= g.latency_s


def test_prefix_cache_gives_the_same_scores_as_the_full_prompt(llm):
    prefix = [
        {"role": "system", "content": "Answer with one word: Yes or No."},
        {"role": "user", "content": "Is water wet?"},
        {"role": "assistant", "content": "Yes"},
    ]
    query = [{"role": "user", "content": "Is fire cold?"}]
    choices = ["Yes", "No", "Maybe not"]  # includes a multi-token label
    cached = llm.choice_logprobs(prefix, query, choices)
    full = llm.choice_logprobs([], prefix + query, choices)
    np.testing.assert_allclose(cached, full, atol=1e-3)


def test_sentence_embeddings_are_normalized_and_cached(tmp_path, monkeypatch):
    monkeypatch.setenv("AIRLAB_DATA_DIR", str(tmp_path))
    from lab_b.encoders import Encoder

    texts = ["a cat sits on the mat", "stock markets fell sharply"]
    first = Encoder(SENTENCE).fit(texts).transform(texts)
    np.testing.assert_allclose(np.linalg.norm(first, axis=1), 1.0, atol=1e-5)
    assert list((tmp_path / "cache" / "embeddings").glob("*.npy"))
    np.testing.assert_array_equal(Encoder(SENTENCE).fit(texts).transform(texts), first)
