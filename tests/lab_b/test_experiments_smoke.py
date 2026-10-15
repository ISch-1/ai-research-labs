"""Every Lab B experiment runs end to end on its offline smoke config."""

from pathlib import Path

import pandas as pd
import pytest

from airlab.config import load_config
from lab_b import classification, rag, representations, search

CONFIGS = Path(__file__).resolve().parents[2] / "lab-b-nlp-llm" / "configs"


def _check(out_dir: Path, name: str, text: str) -> pd.DataFrame:
    assert (out_dir / f"{name}.md").read_text(encoding="utf-8") == text
    for line in text.splitlines():
        if line.startswith("!["):
            assert (out_dir / line[line.index("(") + 1 : -1]).exists()
    return pd.read_csv(out_dir / "results" / f"{name}.csv")


def test_representations(tmp_path):
    cfg = load_config(
        CONFIGS / "b1_smoke.yaml", representations.RepresentationConfig, output_dir=tmp_path
    )
    results = _check(tmp_path, cfg.experiment, representations.run(cfg))
    acc = results[results.metric == "accuracy"].set_index("model").value
    assert (acc > 0.9).all()  # the synthetic topics are keyword-driven


def test_classification(tmp_path):
    cfg = load_config(
        CONFIGS / "b2_smoke.yaml", classification.ClassificationConfig, output_dir=tmp_path
    )
    text = classification.run(cfg)
    results = _check(tmp_path, cfg.experiment, text)
    acc = results[results.metric == "accuracy"]
    tfidf = acc[acc.model == "tfidf"].set_index("fold").value
    assert tfidf.index.astype(int).tolist() == [30, 120, 480]
    few = acc[acc.model == "few:fake"].value.iloc[0]
    zero = acc[acc.model == "zero:fake"].value.iloc[0]
    assert few > zero  # the fake LLM learns keywords from the few-shot examples
    assert "Examples needed" in text


def test_search(tmp_path):
    cfg = load_config(CONFIGS / "b3_smoke.yaml", search.SearchConfig, output_dir=tmp_path)
    text = search.run(cfg)
    results = _check(tmp_path, cfg.experiment, text)
    ndcg = results[results.metric == "ndcg@10"].groupby("model").value.mean()
    assert ndcg["bm25"] > 0.9
    assert "chunk size" in text


def test_rag(tmp_path, monkeypatch):
    cfg = load_config(CONFIGS / "b4_smoke.yaml", rag.RagConfig, output_dir=tmp_path)
    text = rag.run(cfg)
    _check(tmp_path, cfg.experiment, text)
    answers = pd.read_csv(tmp_path / "results" / f"{cfg.experiment}_answers.csv")
    grid = answers[answers.config.isin(rag.DEFAULT_GRID)].groupby("config").contains.mean()
    assert grid["oracle"] > grid["closed"]  # context helps the extractive fake reader
    injection = answers[answers.config.str.startswith("injection / instruction")]
    assert injection.attacked.mean() == pytest.approx(1.0)  # the fake model obeys instructions
    attacks = {c.split(" / ")[1] for c in answers.config if c.startswith("injection")}
    assert attacks == {"instruction", "poisoning"}
    for section in ("Answer quality", "Robustness", "Cost, latency and quality", "LLM-as-judge"):
        assert section in text


def test_classification_report_rebuilds_from_cache_without_models(tmp_path, monkeypatch):
    """What the CI `report` job does: every prediction comes from the shards' caches."""
    from lab_b import llm as llm_module

    cfg = load_config(
        CONFIGS / "b2_smoke.yaml", classification.ClassificationConfig, output_dir=tmp_path / "a"
    )
    first = classification.run(cfg)

    def no_training(*args, **kwargs):
        raise AssertionError("a cached run must not train or load anything")

    monkeypatch.setattr(classification, "fit_tfidf", no_training)
    monkeypatch.setattr(classification, "fit_embed", no_training)
    monkeypatch.setattr(llm_module, "FakeLLM", no_training)  # lazy LLMs are never loaded
    again = classification.run(cfg.model_copy(update={"output_dir": tmp_path / "b"}))
    assert again == first
    assert "Timings measured on:" in again
