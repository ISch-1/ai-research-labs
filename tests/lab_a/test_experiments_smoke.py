"""Every Lab A experiment runs end to end on its offline smoke config."""

from pathlib import Path

import pandas as pd
import pytest

from airlab.config import load_config
from lab_a import anomaly, benchmark, clustering, forecasting, supervised

CONFIGS = Path(__file__).resolve().parents[2] / "lab-a-predictive" / "configs"
RESULT_COLUMNS = ["experiment", "dataset", "model", "fold", "metric", "value"]


def _check_outputs(out_dir: Path, name: str, text: str) -> pd.DataFrame:
    report = out_dir / f"{name}.md"
    assert report.read_text(encoding="utf-8") == text
    for link in [line for line in text.splitlines() if line.startswith("![")]:
        target = link[link.index("(") + 1 : -1]
        assert (out_dir / target).exists(), f"missing figure {target}"
    results = pd.read_csv(out_dir / "results" / f"{name}.csv")
    assert list(results.columns) == RESULT_COLUMNS
    return results


@pytest.mark.parametrize("config", ["a1_smoke.yaml", "a1_smoke_regression.yaml"])
def test_supervised(tmp_path, config):
    cfg = load_config(CONFIGS / config, supervised.SupervisedConfig, output_dir=tmp_path)
    text = supervised.run(cfg)
    results = _check_outputs(tmp_path, cfg.experiment, text)
    assert "### Interpretation" in text
    assert set(results.model) >= set(cfg.models or [])
    # every model must beat the dummy baseline on CV
    primary = supervised.PRIMARY[cfg.task]
    cv = results[(results.fold != "test") & (results.metric == primary)]
    means = cv.groupby("model").value.mean()
    others = means.drop("dummy")
    assert (
        (others > means["dummy"]).all() if primary == "roc_auc" else (others < means["dummy"]).all()
    )


def test_forecasting(tmp_path):
    cfg = load_config(
        CONFIGS / "a2_smoke.yaml",
        forecasting.ForecastConfig,
        output_dir=tmp_path,
        series=["synthetic_seasonal"],
    )
    text = forecasting.run(cfg)
    results = _check_outputs(tmp_path, cfg.experiment, text)
    mase = results[results.metric == "mase"].groupby("model").value.mean()
    # a seasonal series: seasonal models must beat the non-seasonal naive forecast
    assert mase["ets"] < mase["naive"]
    assert mase["seasonal_naive"] < mase["naive"]


def test_clustering(tmp_path):
    cfg = load_config(CONFIGS / "a3_smoke.yaml", clustering.ClusteringConfig, output_dir=tmp_path)
    text = clustering.run(cfg)
    results = _check_outputs(tmp_path, cfg.experiment, text)
    ext = results[(results.metric == "external_ari") & (results.model == "kmeans / log_standard")]
    assert ext.value.iloc[0] > 0.95  # recovers the planted clusters


def test_anomaly(tmp_path):
    cfg = load_config(CONFIGS / "a3_smoke_anomaly.yaml", anomaly.AnomalyConfig, output_dir=tmp_path)
    text = anomaly.run(cfg)
    results = _check_outputs(tmp_path, cfg.experiment, text)
    pr = results[results.metric == "pr_auc"].groupby("model").value.mean()
    assert pr["isolation_forest"] > 0.8  # far above the 5% random baseline


def test_benchmark_and_cross_experiment_analysis(tmp_path):
    # Produce an A3b results file first so the comparison section has something to read.
    anomaly.run(
        load_config(CONFIGS / "a3_smoke_anomaly.yaml", anomaly.AnomalyConfig, output_dir=tmp_path)
    )
    cfg = load_config(
        CONFIGS / "a4_smoke.yaml",
        benchmark.BenchmarkConfig,
        output_dir=tmp_path,
        compare=[
            {
                "title": "anomaly",
                "results": "results/a3_smoke_anomaly.csv",
                "metric": "pr_auc",
                "exclude_models": [],
            },
            {"title": "missing", "results": "results/nope.csv", "metric": "x"},
        ],
    )
    text = benchmark.run(cfg)
    _check_outputs(tmp_path, cfg.experiment, text)
    assert "Friedman" in text
    assert "_Skipped: `results/nope.csv` not found._" in text
