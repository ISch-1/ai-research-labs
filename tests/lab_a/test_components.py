"""Unit tests for Lab A building blocks."""

import numpy as np
import pandas as pd
import pytest

from airlab.reporting import fmt_p, md_table
from lab_a import anomaly, clustering, forecasting
from lab_a import models as Z
from lab_a.datasets import load_anomaly, load_series, load_tabular
from lab_a.supervised import _drop_collinear


def test_md_table_and_fmt_p():
    assert md_table(["a", "b"], [[1, 2]]) == "| a | b |\n|---|---|\n| 1 | 2 |"
    assert fmt_p(0.00001) == "< 0.0001"
    assert fmt_p(0.25) == "0.2500"


@pytest.mark.parametrize(
    "name", ["breast_cancer", "diabetes", "synthetic_linear", "synthetic_hastie"]
)
def test_offline_tabular_datasets(name):
    ds = load_tabular(name)
    assert len(ds.X) == len(ds.y) > 100
    if ds.task == "classification":
        assert set(ds.y.unique()) == {0, 1}
        assert ds.y.mean() <= 0.5  # positive class is the minority one


def test_breast_cancer_positive_class_is_malignant():
    ds = load_tabular("breast_cancer")
    assert ds.y.sum() == 212  # 212 malignant, 357 benign


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_every_default_model_builds_and_fits(task):
    ds = load_tabular("breast_cancer" if task == "classification" else "diabetes")
    X = ds.X.assign(cat=np.where(np.arange(len(ds.X)) % 3 == 0, "a", "b"))  # add a categorical
    X.iloc[0, 0] = np.nan  # and a missing value
    for name in Z.default_models(task):
        model = Z.build(name, task, seed=0).fit(X, ds.y)
        assert len(model.predict(X.head(5))) == 5


def test_incompatible_model_task():
    with pytest.raises(ValueError):
        Z.build("logreg", "regression")


def test_drop_collinear():
    rng = np.random.default_rng(0)
    a = rng.normal(size=200)
    X = pd.DataFrame({"a": a, "a_copy": a + rng.normal(0, 0.01, 200), "b": rng.normal(size=200)})
    kept, dropped = _drop_collinear(X)
    assert list(kept.columns) == ["a", "b"]
    assert dropped == ["a_copy"]


# ------------------------------------------------------------------- forecasting


def test_seasonal_naive_repeats_last_season():
    y = np.arange(36, dtype=float) + np.random.default_rng(0).normal(0, 0.5, 36)
    fc = forecasting._seasonal_naive(y, h=14, m=12, z=1.64)
    np.testing.assert_array_equal(fc.mean[:12], y[24:])
    np.testing.assert_array_equal(fc.mean[12:], y[24:26])
    assert np.all(fc.lower < fc.mean) and np.all(fc.upper > fc.mean)


def test_mase_is_one_for_seasonal_naive_on_a_pure_seasonal_random_walk():
    train = np.tile(np.arange(12.0), 5)
    actual = np.arange(12.0) + 1
    fc = forecasting.Forecast(actual - 1)
    metrics = forecasting.forecast_metrics(actual, fc, train + np.repeat(np.arange(5.0), 12), 12)
    assert metrics["mase"] == pytest.approx(1.0)


def test_needs_log_detects_multiplicative_seasonality():
    t = np.arange(120)
    multiplicative = (100 + 2 * t) * (1 + 0.3 * np.sin(2 * np.pi * t / 12))
    additive = 100 + 2 * t + 30 * np.sin(2 * np.pi * t / 12)
    assert forecasting.needs_log(multiplicative, 12)
    assert not forecasting.needs_log(additive, 12)


def test_ml_forecaster_follows_a_linear_trend():
    from sklearn.linear_model import LinearRegression

    y = 10 + 0.5 * np.arange(100, dtype=float)
    fc = forecasting._ml(LinearRegression)(y, 6, 4, 1.64)
    np.testing.assert_allclose(fc.mean, y[-1] + 0.5 * np.arange(1, 7), atol=1e-6)


def test_synthetic_series_offline():
    ts = load_series("synthetic_seasonal")
    assert ts.season_length == 12 and ts.y.index.freqstr == "MS"


# --------------------------------------------------------- clustering / anomalies


def test_dbscan_eps_is_positive_and_finite():
    X = np.random.default_rng(0).normal(size=(200, 3))
    eps = clustering.dbscan_eps(X, 6)
    assert 0 < eps < 10


def test_internal_scores_ignore_noise():
    X = np.r_[np.zeros((10, 2)), np.ones((10, 2)) * 5, [[100, 100]]]
    labels = np.r_[np.zeros(10), np.ones(10), [-1]].astype(int)
    scores = clustering.internal_scores(
        X + np.random.default_rng(0).normal(0, 0.1, X.shape), labels
    )
    assert scores["n_clusters"] == 2
    assert scores["noise_fraction"] == pytest.approx(1 / 21)
    assert scores["silhouette"] > 0.9


def test_precision_at_n():
    y = np.array([0, 0, 1, 1, 0])
    scores = np.array([0.1, 0.9, 0.8, 0.7, 0.2])
    assert anomaly.precision_at_n(y, scores) == 0.5


@pytest.mark.parametrize("detector", list(anomaly.FAMILY))
def test_every_detector_ranks_obvious_outliers_first(detector):
    rng = np.random.default_rng(0)
    # Isolated outliers in different directions (a tight group of outliers would be a
    # small dense cluster, which density-based detectors such as OCSVM legitimately accept).
    outliers = 12 * np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    X = np.r_[rng.normal(0, 1, (300, 3)), outliers]
    scores = anomaly.score(detector, X, seed=0, fit_subsample=1000)
    assert set(np.argsort(scores)[-5:]) == set(range(300, 305))


def test_load_anomaly_subsample_keeps_rate():
    ds = load_anomaly("synthetic_blobs", max_rows=1000, seed=0)
    assert len(ds.y) == 1000
    assert ds.contamination == pytest.approx(0.05)
