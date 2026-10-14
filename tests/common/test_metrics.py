import numpy as np
import pytest
from sklearn.metrics import accuracy_score, roc_auc_score

from airlab.metrics import CLASSIFICATION_METRICS, bootstrap_ci, report, rmse


def test_bootstrap_ci_contains_point_estimate_and_is_reproducible():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    pred = np.where(rng.random(500) < 0.8, y, 1 - y)

    ci = bootstrap_ci(y, pred, accuracy_score, n_resamples=500, seed=7)
    again = bootstrap_ci(y, pred, accuracy_score, n_resamples=500, seed=7)

    assert ci.low <= ci.value <= ci.high
    assert ci == again
    assert 0.75 < ci.value < 0.85


def _fast_accuracy(y_true, y_pred):
    # Same as accuracy_score without sklearn's input validation (60k calls below).
    return float(np.mean(y_true == y_pred))


def test_bootstrap_ci_covers_true_accuracy_most_of_the_time():
    """The 95% CI should contain the population accuracy in roughly 95% of samples."""
    rng = np.random.default_rng(123)
    true_acc, hits, trials = 0.7, 0, 200
    for i in range(trials):
        y = np.zeros(300, dtype=int)
        pred = (rng.random(300) > true_acc).astype(int)
        ci = bootstrap_ci(y, pred, _fast_accuracy, n_resamples=300, seed=i)
        hits += ci.low <= true_acc <= ci.high
    assert 0.88 <= hits / trials <= 0.99


def test_bootstrap_skips_undefined_resamples():
    y = np.array([0] * 98 + [1] * 2)
    scores = np.linspace(0, 1, 100)
    ci = bootstrap_ci(y, scores, roc_auc_score, n_resamples=200, seed=0)
    assert 0.0 <= ci.low <= ci.high <= 1.0


def test_bootstrap_validates_inputs():
    with pytest.raises(ValueError):
        bootstrap_ci(np.zeros(3), np.zeros(4), accuracy_score)
    with pytest.raises(ValueError):
        bootstrap_ci(np.zeros(3), np.zeros(3), accuracy_score, confidence=1.5)


def test_rmse():
    assert rmse(np.array([0.0, 0.0]), np.array([3.0, 4.0])) == pytest.approx(np.sqrt(12.5))


def test_report_has_every_metric():
    y = np.array([0, 1, 1, 0, 1, 0, 1, 1])
    result = report(y, y, CLASSIFICATION_METRICS, n_resamples=50)
    assert set(result) == set(CLASSIFICATION_METRICS)
    assert all(ci.value == 1.0 for ci in result.values())
