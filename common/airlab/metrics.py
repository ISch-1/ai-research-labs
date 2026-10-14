"""Metric reports with bootstrap confidence intervals.

Every number in a report should come with its uncertainty: a model that is
0.4 points better but whose 95% CI overlaps the baseline has not won anything.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from sklearn import metrics as skm

MetricFn = Callable[[np.ndarray, np.ndarray], float]


@dataclass(frozen=True)
class MetricCI:
    value: float
    low: float
    high: float
    confidence: float = 0.95

    def __str__(self) -> str:
        return f"{self.value:.4f} [{self.low:.4f}, {self.high:.4f}]"


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: MetricFn,
    *,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> MetricCI:
    """Percentile bootstrap CI of ``metric(y_true, y_pred)`` resampling test rows.

    Resamples where the metric is undefined (e.g. ROC-AUC with a single class),
    whether it raises or returns NaN, are skipped instead of spoiling the interval.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")

    generator = np.random.default_rng(seed)
    n = len(y_true)
    stats = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # undefined-metric warnings are handled below
        for _ in range(n_resamples):
            idx = generator.integers(0, n, size=n)
            try:
                value = metric(y_true[idx], y_pred[idx])
            except ValueError:
                continue
            if np.isfinite(value):
                stats.append(value)
    if not stats:
        raise ValueError("metric was undefined on every bootstrap resample")

    alpha = (1 - confidence) / 2
    low, high = np.quantile(stats, [alpha, 1 - alpha])
    return MetricCI(float(metric(y_true, y_pred)), float(low), float(high), confidence)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(skm.mean_squared_error(y_true, y_pred)))


REGRESSION_METRICS: dict[str, MetricFn] = {
    "rmse": rmse,
    "mae": skm.mean_absolute_error,
    "r2": skm.r2_score,
}

CLASSIFICATION_METRICS: dict[str, MetricFn] = {
    "accuracy": skm.accuracy_score,
    "balanced_accuracy": skm.balanced_accuracy_score,
    "f1_macro": lambda t, p: skm.f1_score(t, p, average="macro"),
}

# Metrics that need scores/probabilities instead of hard labels (binary case).
SCORE_METRICS: dict[str, MetricFn] = {
    "roc_auc": skm.roc_auc_score,
    "pr_auc": skm.average_precision_score,
    "brier": skm.brier_score_loss,
}


def report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: dict[str, MetricFn],
    *,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, MetricCI]:
    """Compute each metric with a bootstrap CI."""
    return {
        name: bootstrap_ci(
            y_true, y_pred, fn, n_resamples=n_resamples, confidence=confidence, seed=seed
        )
        for name, fn in metrics.items()
    }
