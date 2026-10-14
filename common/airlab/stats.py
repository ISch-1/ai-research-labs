"""Statistical tests for comparing models.

Which test to use:

=====================================  ===================================
Situation                              Test
=====================================  ===================================
2 models, same CV folds                ``corrected_resampled_ttest``
                                       (or ``wilcoxon_signed_rank``)
2 classifiers, one shared test set     ``mcnemar``
>2 models across several datasets      ``friedman`` + ``nemenyi_cd``
2 forecasters, same horizon            ``diebold_mariano``
=====================================  ===================================

References: Demšar (2006) JMLR 7; Nadeau & Bengio (2003) Mach. Learn. 52;
Dietterich (1998) Neural Comput. 10; Harvey, Leybourne & Newbold (1997).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy import stats as sps


@dataclass(frozen=True)
class TestResult:
    name: str
    statistic: float
    pvalue: float
    details: dict[str, float] = field(default_factory=dict)

    def significant(self, alpha: float = 0.05) -> bool:
        return self.pvalue < alpha


def corrected_resampled_ttest(
    scores_a: np.ndarray, scores_b: np.ndarray, *, n_train: int, n_test: int
) -> TestResult:
    """Nadeau-Bengio corrected t-test for paired scores from (repeated) k-fold CV.

    The plain paired t-test is overconfident on CV scores because training
    sets overlap; the ``n_test / n_train`` term inflates the variance to
    account for that.
    """
    diff = np.asarray(scores_a, dtype=float) - np.asarray(scores_b, dtype=float)
    j = len(diff)
    if j < 2:
        raise ValueError("need at least two paired scores")
    variance = diff.var(ddof=1)
    if variance == 0:
        # Identical differences on every fold: either no difference at all or a
        # perfectly consistent one.
        same = diff.mean() == 0
        return TestResult(
            "corrected_resampled_ttest", 0.0 if same else np.inf, 1.0 if same else 0.0
        )
    t = diff.mean() / np.sqrt((1 / j + n_test / n_train) * variance)
    p = 2 * sps.t.sf(abs(t), df=j - 1)
    return TestResult(
        "corrected_resampled_ttest", float(t), float(p), {"mean_diff": float(diff.mean())}
    )


def wilcoxon_signed_rank(scores_a: np.ndarray, scores_b: np.ndarray) -> TestResult:
    """Non-parametric paired test; no normality assumption on the differences."""
    result = sps.wilcoxon(scores_a, scores_b)
    diff = np.asarray(scores_a, dtype=float) - np.asarray(scores_b, dtype=float)
    return TestResult(
        "wilcoxon_signed_rank",
        float(result.statistic),
        float(result.pvalue),
        {"median_diff": float(np.median(diff))},
    )


def mcnemar(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> TestResult:
    """McNemar's test on the disagreements of two classifiers on one test set.

    Uses the exact binomial test when there are fewer than 25 discordant
    pairs, otherwise the chi-squared approximation with continuity correction.
    """
    y_true, pred_a, pred_b = map(np.asarray, (y_true, pred_a, pred_b))
    correct_a = pred_a == y_true
    correct_b = pred_b == y_true
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    details = {"a_only_correct": float(b), "b_only_correct": float(c)}

    if b + c == 0:
        return TestResult("mcnemar", 0.0, 1.0, details)
    if b + c < 25:
        p = sps.binomtest(b, b + c, 0.5).pvalue
        return TestResult("mcnemar_exact", float(b), float(p), details)
    statistic = (abs(b - c) - 1) ** 2 / (b + c)
    return TestResult("mcnemar", float(statistic), float(sps.chi2.sf(statistic, 1)), details)


@dataclass(frozen=True)
class FriedmanResult:
    statistic: float
    pvalue: float
    average_ranks: dict[str, float]
    n_datasets: int


def friedman(
    scores: np.ndarray, model_names: list[str], *, higher_is_better: bool = True
) -> FriedmanResult:
    """Friedman test over a (n_datasets, n_models) score matrix.

    Rank 1 is the best model on a dataset; ties get the average rank.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2 or scores.shape[1] != len(model_names):
        raise ValueError("scores must be (n_datasets, n_models) matching model_names")
    if scores.shape[1] < 3:
        raise ValueError("Friedman needs at least 3 models; use a paired test for 2")

    ranks = sps.rankdata(-scores if higher_is_better else scores, axis=1)
    result = sps.friedmanchisquare(*scores.T)
    return FriedmanResult(
        statistic=float(result.statistic),
        pvalue=float(result.pvalue),
        average_ranks=dict(zip(model_names, ranks.mean(axis=0).tolist(), strict=True)),
        n_datasets=scores.shape[0],
    )


# Critical values q_alpha for the two-tailed Nemenyi test (Demšar 2006, Table 5),
# indexed by number of models k = 2..10.
_NEMENYI_Q = {
    0.05: [1.960, 2.343, 2.569, 2.728, 2.850, 2.949, 3.031, 3.102, 3.164],
    0.10: [1.645, 2.052, 2.291, 2.459, 2.589, 2.693, 2.780, 2.855, 2.920],
}


def nemenyi_cd(n_models: int, n_datasets: int, alpha: float = 0.05) -> float:
    """Critical difference: two models differ if their average ranks differ by more."""
    if alpha not in _NEMENYI_Q:
        raise ValueError(f"alpha must be one of {sorted(_NEMENYI_Q)}")
    if not 2 <= n_models <= 10:
        raise ValueError("tabulated for 2..10 models")
    q = _NEMENYI_Q[alpha][n_models - 2]
    return float(q * np.sqrt(n_models * (n_models + 1) / (6 * n_datasets)))


def diebold_mariano(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
    *,
    horizon: int = 1,
    loss: Literal["squared", "absolute"] = "squared",
) -> TestResult:
    """Diebold-Mariano test with the Harvey-Leybourne-Newbold small-sample correction.

    A negative statistic means forecaster A has the lower loss.
    """
    e_a = np.asarray(errors_a, dtype=float)
    e_b = np.asarray(errors_b, dtype=float)
    if e_a.shape != e_b.shape:
        raise ValueError("error series must have the same shape")
    if loss == "squared":
        d = e_a**2 - e_b**2
    elif loss == "absolute":
        d = np.abs(e_a) - np.abs(e_b)
    else:
        raise ValueError("loss must be 'squared' or 'absolute'")

    n = len(d)
    if n <= horizon:
        raise ValueError("series too short for this horizon")
    d_mean = d.mean()
    centered = d - d_mean
    autocov = [np.sum(centered[k:] * centered[: n - k]) / n for k in range(horizon)]
    variance = (autocov[0] + 2 * sum(autocov[1:])) / n
    if variance <= 0:
        return TestResult("diebold_mariano", 0.0, 1.0, {"mean_loss_diff": float(d_mean)})

    dm = d_mean / np.sqrt(variance)
    correction = np.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
    statistic = dm * correction
    p = 2 * sps.t.sf(abs(statistic), df=n - 1)
    return TestResult(
        "diebold_mariano", float(statistic), float(p), {"mean_loss_diff": float(d_mean)}
    )
