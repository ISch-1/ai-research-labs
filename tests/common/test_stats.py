import numpy as np
import pytest
from scipy import stats as sps

from airlab import stats


def test_corrected_ttest_is_more_conservative_than_plain_paired_ttest():
    rng = np.random.default_rng(0)
    a = 0.90 + rng.normal(0, 0.01, 30)
    b = a - 0.005 + rng.normal(0, 0.01, 30)

    corrected = stats.corrected_resampled_ttest(a, b, n_train=900, n_test=100)
    plain = sps.ttest_rel(a, b)

    assert corrected.pvalue > plain.pvalue
    assert np.sign(corrected.statistic) == np.sign(plain.statistic)


def test_corrected_ttest_identical_scores():
    a = np.full(10, 0.8)
    result = stats.corrected_resampled_ttest(a, a, n_train=90, n_test=10)
    assert result.pvalue == 1.0


def test_mcnemar_exact_matches_binomial():
    y = np.array([1] * 20)
    a = np.array([1] * 15 + [0] * 5)  # A misses the last 5
    b = np.array([0] * 3 + [1] * 17)  # B misses the first 3
    result = stats.mcnemar(y, a, b)
    # discordant: A-only correct = 3, B-only correct = 5
    assert result.details == {"a_only_correct": 3.0, "b_only_correct": 5.0}
    assert result.pvalue == pytest.approx(sps.binomtest(3, 8, 0.5).pvalue)


def test_mcnemar_detects_clear_difference_with_chi2():
    y = np.ones(200, dtype=int)
    a = np.ones(200, dtype=int)
    b = np.r_[np.zeros(60, dtype=int), np.ones(140, dtype=int)]
    result = stats.mcnemar(y, a, b)
    assert result.name == "mcnemar"
    assert result.significant()


def test_mcnemar_no_disagreement():
    y = np.array([0, 1, 1])
    assert stats.mcnemar(y, y, y).pvalue == 1.0


def test_friedman_ranks_and_scipy_agreement():
    scores = np.array(
        [
            [0.90, 0.85, 0.70],
            [0.88, 0.86, 0.72],
            [0.91, 0.80, 0.75],
            [0.87, 0.89, 0.71],
        ]
    )
    result = stats.friedman(scores, ["a", "b", "c"])
    assert result.average_ranks == {"a": 1.25, "b": 1.75, "c": 3.0}
    assert result.pvalue == pytest.approx(sps.friedmanchisquare(*scores.T).pvalue)


def test_friedman_lower_is_better_flips_ranks():
    scores = np.array([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5], [0.5, 2.0, 4.0]])
    result = stats.friedman(scores, ["a", "b", "c"], higher_is_better=False)
    assert result.average_ranks["a"] == 1.0


def test_friedman_requires_three_models():
    with pytest.raises(ValueError):
        stats.friedman(np.ones((5, 2)), ["a", "b"])


def test_nemenyi_cd_matches_demsar_example():
    # Demšar (2006), section 3.2.4: k=4 classifiers, N=14 datasets -> CD = 1.25
    assert stats.nemenyi_cd(4, 14) == pytest.approx(1.25, abs=0.01)


def test_diebold_mariano_prefers_smaller_errors():
    rng = np.random.default_rng(1)
    good = rng.normal(0, 1.0, 300)
    bad = rng.normal(0, 2.0, 300)
    result = stats.diebold_mariano(good, bad, horizon=1)
    assert result.statistic < 0
    assert result.significant()


def test_diebold_mariano_equal_forecasts_not_significant():
    rng = np.random.default_rng(2)
    e1 = rng.normal(0, 1.0, 300)
    e2 = rng.normal(0, 1.0, 300)
    assert not stats.diebold_mariano(e1, e2, horizon=3).significant()


def test_diebold_mariano_long_horizon_falls_back_to_bartlett():
    # 24 errors at h=12: the rectangular long-run variance is negative here, which
    # used to return p = 1 even though model A is far better.
    rng = np.random.default_rng(3)
    good = rng.normal(0, 0.3, 24)
    bad = 5 + np.sin(np.arange(24) / 2) + rng.normal(0, 0.3, 24)
    result = stats.diebold_mariano(good, bad, horizon=12)
    assert result.statistic < 0
    assert result.significant()


def test_holm_matches_statsmodels():
    from statsmodels.stats.multitest import multipletests

    raw = {"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.005, "e": 0.2}
    expected = multipletests(list(raw.values()), method="holm")[1]
    adjusted = stats.holm(raw)
    assert [adjusted[k] for k in raw] == pytest.approx(expected)
