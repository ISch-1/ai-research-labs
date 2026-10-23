"""A2 — Time-series forecasting: baselines vs. statistical models vs. ML vs. neural networks.

    python -m lab_a.forecasting --config lab-a-predictive/configs/a2_forecasting.yaml

Evaluation is rolling-origin backtesting (never shuffled): the model is refit on
everything before each origin and forecasts the next ``h`` points.

Models
------
* baselines   — ``naive``, ``seasonal_naive``, ``drift``
* statistical — ``ets`` (state-space exponential smoothing, AIC-selected),
                ``sarima`` (orders selected by AIC on the first origin)
* ML          — ``lightgbm`` on lagged differences, recursive multi-step
* neural      — ``mlp`` on the same features

Metrics: MAE, RMSE, sMAPE, MASE (scaled by the in-sample seasonal-naive error)
and empirical coverage of the prediction intervals where a model provides them.
"""

from __future__ import annotations

import itertools
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from pydantic import Field
from scipy import stats as sps
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from airlab.plots import plt, save
from airlab.reporting import fmt_p, md_table, write_results
from airlab.seed import set_seed
from airlab.stats import diebold_mariano, friedman
from airlab.tracking import track_run
from lab_a.common import LabAConfig, Outputs, cli
from lab_a.datasets import TimeSeries, load_series

FAMILY = {
    "naive": "baseline",
    "seasonal_naive": "baseline",
    "drift": "baseline",
    "ets": "statistical",
    "sarima": "statistical",
    "lightgbm": "classical ML",
    "mlp": "neural network",
}


class ForecastConfig(LabAConfig):
    series: list[str]
    models: list[str] = Field(default_factory=lambda: list(FAMILY))
    horizon: dict[int, int] = Field(default_factory=lambda: {12: 12, 4: 8})  # season -> h
    n_origins: int = 4
    interval_level: float = 0.9
    sarima_max_order: int = 2


@dataclass
class Forecast:
    mean: np.ndarray
    lower: np.ndarray | None = None
    upper: np.ndarray | None = None


# ---------------------------------------------------------------------- transforms


def needs_log(y: np.ndarray, m: int) -> bool:
    """Multiplicative seasonality heuristic: the seasonal spread grows with the level."""
    if np.any(y <= 0) or len(y) < 3 * m:
        return False
    frame = pd.Series(y)
    level = frame.rolling(m).mean().dropna()
    spread = frame.rolling(m).std().dropna()
    return bool(np.corrcoef(level, spread)[0, 1] > 0.6)


# ----------------------------------------------------------------------- baselines


def _naive(y: np.ndarray, h: int, m: int, z: float) -> Forecast:
    sigma = np.std(np.diff(y), ddof=1)
    steps = np.arange(1, h + 1)
    mean = np.full(h, y[-1])
    return Forecast(mean, mean - z * sigma * np.sqrt(steps), mean + z * sigma * np.sqrt(steps))


def _seasonal_naive(y: np.ndarray, h: int, m: int, z: float) -> Forecast:
    mean = np.array([y[len(y) - m + (k % m)] for k in range(h)])
    sigma = np.std(y[m:] - y[:-m], ddof=1)
    k = np.arange(h) // m + 1  # number of full seasons ahead
    return Forecast(mean, mean - z * sigma * np.sqrt(k), mean + z * sigma * np.sqrt(k))


def _drift(y: np.ndarray, h: int, m: int, z: float) -> Forecast:
    slope = (y[-1] - y[0]) / (len(y) - 1)
    steps = np.arange(1, h + 1)
    mean = y[-1] + slope * steps
    sigma = np.std(np.diff(y) - slope, ddof=1)
    width = z * sigma * np.sqrt(steps * (1 + steps / (len(y) - 1)))
    return Forecast(mean, mean - width, mean + width)


# --------------------------------------------------------------------- statistical


ETS_CANDIDATES = [
    {"error": "add", "trend": trend, "damped_trend": damped, "seasonal": seasonal}
    for trend, damped in [(None, False), ("add", False), ("add", True)]
    for seasonal in ("add", "mul")
]


def fit_ets(y: pd.Series, m: int) -> tuple[Any, dict[str, Any]]:
    """Fit every ETS candidate and return the AIC-best fit with its specification.

    Multiplicative seasonality is only tried on strictly positive series. Pass a
    Series (not an ndarray): that avoids a statsmodels 0.15 bug in get_prediction.
    """
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel

    positive = bool((y > 0).all())
    best, best_aic, best_spec = None, np.inf, None
    for spec in ETS_CANDIDATES:
        if spec["seasonal"] == "mul" and not positive:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = ETSModel(y, seasonal_periods=m, **spec).fit(disp=False, maxiter=2000)
        except Exception:
            continue
        if np.isfinite(fit.aic) and fit.aic < best_aic:
            best, best_aic, best_spec = fit, fit.aic, spec
    if best is None or best_spec is None:
        raise RuntimeError("no ETS specification could be fitted")
    return best, best_spec


def _ets(y: np.ndarray, h: int, m: int, z: float) -> Forecast:
    best, _ = fit_ets(pd.Series(y), m)
    alpha = 2 * sps.norm.sf(z)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frame = best.get_prediction(start=len(y), end=len(y) + h - 1).summary_frame(alpha=alpha)
    return Forecast(
        frame["mean"].to_numpy(), frame["pi_lower"].to_numpy(), frame["pi_upper"].to_numpy()
    )


def select_sarima_order(y: np.ndarray, m: int, max_order: int) -> tuple[tuple, tuple]:
    """Grid search over (p, q, P, Q) by AIC; d and D fixed by unit-root tests."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.stattools import kpss

    D = 1
    seasonal_diff = y[m:] - y[:-m]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = int(kpss(seasonal_diff, regression="c", nlags="auto")[1] < 0.05)
    best, best_aic = ((1, d, 1), (0, D, 1, m)), np.inf
    for p, q, P, Q in itertools.product(range(max_order + 1), range(max_order + 1), (0, 1), (0, 1)):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = SARIMAX(y, order=(p, d, q), seasonal_order=(P, D, Q, m)).fit(disp=False)
        except Exception:
            continue
        if np.isfinite(fit.aic) and fit.aic < best_aic:
            best, best_aic = ((p, d, q), (P, D, Q, m)), fit.aic
    return best


def _sarima(order: tuple[tuple, tuple]) -> Callable[[np.ndarray, int, int, float], Forecast]:
    def forecast(y: np.ndarray, h: int, m: int, z: float) -> Forecast:
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = SARIMAX(y, order=order[0], seasonal_order=order[1]).fit(disp=False)
            pred = fit.get_forecast(h)
        ci = pred.conf_int(alpha=2 * sps.norm.sf(z))
        return Forecast(np.asarray(pred.predicted_mean), ci[:, 0], ci[:, 1])

    return forecast


# ------------------------------------------------------------------------ ML / NN


def _features(diffs: np.ndarray, t: int, period: int, n_lags: int, phase: int) -> np.ndarray:
    """Lags of the differenced series + seasonal position as sin/cos."""
    lags = diffs[t - n_lags : t][::-1]
    angle = 2 * np.pi * ((phase + t) % period) / period
    return np.r_[lags, np.sin(angle), np.cos(angle)]


def _ml(make: Callable[[], Any]) -> Callable[[np.ndarray, int, int, float], Forecast]:
    """Recursive multi-step forecaster on first differences (trees cannot extrapolate levels)."""

    def forecast(y: np.ndarray, h: int, m: int, z: float) -> Forecast:
        diffs = np.diff(y)
        n_lags = m + 1
        X = np.array([_features(diffs, t, m, n_lags, 1) for t in range(n_lags, len(diffs))])
        target = diffs[n_lags:]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = make().fit(X, target)
        history = list(diffs)
        for _ in range(h):
            t = len(history)
            x = _features(np.asarray(history), t, m, n_lags, 1)
            history.append(float(model.predict(x[None, :])[0]))
        return Forecast(y[-1] + np.cumsum(history[len(diffs) :]))

    return forecast


def _model_fns(seed: int) -> dict[str, Callable[[np.ndarray, int, int, float], Forecast]]:
    return {
        "naive": _naive,
        "seasonal_naive": _seasonal_naive,
        "drift": _drift,
        "ets": _ets,
        "lightgbm": _ml(
            lambda: LGBMRegressor(
                n_estimators=200,
                learning_rate=0.05,
                num_leaves=8,
                min_child_samples=5,
                random_state=seed,
                verbose=-1,
                n_jobs=1,
            )
        ),
        "mlp": _ml(
            lambda: make_pipeline(
                StandardScaler(),
                MLPRegressor(
                    hidden_layer_sizes=(32, 16),
                    alpha=1e-2,
                    max_iter=3000,
                    random_state=seed,
                ),
            )
        ),
    }


# ------------------------------------------------------------------------- metrics


def forecast_metrics(
    actual: np.ndarray, fc: Forecast, train: np.ndarray, m: int
) -> dict[str, float]:
    err = actual - fc.mean
    scale = np.mean(np.abs(train[m:] - train[:-m]))
    out = {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "smape": float(100 * np.mean(2 * np.abs(err) / (np.abs(actual) + np.abs(fc.mean)))),
        "mase": float(np.mean(np.abs(err)) / scale),
    }
    if fc.lower is not None and fc.upper is not None:
        out["coverage"] = float(np.mean((actual >= fc.lower) & (actual <= fc.upper)))
    return out


# --------------------------------------------------------------------- experiment


def diagnostics(ts: TimeSeries, out: Outputs) -> tuple[list[str], str]:
    """Stationarity tests and STL / ACF plots; returns report lines + figure link."""
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from statsmodels.tsa.seasonal import STL
    from statsmodels.tsa.stattools import adfuller, kpss

    y, m = ts.y, ts.season_length
    rows = []
    variants = {
        "level": y,
        "first difference": y.diff().dropna(),
        f"seasonal difference (lag {m})": y.diff(m).dropna(),
    }
    for label, series in variants.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adf_p = adfuller(series, autolag="AIC")[1]
            kpss_p = kpss(series, regression="c", nlags="auto")[1]
        stationary = adf_p < 0.05 and kpss_p >= 0.05
        rows.append([label, fmt_p(adf_p), f"{kpss_p:.3f}", "yes" if stationary else "no"])

    stl = STL(np.log(y) if needs_log(y.to_numpy(), m) else y, period=m, robust=True).fit()
    strength_trend = max(0.0, 1 - np.var(stl.resid) / np.var(stl.trend + stl.resid))
    strength_season = max(0.0, 1 - np.var(stl.resid) / np.var(stl.seasonal + stl.resid))

    fig, axes = plt.subplots(3, 2, figsize=(11, 7))
    for ax, (label, comp) in zip(
        axes[:, 0],
        [("observed", y), ("trend", stl.trend), ("seasonal", stl.seasonal)],
        strict=True,
    ):
        ax.plot(comp.index if hasattr(comp, "index") else y.index, comp)
        ax.set_title(label, fontsize=9)
    plot_acf(y.diff(m).dropna(), ax=axes[0, 1], lags=3 * m, title="ACF (seasonal difference)")
    plot_pacf(y.diff(m).dropna(), ax=axes[1, 1], lags=3 * m, title="PACF (seasonal difference)")
    axes[2, 1].plot(y.index, stl.resid)
    axes[2, 1].set_title("STL remainder", fontsize=9)
    fig.suptitle(ts.name)
    fig.tight_layout()
    path, link = out.figure(f"{ts.name}_diagnostics.png")
    save(fig, path)

    lines = [
        md_table(
            ["Series", "ADF p (H0: unit root)", "KPSS p (H0: stationary)", "Stationary?"], rows
        ),
        "",
        f"STL strength — trend {strength_trend:.2f}, seasonality {strength_season:.2f} "
        f"(0 = none, 1 = dominant).",
        "",
    ]
    return lines, link


def run_series(ts: TimeSeries, cfg: ForecastConfig, out: Outputs, tracker) -> tuple[str, list]:
    m = ts.season_length
    h = cfg.horizon[m]
    y_all = ts.y.to_numpy(float)
    use_log = needs_log(y_all, m)
    z = sps.norm.ppf(0.5 + cfg.interval_level / 2)
    origins = [len(y_all) - h * (cfg.n_origins - i) for i in range(cfg.n_origins)]
    fns = _model_fns(cfg.seed)

    def transform(v: np.ndarray) -> np.ndarray:
        return np.log(v) if use_log else v

    def inverse(fc: Forecast) -> Forecast:
        if not use_log:
            return fc
        return Forecast(
            np.exp(fc.mean),
            None if fc.lower is None else np.exp(fc.lower),
            None if fc.upper is None else np.exp(fc.upper),
        )

    if "sarima" in cfg.models:
        order = select_sarima_order(transform(y_all[: origins[0]]), m, cfg.sarima_max_order)
        fns["sarima"] = _sarima(order)

    rows: list[dict[str, Any]] = []
    metrics: dict[str, list[dict[str, float]]] = {name: [] for name in cfg.models}
    errors: dict[str, list[np.ndarray]] = {name: [] for name in cfg.models}
    last: dict[str, Forecast] = {}
    for i, origin in enumerate(origins):
        train, actual = y_all[:origin], y_all[origin : origin + h]
        for name in cfg.models:
            fc = inverse(fns[name](transform(train), h, m, z))
            result = forecast_metrics(actual, fc, train, m)
            metrics[name].append(result)
            errors[name].append(actual - fc.mean)
            last[name] = fc
            rows += [
                {
                    "experiment": cfg.experiment,
                    "dataset": ts.name,
                    "model": name,
                    "fold": i,
                    "metric": k,
                    "value": v,
                }
                for k, v in result.items()
            ]

    summary = {
        name: {k: float(np.mean([r[k] for r in res if k in r])) for k in res[0]}
        for name, res in metrics.items()
    }
    ranking = sorted(cfg.models, key=lambda n: summary[n]["mase"])
    best = ranking[0]
    tracker.log_metrics({f"{ts.name}.{n}.mase": summary[n]["mase"] for n in cfg.models})

    dm = {
        name: diebold_mariano(np.concatenate(errors[best]), np.concatenate(errors[name]), horizon=h)
        for name in ranking[1:]
    }

    # forecast plot for the last origin
    origin = origins[-1]
    fig, ax = plt.subplots(figsize=(10, 4))
    start = max(0, origin - 4 * m)
    ax.plot(ts.y.index[start:], y_all[start:], color="black", lw=1.2, label="actual")
    future = ts.y.index[origin : origin + h]
    for name in ranking[:4]:
        ax.plot(future, last[name].mean, lw=1.2, label=f"{name} (MASE {summary[name]['mase']:.2f})")
    with_pi = [n for n in ranking if last[n].lower is not None]
    if with_pi:
        top = with_pi[0]
        ax.fill_between(
            future,
            last[top].lower,
            last[top].upper,
            alpha=0.2,
            label=f"{top} {cfg.interval_level:.0%} interval",
        )
    ax.axvline(ts.y.index[origin], color="grey", ls=":", lw=1)
    ax.set_title(f"{ts.name}: last backtest origin")
    ax.legend(fontsize=8, ncol=2)
    path, fc_link = out.figure(f"{ts.name}_forecast.png")
    save(fig, path)

    diag_lines, diag_link = diagnostics(ts, out)
    cols = ["mase", "smape", "mae", "rmse", "coverage"]
    lines = [
        f"## `{ts.name}`",
        "",
        f"{ts.source} · {len(y_all)} observations · season {m} · horizon h = {h} · "
        f"{cfg.n_origins} rolling origins · {'log' if use_log else 'no'} transform"
        + (f" · SARIMA{order[0]}×{order[1][:3]}" if "sarima" in cfg.models else ""),
        "",
        *diag_lines,
        f"![diagnostics]({diag_link})",
        "",
        md_table(
            [
                "Model",
                "Family",
                "MASE",
                "sMAPE %",
                "MAE",
                "RMSE",
                f"{cfg.interval_level:.0%} PI coverage",
                f"DM p vs `{best}`",
            ],
            [
                [
                    f"**{n}**" if n == best else n,
                    FAMILY[n],
                    *(f"{summary[n][c]:.3f}" if c in summary[n] else "—" for c in cols),
                    "—" if n == best else fmt_p(dm[n].pvalue),
                ]
                for n in ranking
            ],
        ),
        "",
        f"![forecast]({fc_link})",
        "",
    ]
    return "\n".join(lines), rows


def run(cfg: ForecastConfig) -> str:
    set_seed(cfg.seed)
    out = Outputs(cfg.output_dir, cfg.experiment)
    sections, rows = [], []
    with track_run(cfg.experiment, cfg.model_dump(mode="json"), cfg.tags) as tracker:
        for name in cfg.series:
            text, series_rows = run_series(load_series(name), cfg, out, tracker)
            sections.append(text)
            rows += series_rows
        results = write_results(rows, out.results)

        mase = (
            results[results.metric == "mase"]
            .pivot_table(index="model", columns="dataset", values="value", aggfunc="mean")
            .loc[cfg.models]
        )
        header = [
            "# A2 — Forecasting: baselines vs. statistical vs. ML vs. neural networks",
            "",
            f"Rolling-origin backtesting · metric MASE (mean over origins; < 1 beats the "
            f"in-sample seasonal naive) · fold-level results in "
            f"[`results/{cfg.experiment}.csv`](results/{cfg.experiment}.csv)",
            "",
        ]
        ranks = mase.rank(axis=0).mean(axis=1).sort_values()
        header.append(
            md_table(
                ["Model", "Family", *mase.columns, "avg rank"],
                [
                    [n, FAMILY[n], *(f"{mase.loc[n, c]:.3f}" for c in mase.columns), f"{r:.2f}"]
                    for n, r in ranks.items()
                ],
            )
        )
        if mase.shape[1] >= 2 and mase.shape[0] >= 3:
            fr = friedman(mase.T.to_numpy(), list(mase.index), higher_is_better=False)
            header += [
                "",
                f"Friedman test across {mase.shape[1]} series: χ² = {fr.statistic:.2f}, "
                f"p = {fmt_p(fr.pvalue)}. Diebold-Mariano p-values per series compare each "
                "model with that series' best (HLN correction, horizon h).",
            ]
        header.append("")
        text = "\n".join(header + sections)
        out.write_report(text)
        tracker.log_artifact(out.report)
    return text


def main() -> None:
    cli(run, ForecastConfig, "A2 forecasting comparison")


if __name__ == "__main__":
    main()
