"""A1 — Regression & classification: statistical models vs. classical ML vs. neural networks.

    python -m lab_a.supervised --config lab-a-predictive/configs/a1_classification.yaml
    python -m lab_a.supervised --config lab-a-predictive/configs/a1_regression.yaml

Per dataset:

1. Stratified hold-out split; the test set is used once, at the end.
2. Repeated k-fold CV of every model with default hyper-parameters
   (Nadeau-Bengio corrected t-test against the CV winner).
3. Optional Optuna tuning of selected models with inner CV on the training split.
4. Test-set metrics with bootstrap CIs; McNemar (classification) or Wilcoxon on
   absolute errors (regression) between the two best models.
5. Diagnostics: calibration curves, permutation importance, SHAP for the best
   tree ensemble, and a classical statistical fit (OLS / Logit with p-values).
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import optuna
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.model_selection import (
    KFold,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_score,
    cross_validate,
    train_test_split,
)

from airlab import metrics as M
from airlab.plots import plt, save
from airlab.reporting import fmt_p, md_table, write_results
from airlab.seed import set_seed
from airlab.stats import TestResult, corrected_resampled_ttest, mcnemar, wilcoxon_signed_rank
from airlab.tracking import track_run
from lab_a import models as Z
from lab_a.common import LabAConfig, Outputs, cli
from lab_a.datasets import TabularDataset, Task, load_tabular

PRIMARY = {"classification": "roc_auc", "regression": "rmse"}
CV_SCORING = {
    "classification": {
        "roc_auc": "roc_auc",
        "accuracy": "accuracy",
        "f1_macro": "f1_macro",
        "neg_brier": "neg_brier_score",
    },
    "regression": {
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "r2": "r2",
    },
}
HIGHER_IS_BETTER = {"roc_auc": True, "accuracy": True, "f1_macro": True, "rmse": False}
TEST_NAMES = {
    "mcnemar": "McNemar",
    "mcnemar_exact": "McNemar (exact)",
    "wilcoxon_signed_rank": "Wilcoxon signed-rank on |error|",
}


class CVConfig(BaseModel):
    n_splits: int = 5
    n_repeats: int = 2


class TuneConfig(BaseModel):
    models: list[str] = Field(default_factory=list)
    n_trials: int = 30
    inner_splits: int = 3


class SupervisedConfig(LabAConfig):
    task: Task
    datasets: list[str]
    models: list[str] | None = None
    skip: dict[str, list[str]] = Field(default_factory=dict)
    test_size: float = 0.2
    max_train_rows: int | None = None
    cv: CVConfig = Field(default_factory=CVConfig)
    tune: TuneConfig = Field(default_factory=TuneConfig)
    interpret: bool = True
    shap_rows: int = 500
    bootstrap_resamples: int = 1000


# ------------------------------------------------------------------------- helpers


def _split(ds: TabularDataset, cfg: SupervisedConfig):
    stratify = ds.y if ds.task == "classification" else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        ds.X, ds.y, test_size=cfg.test_size, stratify=stratify, random_state=cfg.seed
    )
    if cfg.max_train_rows and len(y_tr) > cfg.max_train_rows:
        # Compute budget: every model sees the same subsample of the training split.
        X_tr, _, y_tr, _ = train_test_split(
            X_tr,
            y_tr,
            train_size=cfg.max_train_rows,
            stratify=y_tr if ds.task == "classification" else None,
            random_state=cfg.seed,
        )
    return X_tr, X_te, y_tr, y_te


def _cv_splitter(task: Task, n_splits: int, n_repeats: int, seed: int):
    if task == "classification":
        return RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    return RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)


def _cv_scores(raw: dict[str, np.ndarray], task: Task) -> dict[str, np.ndarray]:
    """sklearn maximises, so error metrics come back negated: flip them."""
    out = {}
    for name in CV_SCORING[task]:
        values = raw[f"test_{name}"]
        if name.startswith("neg_"):
            out[name.removeprefix("neg_")] = -values
        elif name in {"rmse", "mae"}:
            out[name] = -values
        else:
            out[name] = values
    out["fit_time"] = raw["fit_time"]
    return out


def _test_metrics(task: Task, y: np.ndarray, pred: np.ndarray, proba: np.ndarray | None, cfg):
    kw = {"n_resamples": cfg.bootstrap_resamples, "seed": cfg.seed}
    if task == "regression":
        return M.report(y, pred, M.REGRESSION_METRICS, **kw)
    result = M.report(y, pred, M.CLASSIFICATION_METRICS, **kw)
    if proba is not None:
        result |= M.report(y, proba, M.SCORE_METRICS, **kw)
    return result


def _tune(name: str, task: Task, X, y, cfg: SupervisedConfig) -> dict[str, Any]:
    scoring = CV_SCORING[task][PRIMARY[task]]
    splitter = (StratifiedKFold if task == "classification" else KFold)(
        n_splits=cfg.tune.inner_splits, shuffle=True, random_state=cfg.seed
    )

    def objective(trial: optuna.Trial) -> float:
        model = Z.build(name, task, cfg.seed, Z.search_space(name, trial))
        return float(cross_val_score(model, X, y, cv=splitter, scoring=scoring).mean())

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=cfg.seed)
    )
    study.optimize(objective, n_trials=cfg.tune.n_trials)
    return Z.search_space(name, optuna.trial.FixedTrial(study.best_params))


def _drop_collinear(X: pd.DataFrame, threshold: float = 0.9) -> tuple[pd.DataFrame, list[str]]:
    """Greedily drop features with |r| > threshold to an earlier kept feature.

    Coefficients of near-duplicate features are not identifiable: their CIs
    explode and their signs flip. Keeping one representative is the standard fix.
    """
    corr = X.corr().abs().to_numpy()
    kept: list[int] = []
    for j in range(X.shape[1]):
        if all(corr[j, k] <= threshold for k in kept):
            kept.append(j)
    dropped = [c for i, c in enumerate(X.columns) if i not in kept]
    return X.iloc[:, kept], dropped


def _drop_rank_deficient(X: pd.DataFrame, tol: float = 1e-8) -> tuple[pd.DataFrame, list[str]]:
    """Drop columns that are linear combinations of others (QR with column pivoting).

    Catches multi-column redundancy that pairwise correlations miss, e.g. a
    one-hot ``education`` block that reproduces the numeric ``education-num``.
    """
    from scipy.linalg import qr

    _, r, pivots = qr(np.c_[np.ones(len(X)), X.to_numpy()], mode="economic", pivoting=True)
    diag = np.abs(np.diag(r))
    keep = {int(p) - 1 for p, d in zip(pivots, diag, strict=True) if d > tol * diag[0] and p > 0}
    dropped = [c for i, c in enumerate(X.columns) if i not in keep]
    return X.iloc[:, sorted(keep)], dropped


def _statistical_fit(
    task: Task, X: pd.DataFrame, y: pd.Series, seed: int
) -> tuple[pd.DataFrame | None, list[str]]:
    """Classical OLS / Logit on standardized features: coefficients with CIs and p-values.

    Returns ``(None, dropped)`` when the fit is not trustworthy (no convergence,
    perfect separation or non-finite standard errors).
    """
    import statsmodels.api as sm

    prep = Z.preprocessor(scale=True, drop_first=True)
    if len(y) > 20_000:
        X, _, y, _ = train_test_split(X, y, train_size=20_000, random_state=seed)
    Xt = pd.DataFrame(prep.fit_transform(X), columns=prep.get_feature_names_out())
    Xt, dropped = _drop_collinear(Xt.loc[:, Xt.std() > 0])
    Xt, redundant = _drop_rank_deficient(Xt)
    dropped += redundant
    exog = sm.add_constant(Xt, has_constant="add")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if task == "regression":
                fit = sm.OLS(y.to_numpy(), exog).fit()
            else:
                fit = sm.Logit(y.to_numpy(), exog).fit(disp=0, maxiter=500, method="lbfgs")
                if not fit.mle_retvals.get("converged", False):
                    return None, dropped
    except Exception:  # singular design, perfect separation, ...
        return None, dropped
    if not np.all(np.isfinite(fit.bse)) or (
        task == "classification" and fit.params.abs().max() > 20
    ):
        return None, dropped  # quasi-separation: huge coefficients, meaningless CIs
    ci = fit.conf_int()
    table = pd.DataFrame({"coef": fit.params, "low": ci[0], "high": ci[1], "p": fit.pvalues}).drop(
        index="const"
    )
    if task == "classification":
        table[["coef", "low", "high"]] = np.exp(table[["coef", "low", "high"]])  # odds ratios
    order = (np.log(table["coef"]) if task == "classification" else table["coef"]).abs()
    return table.loc[order.sort_values(ascending=False).index], dropped


# --------------------------------------------------------------------- experiment


def run_dataset(
    ds: TabularDataset, cfg: SupervisedConfig, out: Outputs, tracker
) -> tuple[str, list]:
    task = ds.task
    primary = PRIMARY[task]
    higher = HIGHER_IS_BETTER[primary]
    names = [
        m for m in (cfg.models or Z.default_models(task)) if m not in cfg.skip.get(ds.name, [])
    ]
    X_tr, X_te, y_tr, y_te = _split(ds, cfg)
    rows: list[dict[str, Any]] = []

    # --- cross-validation with default hyper-parameters
    splitter = _cv_splitter(task, cfg.cv.n_splits, cfg.cv.n_repeats, cfg.seed)
    cv: dict[str, dict[str, np.ndarray]] = {}
    for name in names:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = cross_validate(
                Z.build(name, task, cfg.seed), X_tr, y_tr, cv=splitter, scoring=CV_SCORING[task]
            )
        cv[name] = _cv_scores(raw, task)
        for metric, values in cv[name].items():
            rows += [
                {
                    "experiment": cfg.experiment,
                    "dataset": ds.name,
                    "model": name,
                    "fold": i,
                    "metric": metric,
                    "value": float(v),
                }
                for i, v in enumerate(values)
            ]
    ranking = sorted(names, key=lambda m: cv[m][primary].mean(), reverse=higher)
    best = ranking[0]
    n_val = len(y_tr) // cfg.cv.n_splits
    tests = {
        m: corrected_resampled_ttest(
            cv[best][primary], cv[m][primary], n_train=len(y_tr) - n_val, n_test=n_val
        )
        for m in ranking[1:]
    }

    # --- tuning
    tuned_params: dict[str, dict[str, Any]] = {}
    for name in cfg.tune.models:
        if name in names:
            tuned_params[name] = _tune(name, task, X_tr, y_tr, cfg)

    # --- test set
    fitted: dict[str, Any] = {}
    preds: dict[str, np.ndarray] = {}
    probas: dict[str, np.ndarray | None] = {}
    test: dict[str, dict[str, M.MetricCI]] = {}
    variants: list[tuple[str, str, dict[str, Any]]] = [(m, m, {}) for m in names] + [
        (f"{m} (tuned)", m, p) for m, p in tuned_params.items()
    ]
    for label, name, params in variants:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = Z.build(name, task, cfg.seed, params).fit(X_tr, y_tr)
        fitted[label] = model
        preds[label] = model.predict(X_te)
        probas[label] = model.predict_proba(X_te)[:, 1] if task == "classification" else None
        test[label] = _test_metrics(task, y_te.to_numpy(), preds[label], probas[label], cfg)
        for metric, ci in test[label].items():
            rows.append(
                {
                    "experiment": cfg.experiment,
                    "dataset": ds.name,
                    "model": label,
                    "fold": "test",
                    "metric": metric,
                    "value": ci.value,
                }
            )
        tracker.log_metrics(
            {f"{ds.name}.{label}.test_{k}": v.value for k, v in test[label].items()}
        )

    test_rank = sorted(test, key=lambda m: test[m][primary].value, reverse=higher)
    a, b = [m for m in test_rank if m in names][:2]  # tuned variants are compared separately
    y_arr = y_te.to_numpy()
    pair: TestResult = (
        mcnemar(y_arr, preds[a], preds[b])
        if task == "classification"
        else wilcoxon_signed_rank(np.abs(y_arr - preds[a]), np.abs(y_arr - preds[b]))
    )
    tuned_vs_default = {
        m: (
            mcnemar(y_arr, preds[f"{m} (tuned)"], preds[m])
            if task == "classification"
            else wilcoxon_signed_rank(
                np.abs(y_arr - preds[f"{m} (tuned)"]), np.abs(y_arr - preds[m])
            )
        )
        for m in tuned_params
    }

    # --- report
    lines = [
        f"## `{ds.name}`",
        "",
        f"{ds.source} · {len(ds.y):,} rows × {ds.X.shape[1]} features "
        f"({ds.n_categorical} categorical) · train {len(y_tr):,} / test {len(y_te):,}"
        + (
            f" · positive rate {ds.y.mean():.1%}"
            if task == "classification"
            else f" · target mean {ds.y.mean():.3g}"
        ),
        "",
        f"### Cross-validation ({cfg.cv.n_repeats}×{cfg.cv.n_splits}-fold, default hyper-parameters)",
        "",
    ]
    metric_cols = list(CV_SCORING[task])
    clean = [c.removeprefix("neg_") for c in metric_cols]
    lines.append(
        md_table(
            ["Model", "Family", *clean, "fit s", f"p vs `{best}`"],
            [
                [
                    f"**{m}**" if m == best else m,
                    Z.FAMILY[m],
                    *(f"{cv[m][c].mean():.4f} ± {cv[m][c].std(ddof=1):.4f}" for c in clean),
                    f"{cv[m]['fit_time'].mean():.2f}",
                    "—" if m == best else fmt_p(tests[m].pvalue),
                ]
                for m in ranking
            ],
        )
    )
    tie = [m for m in ranking[1:] if not tests[m].significant()]
    lines += [
        "",
        f"Corrected t-test on {primary}: "
        + (
            f"`{best}` is not significantly better than {', '.join(f'`{m}`' for m in tie)} (α = 0.05)."
            if tie
            else f"`{best}` is significantly better than every other model (α = 0.05)."
        ),
        "",
    ]
    if tuned_params:
        lines += ["### Tuned hyper-parameters (Optuna, inner CV on the training split)", ""]
        for m, p in tuned_params.items():
            shown = ", ".join(
                f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in p.items()
            )
            lines.append(f"- **{m}** ({cfg.tune.n_trials} trials): {shown}")
        lines.append("")

    test_cols = list(next(iter(test.values())))
    lines += [f"### Test set ({len(y_te):,} rows, 95% bootstrap CI)", ""]
    lines.append(
        md_table(
            ["Model", *test_cols],
            [
                [f"**{m}**" if m == a else m, *(str(test[m][c]) for c in test_cols)]
                for m in test_rank
            ],
        )
    )
    lines += [
        "",
        f"{TEST_NAMES[pair.name]}, best two default models `{a}` vs `{b}`: p = {fmt_p(pair.pvalue)} "
        f"({'significant' if pair.significant() else 'not significant'}).",
    ]
    for m, res in tuned_vs_default.items():
        lines.append(
            f"Tuned vs default `{m}`: p = {fmt_p(res.pvalue)} "
            f"({'significant' if res.significant() else 'not significant'})."
        )
    lines.append("")

    # --- diagnostics
    if task == "classification":
        path, link = out.figure(f"{ds.name}_calibration.png")
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
        for m in test_rank[:4]:
            frac, mean_pred = calibration_curve(y_arr, probas[m], n_bins=10, strategy="quantile")
            ax.plot(
                mean_pred, frac, marker="o", ms=3, label=f"{m} (Brier {test[m]['brier'].value:.3f})"
            )
        ax.set(
            xlabel="mean predicted probability",
            ylabel="observed frequency",
            title=f"Calibration — {ds.name}",
        )
        ax.legend(fontsize=8)
        save(fig, path)
        lines += [f"![calibration]({link})", ""]
    else:
        path, link = out.figure(f"{ds.name}_residuals.png")
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].scatter(y_arr, preds[a], s=4, alpha=0.4)
        lims = [min(y_arr.min(), preds[a].min()), max(y_arr.max(), preds[a].max())]
        axes[0].plot(lims, lims, "k--", lw=1)
        axes[0].set(xlabel="actual", ylabel="predicted", title=f"{a}: predicted vs actual")
        axes[1].hist(y_arr - preds[a], bins=50)
        axes[1].set(xlabel="residual", title="residual distribution")
        save(fig, path)
        lines += [f"![residuals]({link})", ""]

    if cfg.interpret:
        lines += _interpretation(ds, cfg, out, fitted, a, test_rank, X_tr, y_tr, X_te, y_te)
    return "\n".join(lines), rows


def _interpretation(ds, cfg, out, fitted, best, test_rank, X_tr, y_tr, X_te, y_te) -> list[str]:
    task = ds.task
    lines = ["### Interpretation", ""]
    scoring = CV_SCORING[task][PRIMARY[task]]

    # Permutation importance of the best model on the test set (model-agnostic).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        imp = permutation_importance(
            fitted[best], X_te, y_te, scoring=scoring, n_repeats=5, random_state=cfg.seed, n_jobs=1
        )
    order = np.argsort(imp.importances_mean)[::-1][:12]
    path, link = out.figure(f"{ds.name}_permutation_importance.png")
    fig, ax = plt.subplots(figsize=(6, 0.3 * len(order) + 1))
    ax.barh(
        [X_te.columns[i] for i in order][::-1],
        imp.importances_mean[order][::-1],
        xerr=imp.importances_std[order][::-1],
    )
    ax.set(
        xlabel=f"drop in {PRIMARY[task]} when shuffled", title=f"Permutation importance — {best}"
    )
    save(fig, path)
    lines += [
        f"Permutation importance of `{best}` on the test set:",
        "",
        f"![importance]({link})",
        "",
    ]

    # SHAP for the strongest tree ensemble.
    trees = [m for m in test_rank if m.split(" ")[0] in {"lightgbm", "random_forest"}]
    if trees:
        lines += _shap(ds, cfg, out, fitted[trees[0]], trees[0], X_te)

    # Classical statistical model: effect sizes with uncertainty.
    table, dropped = _statistical_fit(task, X_tr, y_tr, cfg.seed)
    kind = (
        "Logit (odds ratios per 1 SD)"
        if task == "classification"
        else "OLS (coefficients per 1 SD)"
    )
    note = (
        f" {len(dropped)} redundant features (|r| > 0.9 or linearly dependent) were dropped "
        f"first: {', '.join(f'`{d}`' for d in dropped[:6])}{'…' if len(dropped) > 6 else ''}."
        if dropped
        else ""
    )
    if table is None:
        lines += [f"{kind}: no trustworthy fit (no convergence or quasi-separation).{note}", ""]
    else:
        lines += [f"{kind}, top effects by magnitude.{note}", ""]
        lines.append(
            md_table(
                ["Feature", "Estimate", "95% CI", "p"],
                [
                    [f, f"{r.coef:.3f}", f"[{r.low:.3f}, {r.high:.3f}]", fmt_p(r.p)]
                    for f, r in table.head(10).iterrows()
                ],
            )
        )
        lines.append("")
    return lines


def _shap(ds, cfg, out, pipeline, label, X_te) -> list[str]:
    import shap

    sample = X_te.sample(min(cfg.shap_rows, len(X_te)), random_state=cfg.seed)
    prep, model = pipeline.named_steps["prep"], pipeline.named_steps["model"]
    Xt = pd.DataFrame(prep.transform(sample), columns=prep.get_feature_names_out())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        values = shap.TreeExplainer(model).shap_values(Xt)
    if isinstance(values, list):  # older shap: one array per class
        values = values[-1]
    if values.ndim == 3:  # (rows, features, classes)
        values = values[..., -1]
    shap.summary_plot(
        values,
        Xt,
        show=False,
        max_display=12,
        plot_size=(7, 5),
        rng=np.random.default_rng(cfg.seed),
    )
    path, link = out.figure(f"{ds.name}_shap.png")
    save(plt.gcf(), path)
    return [f"SHAP values of `{label}` ({len(sample)} test rows):", "", f"![shap]({link})", ""]


def run(cfg: SupervisedConfig) -> str:
    set_seed(cfg.seed)
    out = Outputs(cfg.output_dir, cfg.experiment)
    sections, rows = [], []
    with track_run(cfg.experiment, cfg.model_dump(mode="json"), cfg.tags) as tracker:
        for name in cfg.datasets:
            ds = load_tabular(name)
            if ds.task != cfg.task:
                raise ValueError(f"{name} is a {ds.task} dataset, config task is {cfg.task}")
            text, ds_rows = run_dataset(ds, cfg, out, tracker)
            sections.append(text)
            rows += ds_rows
        results = write_results(rows, out.results)

        primary = PRIMARY[cfg.task]
        cv_mean = (
            results[(results.fold != "test") & (results.metric == primary)]
            .assign(value=lambda d: d.value.astype(float))
            .pivot_table(index="model", columns="dataset", values="value", aggfunc="mean")
        )
        higher = HIGHER_IS_BETTER[primary]
        # Average ranks only over models evaluated on every dataset (see `skip`).
        complete = cv_mean.dropna()
        ranks = complete.rank(ascending=not higher).mean(axis=1).sort_values()
        partial = [m for m in cv_mean.index if m not in complete.index]
        summary = md_table(
            ["Model", "Family", *cv_mean.columns, "avg rank"],
            [
                [
                    m,
                    Z.FAMILY[m],
                    *("—" if pd.isna(v) else f"{v:.4f}" for v in cv_mean.loc[m]),
                    f"{ranks[m]:.2f}" if m in ranks else "— (skipped on some datasets)",
                ]
                for m in [*ranks.index, *partial]
            ],
        )
        header = [
            f"# A1 — {cfg.task.title()}: statistical vs. classical ML vs. neural networks",
            "",
            f"Seed {cfg.seed} · CV {primary} (mean over {cfg.cv.n_repeats}×{cfg.cv.n_splits} folds) · "
            f"lower rank is better · fold-level results in "
            f"[`results/{cfg.experiment}.csv`](results/{cfg.experiment}.csv)",
            "",
            summary,
            "",
        ]
        text = "\n".join(header + sections)
        out.write_report(text)
        tracker.log_artifact(out.report)
    return text


def main() -> None:
    cli(run, SupervisedConfig, "A1 supervised comparison")


if __name__ == "__main__":
    main()
