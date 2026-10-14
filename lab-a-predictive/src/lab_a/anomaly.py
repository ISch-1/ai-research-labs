"""A3b — Unsupervised anomaly detection: statistics vs. classical ML vs. neural networks.

    python -m lab_a.anomaly --config lab-a-predictive/configs/a3_anomaly.yaml

Every detector is fitted **without labels** on the full (contaminated) data and
outputs an anomaly score; labels are used only for evaluation. Randomized
detectors are repeated over several seeds, which become the "folds" of the
results file.

Detectors
---------
* statistical — ``robust_z`` (max robust z-score, median/MAD),
                ``mahalanobis`` (robust covariance, Minimum Covariance Determinant)
* classical   — ``isolation_forest``, ``lof``, ``ocsvm``
* neural      — ``autoencoder`` (MLP reconstruction error)
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from pydantic import Field
from scipy.stats import rankdata
from sklearn.covariance import LedoitWolf, MinCovDet
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import OneClassSVM

from airlab.metrics import bootstrap_ci
from airlab.plots import plt, save
from airlab.reporting import fmt_p, md_table, write_results
from airlab.seed import set_seed
from airlab.stats import friedman
from airlab.tracking import track_run
from lab_a.common import LabAConfig, Outputs, cli
from lab_a.datasets import AnomalyDataset, load_anomaly

FAMILY = {
    "robust_z": "statistical",
    "mahalanobis": "statistical",
    "isolation_forest": "classical ML",
    "lof": "classical ML",
    "ocsvm": "classical ML",
    "autoencoder": "neural network",
}
RANDOMIZED = {"mahalanobis", "isolation_forest", "ocsvm", "autoencoder"}


class AnomalyConfig(LabAConfig):
    datasets: list[str]
    detectors: list[str] = Field(default_factory=lambda: list(FAMILY))
    max_rows: int | None = 20_000
    n_seeds: int = 5
    fit_subsample: int = 5_000  # for the O(n^2) / O(n p^2) fits (OCSVM, MCD)
    bootstrap_resamples: int = 1000


# ------------------------------------------------------------------------ detectors


def _subsample(X: np.ndarray, n: int, seed: int) -> np.ndarray:
    if len(X) <= n:
        return X
    return X[np.random.default_rng(seed).choice(len(X), n, replace=False)]


def score(detector: str, X: np.ndarray, seed: int, fit_subsample: int) -> np.ndarray:
    """Higher score = more anomalous."""
    if detector == "robust_z":
        median = np.median(X, axis=0)
        mad = 1.4826 * np.median(np.abs(X - median), axis=0)
        mad[mad == 0] = np.std(X, axis=0)[mad == 0] + 1e-12  # constant-heavy columns
        return np.max(np.abs(X - median) / mad, axis=1)
    if detector == "mahalanobis":
        fit_on = _subsample(X, fit_subsample, seed)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cov = MinCovDet(random_state=seed).fit(fit_on)
        except (ValueError, np.linalg.LinAlgError):
            cov = LedoitWolf().fit(fit_on)  # singular data (e.g. one-hot columns)
        return cov.mahalanobis(X)
    if detector == "isolation_forest":
        model = IsolationForest(n_estimators=300, random_state=seed, n_jobs=-1).fit(X)
        return -model.score_samples(X)
    if detector == "lof":
        lof = LocalOutlierFactor(n_neighbors=35).fit(X)
        return -lof.negative_outlier_factor_
    if detector == "ocsvm":
        Xs = StandardScaler().fit(X)
        model = OneClassSVM(nu=0.05, gamma="scale").fit(
            Xs.transform(_subsample(X, fit_subsample, seed))
        )
        return -model.score_samples(Xs.transform(X))
    if detector == "autoencoder":
        scaler = StandardScaler().fit(X)
        Z = scaler.transform(X)
        d = X.shape[1]
        width, bottleneck = max(8, 2 * d), max(2, d // 3)
        ae = MLPRegressor(
            hidden_layer_sizes=(width, bottleneck, width),
            alpha=1e-4,
            max_iter=200,
            early_stopping=True,
            random_state=seed,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ae.fit(Z, Z)
        return np.mean((ae.predict(Z) - Z) ** 2, axis=1)
    raise ValueError(f"unknown detector {detector!r}")


def precision_at_n(y: np.ndarray, scores: np.ndarray) -> float:
    """Precision among the n highest scores, n = number of true anomalies (= recall@n)."""
    n = int(y.sum())
    top = np.argsort(scores)[::-1][:n]
    return float(y[top].mean())


# ----------------------------------------------------------------------- experiment


def run_dataset(ds: AnomalyDataset, cfg: AnomalyConfig, out: Outputs) -> tuple[str, list, dict]:
    X = RobustScaler().fit_transform(ds.X)
    X = X[:, np.std(X, axis=0) > 0]
    rows: list[dict[str, Any]] = []
    mean_scores: dict[str, dict[str, float]] = {}
    curves = {}
    ci = {}
    for det in cfg.detectors:
        seeds = range(cfg.n_seeds) if det in RANDOMIZED else [0]
        per_seed = []
        first_scores = None
        for seed in seeds:
            s = score(det, X, cfg.seed + seed, cfg.fit_subsample)
            if first_scores is None:
                first_scores = s
            m = {
                "roc_auc": float(roc_auc_score(ds.y, s)),
                "pr_auc": float(average_precision_score(ds.y, s)),
                "precision_at_n": precision_at_n(ds.y, s),
            }
            per_seed.append(m)
            rows += [
                {
                    "experiment": cfg.experiment,
                    "dataset": ds.name,
                    "model": det,
                    "fold": seed,
                    "metric": k,
                    "value": v,
                }
                for k, v in m.items()
            ]
        mean_scores[det] = {k: float(np.mean([p[k] for p in per_seed])) for k in per_seed[0]}
        mean_scores[det]["pr_auc_seed_std"] = float(np.std([p["pr_auc"] for p in per_seed]))
        assert first_scores is not None
        curves[det] = precision_recall_curve(ds.y, first_scores)
        ci[det] = bootstrap_ci(
            ds.y,
            first_scores,
            average_precision_score,
            n_resamples=cfg.bootstrap_resamples,
            seed=cfg.seed,
        )

    ranking = sorted(cfg.detectors, key=lambda d: -mean_scores[d]["pr_auc"])
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for det in ranking:
        precision, recall, _ = curves[det]
        ax.plot(recall, precision, lw=1.2, label=f"{det} (AP {mean_scores[det]['pr_auc']:.3f})")
    ax.axhline(ds.contamination, color="grey", ls=":", label=f"random ({ds.contamination:.3f})")
    ax.set(xlabel="recall", ylabel="precision", title=f"Precision-recall — {ds.name}")
    ax.legend(fontsize=8)
    path, link = out.figure(f"{ds.name}_pr.png")
    save(fig, path)

    table = md_table(
        [
            "Detector",
            "Family",
            "PR-AUC (mean over seeds)",
            "± seed std",
            "95% bootstrap CI (first seed)",
            "ROC-AUC",
            "Precision@n",
        ],
        [
            [
                f"**{d}**" if d == ranking[0] else d,
                FAMILY[d],
                f"{mean_scores[d]['pr_auc']:.3f}",
                f"{mean_scores[d]['pr_auc_seed_std']:.3f}" if d in RANDOMIZED else "—",
                f"[{ci[d].low:.3f}, {ci[d].high:.3f}]",
                f"{mean_scores[d]['roc_auc']:.3f}",
                f"{mean_scores[d]['precision_at_n']:.3f}",
            ]
            for d in ranking
        ],
    )
    lines = [
        f"## `{ds.name}`",
        "",
        f"{ds.source} · {len(ds.y):,} rows × {X.shape[1]} features · "
        f"contamination {ds.contamination:.2%} (= PR-AUC of a random detector) · "
        f"n = {int(ds.y.sum())} anomalies",
        "",
        table,
        "",
        f"![pr]({link})",
        "",
    ]
    return "\n".join(lines), rows, mean_scores


def run(cfg: AnomalyConfig) -> str:
    set_seed(cfg.seed)
    out = Outputs(cfg.output_dir, cfg.experiment)
    sections, rows, summary = [], [], {}
    with track_run(cfg.experiment, cfg.model_dump(mode="json"), cfg.tags) as tracker:
        for name in cfg.datasets:
            ds = load_anomaly(name, max_rows=cfg.max_rows, seed=cfg.seed)
            text, ds_rows, scores = run_dataset(ds, cfg, out)
            summary[name] = scores
            sections.append(text)
            rows += ds_rows
            tracker.log_metrics({f"{name}.{d}.pr_auc": s["pr_auc"] for d, s in scores.items()})
        write_results(rows, out.results)

        names = list(summary)
        matrix = np.array([[summary[n][d]["pr_auc"] for d in cfg.detectors] for n in names])
        ranks = rankdata(-matrix, axis=1)  # 1 = best; ties share the average rank
        avg_rank = dict(zip(cfg.detectors, ranks.mean(axis=0), strict=True))
        order = sorted(cfg.detectors, key=avg_rank.get)  # type: ignore[arg-type]
        header = [
            "# A3b — Anomaly detection: statistics vs. classical ML vs. neural networks",
            "",
            "Unsupervised: detectors never see labels; labels are used only to score them. "
            "PR-AUC is the headline metric because anomalies are rare. "
            f"Randomized detectors averaged over {cfg.n_seeds} seeds. Results in "
            f"[`results/{cfg.experiment}.csv`](results/{cfg.experiment}.csv).",
            "",
            md_table(
                ["Detector", "Family", *names, "avg rank"],
                [
                    [
                        d,
                        FAMILY[d],
                        *(f"{summary[n][d]['pr_auc']:.3f}" for n in names),
                        f"{avg_rank[d]:.2f}",
                    ]
                    for d in order
                ],
            ),
            "",
        ]
        if len(names) >= 2 and len(cfg.detectors) >= 3:
            fr = friedman(matrix, cfg.detectors)
            header += [
                f"Friedman test on PR-AUC across {len(names)} datasets: "
                f"p = {fmt_p(fr.pvalue)} — with this few datasets only very large rank "
                "differences can reach significance (see A4 for the multi-dataset analysis).",
                "",
            ]
        text = "\n".join(header + sections)
        out.write_report(text)
        tracker.log_artifact(out.report)
    return text


def main() -> None:
    cli(run, AnomalyConfig, "A3 anomaly detection study")


if __name__ == "__main__":
    main()
