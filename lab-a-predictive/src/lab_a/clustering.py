"""A3a — Clustering: how many segments are there, and are they real?

    python -m lab_a.clustering --config lab-a-predictive/configs/a3_clustering.yaml

Revisits a K-Means (K=3) segmentation of the UCI Wholesale-customers data and
asks the questions a single silhouette score cannot answer:

* Does the preprocessing matter? (standardized raw spend vs. log-spend)
* Do other algorithms (GMM, Ward, DBSCAN, HDBSCAN) find the same structure?
* Are the clusters **stable** under resampling (bootstrap ARI)?
* Do they align with a known external label (``Channel``: Horeca vs. Retail)?

A synthetic dataset with known clusters is run first as a sanity check of the
whole procedure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from pydantic import Field
from sklearn.cluster import DBSCAN, HDBSCAN, AgglomerativeClustering, KMeans
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from airlab.plots import plt, save
from airlab.reporting import md_table, write_results
from airlab.seed import set_seed
from airlab.tracking import track_run
from lab_a.common import LabAConfig, Outputs, cli
from lab_a.datasets import WHOLESALE_SPEND, load_wholesale

K_SELECTABLE = ["kmeans", "gmm", "ward"]
DENSITY = ["dbscan", "hdbscan"]


class ClusteringConfig(LabAConfig):
    datasets: list[str] = Field(default_factory=lambda: ["synthetic_blobs", "wholesale"])
    algorithms: list[str] = Field(default_factory=lambda: K_SELECTABLE + DENSITY)
    preprocessing: list[str] = Field(default_factory=lambda: ["standard", "log_standard"])
    k_range: tuple[int, int] = (2, 8)
    n_bootstrap: int = 100
    sample_fraction: float = 0.8


@dataclass
class ClusterData:
    name: str
    raw: pd.DataFrame
    external: pd.Series  # known labels, for external validation only
    external_name: str


def load(name: str, seed: int) -> ClusterData:
    if name == "wholesale":
        df = load_wholesale()
        return ClusterData(name, df[WHOLESALE_SPEND], df["Channel"], "Channel (Horeca/Retail)")
    if name == "synthetic_blobs":
        X, y = make_blobs(
            n_samples=500, centers=4, n_features=5, cluster_std=1.2, random_state=seed
        )
        X = np.exp(X / 4) * 1000  # skewed, positive "spend"-like features
        cols = [f"x{i}" for i in range(X.shape[1])]
        return ClusterData(name, pd.DataFrame(X, columns=cols), pd.Series(y), "true cluster")
    raise KeyError(name)


def transform(raw: pd.DataFrame, kind: str) -> np.ndarray:
    values = raw.to_numpy(float)
    if kind == "log_standard":
        values = np.log1p(values)
    elif kind != "standard":
        raise ValueError(kind)
    return StandardScaler().fit_transform(values)


def dbscan_eps(X: np.ndarray, min_samples: int) -> float:
    """eps at the knee of the sorted k-distance curve (max distance to the chord)."""
    dist = NearestNeighbors(n_neighbors=min_samples).fit(X).kneighbors(X)[0][:, -1]
    d = np.sort(dist)
    x = np.linspace(0, 1, len(d))
    y = (d - d[0]) / (d[-1] - d[0] + 1e-12)
    return float(d[np.argmax(x - y)])


def fit_predict(
    algorithm: str, X: np.ndarray, k: int | None, seed: int, eps: float | None = None
) -> np.ndarray:
    if algorithm == "kmeans":
        return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)
    if algorithm == "gmm":
        return GaussianMixture(n_components=k, n_init=3, random_state=seed).fit(X).predict(X)
    if algorithm == "ward":
        return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
    if algorithm == "dbscan":
        min_samples = 2 * X.shape[1]
        return DBSCAN(eps=eps or dbscan_eps(X, min_samples), min_samples=min_samples).fit_predict(X)
    if algorithm == "hdbscan":
        return HDBSCAN(min_cluster_size=max(10, len(X) // 40)).fit_predict(X)
    raise ValueError(algorithm)


def internal_scores(X: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Silhouette / DB / CH on clustered points; noise (-1) excluded and reported."""
    mask = labels != -1
    n_clusters = len(set(labels[mask]))
    scores = {"n_clusters": float(n_clusters), "noise_fraction": float(1 - mask.mean())}
    if n_clusters >= 2:
        scores |= {
            "silhouette": float(silhouette_score(X[mask], labels[mask])),
            "davies_bouldin": float(davies_bouldin_score(X[mask], labels[mask])),
            "calinski_harabasz": float(calinski_harabasz_score(X[mask], labels[mask])),
        }
    return scores


def is_valid(scores: dict[str, float]) -> bool:
    """A segmentation needs >= 2 clusters covering most points; trivial ones are perfectly 'stable'."""
    return scores["n_clusters"] >= 2 and scores["noise_fraction"] <= 0.5


def bootstrap_stability(
    algorithm: str,
    X: np.ndarray,
    reference: np.ndarray,
    k: int | None,
    eps: float | None,
    n: int,
    fraction: float,
    seed: int,
) -> np.ndarray:
    """ARI between the full-data clustering and refits on random subsamples."""
    rng = np.random.default_rng(seed)
    aris = []
    for b in range(n):
        idx = rng.choice(len(X), int(fraction * len(X)), replace=False)
        labels = fit_predict(algorithm, X[idx], k, seed + b, eps)
        aris.append(adjusted_rand_score(reference[idx], labels))
    return np.array(aris)


def run_dataset(name: str, cfg: ClusteringConfig, out: Outputs) -> tuple[str, list, dict]:
    data = load(name, cfg.seed)
    rows: list[dict[str, Any]] = []
    solutions: dict[str, dict[str, Any]] = {}
    ks = list(range(cfg.k_range[0], cfg.k_range[1] + 1))
    lines = [
        f"## `{name}`",
        "",
        f"{len(data.raw)} rows × {data.raw.shape[1]} features · external label: {data.external_name}",
        "",
    ]

    for prep in cfg.preprocessing:
        X = transform(data.raw, prep)
        # --- choose k by silhouette for the k-based algorithms
        curves: dict[str, list[float]] = {}
        for algo in [a for a in cfg.algorithms if a in K_SELECTABLE]:
            curves[algo] = [silhouette_score(X, fit_predict(algo, X, k, cfg.seed)) for k in ks]
        bic = [
            GaussianMixture(n_components=k, n_init=3, random_state=cfg.seed).fit(X).bic(X)
            for k in ks
        ]
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
        for algo, curve in curves.items():
            axes[0].plot(ks, curve, marker="o", label=algo)
        axes[0].set(xlabel="k", ylabel="silhouette", title=f"Silhouette vs k ({prep})")
        axes[0].legend()
        axes[1].plot(ks, bic, marker="o", color="tab:purple")
        axes[1].set(xlabel="k", ylabel="BIC (lower is better)", title="GMM BIC vs k")
        path, link = out.figure(f"{name}_{prep}_k_selection.png")
        save(fig, path)
        lines += [f"### Preprocessing: `{prep}`", "", f"![k selection]({link})", ""]

        for algo in cfg.algorithms:
            k = ks[int(np.argmax(curves[algo]))] if algo in curves else None
            eps = dbscan_eps(X, 2 * X.shape[1]) if algo == "dbscan" else None
            labels = fit_predict(algo, X, k, cfg.seed, eps)
            scores = internal_scores(X, labels)
            stab = bootstrap_stability(
                algo, X, labels, k, eps, cfg.n_bootstrap, cfg.sample_fraction, cfg.seed
            )
            ext = data.external.to_numpy()
            scores |= {
                "stability_ari": float(stab.mean()),
                "stability_low": float(np.quantile(stab, 0.025)),
                "stability_high": float(np.quantile(stab, 0.975)),
                "external_ari": float(adjusted_rand_score(ext, labels)),
                "external_nmi": float(normalized_mutual_info_score(ext, labels)),
            }
            key = f"{algo} / {prep}"
            solutions[key] = {"labels": labels, "X": X, "k": k, "scores": scores}
            rows += [
                {
                    "experiment": cfg.experiment,
                    "dataset": name,
                    "model": key,
                    "fold": "full",
                    "metric": metric,
                    "value": value,
                }
                for metric, value in scores.items()
            ]
            rows += [
                {
                    "experiment": cfg.experiment,
                    "dataset": name,
                    "model": key,
                    "fold": b,
                    "metric": "bootstrap_ari",
                    "value": float(v),
                }
                for b, v in enumerate(stab)
            ]

    table_rows = []
    order = sorted(
        solutions,
        key=lambda k: (
            not is_valid(solutions[k]["scores"]),
            -solutions[k]["scores"]["stability_ari"],
        ),
    )
    for key in order:
        s = solutions[key]["scores"]
        table_rows.append(
            [
                key if is_valid(s) else f"{key} ⚠ degenerate",
                int(s["n_clusters"]),
                f"{s['noise_fraction']:.0%}",
                f"{s.get('silhouette', float('nan')):.3f}",
                f"{s.get('davies_bouldin', float('nan')):.3f}",
                f"{s['stability_ari']:.3f} [{s['stability_low']:.2f}, {s['stability_high']:.2f}]",
                f"{s['external_ari']:.3f}",
                f"{s['external_nmi']:.3f}",
            ]
        )
    lines += [
        "### Comparison",
        "",
        md_table(
            [
                "Algorithm / preprocessing",
                "k",
                "noise",
                "silhouette ↑",
                "Davies-Bouldin ↓",
                f"bootstrap ARI ({cfg.n_bootstrap}×, 95% interval) ↑",
                "ARI vs external ↑",
                "NMI vs external ↑",
            ],
            table_rows,
        ),
        "",
        "Rows marked ⚠ found fewer than 2 clusters or labelled most points as noise: "
        "such trivial solutions look perfectly stable, so they are ranked last. "
        "Silhouette and Davies-Bouldin are computed in each preprocessing's own feature "
        "space, so they compare algorithms *within* a preprocessing, not across them. "
        "Stability and external agreement are comparable across all rows.",
        "",
    ]
    return "\n".join(lines), rows, solutions


def profile(data: ClusterData, sol: dict[str, Any], key: str, out: Outputs) -> list[str]:
    labels = sol["labels"]
    frame = data.raw.assign(cluster=labels)
    frame = frame[frame.cluster != -1]
    med = frame.groupby("cluster").median().round(0).astype(int)
    size = frame.cluster.value_counts().sort_index()
    ext = pd.crosstab(labels, data.external.to_numpy())
    ext.index.name = "cluster"

    pcs = PCA(n_components=2, random_state=0).fit(sol["X"])
    proj = pcs.transform(sol["X"])
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(proj[:, 0], proj[:, 1], c=labels, cmap="tab10", s=12)
    ax.set(
        xlabel=f"PC1 ({pcs.explained_variance_ratio_[0]:.0%})",
        ylabel=f"PC2 ({pcs.explained_variance_ratio_[1]:.0%})",
        title=key,
    )
    ax.legend(*sc.legend_elements(), title="cluster", fontsize=8)
    path, link = out.figure(f"{data.name}_best_pca.png")
    save(fig, path)

    rows = [[c, int(size[c]), *med.loc[c].tolist()] for c in med.index]
    return [
        f"### Profile of the most stable solution: `{key}`",
        "",
        "Median annual spend per cluster (original units):",
        "",
        md_table(["Cluster", "size", *med.columns], rows),
        "",
        f"Cross-tabulation with {data.external_name}:",
        "",
        md_table(
            ["Cluster", *[str(c) for c in ext.columns]],
            [[i, *r] for i, r in zip(ext.index, ext.to_numpy().tolist(), strict=True)],
        ),
        "",
        f"![pca]({link})",
        "",
    ]


def run(cfg: ClusteringConfig) -> str:
    set_seed(cfg.seed)
    out = Outputs(cfg.output_dir, cfg.experiment)
    sections, rows = [], []
    with track_run(cfg.experiment, cfg.model_dump(mode="json"), cfg.tags) as tracker:
        for name in cfg.datasets:
            text, ds_rows, solutions = run_dataset(name, cfg, out)
            valid = [k for k in solutions if is_valid(solutions[k]["scores"])] or list(solutions)
            best = max(valid, key=lambda k: solutions[k]["scores"]["stability_ari"])
            text += "\n" + "\n".join(profile(load(name, cfg.seed), solutions[best], best, out))
            if name == "wholesale" and "kmeans / standard" in solutions:
                base = solutions["kmeans / standard"]
                sil = base["scores"].get("silhouette", float("nan"))
                text += (
                    "\n### Versus the original K-Means study\n\n"
                    f"`kmeans / standard` reproduces the original setup (K-Means on standardized "
                    f"raw spend); with k chosen by silhouette it gives k = {base['k']} and "
                    f"silhouette {sil:.3f} (original study: K = 3, ≈ 0.548). The table above "
                    "shows how far that solution is from the most stable one.\n"
                )
            tracker.log_metrics(
                {
                    f"{name}.{k}.stability_ari": v["scores"]["stability_ari"]
                    for k, v in solutions.items()
                }
            )
            sections.append(text)
            rows += ds_rows
        write_results(rows, out.results)
        header = [
            "# A3a — Clustering: are the segments real?",
            "",
            f"k chosen by silhouette over k = {cfg.k_range[0]}…{cfg.k_range[1]} · "
            f"stability = mean ARI between the full-data clustering and {cfg.n_bootstrap} refits "
            f"on {cfg.sample_fraction:.0%} subsamples · results in "
            f"[`results/{cfg.experiment}.csv`](results/{cfg.experiment}.csv)",
            "",
        ]
        text = "\n".join(header + sections)
        out.write_report(text)
        tracker.log_artifact(out.report)
    return text


def main() -> None:
    cli(run, ClusteringConfig, "A3 clustering study")


if __name__ == "__main__":
    main()
