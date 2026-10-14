"""A4 — Statistical comparison of models across many datasets.

    python -m lab_a.benchmark --config lab-a-predictive/configs/a4_benchmark.yaml

One dataset can crown a lucky winner; the question "is model X generally
better than model Y?" needs many datasets and a test that treats each dataset
as one observation (Demšar 2006).

1. Runs every classifier with default hyper-parameters on a suite of binary
   OpenML-CC18 datasets (repeated stratified CV, ROC-AUC).
2. For that suite **and** for every other experiment's results file that
   exists (A1, A2, A3b), it reports:
   * average ranks and the Friedman test,
   * the Nemenyi critical difference with a CD diagram,
   * Wilcoxon signed-rank tests of the best model against each other model
     with Holm correction (Benavoli et al. 2016 recommend this over Nemenyi).
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy.stats import wilcoxon
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

from airlab.plots import cd_diagram
from airlab.reporting import fmt_p, md_table, write_results
from airlab.seed import set_seed
from airlab.stats import friedman, holm, nemenyi_cd
from airlab.tracking import track_run
from lab_a import models as Z
from lab_a.anomaly import FAMILY as ANOMALY_FAMILY
from lab_a.common import LabAConfig, Outputs, cli
from lab_a.datasets import BENCHMARK_CLASSIFICATION, load_tabular
from lab_a.forecasting import FAMILY as FORECAST_FAMILY

FAMILY = Z.FAMILY | FORECAST_FAMILY | ANOMALY_FAMILY


class Comparison(BaseModel):
    """A results file from another experiment to analyse across its datasets."""

    title: str
    results: Path
    metric: str
    higher_is_better: bool = True
    exclude_models: list[str] = Field(default_factory=lambda: ["dummy"])
    # Averaging a metric across datasets only makes sense if it is scale-free (AUC, MASE).
    scale_free: bool = True


class BenchmarkConfig(LabAConfig):
    datasets: list[str] = Field(default_factory=lambda: list(BENCHMARK_CLASSIFICATION))
    models: list[str] = Field(
        default_factory=lambda: ["logreg", "tree", "random_forest", "lightgbm", "svm", "knn", "mlp"]
    )
    suite_title: str = "OpenML-CC18 binary suite"
    n_splits: int = 5
    n_repeats: int = 2
    alpha: float = 0.05
    compare: list[Comparison] = Field(default_factory=list)


def run_suite(cfg: BenchmarkConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    cv = RepeatedStratifiedKFold(
        n_splits=cfg.n_splits, n_repeats=cfg.n_repeats, random_state=cfg.seed
    )
    for name in cfg.datasets:
        ds = load_tabular(name)
        for model in cfg.models:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scores = cross_val_score(
                    Z.build(model, "classification", cfg.seed), ds.X, ds.y, cv=cv, scoring="roc_auc"
                )
            rows += [
                {
                    "experiment": cfg.experiment,
                    "dataset": name,
                    "model": model,
                    "fold": i,
                    "metric": "roc_auc",
                    "value": float(v),
                }
                for i, v in enumerate(scores)
            ]
    return pd.DataFrame(rows)


def analyse(
    results: pd.DataFrame,
    *,
    title: str,
    metric: str,
    higher_is_better: bool,
    exclude: list[str],
    scale_free: bool = True,
    alpha: float,
    out: Outputs,
    slug: str,
) -> list[str]:
    """Cross-dataset analysis of a long-format results frame."""
    frame = results[(results.metric == metric) & (results.fold.astype(str) != "test")]
    frame = frame[~frame.model.isin(exclude)]
    table = frame.pivot_table(index="dataset", columns="model", values="value", aggfunc="mean")
    table = table.dropna(axis=1)  # a model must have a score on every dataset
    n, k = table.shape
    lines = [f"## {title}", ""]
    if n < 2 or k < 3:
        return [*lines, f"Not enough data ({n} datasets × {k} models) for a Friedman test.", ""]

    models = list(table.columns)
    fr = friedman(table.to_numpy(), models, higher_is_better=higher_is_better)
    ranks = fr.average_ranks
    order = sorted(models, key=ranks.get)  # type: ignore[arg-type]
    best = order[0]
    rank_matrix = table.rank(axis=1, ascending=not higher_is_better)
    wins_first = {m: int((rank_matrix[m] == 1).sum()) for m in models}

    cd = nemenyi_cd(k, n, alpha) if k <= 10 else float("nan")
    raw_p = {}
    for m in order[1:]:
        diff = table[best] - table[m]
        raw_p[m] = 1.0 if np.allclose(diff, 0) else float(wilcoxon(table[best], table[m]).pvalue)
    adj = holm(raw_p)
    sign = 1 if higher_is_better else -1
    record = {
        m: (
            int((sign * (table[m] - table[best]) > 0).sum()),
            int((sign * (table[best] - table[m]) > 0).sum()),
        )
        for m in order[1:]
    }  # (datasets where m beat the best model, datasets where it lost)

    lines += [
        f"{n} datasets × {k} models · metric `{metric}` "
        f"({'higher' if higher_is_better else 'lower'} is better) · α = {alpha}",
        "",
        md_table(
            [
                "Model",
                "Family",
                "Avg rank",
                f"Mean {metric}" if scale_free else "Datasets won",
                f"W–L vs `{best}`",
                "Wilcoxon p",
                "Holm-adjusted p",
                f"Within CD of `{best}`?",
            ],
            [
                [
                    f"**{m}**" if m == best else m,
                    FAMILY.get(m.split(" ")[0], "—"),
                    f"{ranks[m]:.2f}",
                    f"{table[m].mean():.4f}" if scale_free else f"{wins_first[m]}/{n}",
                    "—" if m == best else f"{record[m][0]}–{record[m][1]}",
                    "—" if m == best else fmt_p(raw_p[m]),
                    "—" if m == best else fmt_p(adj[m]),
                    "—" if m == best else ("yes" if ranks[m] - ranks[best] < cd else "no"),
                ]
                for m in order
            ],
        ),
        "",
        f"Friedman χ² = {fr.statistic:.2f}, p = {fmt_p(fr.pvalue)} → "
        + (
            "the models do **not** all perform the same."
            if fr.pvalue < alpha
            else "no evidence that the models differ; post-hoc results below are descriptive only."
        ),
        f"Nemenyi critical difference = {cd:.2f} ranks. "
        f"W–L: datasets where the model beat / lost to `{best}`.",
        "",
    ]
    if n < 6:
        lines += [
            f"With only {n} datasets the Wilcoxon test cannot reach p < 0.05 "
            "(its smallest attainable p-value is 2 / 2^n); treat this section as descriptive.",
            "",
        ]
    path, link = out.figure(f"{slug}_cd.png")
    cd_diagram(ranks, cd, path, title=title)
    lines += [f"![critical difference diagram]({link})", ""]
    return lines


def run(cfg: BenchmarkConfig) -> str:
    set_seed(cfg.seed)
    out = Outputs(cfg.output_dir, cfg.experiment)
    with track_run(cfg.experiment, cfg.model_dump(mode="json"), cfg.tags) as tracker:
        suite = run_suite(cfg)
        write_results(suite.to_dict("records"), out.results)
        per_dataset = suite.pivot_table(
            index="dataset", columns="model", values="value", aggfunc="mean"
        )
        tracker.log_metrics({f"{m}.mean_roc_auc": per_dataset[m].mean() for m in per_dataset})

        sections = analyse(
            suite,
            title=f"{cfg.suite_title} ({len(cfg.datasets)} datasets, CV ROC-AUC)",
            metric="roc_auc",
            higher_is_better=True,
            exclude=[],
            alpha=cfg.alpha,
            out=out,
            slug="cc18",
        )
        sections += [
            "<details><summary>Per-dataset mean ROC-AUC</summary>",
            "",
            md_table(
                ["Dataset", *per_dataset.columns],
                [[d, *(f"{v:.4f}" for v in r)] for d, r in per_dataset.iterrows()],
            ),
            "",
            "</details>",
            "",
        ]

        for comp in cfg.compare:
            path = comp.results if comp.results.is_absolute() else cfg.output_dir / comp.results
            if not path.exists():
                sections += [
                    f"## {comp.title}",
                    "",
                    f"_Skipped: `{comp.results.as_posix()}` not found._",
                    "",
                ]
                continue
            sections += analyse(
                pd.read_csv(path),
                title=comp.title,
                metric=comp.metric,
                higher_is_better=comp.higher_is_better,
                exclude=comp.exclude_models,
                scale_free=comp.scale_free,
                alpha=cfg.alpha,
                out=out,
                slug=path.stem,
            )

        header = [
            "# A4 — Do some models win in general? A multi-dataset statistical comparison",
            "",
            "Each dataset is one observation: models are ranked within every dataset and the "
            "ranks are compared (Friedman → Nemenyi / Wilcoxon-Holm). Within a dataset, "
            "scores are means over CV folds (A1, suite), backtest origins (A2) or seeds (A3b).",
            "",
        ]
        text = "\n".join(header + sections)
        out.write_report(text)
        tracker.log_artifact(out.report)
    return text


def main() -> None:
    cli(run, BenchmarkConfig, "A4 multi-dataset comparison")


if __name__ == "__main__":
    main()
