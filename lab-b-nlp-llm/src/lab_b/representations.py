"""B1 — Text representations: sparse counts vs. static embeddings vs. sentence transformers.

    python -m lab_b.representations --config lab-b-nlp-llm/configs/b1_representations.yaml

The same classifier (logistic regression, C chosen by 3-fold CV) is trained on
each representation, so differences in test accuracy come from the
representation alone. Also reported: dimensionality, sparsity and encoding
throughput, and a 2-D t-SNE map of two representations.
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import Field
from scipy import sparse
from scipy.stats import rankdata
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegressionCV
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, f1_score

from airlab.config import ExperimentConfig
from airlab.experiment import Outputs, cli
from airlab.metrics import bootstrap_ci
from airlab.plots import plt, save
from airlab.reporting import fmt_p, md_table, write_results
from airlab.seed import set_seed
from airlab.stats import friedman, mcnemar
from airlab.tracking import track_run
from lab_b.datasets import TextClassification, load_classification
from lab_b.encoders import Encoder

DEFAULT_OUTPUT_DIR = Path("lab-b-nlp-llm/reports")
FAMILY = {
    "bow": "sparse",
    "tfidf": "sparse",
    "hashing": "sparse",
    "static": "static embeddings",
    "st": "sentence transformer",
}


def family(spec: str) -> str:
    return FAMILY[spec.split(":")[0]]


class RepresentationConfig(ExperimentConfig):
    output_dir: Path = DEFAULT_OUTPUT_DIR
    datasets: list[str]
    encoders: list[str]
    max_train: int | None = 5000
    max_test: int | None = 2000
    throughput_sample: int = 512
    tsne_dataset: str | None = None
    tsne_encoders: list[str] = Field(default_factory=list)
    tsne_points: int = 600
    bootstrap_resamples: int = 1000


def throughput(spec: str, texts: list[str], fit_texts: list[str]) -> float:
    """Texts per second, measured without the embedding cache."""
    enc = Encoder(spec, cache=False).fit(fit_texts)
    enc.transform(texts[:8])  # warm-up (model load, first batch)
    start = time.perf_counter()
    enc.transform(texts)
    return len(texts) / (time.perf_counter() - start)


def run_dataset(
    ds: TextClassification, cfg: RepresentationConfig, out: Outputs
) -> tuple[list[str], list, dict]:
    rows: list[dict[str, Any]] = []
    results: dict[str, dict[str, Any]] = {}
    for spec in cfg.encoders:
        enc = Encoder(spec).fit(ds.train_texts)
        X_tr, X_te = enc.transform(ds.train_texts), enc.transform(ds.test_texts)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf = LogisticRegressionCV(Cs=[0.1, 1.0, 10.0], cv=3, max_iter=3000, n_jobs=-1).fit(
                X_tr, ds.train_labels
            )
        pred = clf.predict(X_te)
        acc = bootstrap_ci(
            ds.test_labels, pred, accuracy_score, n_resamples=cfg.bootstrap_resamples, seed=cfg.seed
        )
        f1 = f1_score(ds.test_labels, pred, average="macro")
        density = getattr(X_tr, "nnz", np.prod(X_tr.shape)) / np.prod(X_tr.shape)
        tps = throughput(spec, ds.test_texts[: cfg.throughput_sample], ds.train_texts)
        results[spec] = {
            "pred": pred,
            "acc": acc,
            "f1": f1,
            "dim": X_tr.shape[1],
            "density": density,
            "tps": tps,
            "C": float(np.mean(clf.C_)),
        }
        for metric, value in {
            "accuracy": acc.value,
            "f1_macro": f1,
            "dimension": X_tr.shape[1],
            "texts_per_s": tps,
        }.items():
            rows.append(
                {
                    "experiment": cfg.experiment,
                    "dataset": ds.name,
                    "model": enc.label,
                    "fold": "test",
                    "metric": metric,
                    "value": float(value),
                }
            )

    ranking = sorted(cfg.encoders, key=lambda s: -results[s]["acc"].value)
    best = ranking[0]
    tests = {
        s: mcnemar(ds.test_labels, results[best]["pred"], results[s]["pred"]) for s in ranking[1:]
    }
    lines = [
        f"## `{ds.name}`",
        "",
        f"{ds.source} · {ds.n_classes} classes ({', '.join(ds.label_names)}) · "
        f"train {len(ds.train_texts):,} / test {len(ds.test_texts):,}",
        "",
        md_table(
            [
                "Representation",
                "Family",
                "Accuracy (95% CI)",
                "Macro-F1",
                "Dim",
                "Non-zeros",
                "Texts/s (CPU)",
                "McNemar p vs best",
            ],
            [
                [
                    f"**{Encoder(s).label}**" if s == best else Encoder(s).label,
                    family(s),
                    str(results[s]["acc"]),
                    f"{results[s]['f1']:.4f}",
                    f"{results[s]['dim']:,}",
                    f"{results[s]['density']:.2%}",
                    f"{results[s]['tps']:,.0f}",
                    "—" if s == best else fmt_p(tests[s].pvalue),
                ]
                for s in ranking
            ],
        ),
        "",
    ]
    return lines, rows, results


def tsne_figure(
    ds: TextClassification, specs: list[str], cfg: RepresentationConfig, out: Outputs
) -> str:
    rng = np.random.default_rng(cfg.seed)
    idx = rng.choice(len(ds.test_texts), min(cfg.tsne_points, len(ds.test_texts)), replace=False)
    texts = [ds.test_texts[i] for i in idx]
    fig, axes = plt.subplots(1, len(specs), figsize=(5 * len(specs), 4.5))
    for ax, spec in zip(np.atleast_1d(axes), specs, strict=True):
        X = Encoder(spec).fit(ds.train_texts).transform(texts)
        if sparse.issparse(X):
            X = TruncatedSVD(50, random_state=cfg.seed).fit_transform(X)
        Z = TSNE(2, init="pca", random_state=cfg.seed, perplexity=30).fit_transform(np.asarray(X))
        for c, name in enumerate(ds.label_names):
            mask = ds.test_labels[idx] == c
            ax.scatter(Z[mask, 0], Z[mask, 1], s=6, label=name)
        ax.set(title=Encoder(spec).label, xticks=[], yticks=[])
    np.atleast_1d(axes)[0].legend(fontsize=8, markerscale=2)
    fig.suptitle(f"t-SNE of {len(texts)} `{ds.name}` test texts")
    path, link = out.figure(f"{ds.name}_tsne.png")
    save(fig, path)
    return link


def run(cfg: RepresentationConfig) -> str:
    set_seed(cfg.seed)
    out = Outputs(cfg.output_dir, cfg.experiment)
    sections: list[str] = []
    rows: list[dict[str, Any]] = []
    per_dataset: dict[str, dict[str, Any]] = {}
    with track_run(cfg.experiment, cfg.model_dump(mode="json"), cfg.tags) as tracker:
        for name in cfg.datasets:
            ds = load_classification(
                name, max_train=cfg.max_train, max_test=cfg.max_test, seed=cfg.seed
            )
            lines, ds_rows, results = run_dataset(ds, cfg, out)
            if name == cfg.tsne_dataset and cfg.tsne_encoders:
                lines += [f"![t-SNE]({tsne_figure(ds, cfg.tsne_encoders, cfg, out)})", ""]
            sections += lines
            rows += ds_rows
            per_dataset[name] = results
            tracker.log_metrics(
                {f"{name}.{Encoder(s).label}.accuracy": r["acc"].value for s, r in results.items()}
            )
        write_results(rows, out.results)

        # accuracy vs. throughput, all datasets
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for spec in cfg.encoders:
            xs = [per_dataset[d][spec]["tps"] for d in cfg.datasets]
            ys = [per_dataset[d][spec]["acc"].value for d in cfg.datasets]
            ax.scatter(xs, ys, label=Encoder(spec).label, s=40)
        ax.set(
            xscale="log",
            xlabel="texts per second (CPU, log scale)",
            ylabel="test accuracy",
            title="Accuracy vs. encoding speed",
        )
        ax.legend(fontsize=8)
        path, speed_link = out.figure("accuracy_vs_speed.png")
        save(fig, path)

        matrix = np.array(
            [[per_dataset[d][s]["acc"].value for s in cfg.encoders] for d in cfg.datasets]
        )
        ranks = rankdata(-matrix, axis=1).mean(axis=0)  # ties share the average rank
        header = [
            "# B1 — Text representations: sparse vs. static vs. contextual embeddings",
            "",
            "Same classifier for every representation (logistic regression, C ∈ {0.1, 1, 10} by "
            "3-fold CV); test accuracy with 95% bootstrap CI; McNemar against the best "
            f"representation of each dataset. Results in [`results/{cfg.experiment}.csv`](results/{cfg.experiment}.csv).",
            "",
            md_table(
                ["Representation", "Family", *cfg.datasets, "avg rank"],
                [
                    [
                        Encoder(s).label,
                        family(s),
                        *(f"{per_dataset[d][s]['acc'].value:.4f}" for d in cfg.datasets),
                        f"{ranks[i]:.2f}",
                    ]
                    for i, s in sorted(enumerate(cfg.encoders), key=lambda t: ranks[t[0]])
                ],
            ),
            "",
        ]
        if len(cfg.datasets) >= 2 and len(cfg.encoders) >= 3:
            fr = friedman(matrix, [Encoder(s).label for s in cfg.encoders])
            header += [f"Friedman across {len(cfg.datasets)} datasets: p = {fmt_p(fr.pvalue)}.", ""]
        header += [f"![accuracy vs speed]({speed_link})", ""]
        text = "\n".join(header + sections)
        out.write_report(text)
        tracker.log_artifact(out.report)
    return text


def main() -> None:
    cli(run, RepresentationConfig, "B1 text representations")


if __name__ == "__main__":
    main()
