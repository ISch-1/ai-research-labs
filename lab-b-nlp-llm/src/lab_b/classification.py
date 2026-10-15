"""B2 — Text classification: from TF-IDF to fine-tuned transformers to prompting an LLM.

    python -m lab_b.classification --config lab-b-nlp-llm/configs/b2_classification.yaml

Methods (each trained on n labelled examples, n on a log grid → learning curves):

* ``tfidf``     — TF-IDF (1-2 grams) + logistic regression
* ``embed``     — frozen sentence embeddings + logistic regression
* ``finetune``  — DistilBERT fine-tuned end to end (CPU, small n only)
* ``zero:<llm>`` / ``few:<llm>`` — an instruction-tuned LLM picks the label by
  scoring each label's log-probability (no generation, no parsing); few-shot adds
  one labelled example per class, with its KV cache computed once.

The question: how many labelled examples does a classic model need to beat an
LLM that has seen none?
"""

from __future__ import annotations

import hashlib
import math
import re
import time
import warnings
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field
from scipy.stats import rankdata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline

from airlab.config import ExperimentConfig
from airlab.data import data_dir
from airlab.experiment import Outputs, cli, host_description
from airlab.metrics import bootstrap_ci
from airlab.plots import plt, save
from airlab.reporting import fmt_p, md_table, write_results
from airlab.seed import set_seed
from airlab.stats import mcnemar
from airlab.tracking import track_run
from lab_b.datasets import TextClassification, load_classification
from lab_b.encoders import Encoder
from lab_b.llm import LLM, load_llm

DEFAULT_OUTPUT_DIR = Path("lab-b-nlp-llm/reports")


class FinetuneConfig(BaseModel):
    model: str = "distilbert-base-uncased"
    max_n: int = 2000
    max_length: int = 96
    batch_size: int = 16
    learning_rate: float = 5e-5
    min_steps: int = 100


class ClassificationConfig(ExperimentConfig):
    output_dir: Path = DEFAULT_OUTPUT_DIR
    datasets: list[str]
    train_sizes: list[int] = Field(default_factory=lambda: [32, 128, 512, 2048, 8192])
    max_test: int = 300
    embedder: str = "st:BAAI/bge-small-en-v1.5"
    finetune: FinetuneConfig | None = Field(default_factory=FinetuneConfig)
    llms: list[str] = Field(default_factory=list)
    bootstrap_resamples: int = 1000


# ------------------------------------------------------------------------- methods


def _logreg() -> LogisticRegression:
    return LogisticRegression(C=10.0, max_iter=3000)


def fit_tfidf(train: list[str], y: np.ndarray, test: list[str]) -> tuple[np.ndarray, float, float]:
    """Returns (predictions, train seconds, inference seconds for the whole test set)."""
    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1), _logreg()
    )
    t0 = time.perf_counter()
    model.fit(train, y)
    t1 = time.perf_counter()
    pred = model.predict(test)
    return pred, t1 - t0, time.perf_counter() - t1


def fit_embed(
    spec: str, train: list[str], y: np.ndarray, test: list[str]
) -> tuple[np.ndarray, float, float]:
    # No embedding cache here: encoding time is part of what is being measured.
    enc = Encoder(spec, cache=False).fit(train)
    enc.transform(train[:2])  # model load / warm-up is not training time
    t0 = time.perf_counter()
    clf = _logreg().fit(enc.transform(train), y)
    t1 = time.perf_counter()
    pred = clf.predict(enc.transform(test))
    return pred, t1 - t0, time.perf_counter() - t1


def fit_finetune(
    cfg: FinetuneConfig, train: list[str], y: np.ndarray, test: list[str], n_classes: int, seed: int
) -> tuple[np.ndarray, float, float]:
    """Plain PyTorch loop: AdamW, linear warm-up/decay, enough epochs for ``min_steps`` updates."""
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(cfg.model)
    model = AutoModelForSequenceClassification.from_pretrained(cfg.model, num_labels=n_classes)

    def batches(texts: list[str], labels: np.ndarray | None, shuffle: bool):
        order = (
            np.random.default_rng(seed).permutation(len(texts))
            if shuffle
            else np.arange(len(texts))
        )
        for i in range(0, len(texts), cfg.batch_size):
            idx = order[i : i + cfg.batch_size]
            enc = tok(
                [texts[j] for j in idx],
                truncation=True,
                max_length=cfg.max_length,
                padding=True,
                return_tensors="pt",
            )
            if labels is not None:
                enc["labels"] = torch.tensor(labels[idx])
            yield enc

    steps_per_epoch = math.ceil(len(train) / cfg.batch_size)
    epochs = max(3, math.ceil(cfg.min_steps / steps_per_epoch))
    total = epochs * steps_per_epoch
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)
    t0 = time.perf_counter()
    model.train()
    for epoch in range(epochs):
        for batch in batches(train, y, shuffle=True):
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
        seed += epoch  # different shuffle each epoch
    t1 = time.perf_counter()
    model.eval()
    preds = []
    with torch.inference_mode():
        for batch in batches(test, None, shuffle=False):
            preds.append(model(**batch).logits.argmax(-1).numpy())
    return np.concatenate(preds), t1 - t0, time.perf_counter() - t1


def llm_messages(
    ds: TextClassification, few_shot: bool, seed: int
) -> tuple[list[dict[str, str]], list[str]]:
    """System prompt (+ one example per class) and the label strings the LLM chooses from."""
    labels = [n.capitalize() for n in ds.label_names]
    prefix = [
        {
            "role": "system",
            "content": "Classify the text into exactly one of these categories: "
            + ", ".join(labels)
            + ". Answer with the category name only.",
        }
    ]
    if few_shot:
        rng = np.random.default_rng(seed)
        for c, label in enumerate(labels):
            i = int(rng.choice(np.flatnonzero(ds.train_labels == c)))
            prefix += [
                {"role": "user", "content": f"Text: {ds.train_texts[i]}"},
                {"role": "assistant", "content": label},
            ]
    return prefix, labels


def predict_llm(
    llm: LLM, ds: TextClassification, test: list[str], few_shot: bool, seed: int
) -> tuple[np.ndarray, float, float]:
    """Predictions, 0 training seconds, and the *median* per-text latency times n.

    The median makes the cost estimate robust to one-off stalls (e.g. the
    laptop sleeping mid-run); the KV cache of the few-shot prefix is built by a
    warm-up call and is excluded from the per-text latency.
    """
    prefix, labels = llm_messages(ds, few_shot, seed)
    if prefix:
        llm.choice_logprobs(prefix, [{"role": "user", "content": "Text: warm-up"}], labels)
    preds, latencies = [], []
    for t in test:
        t0 = time.perf_counter()
        scores = llm.choice_logprobs(prefix, [{"role": "user", "content": f"Text: {t}"}], labels)
        latencies.append(time.perf_counter() - t0)
        preds.append(int(np.argmax(scores)))
    return np.array(preds), 0.0, float(np.median(latencies)) * len(test)


# ---------------------------------------------------------------------- experiment


HOSTS: set[str] = set()  # machines whose timings appear in the current report


class PredictionCache:
    """Per-method predictions and timings on disk: an interrupted run resumes, a re-run is free.

    Each entry also stores the machine it was measured on, so a report assembled
    from caches (e.g. shards run on CI) states where its timings come from.
    """

    def __init__(self, ds: TextClassification, cfg: ClassificationConfig) -> None:
        key = "\x1f".join([ds.name, str(cfg.seed), *ds.test_texts])
        digest = hashlib.sha1(key.encode()).hexdigest()[:16]
        self.dir = data_dir() / "cache" / "b2" / f"{ds.name}-{digest}"
        self.dir.mkdir(parents=True, exist_ok=True)

    def get_or_run(
        self, key: str, job: Callable[[], tuple[np.ndarray, float, float]]
    ) -> tuple[np.ndarray, float, float]:
        safe = re.sub(r"[^\w.-]", "_", key)
        path = self.dir / f"{safe}.npz"
        if path.exists():
            data = np.load(path)
            HOSTS.add(str(data["host"]) if "host" in data else "unknown")
            return data["pred"], float(data["train_s"]), float(data["infer_s"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred, train_s, infer_s = job()
        host = host_description()
        HOSTS.add(host)
        np.savez(path, pred=pred, train_s=train_s, infer_s=infer_s, host=np.array(host))
        return pred, train_s, infer_s


def run_dataset(
    ds: TextClassification, cfg: ClassificationConfig, llms: dict[str, LLM], out: Outputs
) -> tuple[list[str], list, dict]:
    y_te = ds.test_labels
    rows: list[dict[str, Any]] = []
    curve: dict[str, dict[int, float]] = {}
    final: dict[str, dict[str, Any]] = {}  # method -> best-n predictions & stats
    rng = np.random.default_rng(cfg.seed)
    order = rng.permutation(len(ds.train_texts))  # nested training subsets: n=32 ⊂ n=128 ⊂ ...

    def record(method: str, n: int, pred: np.ndarray, train_s: float, infer_s: float) -> None:
        acc = float(accuracy_score(y_te, pred))
        curve.setdefault(method, {})[n] = acc
        final[method] = {"n": n, "pred": pred, "train_s": train_s, "ms": 1000 * infer_s / len(y_te)}
        for metric, value in {
            "accuracy": acc,
            "f1_macro": f1_score(y_te, pred, average="macro"),
            "train_s": train_s,
        }.items():
            rows.append(
                {
                    "experiment": cfg.experiment,
                    "dataset": ds.name,
                    "model": method,
                    "fold": n,
                    "metric": metric,
                    "value": float(value),
                }
            )

    cache = PredictionCache(ds, cfg)
    sizes = [n for n in cfg.train_sizes if n <= len(ds.train_texts)]
    for n in sizes:
        idx = order[:n]
        if len(np.unique(ds.train_labels[idx])) < ds.n_classes:
            continue
        train, y = [ds.train_texts[i] for i in idx], ds.train_labels[idx]
        jobs: list[tuple[str, Callable[[], tuple[np.ndarray, float, float]]]] = [
            ("tfidf", partial(fit_tfidf, train, y, ds.test_texts)),
            ("embed", partial(fit_embed, cfg.embedder, train, y, ds.test_texts)),
        ]
        if cfg.finetune and n <= cfg.finetune.max_n:
            jobs.append(
                (
                    "finetune",
                    partial(
                        fit_finetune, cfg.finetune, train, y, ds.test_texts, ds.n_classes, cfg.seed
                    ),
                )
            )
        for method, job in jobs:
            record(method, n, *cache.get_or_run(f"{method}-{n}", job))

    for llm in llms.values():
        for few in (False, True):
            method = f"{'few' if few else 'zero'}:{llm.name}"
            job = partial(predict_llm, llm, ds, ds.test_texts, few, cfg.seed)
            record(method, ds.n_classes if few else 0, *cache.get_or_run(method, job))

    # --- report
    acc_ci = {
        m: bootstrap_ci(
            y_te, f["pred"], accuracy_score, n_resamples=cfg.bootstrap_resamples, seed=cfg.seed
        )
        for m, f in final.items()
    }
    ranking = sorted(final, key=lambda m: -acc_ci[m].value)
    best = ranking[0]
    tests = {m: mcnemar(y_te, final[best]["pred"], final[m]["pred"]) for m in ranking[1:]}

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for method in ("tfidf", "embed", "finetune"):
        if method in curve:
            ns = sorted(curve[method])
            ax.plot(ns, [curve[method][n] for n in ns], marker="o", label=method)
    for m in final:
        if ":" in m:
            ax.axhline(
                curve[m][final[m]["n"]], ls="--" if m.startswith("zero") else ":", lw=1.2, label=m
            )
    ax.set(
        xscale="log",
        xlabel="labelled training examples (log scale)",
        ylabel="test accuracy",
        title=f"Learning curves — {ds.name}",
    )
    ax.legend(fontsize=7, ncol=2)
    path, link = out.figure(f"{ds.name}_learning_curve.png")
    save(fig, path)

    crossover = []
    for m in [m for m in final if ":" in m]:
        target = curve[m][final[m]["n"]]
        for method in [m for m in ("tfidf", "embed", "finetune") if m in curve]:
            beat = [n for n in sorted(curve[method]) if curve[method][n] > target]
            crossover.append([m, method, f"{beat[0]:,}" if beat else f"> {max(curve[method]):,}"])

    lines = [
        f"## `{ds.name}`",
        "",
        f"{ds.source} · {ds.n_classes} classes · test {len(y_te):,} texts · training subsets are nested",
        "",
        md_table(
            [
                "Method",
                "Labelled examples",
                "Accuracy (95% CI)",
                "Train s",
                "Inference ms/text",
                "McNemar p vs best",
            ],
            [
                [
                    f"**{m}**" if m == best else m,
                    f"{final[m]['n']:,}",
                    str(acc_ci[m]),
                    f"{final[m]['train_s']:.1f}",
                    f"{final[m]['ms']:.1f}" if final[m]["ms"] else "—",
                    "—" if m == best else fmt_p(tests[m].pvalue),
                ]
                for m in ranking
            ],
        ),
        "",
        f"![learning curve]({link})",
        "",
    ]
    if crossover:
        lines += [
            "Labelled examples a trained model needs to beat each LLM setting:",
            "",
            md_table(["LLM setting", "Trained method", "Examples needed"], crossover),
            "",
        ]
    return lines, rows, {m: acc_ci[m].value for m in final}


def run(cfg: ClassificationConfig) -> str:
    set_seed(cfg.seed)
    out = Outputs(cfg.output_dir, cfg.experiment)
    llms = {spec: load_llm(spec, lazy=True) for spec in cfg.llms}
    HOSTS.clear()
    sections: list[str] = []
    rows: list[dict[str, Any]] = []
    summary: dict[str, dict[str, float]] = {}
    with track_run(cfg.experiment, cfg.model_dump(mode="json"), cfg.tags) as tracker:
        for name in cfg.datasets:
            ds = load_classification(
                name, max_train=max(cfg.train_sizes), max_test=cfg.max_test, seed=cfg.seed
            )
            lines, ds_rows, best_acc = run_dataset(ds, cfg, llms, out)
            sections += lines
            rows += ds_rows
            summary[name] = best_acc
            tracker.log_metrics({f"{name}.{m}.accuracy": v for m, v in best_acc.items()})
        write_results(rows, out.results)
        methods = sorted({m for s in summary.values() for m in s})
        matrix = np.array([[summary[d].get(m, np.nan) for m in methods] for d in cfg.datasets])
        ranks = np.nanmean(
            np.where(np.isnan(matrix), np.nan, rankdata(-np.nan_to_num(matrix, nan=-1), axis=1)),
            axis=0,
        )
        header = [
            "# B2 — Text classification: TF-IDF → embeddings → fine-tuning → prompting",
            "",
            "Accuracy of each method at its largest training size (LLM rows use 0 or one example "
            "per class). LLMs choose a label by comparing label log-probabilities, so every "
            f"answer is valid. Results per training size in [`results/{cfg.experiment}.csv`](results/{cfg.experiment}.csv).",
            "",
            "Timings measured on: " + "; ".join(sorted(HOSTS)) + ". LLM inference time is the "
            "median per text (robust to one-off stalls).",
            "",
            md_table(
                ["Method", *cfg.datasets, "avg rank"],
                [
                    [
                        m,
                        *(f"{summary[d][m]:.4f}" if m in summary[d] else "—" for d in cfg.datasets),
                        f"{ranks[i]:.2f}",
                    ]
                    for i, m in sorted(enumerate(methods), key=lambda t: ranks[t[0]])
                ],
            ),
            "",
        ]
        text = "\n".join(header + sections)
        out.write_report(text)
        tracker.log_artifact(out.report)
    return text


def main() -> None:
    cli(run, ClassificationConfig, "B2 text classification")


if __name__ == "__main__":
    main()
