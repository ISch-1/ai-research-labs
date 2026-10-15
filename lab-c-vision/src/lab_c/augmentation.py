"""C3 — Data augmentation: which transformations help, and how much more when data are scarce?

    python -m lab_c.augmentation --config lab-c-vision/configs/c3_augmentation.yaml

The same small CNN is trained on CIFAR-10 with each augmentation recipe and on
nested fractions of the training set. Reported: test accuracy (bootstrap CI),
the gain over no augmentation (McNemar), and the generalisation gap (training
minus test accuracy) — augmentation as a regulariser.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pydantic import Field

from airlab.config import ExperimentConfig
from airlab.experiment import Outputs, cli
from airlab.metrics import bootstrap_ci
from airlab.plots import plt, save
from airlab.reporting import fmt_p, md_table, write_results
from airlab.seed import set_seed
from airlab.stats import mcnemar
from airlab.tracking import track_run
from lab_c.augment import augment
from lab_c.datasets import load_images
from lab_c.nets import SmallCNN
from lab_c.train import TrainConfig, predict, train_or_load

DEFAULT_OUTPUT_DIR = Path("lab-c-vision/reports")


class AugmentationConfig(ExperimentConfig):
    output_dir: Path = DEFAULT_OUTPUT_DIR
    dataset: str = "cifar10"
    variants: list[str] = Field(
        default_factory=lambda: [
            "none",
            "flip+crop",
            "flip+crop+color",
            "flip+crop+cutout",
            "flip+crop+mixup",
            "flip+crop+cutmix",
        ]
    )
    fractions: list[float] = Field(default_factory=lambda: [0.1, 0.25, 1.0])
    # fewer images -> more epochs, so every run gets a comparable number of updates
    epochs: dict[float, int] = Field(default_factory=lambda: {0.1: 40, 0.25: 25, 1.0: 15})
    width: int = 32
    lr: float = 2e-3
    batch_size: int = 128
    max_test: int | None = None
    gap_sample: int = 5000
    bootstrap_resamples: int = 1000


def run(cfg: AugmentationConfig) -> str:
    set_seed(cfg.seed)
    out = Outputs(cfg.output_dir, cfg.experiment)
    full = load_images(cfg.dataset, "train")
    test = load_images(cfg.dataset, "test")
    if cfg.max_test:
        test = test.stratified_fraction(cfg.max_test / len(test), cfg.seed)
    y = test.labels.numpy()
    rows: list[dict[str, Any]] = []
    res: dict[tuple[str, float], dict[str, Any]] = {}

    with track_run(cfg.experiment, cfg.model_dump(mode="json"), cfg.tags) as tracker:
        for fraction in cfg.fractions:
            train = full.stratified_fraction(fraction, cfg.seed)
            gap_set = train.subset(
                torch.randperm(len(train), generator=torch.Generator().manual_seed(cfg.seed))[
                    : cfg.gap_sample
                ]
            )
            for variant in cfg.variants:
                tc = TrainConfig(
                    epochs=cfg.epochs[fraction],
                    lr=cfg.lr,
                    batch_size=cfg.batch_size,
                    augment=variant,
                    seed=cfg.seed,
                )
                build = partial(SmallCNN, train.images.shape[1], train.n_classes, width=cfg.width)
                model, hist = train_or_load(
                    f"aug-{variant.replace('+', '_')}-{fraction}", build, train, tc
                )
                pred = predict(model, test).argmax(1).numpy()
                train_acc = float(
                    (predict(model, gap_set).argmax(1) == gap_set.labels).float().mean()
                )
                acc = bootstrap_ci(
                    y,
                    pred,
                    lambda a, b: float(np.mean(a == b)),
                    n_resamples=cfg.bootstrap_resamples,
                    seed=cfg.seed,
                )
                res[(variant, fraction)] = {
                    "pred": pred,
                    "acc": acc,
                    "gap": train_acc - acc.value,
                    "minutes": hist.train_seconds / 60,
                    "host": hist.host,
                }
                for metric, value in (
                    ("accuracy", acc.value),
                    ("train_accuracy", train_acc),
                    ("train_minutes", hist.train_seconds / 60),
                ):
                    rows.append(
                        {
                            "experiment": cfg.experiment,
                            "dataset": f"{cfg.dataset}@{fraction}",
                            "model": variant,
                            "fold": "test",
                            "metric": metric,
                            "value": value,
                        }
                    )
                tracker.log_metrics({f"{variant}@{fraction}.accuracy": acc.value})
        write_results(rows, out.results)

        # accuracy vs. fraction, one line per recipe
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        for variant in cfg.variants:
            ax.plot(
                cfg.fractions,
                [res[(variant, f)]["acc"].value for f in cfg.fractions],
                marker="o",
                label=variant,
            )
        ax.set(
            xscale="log",
            xlabel="fraction of the training set (log scale)",
            ylabel="test accuracy",
            title="Augmentation × data size",
        )
        ax.set_xticks(cfg.fractions, [f"{f:.0%}" for f in cfg.fractions])
        ax.legend(fontsize=8)
        path, curve_link = out.figure("accuracy_vs_fraction.png")
        save(fig, path)

        # what each recipe does to a batch
        g = torch.Generator().manual_seed(cfg.seed)
        sample = full.images[:8].float() / 255
        fig, axes = plt.subplots(len(cfg.variants), 8, figsize=(8, 1.05 * len(cfg.variants)), squeeze=False)
        for row, variant in enumerate(cfg.variants):
            xa, _ = augment(sample, full.labels[:8], variant, full.n_classes, g)
            for c in range(8):
                ax = axes[row, c]
                img = xa[c].permute(1, 2, 0).numpy()
                ax.imshow(img.squeeze(), cmap="gray" if img.shape[-1] == 1 else None)
                ax.set_xticks([])
                ax.set_yticks([])
            axes[row, 0].set_ylabel(variant, rotation=0, ha="right", va="center", fontsize=7)
        path, grid_link = out.figure("augmentation_examples.png")
        save(fig, path)

        table = []
        for fraction in cfg.fractions:
            base = res[("none", fraction)] if "none" in cfg.variants else None
            for variant in sorted(cfg.variants, key=lambda v: -res[(v, fraction)]["acc"].value):
                r = res[(variant, fraction)]
                gain = (
                    "—"
                    if base is None or variant == "none"
                    else f"{r['acc'].value - base['acc'].value:+.4f}"
                )
                p = (
                    "—"
                    if base is None or variant == "none"
                    else fmt_p(mcnemar(y, r["pred"], base["pred"]).pvalue)
                )
                table.append(
                    [
                        f"{fraction:.0%}",
                        variant,
                        str(r["acc"]),
                        gain,
                        p,
                        f"{r['gap']:+.3f}",
                        f"{r['minutes']:.1f}",
                    ]
                )
        hosts = sorted({r["host"] for r in res.values() if r["host"]})
        header = [
            "# C3 — Data augmentation × data size",
            "",
            f"SmallCNN (width {cfg.width}) on {full.source}; nested stratified fractions of the "
            f"{len(full):,} training images; epochs per fraction {cfg.epochs}; {len(test):,} test images. "
            f"Training on: {'; '.join(hosts) or 'unknown'}. Results in [`results/{cfg.experiment}.csv`](results/{cfg.experiment}.csv).",
            "",
            md_table(
                [
                    "Data",
                    "Augmentation",
                    "Test accuracy (95% CI)",
                    "Gain vs none",
                    "McNemar p vs none",
                    "Train - test accuracy",
                    "Train min",
                ],
                table,
            ),
            "",
            f"![accuracy vs fraction]({curve_link})",
            "",
            "What each recipe does to the same eight training images:",
            "",
            f"![augmentation examples]({grid_link})",
            "",
        ]
        text = "\n".join(header)
        out.write_report(text)
        tracker.log_artifact(out.report)
    return text


def main() -> None:
    cli(run, AugmentationConfig, "C3 data augmentation")


if __name__ == "__main__":
    main()
