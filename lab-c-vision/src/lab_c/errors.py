"""C4 — Error analysis: confusions, confident mistakes, calibration and robustness.

    python -m lab_c.errors --config lab-c-vision/configs/c4_errors.yaml

Uses the Pets models trained in C2 (cached), so it runs in minutes:

* confusion matrix and the most-confused breed pairs,
* per-class F1 (best and worst breeds),
* the most confident mistakes, shown,
* calibration: expected calibration error (ECE), reliability diagram, and
  temperature scaling fitted on the validation split,
* robustness: accuracy under Gaussian noise, blur, low contrast and darkening
  at three severities — pretrained vs. trained from scratch.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from pydantic import Field
from sklearn.metrics import confusion_matrix, f1_score

from airlab.config import ExperimentConfig
from airlab.experiment import Outputs, cli
from airlab.plots import plt, save
from airlab.reporting import md_table, write_results
from airlab.seed import set_seed
from airlab.tracking import track_run
from lab_c.datasets import ImageSet
from lab_c.train import predict
from lab_c.transfer import FT_MODES, PetsConfig, finetune_modes, finetuned, pets_splits, scratch_cnn

DEFAULT_OUTPUT_DIR = Path("lab-c-vision/reports")
CORRUPTIONS = ("gaussian_noise", "blur", "contrast", "dark")


class ErrorsConfig(ExperimentConfig):
    output_dir: Path = DEFAULT_OUTPUT_DIR
    pets: PetsConfig = Field(default_factory=PetsConfig)
    main_model: str = "auto"  # e.g. "resnet18 · LP-FT", or "auto" = best validation accuracy
    n_confident_errors: int = 12
    ece_bins: int = 15


# ------------------------------------------------------------------- calibration


def ece(probs: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    """Expected calibration error: |accuracy - confidence| averaged over confidence bins."""
    conf, pred = probs.max(1), probs.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in itertools.pairwise(edges):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            total += mask.mean() * abs((pred[mask] == labels[mask]).mean() - conf[mask].mean())
    return float(total)


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Single temperature minimising validation NLL (Guo et al., 2017)."""
    logits, labels = logits.clone(), labels.clone()  # may come from inference mode
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure() -> torch.Tensor:
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.detach().exp())


def reliability(ax, probs: np.ndarray, labels: np.ndarray, bins: int, label: str) -> None:
    conf, correct = probs.max(1), probs.argmax(1) == labels
    edges = np.linspace(0, 1, bins + 1)
    centers, accs = [], []
    for lo, hi in itertools.pairwise(edges):
        mask = (conf > lo) & (conf <= hi)
        if mask.sum() >= 5:
            centers.append(conf[mask].mean())
            accs.append(correct[mask].mean())
    ax.plot(centers, accs, marker="o", ms=3, label=label)


# -------------------------------------------------------------------- robustness


def corrupt(images: torch.Tensor, kind: str, severity: int, seed: int = 0) -> torch.Tensor:
    """uint8 in, uint8 out. Severity 1..3."""
    x = images.float() / 255
    if kind == "gaussian_noise":
        g = torch.Generator().manual_seed(seed)
        x = x + torch.randn(x.shape, generator=g) * (0.04, 0.08, 0.16)[severity - 1]
    elif kind == "blur":
        sigma = (0.8, 1.5, 3.0)[severity - 1]
        radius = int(3 * sigma)
        k = torch.exp(-(torch.arange(-radius, radius + 1) ** 2) / (2 * sigma**2))
        k = (k / k.sum()).to(x.dtype)
        c = x.shape[1]
        x = F.conv2d(
            F.pad(x, (radius, radius, 0, 0), mode="reflect"),
            k.view(1, 1, 1, -1).repeat(c, 1, 1, 1),
            groups=c,
        )
        x = F.conv2d(
            F.pad(x, (0, 0, radius, radius), mode="reflect"),
            k.view(1, 1, -1, 1).repeat(c, 1, 1, 1),
            groups=c,
        )
    elif kind == "contrast":
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) * (0.5, 0.3, 0.15)[severity - 1] + mean
    elif kind == "dark":
        x = x * (0.6, 0.4, 0.2)[severity - 1]
    else:
        raise ValueError(kind)
    return (x.clamp(0, 1) * 255).round().to(torch.uint8)


def _with_images(data: ImageSet, images: torch.Tensor) -> ImageSet:
    return ImageSet(data.name, images, data.labels, data.classes, data.source)


# ---------------------------------------------------------------------- experiment


def run(cfg: ErrorsConfig) -> str:
    set_seed(cfg.seed)
    out = Outputs(cfg.output_dir, cfg.experiment)
    pc = cfg.pets
    train, val, test = pets_splits(pc, cfg.seed)
    n_classes = train.n_classes
    y_te = test.labels.numpy()

    models: dict[str, torch.nn.Module] = {
        f"{name} · {FT_MODES[mode]}": finetuned(
            name, pc, train, n_classes, mode, val, test, cfg.seed
        )[0]
        for mode in finetune_modes(pc)
        for name in pc.finetune
    }
    val_acc = {
        name: float((predict(m, val).argmax(1) == val.labels).float().mean())
        for name, m in models.items()
    }
    main = max(val_acc, key=val_acc.get) if cfg.main_model == "auto" else cfg.main_model  # type: ignore[arg-type]
    compared = {
        main: models[main],
        "SmallCNN (scratch)": scratch_cnn(pc, train, n_classes)[0],
    }
    rows: list[dict[str, Any]] = []

    with track_run(cfg.experiment, cfg.model_dump(mode="json"), cfg.tags) as tracker:
        logits = predict(models[main], test)
        probs = logits.softmax(1).numpy()
        pred = probs.argmax(1)
        names = [c.replace("_", " ") for c in test.classes]

        # --- confusions
        cm = confusion_matrix(y_te, pred, labels=range(n_classes))
        norm = cm / cm.sum(1, keepdims=True).clip(min=1)
        fig, ax = plt.subplots(figsize=(9, 8))
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(n_classes), names, rotation=90, fontsize=6)
        ax.set_yticks(range(n_classes), names, fontsize=6)
        ax.set(
            xlabel="predicted", ylabel="true", title=f"Confusion matrix (row-normalised) — {main}"
        )
        path, cm_link = out.figure("confusion_matrix.png")
        save(fig, path)
        off = cm.copy()
        np.fill_diagonal(off, 0)
        pairs = sorted(
            (
                (int(off[i, j] + off[j, i]), i, j)
                for i in range(n_classes)
                for j in range(i + 1, n_classes)
            ),
            reverse=True,
        )[:8]
        f1 = f1_score(y_te, pred, average=None, labels=range(n_classes))
        order = np.argsort(f1)

        # --- confident mistakes
        wrong = np.flatnonzero(pred != y_te)
        top = wrong[np.argsort(-probs[wrong].max(1))][: cfg.n_confident_errors]
        cols = 6
        fig, axes = plt.subplots(
            int(np.ceil(len(top) / cols)), cols, figsize=(12, 2.4 * np.ceil(len(top) / cols))
        )
        for ax, i in zip(np.atleast_1d(axes).ravel(), top, strict=False):
            ax.imshow(test.images[i].permute(1, 2, 0).numpy())
            ax.set_title(
                f"true: {names[y_te[i]]}\npred: {names[pred[i]]} ({probs[i].max():.0%})", fontsize=7
            )
            ax.axis("off")
        for ax in np.atleast_1d(axes).ravel()[len(top) :]:
            ax.axis("off")
        path, err_link = out.figure("confident_errors.png")
        save(fig, path)

        # --- calibration (both models)
        cal_rows = []
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
        for label, model in compared.items():
            lt, lv = predict(model, test), predict(model, val)
            t = fit_temperature(lv, val.labels)
            p_raw, p_cal = lt.softmax(1).numpy(), (lt / t).softmax(1).numpy()
            nll = lambda p: float(-np.log(p[np.arange(len(y_te)), y_te] + 1e-12).mean())  # noqa: E731
            cal_rows.append(
                [
                    label,
                    f"{t:.2f}",
                    f"{ece(p_raw, y_te, cfg.ece_bins):.3f} → {ece(p_cal, y_te, cfg.ece_bins):.3f}",
                    f"{nll(p_raw):.3f} → {nll(p_cal):.3f}",
                    f"{(p_raw.argmax(1) == y_te).mean():.3f}",
                ]
            )
            reliability(ax, p_raw, y_te, cfg.ece_bins, f"{label}, raw")
            reliability(ax, p_cal, y_te, cfg.ece_bins, f"{label}, T = {t:.2f}")
            rows += [
                {
                    "experiment": cfg.experiment,
                    "dataset": "pets",
                    "model": label,
                    "fold": "test",
                    "metric": m,
                    "value": v,
                }
                for m, v in (
                    ("ece_raw", ece(p_raw, y_te, cfg.ece_bins)),
                    ("ece_calibrated", ece(p_cal, y_te, cfg.ece_bins)),
                    ("temperature", t),
                )
            ]
        ax.set(xlabel="confidence", ylabel="accuracy", title="Reliability diagram (test)")
        ax.legend(fontsize=7)
        path, rel_link = out.figure("reliability.png")
        save(fig, path)

        # --- robustness
        rob: dict[str, dict[tuple[str, int], float]] = {k: {} for k in compared}
        clean = {
            k: float((predict(m, test).argmax(1) == test.labels).float().mean())
            for k, m in compared.items()
        }
        for kind in CORRUPTIONS:
            for severity in (1, 2, 3):
                corrupted = _with_images(test, corrupt(test.images, kind, severity, cfg.seed))
                for k, m in compared.items():
                    acc = float(
                        (predict(m, corrupted).argmax(1) == corrupted.labels).float().mean()
                    )
                    rob[k][(kind, severity)] = acc
                    rows.append(
                        {
                            "experiment": cfg.experiment,
                            "dataset": f"pets/{kind}",
                            "model": k,
                            "fold": severity,
                            "metric": "accuracy",
                            "value": acc,
                        }
                    )
        fig, axes = plt.subplots(1, len(CORRUPTIONS), figsize=(13, 3.2), sharey=True)
        for ax, kind in zip(axes, CORRUPTIONS, strict=True):
            for k in compared:
                ax.plot(
                    [0, 1, 2, 3],
                    [clean[k]] + [rob[k][(kind, s)] for s in (1, 2, 3)],
                    marker="o",
                    label=k,
                )
            ax.set(title=kind.replace("_", " "), xlabel="severity", xticks=[0, 1, 2, 3])
        axes[0].set_ylabel("test accuracy")
        axes[0].legend(fontsize=7)
        path, rob_link = out.figure("robustness.png")
        save(fig, path)
        examples = torch.stack([corrupt(test.images[:1], k, 3, cfg.seed)[0] for k in CORRUPTIONS])
        fig, axes = plt.subplots(1, len(CORRUPTIONS) + 1, figsize=(10, 2.2))
        for ax, img, title in zip(
            axes,
            [test.images[0], *examples],
            ["clean", *[f"{k} (3)" for k in CORRUPTIONS]],
            strict=True,
        ):
            ax.imshow(img.permute(1, 2, 0).numpy())
            ax.set_title(title.replace("_", " "), fontsize=8)
            ax.axis("off")
        path, ex_link = out.figure("corruption_examples.png")
        save(fig, path)

        write_results(rows, out.results)
        tracker.log_metrics({"main_accuracy": float((pred == y_te).mean())})
        rel_drop = {
            k: np.mean([rob[k][(kind, 3)] / clean[k] for kind in CORRUPTIONS]) for k in compared
        }
        lines = [
            "# C4 — Error analysis, calibration and robustness",
            "",
            f"Oxford-IIIT Pets test set ({len(test):,} images, 37 breeds). Main model: **{main}** "
            f"fine-tuned in C2, chosen by validation accuracy ({', '.join(f'{k} {v:.1%}' for k, v in val_acc.items())}). "
            f"Results in [`results/{cfg.experiment}.csv`](results/{cfg.experiment}.csv).",
            "",
            "## Where it goes wrong",
            "",
            f"Test accuracy {(pred == y_te).mean():.1%}. Most-confused breed pairs (both directions):",
            "",
            md_table(
                ["Breed A", "Breed B", "Confusions"], [[names[i], names[j], n] for n, i, j in pairs]
            ),
            "",
            md_table(
                ["Hardest breeds (F1)", "Easiest breeds (F1)"],
                [
                    [f"{names[a]} {f1[a]:.2f}", f"{names[b]} {f1[b]:.2f}"]
                    for a, b in zip(order[:5], order[::-1][:5], strict=True)
                ],
            ),
            "",
            f"![confusion matrix]({cm_link})",
            "",
            f"The {len(top)} most confident mistakes:",
            "",
            f"![confident errors]({err_link})",
            "",
            "## Calibration",
            "",
            "Expected calibration error (ECE, 15 bins) and negative log-likelihood before → after "
            "temperature scaling; the temperature is fitted on the validation split, so test accuracy "
            "is unchanged.",
            "",
            md_table(["Model", "Temperature", "ECE", "NLL", "Accuracy"], cal_rows),
            "",
            f"![reliability]({rel_link})",
            "",
            "## Robustness to corruptions",
            "",
            "Test accuracy under synthetic corruptions (severity 1–3), clean accuracy at severity 0. "
            "Mean share of clean accuracy kept at severity 3: "
            + ", ".join(f"{k} {v:.0%}" for k, v in rel_drop.items())
            + ".",
            "",
            md_table(
                ["Model", "Clean", *[f"{k.replace('_', ' ')} (3)" for k in CORRUPTIONS]],
                [
                    [k, f"{clean[k]:.3f}", *(f"{rob[k][(kind, 3)]:.3f}" for kind in CORRUPTIONS)]
                    for k in compared
                ],
            ),
            "",
            f"![robustness]({rob_link})",
            "",
            f"![corruption examples]({ex_link})",
            "",
        ]
        text = "\n".join(lines)
        out.write_report(text)
        tracker.log_artifact(out.report)
    return text


def main() -> None:
    cli(run, ErrorsConfig, "C4 error analysis")


if __name__ == "__main__":
    main()
