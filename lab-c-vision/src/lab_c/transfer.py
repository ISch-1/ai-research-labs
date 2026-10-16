"""C2 — CNNs and transfer learning.

    python -m lab_c.transfer --config lab-c-vision/configs/c2_transfer.yaml

A. Inductive bias (CIFAR-10): an MLP and a small CNN trained with the same
   budget and augmentation. Convolutions share weights across positions; the MLP
   has to learn every position separately.
B. Transfer learning (Oxford-IIIT Pets, 37 breeds, ~3.3k training images):
   * a small CNN trained from scratch,
   * ImageNet-pretrained backbones used frozen as feature extractors with a
     logistic-regression head ("linear probe"),
   * the same backbones fine-tuned end to end.
   Accuracy with bootstrap CIs, McNemar vs. the best, trainable parameters,
   training time and inference throughput on CPU.

Fine-tuned models are cached and reused by C4 (error analysis) and C5
(interpretability).
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pydantic import BaseModel, Field
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from airlab.config import ExperimentConfig
from airlab.data import data_dir
from airlab.experiment import Outputs, cli
from airlab.metrics import bootstrap_ci
from airlab.plots import plt, save
from airlab.reporting import fmt_p, md_table, write_results
from airlab.seed import set_seed
from airlab.stats import mcnemar
from airlab.tracking import track_run
from lab_c.datasets import ImageSet, load_images
from lab_c.nets import MLP, SmallCNN, n_params, normalization, pretrained
from lab_c.train import (
    TrainConfig,
    cache_key,
    classifier_head,
    predict,
    throughput,
    to_input,
    train_or_load,
)

DEFAULT_OUTPUT_DIR = Path("lab-c-vision/reports")
CAT_BREEDS = {
    "abyssinian",
    "bengal",
    "birman",
    "bombay",
    "british_shorthair",
    "egyptian_mau",
    "maine_coon",
    "persian",
    "ragdoll",
    "russian_blue",
    "siamese",
    "sphynx",
}


class CifarConfig(BaseModel):
    dataset: str = "cifar10"
    max_train: int | None = None
    mlp_hidden: list[int] = Field(default_factory=lambda: [512, 256])
    cnn_width: int = 32
    train: TrainConfig = Field(
        default_factory=lambda: TrainConfig(epochs=15, lr=2e-3, augment="flip+crop")
    )


class PetsConfig(BaseModel):
    dataset: str = "pets"
    include_scratch: bool = True
    ft_modes: list[str] | None = None  # subset of naive / lpft / lpft_bn; None = all configured
    image_size: int = 160
    val_fraction: float = 0.1
    max_train: int | None = None
    max_test: int | None = None
    scratch_width: int = 32
    scratch_input: int = 112
    scratch: TrainConfig = Field(
        default_factory=lambda: TrainConfig(
            epochs=20, lr=2e-3, batch_size=64, augment="resized+flip+color"
        )
    )
    probe: list[str] = Field(
        default_factory=lambda: [
            "resnet18",
            "mobilenetv3_small_100",
            "efficientnet_b0",
            "deit_tiny_patch16_224",
        ]
    )
    finetune: list[str] = Field(
        default_factory=lambda: ["mobilenetv3_small_100", "resnet18", "efficientnet_b0"]
    )
    finetune_train: TrainConfig = Field(
        default_factory=lambda: TrainConfig(
            epochs=5,
            lr=3e-4,
            batch_size=32,
            weight_decay=0.05,
            label_smoothing=0.1,
            augment="resized+flip",
        )
    )
    # LP-FT: start from the converged linear probe, then fine-tune everything at a low rate (None = skip)
    lpft_train: TrainConfig | None = Field(
        default_factory=lambda: TrainConfig(
            epochs=3,
            lr=1e-4,
            batch_size=32,
            weight_decay=0.05,
            label_smoothing=0.1,
            augment="resized+flip",
        )
    )


DIAGNOSTIC_VARIANTS: dict[str, dict[str, Any]] = {
    "as in LP-FT": {},
    "learning rate ÷10": {"lr_factor": 0.1},
    "BatchNorm statistics frozen": {"freeze_bn": True},
    "flip only (no random crops)": {"augment": "flip"},
    "no label smoothing": {"label_smoothing": 0.0},
}


class TransferConfig(ExperimentConfig):
    output_dir: Path = DEFAULT_OUTPUT_DIR
    cifar: CifarConfig | None = Field(default_factory=CifarConfig)
    pets: PetsConfig | None = Field(default_factory=PetsConfig)
    tsne_points: int = 800
    bootstrap_resamples: int = 1000
    # One epoch of LP-FT per variant, to find out why fine-tuning loses to the probe (None = skip)
    diagnostic_model: str | None = None


def pets_splits(pc: PetsConfig, seed: int) -> tuple[ImageSet, ImageSet, ImageSet]:
    """Train / validation / test, identical for C2, C4 and C5."""
    full = load_images(pc.dataset, "train", pc.image_size)
    if pc.max_train:
        full = full.stratified_fraction(pc.max_train / len(full), seed)
    train, val = full.split(pc.val_fraction, seed)
    test = load_images(pc.dataset, "test", pc.image_size)
    if pc.max_test:
        test = test.stratified_fraction(pc.max_test / len(test), seed)
    return train, val, test


def head_from_probe(
    model: torch.nn.Module, clf: LogisticRegression, scaler: StandardScaler
) -> torch.nn.Module:
    """Load a fitted logistic regression on standardized features into the model's linear head.

    logits = ((f - mu) / s) @ W.T + b  =  f @ (W / s).T + (b - W @ (mu / s))
    """
    head = classifier_head(model)
    W = torch.as_tensor(clf.coef_ / scaler.scale_, dtype=torch.float32)
    b = torch.as_tensor(
        clf.intercept_ - clf.coef_ @ (scaler.mean_ / scaler.scale_), dtype=torch.float32
    )
    assert isinstance(head, torch.nn.Linear) and head.weight.shape == W.shape
    with torch.no_grad():
        head.weight.copy_(W)
        head.bias.copy_(b)
    return model


def finetuned(
    name: str,
    pc: PetsConfig,
    train: ImageSet,
    n_classes: int,
    mode: str = "naive",
    val: ImageSet | None = None,
    test: ImageSet | None = None,
    seed: int = 0,
) -> tuple[torch.nn.Module, Any]:
    """``naive``: new random head, one learning rate for everything.

    ``lpft`` (Kumar et al., 2022): the head starts from the converged linear probe, then
    the whole network is fine-tuned at a low learning rate. Needs ``val`` (and ``test``,
    only to reuse the cached probe features).
    """
    cfg = _recipe(pc, mode)
    if cfg is None:
        raise ValueError(f"fine-tuning mode {mode!r} is not configured")
    if mode == "naive":
        return train_or_load(
            f"ft-{name}", lambda: pretrained(name, n_classes, pc.image_size), train, cfg
        )
    if val is None or test is None:
        raise ValueError("LP-FT needs the validation and test splits for its linear probe")
    probe = linear_probe(name, pc, train, val, test, seed)

    def build() -> torch.nn.Module:
        return head_from_probe(
            pretrained(name, n_classes, pc.image_size), probe["clf"], probe["scaler"]
        )

    return train_or_load(f"{mode.replace('_', '')}-{name}", build, train, cfg)


FT_MODES = {"naive": "fine-tuned", "lpft": "LP-FT", "lpft_bn": "LP-FT, frozen BN"}


def finetune_modes(pc: PetsConfig) -> list[str]:
    available = ["naive"] + (["lpft", "lpft_bn"] if pc.lpft_train else [])
    return [m for m in available if pc.ft_modes is None or m in pc.ft_modes]


def _recipe(pc: PetsConfig, mode: str) -> TrainConfig | None:
    if mode == "naive":
        return pc.finetune_train
    if pc.lpft_train is None:
        return None
    return pc.lpft_train.model_copy(update={"freeze_bn": mode == "lpft_bn"})


def scratch_cnn(pc: PetsConfig, train: ImageSet, n_classes: int) -> tuple[torch.nn.Module, Any]:
    build = lambda: SmallCNN(3, n_classes, width=pc.scratch_width, input_size=pc.scratch_input)  # noqa: E731
    return train_or_load("scratch-cnn", build, train, pc.scratch)


@torch.inference_mode()
def embed(model: torch.nn.Module, data: ImageSet, batch_size: int = 64) -> np.ndarray:
    model.eval()
    mean, std = normalization(model)
    return torch.cat(
        [
            model(to_input(data.images[i : i + batch_size], mean, std))
            for i in range(0, len(data), batch_size)
        ]
    ).numpy()


def linear_probe(
    name: str, pc: PetsConfig, train: ImageSet, val: ImageSet, test: ImageSet, seed: int
) -> dict[str, Any]:
    """Frozen ImageNet features + logistic regression (C chosen on the validation split)."""
    key = cache_key("probe", name, pc.image_size, len(train), len(val), len(test), seed)
    path = data_dir() / "cache" / "lab_c" / "features" / f"{name}-{key}.npz"
    backbone = pretrained(name, 0, pc.image_size)
    if path.exists():
        f = np.load(path)
        f_tr, f_val, f_te, extract_s = f["train"], f["val"], f["test"], float(f["seconds"])
    else:
        start = time.perf_counter()
        f_tr, f_val, f_te = embed(backbone, train), embed(backbone, val), embed(backbone, test)
        extract_s = time.perf_counter() - start
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, train=f_tr, val=f_val, test=f_te, seconds=extract_s)
    scaler = StandardScaler().fit(f_tr)
    start = time.perf_counter()
    best = None
    for c in (0.01, 0.1, 1.0):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf = LogisticRegression(C=c, max_iter=3000).fit(
                scaler.transform(f_tr), train.labels.numpy()
            )
        acc = float((clf.predict(scaler.transform(f_val)) == val.labels.numpy()).mean())
        if best is None or acc > best[0]:
            best = (acc, c, clf)
    assert best is not None
    head_s = time.perf_counter() - start
    return {
        "pred": best[2].predict(scaler.transform(f_te)),
        "train_s": extract_s * len(train) / (len(train) + len(val) + len(test)) + head_s,
        "trainable": f_tr.shape[1] * train.n_classes + train.n_classes,
        "total": n_params(backbone),
        "img_s": throughput(backbone, test),
        "features": f_te,
        "note": f"C = {best[1]}",
        "clf": best[2],
        "scaler": scaler,
    }


def part_cifar(cc: CifarConfig, cfg: TransferConfig) -> tuple[list[str], list, dict]:
    train, test = load_images(cc.dataset, "train"), load_images(cc.dataset, "test")
    if cc.max_train:
        train = train.stratified_fraction(cc.max_train / len(train), cfg.seed)
    in_dim = int(np.prod(train.images.shape[1:]))
    models = {
        "MLP": lambda: MLP(in_dim, cc.mlp_hidden, train.n_classes, dropout=0.2),
        "SmallCNN": lambda: SmallCNN(train.images.shape[1], train.n_classes, width=cc.cnn_width),
    }
    res: dict[str, dict[str, Any]] = {}
    for name, build in models.items():
        model, hist = train_or_load(f"cifar-{name}", build, train, cc.train)
        res[name] = {
            "pred": predict(model, test).argmax(1).numpy(),
            "train_s": hist.train_seconds,
            "params": n_params(model),
            "img_s": throughput(model, test),
        }
    y = test.labels.numpy()
    acc = {
        k: bootstrap_ci(
            y,
            r["pred"],
            lambda a, b: float(np.mean(a == b)),
            n_resamples=cfg.bootstrap_resamples,
            seed=cfg.seed,
        )
        for k, r in res.items()
    }
    test_ = mcnemar(y, res["SmallCNN"]["pred"], res["MLP"]["pred"])
    rows = [
        {
            "experiment": cfg.experiment,
            "dataset": train.name,
            "model": k,
            "fold": "test",
            "metric": "accuracy",
            "value": v.value,
        }
        for k, v in acc.items()
    ]
    lines = [
        "## A. Inductive bias — MLP vs. CNN on CIFAR-10",
        "",
        f"{train.source} · {len(train):,} training / {len(test):,} test images · same budget for both: "
        f"{cc.train.epochs} epochs, AdamW, cosine schedule, `{cc.train.augment}` augmentation.",
        "",
        md_table(
            ["Model", "Parameters", "Test accuracy (95% CI)", "Train min", "Images/s (CPU)"],
            [
                [
                    k,
                    f"{r['params']:,}",
                    str(acc[k]),
                    f"{r['train_s'] / 60:.1f}",
                    f"{r['img_s']:,.0f}",
                ]
                for k, r in res.items()
            ],
        ),
        "",
        f"McNemar SmallCNN vs. MLP: p = {fmt_p(test_.pvalue)}.",
        "",
    ]
    return lines, rows, res


def part_pets(pc: PetsConfig, cfg: TransferConfig, out: Outputs) -> tuple[list[str], list]:
    train, val, test = pets_splits(pc, cfg.seed)
    n_classes = train.n_classes
    res: dict[str, dict[str, Any]] = {}

    if pc.include_scratch:
        model, hist = scratch_cnn(pc, train, n_classes)
        res["SmallCNN from scratch"] = {
            "pred": predict(model, test).argmax(1).numpy(),
            "train_s": hist.train_seconds,
            "trainable": n_params(model),
            "total": n_params(model),
            "img_s": throughput(model, test),
            "features": embed_scratch(model, test),
            "note": f"{pc.scratch.epochs} epochs at {pc.scratch_input}px",
        }
    for name in pc.probe:
        res[f"{name} · linear probe"] = linear_probe(name, pc, train, val, test, cfg.seed)
    for mode in finetune_modes(pc):
        tc = _recipe(pc, mode)
        assert tc is not None
        note = (
            f"random head, {tc.epochs} epochs, lr {tc.lr:g}"
            if mode == "naive"
            else f"head = linear probe, then {tc.epochs} epochs, lr {tc.lr:g}"
            + (", BatchNorm statistics frozen" if tc.freeze_bn else "")
        )
        for name in pc.finetune:
            model, hist = finetuned(name, pc, train, n_classes, mode, val, test, cfg.seed)
            res[f"{name} · {FT_MODES[mode]}"] = {
                "pred": predict(model, test).argmax(1).numpy(),
                "train_s": hist.train_seconds,
                "trainable": n_params(model),
                "total": n_params(model),
                "img_s": throughput(model, test),
                "note": note,
            }

    y = test.labels.numpy()
    acc = {
        k: bootstrap_ci(
            y,
            r["pred"],
            lambda a, b: float(np.mean(a == b)),
            n_resamples=cfg.bootstrap_resamples,
            seed=cfg.seed,
        )
        for k, r in res.items()
    }
    ranking = sorted(res, key=lambda k: -acc[k].value)
    best = ranking[0]
    tests = {k: mcnemar(y, res[best]["pred"], res[k]["pred"]) for k in ranking[1:]}
    rows = [
        {
            "experiment": cfg.experiment,
            "dataset": train.name,
            "model": k,
            "fold": "test",
            "metric": m,
            "value": float(v),
        }
        for k, r in res.items()
        for m, v in (
            ("accuracy", acc[k].value),
            ("train_s", r["train_s"]),
            ("images_per_s", r["img_s"]),
            ("trainable_params", r["trainable"]),
        )
    ]

    lines = [
        "## B. Transfer learning — Oxford-IIIT Pets (37 breeds)",
        "",
        f"{train.source} · {len(train):,} training / {len(val):,} validation / {len(test):,} test "
        f"images at {pc.image_size}px. Pretrained backbones: ImageNet weights from timm.",
        "",
        md_table(
            [
                "Method",
                "Trainable / total params",
                "Test accuracy (95% CI)",
                "Train min",
                "Images/s (CPU)",
                "Notes",
                "McNemar p vs best",
            ],
            [
                [
                    f"**{k}**" if k == best else k,
                    f"{res[k]['trainable']:,} / {res[k]['total']:,}",
                    str(acc[k]),
                    f"{res[k]['train_s'] / 60:.1f}",
                    f"{res[k]['img_s']:,.0f}",
                    res[k]["note"],
                    "—" if k == best else fmt_p(tests[k].pvalue),
                ]
                for k in ranking
            ],
        ),
        "",
    ]

    if cfg.diagnostic_model and pc.lpft_train is not None:
        lines += finetune_diagnostic(cfg, train, val, test)

    # t-SNE: scratch features vs. the strongest frozen backbone's features
    probes = [k for k in ranking if "linear probe" in k]
    if probes and cfg.tsne_points and "SmallCNN from scratch" in res:
        rng = np.random.default_rng(cfg.seed)
        idx = rng.choice(len(test), min(cfg.tsne_points, len(test)), replace=False)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
        for ax, key in zip(axes, ["SmallCNN from scratch", probes[0]], strict=True):
            Z = TSNE(2, init="pca", random_state=cfg.seed, perplexity=30).fit_transform(
                res[key]["features"][idx]
            )
            cats = np.array([test.classes[c] in CAT_BREEDS for c in y[idx]])
            for mask, marker in ((cats, "o"), (~cats, "^")):
                ax.scatter(
                    Z[mask, 0],
                    Z[mask, 1],
                    c=y[idx][mask],
                    cmap="tab20",
                    vmin=0,
                    vmax=n_classes - 1,
                    s=8,
                    alpha=0.8,
                    marker=marker,
                )
            ax.set(title=key, xticks=[], yticks=[])
        fig.suptitle(
            f"t-SNE of {len(idx)} test images, coloured by breed (circles: cats, triangles: dogs)"
        )
        path, link = out.figure("pets_tsne.png")
        save(fig, path)
        knn = {}
        for key in ["SmallCNN from scratch", probes[0]]:
            feats = StandardScaler().fit_transform(res[key]["features"])
            knn[key] = float(cross_val_score(KNeighborsClassifier(5), feats, y, cv=5).mean())
        lines += [
            "How well do the feature spaces separate the breeds by themselves? 5-nearest-neighbour "
            "accuracy on the test images' features (5-fold CV): "
            + ", ".join(f"{k} {v:.1%}" for k, v in knn.items())
            + ".",
            "",
            f"![t-SNE]({link})",
            "",
        ]
    return lines, rows


def _accuracy(model: torch.nn.Module, data: ImageSet) -> float:
    return float((predict(model, data).argmax(1) == data.labels).float().mean())


def finetune_diagnostic(
    cfg: TransferConfig, train: ImageSet, val: ImageSet, test: ImageSet
) -> list[str]:
    pc, name = cfg.pets, cfg.diagnostic_model
    assert pc is not None and pc.lpft_train is not None and name is not None
    probe = linear_probe(name, pc, train, val, test, cfg.seed)
    y_val, y_te = val.labels.numpy(), test.labels.numpy()
    rows = [
        [
            "linear probe, no fine-tuning",
            f"{(probe['clf'].predict(probe['scaler'].transform(embed(pretrained(name, 0, pc.image_size), val))) == y_val).mean():.3f}",
            f"{(probe['pred'] == y_te).mean():.3f}",
        ]
    ]
    base = pc.lpft_train.model_copy(update={"epochs": 1, "schedule": "constant"})
    for label, change in DIAGNOSTIC_VARIANTS.items():
        change = dict(change)
        factor = change.pop("lr_factor", 1.0)
        tc = base.model_copy(update={**change, "lr": base.lr * factor})

        def build() -> torch.nn.Module:
            return head_from_probe(
                pretrained(name, train.n_classes, pc.image_size), probe["clf"], probe["scaler"]
            )

        model, _ = train_or_load(f"diag-{name}", build, train, tc)
        rows.append(
            [
                f"{label} (lr {tc.lr:g})",
                f"{_accuracy(model, val):.3f}",
                f"{_accuracy(model, test):.3f}",
            ]
        )
    return [
        f"### Why does fine-tuning lose to the probe? (`{name}`, one epoch from the probe head)",
        "",
        md_table(["Variant", "Validation accuracy", "Test accuracy"], rows),
        "",
    ]


@torch.inference_mode()
def embed_scratch(model: torch.nn.Module, data: ImageSet, batch_size: int = 128) -> np.ndarray:
    assert isinstance(model, SmallCNN)
    model.eval()
    mean, std = normalization(model)
    return torch.cat(
        [
            model.features(to_input(data.images[i : i + batch_size], mean, std))
            for i in range(0, len(data), batch_size)
        ]
    ).numpy()


def run(cfg: TransferConfig) -> str:
    set_seed(cfg.seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    out = Outputs(cfg.output_dir, cfg.experiment)
    sections: list[str] = []
    rows: list[dict[str, Any]] = []
    with track_run(cfg.experiment, cfg.model_dump(mode="json"), cfg.tags) as tracker:
        if cfg.cifar:
            lines, r, _ = part_cifar(cfg.cifar, cfg)
            sections += lines
            rows += r
        if cfg.pets:
            lines, r = part_pets(cfg.pets, cfg, out)
            sections += lines
            rows += r
        write_results(rows, out.results)
        tracker.log_metrics(
            {
                f"{r['dataset']}.{r['model']}.{r['metric']}": r["value"]
                for r in rows
                if r["metric"] == "accuracy"
            }
        )
        header = [
            "# C2 — CNNs and transfer learning",
            "",
            f"All training and timing on CPU. Results in [`results/{cfg.experiment}.csv`](results/{cfg.experiment}.csv).",
            "",
        ]
        text = "\n".join(header + sections)
        out.write_report(text)
        tracker.log_artifact(out.report)
    return text


def main() -> None:
    cli(run, TransferConfig, "C2 CNNs and transfer learning")


if __name__ == "__main__":
    main()
