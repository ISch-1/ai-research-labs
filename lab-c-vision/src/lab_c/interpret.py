"""C5 — Visual interpretability: what is the model looking at, and can we trust the maps?

    python -m lab_c.interpret --config lab-c-vision/configs/c5_interpret.yaml

* Attribution methods, implemented here: vanilla gradient saliency, Integrated
  Gradients (Sundararajan et al., 2017) and Grad-CAM (Selvaraju et al., 2017).
* Faithfulness: the deletion test — remove the pixels a map ranks highest, in
  steps, and track the predicted class's probability. A faithful map makes it
  fall faster than removing random pixels (lower area under the curve).
* Shortcut learning: a CNN trained on CIFAR-10 where a small coloured patch in
  the corner reveals the class in 95% of training images. It reaches high
  accuracy, collapses when the patch lies, and Grad-CAM shows where it looks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field
from scipy.stats import wilcoxon
from torch import nn

from airlab.config import ExperimentConfig
from airlab.experiment import Outputs, cli
from airlab.plots import plt, save
from airlab.reporting import fmt_p, md_table, write_results
from airlab.seed import set_seed
from airlab.stats import holm
from airlab.tracking import track_run
from lab_c.datasets import ImageSet, load_images
from lab_c.nets import SmallCNN, gradcam_layer, normalization
from lab_c.train import TrainConfig, predict, to_input, train_or_load
from lab_c.transfer import PetsConfig, finetuned, pets_splits

DEFAULT_OUTPUT_DIR = Path("lab-c-vision/reports")
METHODS = ("saliency", "integrated_gradients", "grad_cam", "random")


# --------------------------------------------------------------------- attributions


def saliency(model: nn.Module, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """|∂ score / ∂ pixel|, max over channels. x: normalized (B, C, H, W) -> (B, H, W)."""
    x = x.clone().requires_grad_(True)
    model(x).gather(1, target[:, None]).sum().backward()
    assert x.grad is not None
    return x.grad.abs().amax(1)


def integrated_gradients(
    model: nn.Module, x: torch.Tensor, target: torch.Tensor, baseline: torch.Tensor, steps: int = 32
) -> torch.Tensor:
    """Path integral of gradients from ``baseline`` to ``x`` (Riemann, midpoint rule), summed |·| over channels."""
    attributions = []
    alphas = (torch.arange(steps, dtype=x.dtype) + 0.5) / steps
    for xi, ti, bi in zip(x, target, baseline, strict=True):
        path = (bi + alphas.view(-1, 1, 1, 1) * (xi - bi)).requires_grad_(True)
        model(path)[:, ti].sum().backward()
        avg_grad = path.grad.mean(0)
        attributions.append(((xi - bi) * avg_grad).abs().sum(0))
    return torch.stack(attributions)


def grad_cam(
    model: nn.Module, x: torch.Tensor, target: torch.Tensor, layer: nn.Module
) -> torch.Tensor:
    """ReLU(Σ_k mean(∂y/∂A_k) · A_k) at ``layer``, upsampled to the input size."""
    store: dict[str, torch.Tensor] = {}

    def keep(_module: nn.Module, _inputs: Any, output: torch.Tensor) -> None:
        # A tensor hook (not a module backward hook) also works when in-place ReLUs follow.
        store["act"] = output
        output.register_hook(lambda grad: store.__setitem__("grad", grad))

    handle = layer.register_forward_hook(keep)
    try:
        with torch.enable_grad():
            model(x.detach().requires_grad_(True)).gather(1, target[:, None]).sum().backward()
    finally:
        handle.remove()
    weights = store["grad"].mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * store["act"]).sum(1, keepdim=True))
    return F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[
        :, 0
    ].detach()


def attribute(
    method: str, model: nn.Module, x: torch.Tensor, target: torch.Tensor, ig_steps: int, seed: int
) -> torch.Tensor:
    model.zero_grad(set_to_none=True)
    if method == "saliency":
        return saliency(model, x, target)
    if method == "integrated_gradients":
        return integrated_gradients(
            model, x, target, torch.zeros_like(x), ig_steps
        )  # zero = the dataset mean colour
    if method == "grad_cam":
        return grad_cam(model, x, target, gradcam_layer(model))
    g = torch.Generator().manual_seed(seed)
    return torch.rand(x.shape[0], *x.shape[2:], generator=g)


@torch.inference_mode()
def deletion_curve(
    model: nn.Module, x: torch.Tensor, target: torch.Tensor, attr: torch.Tensor, steps: int = 10
) -> np.ndarray:
    """Probability of ``target`` as the top 0%, 10%, …, 100% of pixels (by ``attr``) are set to 0 (= the mean)."""
    b, _, h, w = x.shape
    order = attr.reshape(b, -1).argsort(dim=1, descending=True)
    probs = []
    for k in np.linspace(0, h * w, steps + 1).astype(int):
        mask = torch.ones(b, h * w)
        mask.scatter_(1, order[:, :k], 0.0)
        xk = x * mask.view(b, 1, h, w)
        probs.append(model(xk).softmax(1).gather(1, target[:, None])[:, 0])
    return torch.stack(probs, 1).numpy()


# ------------------------------------------------------------------------ shortcut


PALETTE = torch.tensor(
    [
        [230, 25, 75],
        [60, 180, 75],
        [255, 225, 25],
        [0, 130, 200],
        [245, 130, 48],
        [145, 30, 180],
        [70, 240, 240],
        [240, 50, 230],
        [210, 245, 60],
        [250, 190, 212],
    ],
    dtype=torch.uint8,
)


def add_patch(
    data: ImageSet, colour_of: torch.Tensor, share: float, size: int, seed: int, name: str
) -> ImageSet:
    """Paint a ``size``×``size`` square in the top-left corner, coloured by ``colour_of`` (class ids)."""
    g = torch.Generator().manual_seed(seed)
    images = data.images.clone()
    chosen = torch.rand(len(data), generator=g) < share
    colours = PALETTE[colour_of[chosen] % len(PALETTE)]
    images[chosen, :, :size, :size] = colours[:, :, None, None]
    return ImageSet(name, images, data.labels, data.classes, data.source)


class ShortcutConfig(BaseModel):
    dataset: str = "cifar10"
    patch_size: int = 5
    patch_share: float = 0.95
    width: int = 32
    train: TrainConfig = Field(
        default_factory=lambda: TrainConfig(epochs=8, lr=2e-3, augment="flip")
    )
    max_train: int | None = None
    max_test: int | None = None


class InterpretConfig(ExperimentConfig):
    output_dir: Path = DEFAULT_OUTPUT_DIR
    pets: PetsConfig | None = Field(default_factory=PetsConfig)
    model: str = "resnet18"
    mode: str = "lpft"  # "naive" or "lpft" fine-tuning (see C2)
    n_images: int = 100
    n_shown: int = 6
    ig_steps: int = 32
    deletion_steps: int = 10
    shortcut: ShortcutConfig | None = Field(default_factory=ShortcutConfig)


def _overlay(ax, image: torch.Tensor, heat: torch.Tensor | None, title: str) -> None:
    ax.imshow(image.permute(1, 2, 0).numpy())
    if heat is not None:
        h = heat.numpy()
        ax.imshow((h - h.min()) / (h.max() - h.min() + 1e-12), cmap="jet", alpha=0.45)
    ax.set_title(title, fontsize=7)
    ax.axis("off")


def pets_section(cfg: InterpretConfig, out: Outputs) -> tuple[list[str], list]:
    pc = cfg.pets
    assert pc is not None
    train, _, test = pets_splits(pc, cfg.seed)
    model, _ = finetuned(cfg.model, pc, train, train.n_classes, cfg.mode)
    model.eval()
    mean, std = normalization(model)
    rng = np.random.default_rng(cfg.seed)
    idx = torch.as_tensor(rng.choice(len(test), min(cfg.n_images, len(test)), replace=False))
    x = to_input(test.images[idx], mean, std)
    with torch.no_grad():
        target = model(x).argmax(1)  # explain the model's own prediction

    curves: dict[str, np.ndarray] = {}
    shown: dict[str, torch.Tensor] = {}
    for method in METHODS:
        attrs, cs = [], []
        for i in range(0, len(x), 16):
            a = attribute(
                method, model, x[i : i + 16], target[i : i + 16], cfg.ig_steps, cfg.seed + i
            )
            cs.append(
                deletion_curve(model, x[i : i + 16], target[i : i + 16], a, cfg.deletion_steps)
            )
            attrs.append(a)
        curves[method] = np.concatenate(cs)
        shown[method] = torch.cat(attrs)[: cfg.n_shown]
    auc = {
        m: curves[m].mean(1) for m in METHODS
    }  # mean probability along the curve (lower = more faithful)
    raw = {m: float(wilcoxon(auc[m], auc["random"]).pvalue) for m in METHODS if m != "random"}
    adj = holm(raw)
    rows = [
        {
            "experiment": cfg.experiment,
            "dataset": "pets",
            "model": m,
            "fold": int(i),
            "metric": "deletion_auc",
            "value": float(v),
        }
        for m in METHODS
        for i, v in zip(idx.tolist(), auc[m], strict=True)
    ]

    names = [c.replace("_", " ") for c in test.classes]
    fig, axes = plt.subplots(cfg.n_shown, 4, figsize=(8, 2.1 * cfg.n_shown))
    for r in range(cfg.n_shown):
        img = test.images[idx[r]]
        label = f"pred: {names[target[r]]}" + (
            "" if target[r] == test.labels[idx[r]] else f"\n(true: {names[test.labels[idx[r]]]})"
        )
        _overlay(axes[r, 0], img, None, label)
        for c, method in enumerate(("saliency", "integrated_gradients", "grad_cam"), start=1):
            _overlay(axes[r, c], img, shown[method][r], method.replace("_", " ") if r == 0 else "")
    path, maps_link = out.figure("attribution_maps.png")
    save(fig, path)

    fig, ax = plt.subplots(figsize=(5.5, 4))
    xs = np.linspace(0, 100, cfg.deletion_steps + 1)
    for m in METHODS:
        ax.plot(
            xs,
            curves[m].mean(0),
            marker="o",
            ms=3,
            label=f"{m.replace('_', ' ')} (AUC {auc[m].mean():.3f})",
        )
    ax.set(
        xlabel="% of pixels removed (highest attribution first)",
        ylabel="probability of predicted class",
        title="Deletion test",
    )
    ax.legend(fontsize=8)
    path, del_link = out.figure("deletion_curves.png")
    save(fig, path)

    lines = [
        f"## Attribution maps — {cfg.model} ({cfg.mode}) on Pets",
        "",
        f"{len(idx)} random test images; each map explains the model's own prediction. Deletion AUC = "
        "mean probability of the predicted class while removing pixels in the map's order (lower = "
        "the map found pixels the model actually relies on). Wilcoxon vs. random order, Holm-adjusted.",
        "",
        md_table(
            ["Method", "Deletion AUC (mean)", "Holm p vs random"],
            [
                [
                    m.replace("_", " "),
                    f"{auc[m].mean():.3f}",
                    "—" if m == "random" else fmt_p(adj[m]),
                ]
                for m in sorted(METHODS, key=lambda m: auc[m].mean())
            ],
        ),
        "",
        f"![deletion curves]({del_link})",
        "",
        f"![attribution maps]({maps_link})",
        "",
    ]
    return lines, rows


def shortcut_section(cfg: InterpretConfig, out: Outputs) -> tuple[list[str], list]:
    sc = cfg.shortcut
    assert sc is not None
    train = load_images(sc.dataset, "train")
    test = load_images(sc.dataset, "test")
    if sc.max_train:
        train = train.stratified_fraction(sc.max_train / len(train), cfg.seed)
    if sc.max_test:
        test = test.stratified_fraction(sc.max_test / len(test), cfg.seed)
    shortcut_train = add_patch(
        train, train.labels, sc.patch_share, sc.patch_size, cfg.seed, f"{train.name}_shortcut"
    )
    build = lambda: SmallCNN(train.images.shape[1], train.n_classes, width=sc.width)  # noqa: E731
    models = {
        "trained on clean images": train_or_load("shortcut-clean", build, train, sc.train)[0],
        "trained with the shortcut patch": train_or_load(
            "shortcut-patched", build, shortcut_train, sc.train
        )[0],
    }
    wrong = (test.labels + 1) % test.n_classes
    tests = {
        "clean": test,
        "patch = true class": add_patch(test, test.labels, 1.0, sc.patch_size, cfg.seed, test.name),
        "patch = wrong class": add_patch(test, wrong, 1.0, sc.patch_size, cfg.seed, test.name),
    }
    acc = {
        (m, t): float((predict(model, data).argmax(1) == data.labels).float().mean())
        for m, model in models.items()
        for t, data in tests.items()
    }
    follows = {
        m: float((predict(model, tests["patch = wrong class"]).argmax(1) == wrong).float().mean())
        for m, model in models.items()
    }

    # Grad-CAM mass inside the patch, on images whose patch lies
    lying = tests["patch = wrong class"]
    k = min(200, len(lying))
    mass, cams = {}, {}
    for m, model in models.items():
        mean, std = normalization(model)
        x = to_input(lying.images[:k], mean, std)
        with torch.no_grad():
            target = model(x).argmax(1)
        cam = grad_cam(model, x, target, gradcam_layer(model))
        total = cam.sum(dim=(1, 2)).clamp(min=1e-12)
        mass[m] = float((cam[:, : sc.patch_size, : sc.patch_size].sum(dim=(1, 2)) / total).mean())
        cams[m] = cam[:6]
    area = sc.patch_size**2 / (test.images.shape[-1] ** 2)

    fig, axes = plt.subplots(2, 6, figsize=(12, 4.4))
    for r, m in enumerate(models):
        for c in range(6):
            _overlay(axes[r, c], lying.images[c], cams[m][c], m if c == 0 else "")
    path, cam_link = out.figure("shortcut_gradcam.png")
    save(fig, path)

    rows = [
        {
            "experiment": cfg.experiment,
            "dataset": f"shortcut/{t}",
            "model": m,
            "fold": "test",
            "metric": "accuracy",
            "value": v,
        }
        for (m, t), v in acc.items()
    ]
    lines = [
        "## Shortcut learning",
        "",
        f"{train.source}: in the shortcut training set {sc.patch_share:.0%} of images carry a "
        f"{sc.patch_size}×{sc.patch_size} patch in the top-left corner whose colour encodes the class. "
        f"Both CNNs are otherwise identical ({sc.train.epochs} epochs).",
        "",
        md_table(
            [
                "Model",
                *[f"Accuracy, {t}" for t in tests],
                "Predicts the patch's (wrong) class",
                f"Grad-CAM mass in the patch (area {area:.1%})",
            ],
            [
                [m, *(f"{acc[(m, t)]:.3f}" for t in tests), f"{follows[m]:.1%}", f"{mass[m]:.1%}"]
                for m in models
            ],
        ),
        "",
        "Grad-CAM on test images whose patch names the wrong class (top: clean model, bottom: shortcut model):",
        "",
        f"![shortcut grad-cam]({cam_link})",
        "",
    ]
    return lines, rows


def run(cfg: InterpretConfig) -> str:
    set_seed(cfg.seed)
    out = Outputs(cfg.output_dir, cfg.experiment)
    sections: list[str] = []
    rows: list[dict[str, Any]] = []
    with track_run(cfg.experiment, cfg.model_dump(mode="json"), cfg.tags) as tracker:
        if cfg.pets:
            lines, r = pets_section(cfg, out)
            sections += lines
            rows += r
        if cfg.shortcut:
            lines, r = shortcut_section(cfg, out)
            sections += lines
            rows += r
        write_results(rows, out.results)
        tracker.log_metrics(
            {f"{r['dataset']}.{r['model']}": r["value"] for r in rows if r["metric"] == "accuracy"}
        )
        header = [
            "# C5 — Interpretability: attribution maps, their faithfulness, and shortcut learning",
            "",
            f"Saliency, Integrated Gradients ({cfg.ig_steps} steps, mean-colour baseline) and Grad-CAM are "
            f"implemented in [`interpret.py`](../src/lab_c/interpret.py). Results in [`results/{cfg.experiment}.csv`](results/{cfg.experiment}.csv).",
            "",
        ]
        text = "\n".join(header + sections)
        out.write_report(text)
        tracker.log_artifact(out.report)
    return text


def main() -> None:
    cli(run, InterpretConfig, "C5 interpretability")


if __name__ == "__main__":
    main()
