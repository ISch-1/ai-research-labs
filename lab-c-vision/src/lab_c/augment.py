"""Batch data augmentation on float tensors (B, C, H, W) in [0, 1], vectorized on the CPU.

A spec is a ``+``-joined list of ops, e.g. ``"flip+crop+color"``:

* ``flip``    — random horizontal flip
* ``crop``    — random translation up to 1/8 of the side with reflection padding
* ``resized`` — random-resized crop (scale 0.6–1), the standard for ImageNet-style models
* ``color``   — brightness, contrast and saturation jitter (±40%)
* ``cutout``  — one random square (1/4 of the side) set to the image mean
* ``mixup`` / ``cutmix`` — mix pairs of images and their labels (needs the labels)

``None`` or ``"none"`` leaves the batch unchanged.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

OPS = {"flip", "crop", "resized", "color", "cutout", "mixup", "cutmix"}


def parse(spec: str | None) -> set[str]:
    if not spec or spec == "none":
        return set()
    ops = set(spec.split("+"))
    if unknown := ops - OPS:
        raise ValueError(f"unknown augmentation(s) {sorted(unknown)}; choose from {sorted(OPS)}")
    if {"mixup", "cutmix"} <= ops:
        raise ValueError("use mixup or cutmix, not both")
    return ops


def _affine(x: torch.Tensor, ops: set[str], g: torch.Generator) -> torch.Tensor:
    """Flip / translate / scale in one ``grid_sample`` call."""
    b = x.shape[0]
    theta = torch.zeros(b, 2, 3)
    scale = torch.ones(b)
    if "resized" in ops:
        scale = torch.empty(b).uniform_(0.6, 1.0, generator=g).sqrt()  # area fraction -> side
    theta[:, 0, 0] = scale
    theta[:, 1, 1] = scale
    if "flip" in ops:
        theta[:, 0, 0] *= torch.where(torch.rand(b, generator=g) < 0.5, -1.0, 1.0)
    shift = (1 - scale) if "resized" in ops else torch.full((b,), 0.25 if "crop" in ops else 0.0)
    theta[:, :, 2] = (torch.rand(b, 2, generator=g) * 2 - 1) * shift[:, None]
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)


def _color(x: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    b = x.shape[0]
    factor = lambda: torch.empty(b, 1, 1, 1).uniform_(0.6, 1.4, generator=g)  # noqa: E731
    x = x * factor()
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x - mean) * factor() + mean
    if x.shape[1] == 3:
        gray = x.mean(dim=1, keepdim=True)
        x = (x - gray) * factor() + gray
    return x.clamp(0, 1)


def _cutout(x: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    b, _, h, w = x.shape
    side = max(1, h // 4)
    cy = torch.randint(0, h, (b,), generator=g)
    cx = torch.randint(0, w, (b,), generator=g)
    ys = torch.arange(h)[None, :, None]
    xs = torch.arange(w)[None, None, :]
    mask = ((ys - cy[:, None, None]).abs() <= side // 2) & (
        (xs - cx[:, None, None]).abs() <= side // 2
    )
    return torch.where(mask[:, None], x.mean(dim=(2, 3), keepdim=True), x)


def augment(
    x: torch.Tensor, y: torch.Tensor, spec: str | None, n_classes: int, g: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns augmented images and (soft) one-hot targets."""
    ops = parse(spec)
    targets = F.one_hot(y, n_classes).float()
    if ops & {"flip", "crop", "resized"}:
        x = _affine(x, ops, g)
    if "color" in ops:
        x = _color(x, g)
    if "cutout" in ops:
        x = _cutout(x, g)
    if ops & {"mixup", "cutmix"}:
        perm = torch.randperm(x.shape[0], generator=g)
        lam = float(torch.distributions.Beta(1.0, 1.0).sample())
        if "mixup" in ops:
            x = lam * x + (1 - lam) * x[perm]
        else:
            _, _, h, w = x.shape
            ch, cw = int(h * (1 - lam) ** 0.5), int(w * (1 - lam) ** 0.5)
            y0 = int(torch.randint(0, h - ch + 1, (1,), generator=g))
            x0 = int(torch.randint(0, w - cw + 1, (1,), generator=g))
            x = x.clone()
            x[:, :, y0 : y0 + ch, x0 : x0 + cw] = x[perm, :, y0 : y0 + ch, x0 : x0 + cw]
            lam = 1 - (ch * cw) / (h * w)
        targets = lam * targets + (1 - lam) * targets[perm]
    return x, targets
