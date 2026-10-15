"""Image datasets for Lab C as uint8 tensors, cached on disk for fast reloads.

Every loader returns an ``ImageSet``: images ``(N, C, H, W)`` uint8, labels int64
and class names. Hugging Face datasets are decoded once and stored as ``.pt``
under ``<data_dir>/cache/lab_c``. ``synthetic_shapes`` is generated offline for
tests and CI smoke runs.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from airlab.data import data_dir

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


@dataclass
class ImageSet:
    name: str
    images: torch.Tensor  # (N, C, H, W) uint8
    labels: torch.Tensor  # (N,) int64
    classes: list[str]
    source: str

    def __len__(self) -> int:
        return len(self.labels)

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    def subset(self, idx: np.ndarray | torch.Tensor) -> ImageSet:
        idx = torch.as_tensor(idx)
        return ImageSet(self.name, self.images[idx], self.labels[idx], self.classes, self.source)

    def stratified_fraction(self, fraction: float, seed: int) -> ImageSet:
        """The same fraction of every class (at least one image each)."""
        if fraction >= 1:
            return self
        rng = np.random.default_rng(seed)
        labels = self.labels.numpy()
        keep = np.concatenate(
            [
                rng.choice(
                    np.flatnonzero(labels == c),
                    max(1, round(fraction * np.sum(labels == c))),
                    replace=False,
                )
                for c in range(self.n_classes)
            ]
        )
        return self.subset(np.sort(keep))

    def split(self, val_fraction: float, seed: int) -> tuple[ImageSet, ImageSet]:
        """Stratified train/validation split (validation is used for early stopping and calibration)."""
        rng = np.random.default_rng(seed)
        labels = self.labels.numpy()
        val = np.concatenate(
            [
                rng.choice(idx, max(1, round(val_fraction * len(idx))), replace=False)
                for idx in (np.flatnonzero(labels == c) for c in range(self.n_classes))
            ]
        )
        mask = np.ones(len(labels), dtype=bool)
        mask[val] = False
        return self.subset(np.flatnonzero(mask)), self.subset(np.sort(val))


def _cache_path(key: str):
    path = data_dir() / "cache" / "lab_c" / f"{key}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _to_square(image, size: int):
    """Resize the shorter side to ``size`` and centre-crop a square."""
    from PIL import Image

    image = image.convert("RGB")
    w, h = image.size
    scale = size / min(w, h)
    image = image.resize(
        (max(size, round(w * scale)), max(size, round(h * scale))), Image.Resampling.BILINEAR
    )
    w, h = image.size
    left, top = (w - size) // 2, (h - size) // 2
    return image.crop((left, top, left + size, top + size))


def _from_hf(
    name: str, path: str, split: str, image_col: str, size: int | None, gray: bool = False
) -> ImageSet:
    cache = _cache_path(f"{name}-{split}-{size or 'native'}")
    if cache.exists():
        blob = torch.load(cache, weights_only=False)
        return ImageSet(name, blob["images"], blob["labels"], blob["classes"], blob["source"])

    from datasets import load_dataset

    ds = load_dataset(path, split=split, cache_dir=str(data_dir() / "hf"))
    classes = list(ds.features["label"].names)
    arrays = []
    for item in ds:
        img = item[image_col]
        if gray:
            arr = np.asarray(img.convert("L"))[None]
        else:
            arr = np.asarray(_to_square(img, size) if size else img.convert("RGB")).transpose(
                2, 0, 1
            )
        arrays.append(arr)
    images = torch.from_numpy(np.stack(arrays)).to(torch.uint8)
    labels = torch.as_tensor(ds["label"], dtype=torch.int64)
    source = f"HF `{path}` ({split})"
    torch.save({"images": images, "labels": labels, "classes": classes, "source": source}, cache)
    return ImageSet(name, images, labels, classes, source)


def synthetic_shapes(n: int = 600, size: int = 32, seed: int = 0) -> ImageSet:
    """Circles, squares and triangles of random colour, size and position on noisy backgrounds."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    images = np.empty((n, 3, size, size), dtype=np.uint8)
    labels = rng.integers(0, 3, n)
    for i, label in enumerate(labels):
        canvas = rng.integers(0, 60, (3, size, size))
        r = rng.uniform(size * 0.2, size * 0.32)
        cx, cy = rng.uniform(r + 1, size - r - 1, 2)
        if label == 0:
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r**2
        elif label == 1:
            mask = (np.abs(xx - cx) <= r * 0.8) & (np.abs(yy - cy) <= r * 0.8)
        else:
            mask = (
                (yy - (cy - r) >= 0)
                & (yy <= cy + r * 0.7)
                & (np.abs(xx - cx) <= (yy - (cy - r)) * 0.6)
            )
        color = rng.integers(120, 256, 3)
        for c in range(3):
            canvas[c][mask] = color[c]
        images[i] = canvas
    return ImageSet(
        "synthetic_shapes",
        torch.from_numpy(images),
        torch.as_tensor(labels, dtype=torch.int64),
        ["circle", "square", "triangle"],
        "synthetic",
    )


LOADERS: dict[str, Callable[[str, int | None], ImageSet]] = {
    "fashion_mnist": lambda split, size: _from_hf(
        "fashion_mnist", "zalando-datasets/fashion_mnist", split, "image", None, gray=True
    ),
    "cifar10": lambda split, size: _from_hf("cifar10", "uoft-cs/cifar10", split, "img", None),
    "pets": lambda split, size: _from_hf(
        "pets", "timm/oxford-iiit-pet", split, "image", size or 160
    ),
    "synthetic_shapes": lambda split, size: synthetic_shapes(
        600 if split == "train" else 300, size or 32, seed=0 if split == "train" else 1
    ),
}


def load_images(name: str, split: str, size: int | None = None) -> ImageSet:
    """``split`` is ``train`` or ``test``; ``size`` only applies to variable-size datasets (Pets)."""
    return LOADERS[name](split, size)
