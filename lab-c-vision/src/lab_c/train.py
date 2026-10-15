"""Training, prediction and timing for image classifiers, with an on-disk model cache.

``train_or_load(key, build, data, cfg)`` trains once and afterwards reloads the
weights and the training history, so long experiments resume after an
interruption and later experiments (C4, C5) reuse the models trained in C2.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from pydantic import BaseModel
from torch import nn

from airlab.data import data_dir
from airlab.experiment import host_description
from lab_c.augment import augment
from lab_c.datasets import ImageSet
from lab_c.nets import normalization


class TrainConfig(BaseModel):
    epochs: int = 10
    lr: float = 1e-3
    batch_size: int = 128
    weight_decay: float = 5e-4
    optimizer: Literal["sgd", "momentum", "adam", "adamw"] = "adamw"
    schedule: Literal["cosine", "constant"] = "cosine"
    label_smoothing: float = 0.0
    augment: str | None = None
    seed: int = 0


@dataclass
class History:
    train_loss: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)
    epoch_seconds: list[float] = field(default_factory=list)
    host: str = ""

    @property
    def train_seconds(self) -> float:
        return float(sum(self.epoch_seconds))


def to_input(images: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """uint8 (B, C, H, W) -> normalized float32."""
    return (images.float() / 255 - mean) / std


def _optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "momentum":
        return torch.optim.SGD(
            params, lr=cfg.lr, momentum=0.9, nesterov=True, weight_decay=cfg.weight_decay
        )
    if cfg.optimizer == "adam":
        return torch.optim.Adam(params, lr=cfg.lr)
    return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)


def fit(
    model: nn.Module, train: ImageSet, cfg: TrainConfig, val: ImageSet | None = None
) -> History:
    torch.manual_seed(cfg.seed)
    g = torch.Generator().manual_seed(cfg.seed)
    mean, std = normalization(model, train.images.shape[1])
    opt = _optimizer(model, cfg)
    steps = cfg.epochs * int(np.ceil(len(train) / cfg.batch_size))
    sched = (
        torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps) if cfg.schedule == "cosine" else None
    )
    history = History(host=host_description())
    for _ in range(cfg.epochs):
        start = time.perf_counter()
        model.train()
        losses = []
        for idx in torch.randperm(len(train), generator=g).split(cfg.batch_size):
            x = train.images[idx].float() / 255
            x, target = augment(x, train.labels[idx], cfg.augment, train.n_classes, g)
            logits = model((x - mean) / std)
            loss = F.cross_entropy(logits, target, label_smoothing=cfg.label_smoothing)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if sched:
                sched.step()
            losses.append(loss.item())
        history.epoch_seconds.append(time.perf_counter() - start)
        history.train_loss.append(float(np.mean(losses)))
        if val is not None:
            history.val_acc.append(
                float((predict(model, val).argmax(1) == val.labels).float().mean())
            )
    return history


@torch.inference_mode()
def predict(model: nn.Module, data: ImageSet, batch_size: int = 256) -> torch.Tensor:
    """Logits for every image."""
    model.eval()
    mean, std = normalization(model, data.images.shape[1])
    return torch.cat(
        [
            model(to_input(data.images[i : i + batch_size], mean, std))
            for i in range(0, len(data), batch_size)
        ]
    )


@torch.inference_mode()
def throughput(model: nn.Module, data: ImageSet, n: int = 256, batch_size: int = 64) -> float:
    """Images per second at inference, after a warm-up batch."""
    model.eval()
    sample = data.subset(torch.arange(min(n, len(data))))
    mean, std = normalization(model, data.images.shape[1])
    model(to_input(sample.images[:batch_size], mean, std))
    start = time.perf_counter()
    for i in range(0, len(sample), batch_size):
        model(to_input(sample.images[i : i + batch_size], mean, std))
    return len(sample) / (time.perf_counter() - start)


def cache_key(*parts: object) -> str:
    text = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha1(text.encode()).hexdigest()[:16]


def train_or_load(
    name: str,
    build: Callable[[], nn.Module],
    train: ImageSet,
    cfg: TrainConfig,
    val: ImageSet | None = None,
) -> tuple[nn.Module, History]:
    """Train ``build()`` on ``train`` unless a model with the same name, data and config is cached."""
    key = cache_key(
        name, train.name, len(train), cfg.model_dump(), None if val is None else len(val)
    )
    path = data_dir() / "cache" / "lab_c" / "models" / f"{name}-{key}.pt"
    model = build()
    if path.exists():
        blob = torch.load(path, weights_only=False)
        model.load_state_dict(blob["state"])
        model.eval()
        return model, History(**blob["history"])
    history = fit(model, train, cfg, val)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "history": asdict(history)}, path)
    model.eval()
    return model, history
