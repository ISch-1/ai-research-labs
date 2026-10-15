"""Network architectures for Lab C: MLPs, a small CNN, and pretrained models via timm."""

from __future__ import annotations

from typing import Any, Literal, cast

import torch
import torch.nn.functional as F
from torch import nn

Activation = Literal["relu", "gelu", "tanh", "sigmoid"]
Init = Literal["default", "small_normal", "xavier", "he"]

_ACT = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid}


class MLP(nn.Module):
    """Fully connected network on flattened images."""

    def __init__(
        self,
        in_dim: int,
        hidden: list[int],
        n_classes: int,
        *,
        activation: Activation = "relu",
        batchnorm: bool = False,
        dropout: float = 0.0,
        init: Init = "default",
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Flatten()]
        for d_in, d_out in zip([in_dim, *hidden[:-1]], hidden, strict=True):
            layers.append(nn.Linear(d_in, d_out))
            if batchnorm:
                layers.append(nn.BatchNorm1d(d_out))
            layers.append(_ACT[activation]())
            if dropout:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden[-1] if hidden else in_dim, n_classes))
        self.net = nn.Sequential(*layers)
        if init != "default":
            for m in self.net:
                if isinstance(m, nn.Linear):
                    if init == "small_normal":
                        nn.init.normal_(m.weight, 0, 0.01)
                    elif init == "xavier":
                        nn.init.xavier_normal_(m.weight)
                    else:
                        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


class SmallCNN(nn.Module):
    """Three VGG-style stages (width w, 2w, 4w) with global average pooling.

    ``last_conv`` is exposed for Grad-CAM; ``features`` returns the pooled embedding.
    """

    def __init__(
        self,
        in_channels: int,
        n_classes: int,
        width: int = 32,
        dropout: float = 0.2,
        input_size: int | None = None,
    ) -> None:
        super().__init__()
        self.input_size = input_size  # images are resized to this side first (CPU budget)
        self.stages = nn.Sequential(
            _block(in_channels, width),
            nn.MaxPool2d(2),
            _block(width, 2 * width),
            nn.MaxPool2d(2),
            _block(2 * width, 4 * width),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout), nn.Linear(4 * width, n_classes)
        )

    @property
    def last_conv(self) -> nn.Module:
        last_stage = cast(nn.Sequential, self.stages[-1])
        return last_stage[-3]  # the final Conv2d of the last stage

    def _resize(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_size and x.shape[-1] != self.input_size:
            x = F.interpolate(
                x, size=self.input_size, mode="bilinear", antialias=True, align_corners=False
            )
        return x

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.stages(self._resize(x))).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(self.stages(self._resize(x))))


def pretrained(name: str, n_classes: int, img_size: int) -> nn.Module:
    """An ImageNet-pretrained timm model with a fresh classification head."""
    import timm

    kwargs: dict[str, Any] = (
        {"img_size": img_size} if any(k in name for k in ("vit", "deit")) else {}
    )
    return cast(
        nn.Module, timm.create_model(name, pretrained=True, num_classes=n_classes, **kwargs)
    )


def normalization(model: nn.Module, channels: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean/std the model expects (timm's data config, or 0.5 for models trained here)."""
    cfg = getattr(model, "pretrained_cfg", None) or {}
    mean = torch.tensor(cfg.get("mean", (0.5,) * channels)).view(1, -1, 1, 1)
    std = torch.tensor(cfg.get("std", (0.5,) * channels)).view(1, -1, 1, 1)
    return mean, std


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def gradcam_layer(model: nn.Module) -> nn.Module:
    """The convolutional layer Grad-CAM should hook for each supported architecture."""
    if isinstance(model, SmallCNN):
        return model.last_conv
    blocks: Any = getattr(model, "layer4", None)  # ResNets
    if blocks is None and type(model).__name__ in {"EfficientNet", "MobileNetV3"}:
        # The last block is still spatial; MobileNetV3's conv_head runs after pooling.
        blocks = getattr(model, "blocks", None)
    if blocks is not None:
        return cast(nn.Module, blocks[-1])
    raise ValueError(f"no Grad-CAM layer known for {type(model).__name__}")
