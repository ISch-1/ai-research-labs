"""Model loading, calibrated prediction and attribution maps for the demo (PyTorch + timm only).

Self-contained on purpose — the Space does not install the lab. The attribution
methods are the same as ``lab_c.interpret`` (a test checks that they agree).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file
from torch import nn

HERE = Path(__file__).resolve().parent


def to_square(image: Image.Image, size: int) -> Image.Image:
    """Resize the shorter side to ``size`` and centre-crop a square (as in training)."""
    image = image.convert("RGB")
    w, h = image.size
    scale = size / min(w, h)
    image = image.resize(
        (max(size, round(w * scale)), max(size, round(h * scale))), Image.Resampling.BILINEAR
    )
    w, h = image.size
    left, top = (w - size) // 2, (h - size) // 2
    return image.crop((left, top, left + size, top + size))


def grad_cam_layer(model: nn.Module) -> nn.Module:
    """Last spatial block: ``layer4`` for ResNets, ``blocks[-1]`` for EfficientNet/MobileNetV3."""
    layer = getattr(model, "layer4", None)
    return layer if layer is not None else model.blocks[-1]


class PetClassifier:
    def __init__(self, model_dir: str | Path = HERE) -> None:
        model_dir = Path(model_dir)
        self.config: dict[str, Any] = json.loads((model_dir / "config.json").read_text("utf-8"))
        self.classes: list[str] = self.config["classes"]
        self.size: int = self.config["image_size"]
        self.temperature: float = self.config["temperature"]
        self.mean = torch.tensor(self.config["mean"]).view(1, 3, 1, 1)
        self.std = torch.tensor(self.config["std"]).view(1, 3, 1, 1)
        self.model: nn.Module = timm.create_model(
            self.config["architecture"], pretrained=False, num_classes=len(self.classes)
        )
        self.model.load_state_dict(load_file(model_dir / "model.safetensors"))
        self.model.eval()

    def preprocess(self, image: Image.Image) -> tuple[Image.Image, torch.Tensor]:
        """The square crop the model sees, and its normalised (1, 3, H, W) tensor."""
        crop = to_square(image, self.size)
        x = torch.from_numpy(np.asarray(crop).copy()).permute(2, 0, 1)[None].float() / 255
        return crop, (x - self.mean) / self.std

    @torch.inference_mode()
    def probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """Class probabilities after temperature scaling (fitted on the validation split)."""
        return (self.model(x) / self.temperature).softmax(1)[0]

    def grad_cam(self, x: torch.Tensor, target: int) -> torch.Tensor:
        """ReLU(Σ_k mean(∂y/∂A_k) · A_k) at the last spatial block, upsampled to the input."""
        store: dict[str, torch.Tensor] = {}

        def keep(_module: nn.Module, _inputs: Any, output: torch.Tensor) -> None:
            store["act"] = output
            output.register_hook(lambda grad: store.__setitem__("grad", grad))

        handle = grad_cam_layer(self.model).register_forward_hook(keep)
        try:
            with torch.enable_grad():
                self.model(x.detach().requires_grad_(True))[0, target].backward()
        finally:
            handle.remove()
            self.model.zero_grad(set_to_none=True)
        weights = store["grad"].mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * store["act"]).sum(1, keepdim=True))
        return F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[
            0, 0
        ].detach()

    def integrated_gradients(self, x: torch.Tensor, target: int, steps: int = 32) -> torch.Tensor:
        """|x - baseline| · mean gradient along the straight path (baseline 0 = the mean colour)."""
        baseline = torch.zeros_like(x[0])
        alphas = (torch.arange(steps, dtype=torch.float32) + 0.5) / steps
        grads = torch.zeros_like(x[0])
        with torch.enable_grad():
            for chunk in alphas.split(8):
                path = (baseline + chunk.view(-1, 1, 1, 1) * (x[0] - baseline)).requires_grad_(True)
                self.model(path)[:, target].sum().backward()
                grads += path.grad.sum(0)
        self.model.zero_grad(set_to_none=True)
        return ((x[0] - baseline) * grads / steps).abs().sum(0).detach()


def overlay(crop: Image.Image, heat: torch.Tensor, smooth: int = 0) -> Image.Image:
    """Heat map (0-1 after clipping at the 99th percentile) blended over the crop."""
    if smooth:
        heat = F.avg_pool2d(heat[None, None], smooth, stride=1, padding=smooth // 2)[0, 0]
        heat = heat[: crop.size[1], : crop.size[0]]
    h = heat.numpy()
    h = np.clip(h / (np.percentile(h, 99) + 1e-12), 0, 1)
    # A small "inferno"-like colour ramp, so the demo needs no matplotlib.
    ramp = np.array([[0, 0, 4], [87, 16, 110], [188, 55, 84], [249, 142, 9], [252, 255, 164]])
    pos = h * (len(ramp) - 1)
    lo = np.floor(pos).astype(int).clip(0, len(ramp) - 2)
    frac = (pos - lo)[..., None]
    colour = ramp[lo] * (1 - frac) + ramp[lo + 1] * frac
    alpha = 0.25 + 0.5 * h[..., None]
    blended = np.asarray(crop, dtype=np.float32) * (1 - alpha) + colour * alpha
    return Image.fromarray(blended.clip(0, 255).astype(np.uint8))
