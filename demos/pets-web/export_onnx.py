"""Build the in-browser demo (a static Hugging Face Space) from the exported PyTorch model.

    python -m lab_c.export --config lab-c-vision/configs/export_demo.yaml   # weights first
    python demos/pets-web/export_onnx.py

Writes ``model.onnx`` with two outputs — the logits and one class-activation map per
breed — checks it against PyTorch with ONNX Runtime, and copies ``config.json`` and the
example images (listed in ``examples.json``) from ``demos/pets-gradcam``, so this
folder is the whole Space.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
GRADIO_DEMO = HERE.parent / "pets-gradcam"
OPSET = 17
IR_VERSION = 10  # what onnxruntime-web reads without complaint; the graph needs nothing newer


class LogitsAndCams(nn.Module):
    """Logits, plus ``Σ_c W_kc · F_c`` for every class k on the final feature map F.

    With global average pooling followed by one linear layer, ∂y_k/∂F_chw = W_kc / HW at
    every position, so Grad-CAM on F is exactly ReLU of this map (up to the constant 1/HW,
    which the normalisation for display removes) and the browser needs no gradients.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        head = model.get_classifier()  # type: ignore[operator]
        if not isinstance(head, nn.Linear):
            raise TypeError(f"expected a linear classifier, got {type(head).__name__}")
        self.model = model
        self.weight = head.weight

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.model.forward_features(x)  # type: ignore[operator]
        logits = self.model.forward_head(features)  # type: ignore[operator]
        return logits, torch.einsum("kc,bchw->bkhw", self.weight, features)


def _classifier(model_dir: Path) -> Any:
    sys.path.insert(0, str(GRADIO_DEMO))
    try:
        from inference import PetClassifier
    finally:
        sys.path.remove(str(GRADIO_DEMO))
    return PetClassifier(model_dir)


def export(model_dir: Path = GRADIO_DEMO, out_dir: Path = HERE) -> str:
    import onnx
    import onnxruntime as ort
    from PIL import Image

    clf = _classifier(model_dir)
    net = LogitsAndCams(clf.model).eval()
    examples = sorted((model_dir / "examples").glob("*.jpg"))
    rng = np.random.default_rng(0)
    probe = Image.fromarray(rng.integers(0, 255, (clf.size, clf.size, 3), dtype=np.uint8))
    inputs = [clf.preprocess(probe)[1]] + [clf.preprocess(Image.open(p))[1] for p in examples]

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "model.onnx"
    torch.onnx.export(
        net,
        (inputs[0],),
        str(path),
        input_names=["image"],
        output_names=["logits", "cams"],
        opset_version=OPSET,
        dynamo=False,
    )
    proto = onnx.load(str(path))
    proto.ir_version = min(proto.ir_version, IR_VERSION)
    onnx.checker.check_model(proto)
    onnx.save(proto, str(path))

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    worst = 0.0
    for x in inputs:
        with torch.no_grad():
            logits, cams = (t.numpy() for t in net(x))
        o_logits, o_cams = session.run(None, {"image": x.numpy()})
        np.testing.assert_allclose(o_logits, logits, rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(o_cams, cams, rtol=1e-3, atol=1e-2 * np.abs(cams).max())
        assert o_logits.argmax() == logits.argmax()
        worst = max(worst, float(np.abs(o_logits - logits).max()))

    shutil.copy(model_dir / "config.json", out_dir / "config.json")
    if (out_dir / "examples").exists():
        shutil.rmtree(out_dir / "examples")
    shutil.copytree(model_dir / "examples", out_dir / "examples")
    # A static page cannot list a folder: name the examples, the known failure last.
    names = sorted((p.name for p in examples), key=lambda n: (n.endswith("_hard.jpg"), n))
    (out_dir / "examples.json").write_text(json.dumps(names), encoding="utf-8")
    size = path.stat().st_size / 2**20
    return (
        f"{path} ({size:.1f} MB, opset {OPSET}): ONNX Runtime matches PyTorch on "
        f"{len(inputs)} images (max logit difference {worst:.1e})"
    )


if __name__ == "__main__":
    print(export())
