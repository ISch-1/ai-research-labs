"""The Gradio demo's inference code agrees with Lab C (offline), and the export round-trips."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

timm = pytest.importorskip("timm")
pytest.importorskip("safetensors")

from safetensors.torch import save_file  # noqa: E402

from lab_c.interpret import grad_cam, integrated_gradients  # noqa: E402
from lab_c.nets import gradcam_layer, normalization  # noqa: E402

DEMO = Path(__file__).resolve().parents[2] / "demos" / "pets-gradcam"


def _inference():
    spec = importlib.util.spec_from_file_location("pets_demo_inference", DEMO / "inference.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_demo_model(path: Path, arch: str = "mobilenetv3_small_050", n_classes: int = 4):
    torch.manual_seed(0)
    model = timm.create_model(arch, pretrained=False, num_classes=n_classes).eval()
    save_file(
        {k: v.contiguous() for k, v in model.state_dict().items()}, path / "model.safetensors"
    )
    mean, std = normalization(model)
    config = {
        "architecture": arch,
        "classes": [f"Breed {i}" for i in range(n_classes)],
        "image_size": 64,
        "mean": mean.flatten().tolist(),
        "std": std.flatten().tolist(),
        "temperature": 0.5,
    }
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return model


def test_demo_matches_lab_c_attributions(tmp_path):
    from PIL import Image

    inf = _inference()
    model = _write_demo_model(tmp_path)
    clf = inf.PetClassifier(tmp_path)

    rng = np.random.default_rng(0)
    photo = Image.fromarray(rng.integers(0, 255, (90, 120, 3), dtype=np.uint8))
    crop, x = clf.preprocess(photo)
    assert crop.size == (64, 64) and x.shape == (1, 3, 64, 64)

    with torch.no_grad():
        logits = model(x)
    torch.testing.assert_close(clf.probabilities(x), (logits / 0.5).softmax(1)[0])

    target = torch.tensor([int(logits.argmax())])
    torch.testing.assert_close(
        clf.grad_cam(x, int(target)), grad_cam(model, x, target, gradcam_layer(model))[0]
    )
    torch.testing.assert_close(
        clf.integrated_gradients(x, int(target), steps=8),
        integrated_gradients(model, x, target, torch.zeros_like(x), steps=8)[0],
        rtol=1e-4,
        atol=1e-6,
    )
    shown = inf.overlay(crop, clf.grad_cam(x, int(target)))
    assert shown.size == crop.size and shown.mode == "RGB"


@pytest.mark.hf
def test_export_round_trips_into_the_demo(tmp_path):
    """``lab_c.export`` writes weights the demo loads to the same predictions."""
    from lab_c import export, transfer
    from lab_c.train import TrainConfig, predict

    pc = transfer.PetsConfig(
        dataset="synthetic_shapes",
        image_size=32,
        probe=[],
        finetune=["mobilenetv3_small_050"],
        finetune_train=TrainConfig(epochs=1, batch_size=64, augment="flip"),
        lpft_train=None,
    )
    cfg = export.ExportConfig(
        experiment="export",
        output_dir=tmp_path,
        pets=pc,
        model="mobilenetv3_small_050",
        mode="naive",
    )
    assert "test accuracy" in export.run(cfg)
    config = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert config["architecture"] == "mobilenetv3_small_050" and config["temperature"] > 0

    train, _, test = transfer.pets_splits(pc, cfg.seed)
    model, _ = transfer.finetuned("mobilenetv3_small_050", pc, train, train.n_classes)  # cached
    clf = _inference().PetClassifier(tmp_path)
    x = (test.images[:8].float() / 255 - clf.mean) / clf.std
    with torch.no_grad():
        torch.testing.assert_close(clf.model(x), predict(model, test.subset(np.arange(8))))


def _web_export():
    spec = importlib.util.spec_from_file_location(
        "pets_web_export", DEMO.parent / "pets-web" / "export_onnx.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_in_browser_cams_equal_grad_cam_on_the_last_feature_map():
    """Global average pooling + linear head: Grad-CAM needs no gradients at that layer."""
    torch.manual_seed(0)
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=5).eval()
    net = _web_export().LogitsAndCams(model)
    x = torch.randn(1, 3, 64, 64)
    logits, cams = net(x)
    torch.testing.assert_close(logits, model(x))

    k = int(logits.argmax())
    features = model.forward_features(x).detach().requires_grad_(True)
    model.forward_head(features)[0, k].backward()
    weights = features.grad.mean(dim=(2, 3), keepdim=True)
    grad_cam = torch.relu((weights * features).sum(1))[0]
    h, w = features.shape[-2:]
    torch.testing.assert_close(torch.relu(cams[0, k]) / (h * w), grad_cam.detach())


def test_onnx_export_matches_pytorch(tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from PIL import Image

    model_dir = tmp_path / "model"
    (model_dir / "examples").mkdir(parents=True)
    _write_demo_model(model_dir, arch="efficientnet_b0")
    rng = np.random.default_rng(1)
    for name in ("cat.jpg", "dog_hard.jpg"):
        Image.fromarray(rng.integers(0, 255, (70, 90, 3), dtype=np.uint8)).save(
            model_dir / "examples" / name
        )
    out = tmp_path / "web"
    assert "ONNX Runtime matches PyTorch" in _web_export().export(model_dir, out)
    assert (out / "model.onnx").exists() and (out / "config.json").exists()
    assert json.loads((out / "examples.json").read_text()) == ["cat.jpg", "dog_hard.jpg"]
