"""Export a Pets model for the demo in ``demos/pets-gradcam`` (a Hugging Face Space).

    python -m lab_c.export --config lab-c-vision/configs/export_demo.yaml

Writes ``model.safetensors`` (weights only, no pickle), ``config.json``
(architecture, breeds, preprocessing, the temperature fitted on the validation
split and the held-out accuracy) and a few test images as examples. The model is
the one C2 trained and C4 analysed, loaded from the cache — nothing is retrained.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from pydantic import Field
from safetensors.torch import save_file

from airlab.config import ExperimentConfig
from airlab.experiment import cli
from airlab.seed import set_seed
from lab_c.errors import fit_temperature
from lab_c.nets import normalization
from lab_c.train import predict
from lab_c.transfer import FT_MODES, PetsConfig, finetuned, pets_splits

SCHEMA_VERSION = 1


class ExportConfig(ExperimentConfig):
    output_dir: Path = Path("demos/pets-gradcam")
    pets: PetsConfig = Field(default_factory=PetsConfig)
    model: str = "efficientnet_b0"
    mode: str = "lpft_bn"
    n_examples: int = 6  # the last one is a mistake from the most-confused breed pair


def breed(name: str) -> str:
    """``american_pit_bull_terrier`` -> ``American Pit Bull Terrier``."""
    return name.replace("_", " ").title()


def run(cfg: ExportConfig) -> str:
    set_seed(cfg.seed)
    pc = cfg.pets
    train, val, test = pets_splits(pc, cfg.seed)
    model, _ = finetuned(cfg.model, pc, train, train.n_classes, cfg.mode, val, test, cfg.seed)
    model.eval()

    temperature = fit_temperature(predict(model, val), val.labels)
    logits = predict(model, test)
    pred = logits.argmax(1)
    accuracy = float((pred == test.labels).float().mean())
    mean, std = normalization(model)

    out = cfg.output_dir
    (out / "examples").mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, out / "model.safetensors")
    config = {
        "schema_version": SCHEMA_VERSION,
        "architecture": cfg.model,
        "training": f"{FT_MODES[cfg.mode]} on Oxford-IIIT Pets (Lab C, module C2)",
        "classes": [breed(c) for c in test.classes],
        "image_size": pc.image_size,
        "mean": [round(v, 4) for v in mean.flatten().tolist()],
        "std": [round(v, 4) for v in std.flatten().tolist()],
        "temperature": round(temperature, 4),
        "test_accuracy": round(accuracy, 4),
        "test_images": len(test),
    }
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    # Examples: confident, correct predictions from different breeds (cats and dogs
    # alternate in the class list) and one mistake between birman and ragdoll.
    from PIL import Image

    probs = (logits / temperature).softmax(1)
    rng = np.random.default_rng(cfg.seed)
    correct = (pred == test.labels).numpy() & (probs.max(1).values > 0.9).numpy()
    chosen: list[tuple[int, str]] = []
    for c in rng.permutation(test.n_classes):
        candidates = np.flatnonzero(correct & (test.labels.numpy() == c))
        if len(candidates) and len(chosen) < cfg.n_examples - 1:
            chosen.append((int(rng.choice(candidates)), test.classes[c]))
    confused = {test.classes.index(b) for b in ("birman", "ragdoll") if b in test.classes}
    mistakes = [
        i
        for i in np.flatnonzero((pred != test.labels).numpy())
        if int(test.labels[i]) in confused and int(pred[i]) in confused
    ]
    if mistakes:
        chosen.append((int(mistakes[0]), f"{test.classes[int(test.labels[mistakes[0]])]}_hard"))
    for old in (out / "examples").glob("*.jpg"):
        old.unlink()
    for i, name in chosen:
        image = Image.fromarray(test.images[i].permute(1, 2, 0).numpy())
        image.save(out / "examples" / f"{name}.jpg", quality=92)

    return "\n".join(
        [
            f"Exported {cfg.model} · {FT_MODES[cfg.mode]} to {out}",
            f"test accuracy {accuracy:.4f} on {len(test)} images, temperature {temperature:.3f}",
            f"examples: {', '.join(name for _, name in chosen)}",
        ]
    )


def main() -> None:
    cli(run, ExportConfig, "Export a Pets model for the Gradio demo")


if __name__ == "__main__":
    main()
