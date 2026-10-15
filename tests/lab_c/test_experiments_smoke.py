"""Lab C experiments end to end on their offline smoke configs."""

from pathlib import Path

import pandas as pd
import pytest

from airlab.config import load_config
from lab_c import augmentation, fundamentals, interpret, transfer

CONFIGS = Path(__file__).resolve().parents[2] / "lab-c-vision" / "configs"


def _check(out_dir: Path, name: str, text: str) -> pd.DataFrame:
    assert (out_dir / f"{name}.md").read_text(encoding="utf-8") == text
    for line in text.splitlines():
        if line.startswith("!["):
            assert (out_dir / line[line.index("(") + 1 : -1]).exists()
    return pd.read_csv(out_dir / "results" / f"{name}.csv")


def test_fundamentals(tmp_path):
    cfg = load_config(
        CONFIGS / "c1_smoke.yaml", fundamentals.FundamentalsConfig, output_dir=tmp_path
    )
    text = fundamentals.run(cfg)
    _check(tmp_path, cfg.experiment, text)
    assert "Gradient check" in text and "Ablations" in text


def test_transfer_offline_part(tmp_path):
    cfg = load_config(CONFIGS / "c2_smoke.yaml", transfer.TransferConfig, output_dir=tmp_path)
    results = _check(tmp_path, cfg.experiment, transfer.run(cfg))
    acc = results[results.metric == "accuracy"].set_index("model").value
    assert acc["SmallCNN"] > acc["MLP"]


def test_augmentation(tmp_path):
    cfg = load_config(
        CONFIGS / "c3_smoke.yaml", augmentation.AugmentationConfig, output_dir=tmp_path
    )
    results = _check(tmp_path, cfg.experiment, augmentation.run(cfg))
    assert set(results.model) == set(cfg.variants)


def test_interpret_shortcut(tmp_path):
    cfg = load_config(CONFIGS / "c5_smoke.yaml", interpret.InterpretConfig, output_dir=tmp_path)
    text = interpret.run(cfg)
    _check(tmp_path, cfg.experiment, text)
    assert "Shortcut learning" in text


@pytest.mark.hf
def test_transfer_errors_and_maps_with_a_small_pretrained_model(tmp_path):
    """C2 pets part, C4 and C5 maps with a ~2.5M-parameter ImageNet model on synthetic images."""
    from lab_c import errors
    from lab_c.train import TrainConfig

    pc = transfer.PetsConfig(
        dataset="synthetic_shapes",
        image_size=32,
        scratch_width=4,
        scratch_input=32,
        scratch=TrainConfig(epochs=1, batch_size=64),
        probe=["mobilenetv3_small_050"],
        finetune=["mobilenetv3_small_050"],
        finetune_train=TrainConfig(epochs=1, batch_size=64, augment="flip"),
    )
    t = transfer.run(
        transfer.TransferConfig(
            experiment="c2",
            output_dir=tmp_path,
            cifar=None,
            pets=pc,
            tsne_points=60,
            bootstrap_resamples=50,
        )
    )
    assert "linear probe" in t and "fine-tuned" in t
    e = errors.run(
        errors.ErrorsConfig(experiment="c4", output_dir=tmp_path, pets=pc, n_confident_errors=4)
    )
    assert "Calibration" in e and "Robustness" in e
    i = interpret.run(
        interpret.InterpretConfig(
            experiment="c5",
            output_dir=tmp_path,
            pets=pc,
            model="mobilenetv3_small_050",
            n_images=8,
            n_shown=2,
            ig_steps=4,
            shortcut=None,
        )
    )
    assert "Deletion AUC" in i


def test_augmentation_with_a_single_variant(tmp_path):
    """What one CI shard runs: a single recipe (one-row figure grid)."""
    cfg = load_config(
        CONFIGS / "c3_smoke.yaml",
        augmentation.AugmentationConfig,
        output_dir=tmp_path,
        variants=["flip+crop"],
        fractions=[1.0],
    )
    _check(tmp_path, cfg.experiment, augmentation.run(cfg))
