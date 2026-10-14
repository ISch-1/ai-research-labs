"""Runs the Lab A baseline end to end on the smoke config."""

from pathlib import Path

from airlab.config import load_config
from lab_a.baseline import BaselineConfig, run

SMOKE = Path(__file__).resolve().parents[2] / "lab-a-predictive" / "configs" / "baseline_smoke.yaml"


def test_baseline_smoke(tmp_path):
    config = load_config(SMOKE, BaselineConfig, report_path=tmp_path / "report.md")
    text = run(config)

    assert "## Cross-validation" in text
    assert "McNemar" in text
    assert (tmp_path / "report.md").read_text(encoding="utf-8") == text
    # On wine a scaled logistic regression must beat the majority-class baseline.
    cv_section = text.split("## Held-out")[0]
    assert cv_section.index("| logreg") < cv_section.index("| dummy")
