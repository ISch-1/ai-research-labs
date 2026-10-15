"""Lab A defaults on top of the shared experiment plumbing in ``airlab.experiment``."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field

from airlab.config import ExperimentConfig
from airlab.experiment import Outputs, cli

__all__ = ["DEFAULT_OUTPUT_DIR", "LabAConfig", "Outputs", "cli"]

DEFAULT_OUTPUT_DIR = Path("lab-a-predictive/reports")


class LabAConfig(ExperimentConfig):
    output_dir: Path = Field(default=DEFAULT_OUTPUT_DIR)
