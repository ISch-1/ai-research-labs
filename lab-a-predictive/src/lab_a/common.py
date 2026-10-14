"""Plumbing shared by every Lab A experiment: output locations and the CLI."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from pydantic import Field

from airlab.config import ExperimentConfig, load_config

DEFAULT_OUTPUT_DIR = Path("lab-a-predictive/reports")


class LabAConfig(ExperimentConfig):
    output_dir: Path = Field(default=DEFAULT_OUTPUT_DIR)


@dataclass(frozen=True)
class Outputs:
    """Where an experiment writes: ``<dir>/<name>.md``, ``results/``, ``figures/<name>/``."""

    root: Path
    name: str

    @property
    def report(self) -> Path:
        return self.root / f"{self.name}.md"

    @property
    def results(self) -> Path:
        return self.root / "results" / f"{self.name}.csv"

    @property
    def figures(self) -> Path:
        path = self.root / "figures" / self.name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def figure(self, filename: str) -> tuple[Path, str]:
        """Absolute path to save to, and the relative link to use in the report."""
        return self.figures / filename, f"figures/{self.name}/{filename}"

    def write_report(self, text: str) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        self.report.write_text(text, encoding="utf-8")
        return self.report


C = TypeVar("C", bound=ExperimentConfig)


def cli(run: Callable[[C], str], config_model: type[C], description: str) -> None:
    """``python -m lab_a.<experiment> --config path.yaml``; prints the report."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    text = run(load_config(args.config, config_model))
    # Windows consoles default to cp1252, which cannot print the report's Unicode.
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    print(text)
