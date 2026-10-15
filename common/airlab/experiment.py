"""Plumbing shared by every lab's experiments: output locations and the CLI."""

from __future__ import annotations

import argparse
import os
import platform
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import yaml

from airlab.config import ExperimentConfig, load_config


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


def host_description() -> str:
    """CPU model and core count, so reported timings say what hardware produced them."""
    cpu = platform.processor() or platform.machine()
    try:  # Linux (e.g. GitHub runners): the readable model name lives in /proc/cpuinfo
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    if platform.system() == "Windows":
        try:
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            )
            cpu = str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    return f"{cpu}, {os.cpu_count()} logical CPUs, {platform.system()}"


C = TypeVar("C", bound=ExperimentConfig)


def parse_overrides(pairs: Sequence[str]) -> dict[str, Any]:
    """``["datasets=[sst2]", "max_test=50"]`` -> ``{"datasets": ["sst2"], "max_test": 50}``.

    Values are parsed as YAML, so lists, numbers, booleans and ``null`` work.
    """
    overrides: dict[str, Any] = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or not key:
            raise ValueError(f"--set expects KEY=VALUE, got {pair!r}")
        overrides[key.strip()] = yaml.safe_load(value)
    return overrides


def cli(run: Callable[[C], str], config_model: type[C], description: str) -> None:
    """``python -m <lab>.<experiment> --config path.yaml [--set key=value ...]``; prints the report."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config value (YAML syntax), e.g. --set 'datasets=[sst2]'",
    )
    args = parser.parse_args()
    text = run(load_config(args.config, config_model, **parse_overrides(args.overrides)))
    # Windows consoles default to cp1252, which cannot print the report's Unicode.
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    print(text)
