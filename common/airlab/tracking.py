"""Experiment tracking with MLflow, falling back to plain JSON files.

``AIRLAB_TRACKING`` selects the backend:

* ``mlflow`` — log to MLflow (``MLFLOW_TRACKING_URI`` or the local ``mlruns/``)
* ``json``   — write ``runs/<experiment>/<timestamp>/run.json``; no extra deps
* ``auto``   — MLflow if installed, otherwise JSON (default)
* ``off``    — record nothing (used by smoke tests)
"""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from airlab.config import flatten
from airlab.data import REPO_ROOT


class Run:
    """Common interface over both backends."""

    def log_params(self, params: dict[str, Any]) -> None:
        raise NotImplementedError

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        raise NotImplementedError

    def log_artifact(self, path: str | Path) -> None:
        raise NotImplementedError


class NullRun(Run):
    def log_params(self, params: dict[str, Any]) -> None:
        pass

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        pass

    def log_artifact(self, path: str | Path) -> None:
        pass


class JsonRun(Run):
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.record: dict[str, Any] = {"params": {}, "metrics": [], "artifacts": []}

    def log_params(self, params: dict[str, Any]) -> None:
        self.record["params"].update(flatten(params))

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        self.record["metrics"].append({"step": step, **{k: float(v) for k, v in metrics.items()}})

    def log_artifact(self, path: str | Path) -> None:
        source = Path(path)
        shutil.copy2(source, self.run_dir / source.name)
        self.record["artifacts"].append(source.name)

    def save(self) -> None:
        (self.run_dir / "run.json").write_text(json.dumps(self.record, indent=2, default=str))


class MlflowRun(Run):
    def __init__(self, mlflow: Any) -> None:
        self._mlflow = mlflow

    def log_params(self, params: dict[str, Any]) -> None:
        self._mlflow.log_params(flatten(params))

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        self._mlflow.log_metrics({k: float(v) for k, v in metrics.items()}, step=step)

    def log_artifact(self, path: str | Path) -> None:
        self._mlflow.log_artifact(str(path))


def _backend() -> str:
    choice = os.environ.get("AIRLAB_TRACKING", "auto").lower()
    if choice != "auto":
        return choice
    try:
        import mlflow  # noqa: F401
    except ImportError:
        return "json"
    return "mlflow"


@contextmanager
def track_run(
    experiment: str,
    params: dict[str, Any] | None = None,
    tags: dict[str, str] | None = None,
) -> Iterator[Run]:
    """Open a tracked run; parameters are logged immediately."""
    backend = _backend()
    run: Run

    if backend == "off":
        yield NullRun()
    elif backend == "json":
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        run_dir = REPO_ROOT / "runs" / experiment / stamp
        run_dir.mkdir(parents=True)
        run = JsonRun(run_dir)
        run.record["tags"] = tags or {}
        run.log_params(params or {})
        try:
            yield run
        finally:
            run.save()
    elif backend == "mlflow":
        import mlflow

        mlflow.set_experiment(experiment)
        with mlflow.start_run(tags=tags):
            run = MlflowRun(mlflow)
            run.log_params(params or {})
            yield run
    else:
        raise ValueError(f"Unknown AIRLAB_TRACKING backend: {backend!r}")
