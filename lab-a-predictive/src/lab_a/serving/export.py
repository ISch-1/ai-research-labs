"""Train the models that the API serves and write them to a model directory.

    python -m lab_a.serving.export --config lab-a-predictive/configs/a5_export.yaml

Each model becomes ``<model_dir>/<name>/model.joblib`` plus ``metadata.json``
(input schema, metrics on a held-out split, library versions, git commit), so
the API can validate requests and ``GET /models`` can describe what it serves.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal

import joblib
import numpy as np
import pandas as pd
import sklearn
from pydantic import BaseModel, Field
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

from airlab import metrics as M
from airlab.config import ExperimentConfig
from airlab.seed import set_seed
from lab_a import models as Z
from lab_a.common import cli
from lab_a.datasets import load_anomaly, load_series, load_tabular

SCHEMA_VERSION = 1


class TabularSpec(BaseModel):
    kind: Literal["tabular"]
    name: str
    dataset: str
    model: str = "lightgbm"
    params: dict[str, Any] = Field(default_factory=dict)


class ForecastSpec(BaseModel):
    kind: Literal["forecast"]
    name: str
    series: str


class AnomalySpec(BaseModel):
    kind: Literal["anomaly"]
    name: str
    dataset: str
    contamination: float | None = None  # threshold quantile; default = dataset's rate
    max_rows: int | None = 50_000


class ExportConfig(ExperimentConfig):
    model_dir: Path = Path("models")
    models: list[Annotated[TabularSpec | ForecastSpec | AnomalySpec, Field(discriminator="kind")]]


def _git_commit() -> str | None:
    if commit := os.environ.get("GIT_COMMIT"):  # set by the Docker build
        return commit
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _provenance(seed: int) -> dict[str, Any]:
    return {
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "seed": seed,
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "schema_version": SCHEMA_VERSION,
    }


def _feature_schema(X: pd.DataFrame) -> list[dict[str, Any]]:
    schema = []
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            schema.append({"name": col, "type": "number"})
        else:
            values = sorted(str(v) for v in X[col].dropna().unique())
            schema.append({"name": col, "type": "category", "categories": values})
    return schema


def export_tabular(spec: TabularSpec, seed: int) -> tuple[Any, dict[str, Any]]:
    ds = load_tabular(spec.dataset)
    X_tr, X_te, y_tr, y_te = train_test_split(
        ds.X,
        ds.y,
        test_size=0.2,
        random_state=seed,
        stratify=ds.y if ds.task == "classification" else None,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = Z.build(spec.model, ds.task, seed, spec.params).fit(X_tr, y_tr)
    pred = model.predict(X_te)
    if ds.task == "classification":
        proba = model.predict_proba(X_te)[:, 1]
        holdout = {
            "roc_auc": float(roc_auc_score(y_te, proba)),
            "accuracy": float(np.mean(pred == y_te.to_numpy())),
        }
    else:
        holdout = {k: float(fn(y_te.to_numpy(), pred)) for k, fn in M.REGRESSION_METRICS.items()}
    # Refit on all data for serving; the hold-out metrics describe the procedure.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = Z.build(spec.model, ds.task, seed, spec.params).fit(ds.X, ds.y)
    meta: dict[str, Any] = {
        "kind": "tabular",
        "task": ds.task,
        "estimator": spec.model,
        "dataset": ds.name,
        "source": ds.source,
        "n_train_rows": len(ds.y),
        "features": _feature_schema(ds.X),
        "holdout_metrics": holdout,
    }
    if ds.task == "classification":
        meta["positive_class_rate"] = float(ds.y.mean())
    return model, meta


def export_forecast(spec: ForecastSpec, seed: int) -> tuple[Any, dict[str, Any]]:
    """AIC-selected ETS, fitted exactly as in A2: on the log scale when the seasonal
    spread grows with the level. ``transform`` in the metadata tells the API to
    exponentiate the forecasts (the point forecast is then the median)."""
    from lab_a.forecasting import fit_ets, needs_log

    ts = load_series(spec.series)
    y = ts.y.astype(float)
    use_log = needs_log(y.to_numpy(), ts.season_length)
    try:
        best, best_spec = fit_ets(np.log(y) if use_log else y, ts.season_length)
    except RuntimeError as exc:
        raise RuntimeError(f"could not fit ETS to {spec.series}") from exc
    meta = {
        "kind": "forecast",
        "estimator": "ets",
        "ets_spec": best_spec,
        "transform": "log" if use_log else "none",
        "aic": float(best.aic),  # on the transformed scale
        "series": ts.name,
        "source": ts.source,
        "frequency": ts.y.index.freqstr,
        "season_length": ts.season_length,
        "last_observation": ts.y.index[-1].isoformat(),
        "n_observations": len(y),
    }
    return best, meta


def export_anomaly(spec: AnomalySpec, seed: int) -> tuple[Any, dict[str, Any]]:
    ds = load_anomaly(spec.dataset, max_rows=spec.max_rows, seed=seed)
    model = make_pipeline(
        RobustScaler(), IsolationForest(n_estimators=300, random_state=seed, n_jobs=-1)
    ).fit(ds.X)
    scores = -model.score_samples(ds.X)
    contamination = spec.contamination or ds.contamination
    threshold = float(np.quantile(scores, 1 - contamination))
    flagged = scores >= threshold
    meta = {
        "kind": "anomaly",
        "estimator": "isolation_forest",
        "dataset": ds.name,
        "source": ds.source,
        "n_features": int(ds.X.shape[1]),
        "threshold": threshold,
        "contamination": float(contamination),
        "in_sample_metrics": {
            "roc_auc": float(roc_auc_score(ds.y, scores)),
            "pr_auc": float(average_precision_score(ds.y, scores)),
            "precision_at_threshold": float(ds.y[flagged].mean()),
            "recall_at_threshold": float(flagged[ds.y == 1].mean()),
        },
    }
    return model, meta


EXPORTERS = {"tabular": export_tabular, "forecast": export_forecast, "anomaly": export_anomaly}


def run(cfg: ExportConfig) -> str:
    set_seed(cfg.seed)
    lines = [f"Exported to `{cfg.model_dir}`:", ""]
    for spec in cfg.models:
        model, meta = EXPORTERS[spec.kind](spec, cfg.seed)  # type: ignore[operator]
        meta = {"name": spec.name, **meta, **_provenance(cfg.seed)}
        target = cfg.model_dir / spec.name
        target.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, target / "model.joblib")
        (target / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        summary = (
            meta.get("holdout_metrics") or meta.get("in_sample_metrics") or {"aic": meta.get("aic")}
        )
        shown = ", ".join(f"{k}={v:.4f}" for k, v in summary.items())
        lines.append(f"- `{spec.name}` ({spec.kind}, {meta['estimator']}): {shown}")
    return "\n".join(lines)


def main() -> None:
    cli(run, ExportConfig, "Export models for the Lab A API")


if __name__ == "__main__":
    main()
