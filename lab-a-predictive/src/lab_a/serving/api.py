"""FastAPI service for the models exported by ``lab_a.serving.export``.

    uvicorn lab_a.serving.api:app --port 8000          # MODEL_DIR=models by default
    open http://localhost:8000/docs                    # interactive OpenAPI docs

Request bodies for tabular models are validated against each model's own
feature schema (types and allowed categories), generated from its metadata.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, create_model

log = logging.getLogger("lab_a.api")


@dataclass
class LoadedModel:
    name: str
    model: Any
    meta: dict[str, Any]
    request_schema: type[BaseModel] | None = None


def _instance_schema(meta: dict[str, Any]) -> type[BaseModel]:
    """A pydantic model with one optional field per feature (missing values are imputed)."""
    fields: dict[str, Any] = {}
    for feature in meta["features"]:
        if feature["type"] == "number":
            annotation: Any = float | None
        else:
            annotation = Literal[tuple(feature["categories"])] | None
        fields[feature["name"]] = (annotation, None)
    return create_model(
        f"{meta['name']}_instance",
        __config__=ConfigDict(extra="forbid", protected_namespaces=()),
        **fields,
    )


def load_registry(model_dir: Path) -> dict[str, LoadedModel]:
    registry = {}
    for meta_path in sorted(model_dir.glob("*/metadata.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        loaded = LoadedModel(meta["name"], joblib.load(meta_path.parent / "model.joblib"), meta)
        if meta["kind"] == "tabular":
            loaded.request_schema = _instance_schema(meta)
        registry[loaded.name] = loaded
        log.info("loaded model %s (%s)", loaded.name, meta["kind"])
    return registry


# ------------------------------------------------------------------- API schemas


class PredictRequest(BaseModel):
    instances: list[dict[str, Any]] = Field(min_length=1, max_length=1000)


class Prediction(BaseModel):
    prediction: float
    probability: float | None = None


class PredictResponse(BaseModel):
    model: str
    predictions: list[Prediction]


class ForecastRequest(BaseModel):
    horizon: int = Field(12, ge=1, le=60)
    level: float = Field(0.9, gt=0, lt=1)


class ForecastPoint(BaseModel):
    date: str
    mean: float
    lower: float
    upper: float


class ForecastResponse(BaseModel):
    model: str
    level: float
    points: list[ForecastPoint]


class AnomalyRequest(BaseModel):
    instances: list[list[float]] = Field(min_length=1, max_length=10_000)


class AnomalyResult(BaseModel):
    score: float
    is_anomaly: bool


class AnomalyResponse(BaseModel):
    model: str
    threshold: float
    results: list[AnomalyResult]


# --------------------------------------------------------------------------- app


def create_app(model_dir: str | Path | None = None) -> FastAPI:
    directory = Path(model_dir or os.environ.get("MODEL_DIR", "models"))

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.models = load_registry(directory)
        yield

    app = FastAPI(
        title="AI Research Labs — Lab A model API",
        version="1.0.0",
        description="Serves the regression, classification, forecasting and anomaly models "
        "trained in Lab A. Every response carries an `X-Process-Time-Ms` header.",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def timing(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = (time.perf_counter() - start) * 1000
        response.headers["X-Process-Time-Ms"] = f"{elapsed:.2f}"
        log.info("%s %s %d %.1fms", request.method, request.url.path, response.status_code, elapsed)
        return response

    def get(request: Request, name: str, kind: str) -> LoadedModel:
        registry: dict[str, LoadedModel] = request.app.state.models
        if name not in registry:
            raise HTTPException(404, f"model {name!r} not found; available: {sorted(registry)}")
        loaded = registry[name]
        if loaded.meta["kind"] != kind:
            raise HTTPException(400, f"model {name!r} is a {loaded.meta['kind']} model, not {kind}")
        return loaded

    @app.get("/health")
    def health(request: Request) -> dict[str, Any]:
        return {"status": "ok", "models": len(request.app.state.models)}

    @app.get("/models")
    def list_models(request: Request) -> list[dict[str, Any]]:
        return [
            {k: m.meta[k] for k in ("name", "kind", "estimator", "trained_at") if k in m.meta}
            for m in request.app.state.models.values()
        ]

    @app.get("/models/{name}")
    def model_metadata(request: Request, name: str) -> dict[str, Any]:
        registry = request.app.state.models
        if name not in registry:
            raise HTTPException(404, f"model {name!r} not found")
        return registry[name].meta

    @app.post("/predict/{name}")
    def predict(request: Request, name: str, body: PredictRequest) -> PredictResponse:
        loaded = get(request, name, "tabular")
        assert loaded.request_schema is not None
        rows = []
        for i, instance in enumerate(body.instances):
            try:
                rows.append(loaded.request_schema.model_validate(instance).model_dump())
            except Exception as exc:  # pydantic.ValidationError
                raise HTTPException(422, f"instance {i}: {exc}") from exc
        columns = [f["name"] for f in loaded.meta["features"]]
        frame = pd.DataFrame(rows, columns=columns)
        numeric = [f["name"] for f in loaded.meta["features"] if f["type"] == "number"]
        frame[numeric] = frame[numeric].astype(float)
        preds = loaded.model.predict(frame)
        if loaded.meta["task"] == "classification":
            proba = loaded.model.predict_proba(frame)[:, 1]
            out = [
                Prediction(prediction=float(p), probability=float(q))
                for p, q in zip(preds, proba, strict=True)
            ]
        else:
            out = [Prediction(prediction=float(p)) for p in preds]
        return PredictResponse(model=name, predictions=out)

    @app.post("/forecast/{name}")
    def forecast(request: Request, name: str, body: ForecastRequest) -> ForecastResponse:
        loaded = get(request, name, "forecast")
        n = loaded.meta["n_observations"]
        frame = loaded.model.get_prediction(start=n, end=n + body.horizon - 1).summary_frame(
            alpha=1 - body.level
        )[["mean", "pi_lower", "pi_upper"]]
        if loaded.meta.get("transform") == "log":  # fitted on log(y): back-transform
            frame = np.exp(frame)
        points = [
            ForecastPoint(
                date=pd.Timestamp(idx).date().isoformat(),
                mean=float(row["mean"]),
                lower=float(row["pi_lower"]),
                upper=float(row["pi_upper"]),
            )
            for idx, row in frame.iterrows()
        ]
        return ForecastResponse(model=name, level=body.level, points=points)

    @app.post("/anomaly/{name}")
    def anomaly(request: Request, name: str, body: AnomalyRequest) -> AnomalyResponse:
        loaded = get(request, name, "anomaly")
        X = np.asarray(body.instances, dtype=float)
        expected = loaded.meta["n_features"]
        if X.ndim != 2 or X.shape[1] != expected:
            raise HTTPException(422, f"each instance must have {expected} features")
        scores = -loaded.model.score_samples(X)
        threshold = loaded.meta["threshold"]
        return AnomalyResponse(
            model=name,
            threshold=threshold,
            results=[
                AnomalyResult(score=float(s), is_anomaly=bool(s >= threshold)) for s in scores
            ],
        )

    return app


logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
app = create_app()
