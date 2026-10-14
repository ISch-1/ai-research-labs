"""Exports the offline smoke models and exercises every API endpoint."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from airlab.config import load_config
from lab_a.serving import api, export

CONFIG = (
    Path(__file__).resolve().parents[2] / "lab-a-predictive" / "configs" / "a5_export_smoke.yaml"
)


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    model_dir = tmp_path_factory.mktemp("models")
    export.run(load_config(CONFIG, export.ExportConfig, model_dir=model_dir))
    with TestClient(api.create_app(model_dir)) as c:
        yield c


def test_health_and_listing(client):
    assert client.get("/health").json() == {"status": "ok", "models": 4}
    names = {m["name"] for m in client.get("/models").json()}
    assert names == {"breast_cancer", "diabetes", "synthetic_seasonal", "synthetic_blobs"}


def test_metadata_has_schema_and_provenance(client):
    meta = client.get("/models/breast_cancer").json()
    assert meta["task"] == "classification"
    assert meta["holdout_metrics"]["roc_auc"] > 0.95
    assert {"trained_at", "scikit_learn", "seed"} <= set(meta)
    assert all(f["type"] == "number" for f in meta["features"])


def test_classification_prediction(client):
    meta = client.get("/models/breast_cancer").json()
    instance = {f["name"]: 1.0 for f in meta["features"]}
    instance["mean radius"] = None  # missing values are imputed
    response = client.post("/predict/breast_cancer", json={"instances": [instance, instance]})
    assert response.status_code == 200
    preds = response.json()["predictions"]
    assert len(preds) == 2
    assert 0 <= preds[0]["probability"] <= 1
    assert preds[0]["prediction"] in (0, 1)
    assert float(response.headers["X-Process-Time-Ms"]) >= 0


def test_regression_prediction(client):
    meta = client.get("/models/diabetes").json()
    instance = {f["name"]: 0.0 for f in meta["features"]}
    response = client.post("/predict/diabetes", json={"instances": [instance]})
    pred = response.json()["predictions"][0]
    assert pred["probability"] is None
    assert 100 < pred["prediction"] < 200  # all-zero (=mean) features -> near the target mean


def test_prediction_validation_errors(client):
    bad_type = client.post("/predict/diabetes", json={"instances": [{"age": "old"}]})
    assert bad_type.status_code == 422
    unknown = client.post("/predict/diabetes", json={"instances": [{"not_a_feature": 1}]})
    assert unknown.status_code == 422
    empty = client.post("/predict/diabetes", json={"instances": []})
    assert empty.status_code == 422


def test_wrong_kind_and_missing_model(client):
    assert client.post("/predict/synthetic_seasonal", json={"instances": [{}]}).status_code == 400
    assert client.post("/predict/nope", json={"instances": [{}]}).status_code == 404


def test_forecast(client):
    response = client.post("/forecast/synthetic_seasonal", json={"horizon": 6, "level": 0.8})
    body = response.json()
    assert response.status_code == 200
    assert len(body["points"]) == 6
    first = body["points"][0]
    assert first["date"] == "2020-01-01"  # series ends 2019-12
    assert first["lower"] < first["mean"] < first["upper"]
    assert client.post("/forecast/synthetic_seasonal", json={"horizon": 0}).status_code == 422


def test_anomaly(client):
    meta = client.get("/models/synthetic_blobs").json()
    normal_point = [0.0] * meta["n_features"]
    far_point = [1e3] * meta["n_features"]
    body = client.post(
        "/anomaly/synthetic_blobs", json={"instances": [normal_point, far_point]}
    ).json()
    scores = [r["score"] for r in body["results"]]
    assert scores[1] > scores[0]
    assert body["results"][1]["is_anomaly"] is True
    wrong = client.post("/anomaly/synthetic_blobs", json={"instances": [[1.0, 2.0]]})
    assert wrong.status_code == 422


def test_metadata_is_valid_json_on_disk(tmp_path):
    export.run(
        load_config(
            CONFIG,
            export.ExportConfig,
            model_dir=tmp_path,
            models=[
                {"kind": "tabular", "name": "bc", "dataset": "breast_cancer", "model": "logreg"},
            ],
        )
    )
    meta = json.loads((tmp_path / "bc" / "metadata.json").read_text())
    assert meta["name"] == "bc" and (tmp_path / "bc" / "model.joblib").exists()
