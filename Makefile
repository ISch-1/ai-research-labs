.PHONY: setup lint format typecheck test smoke baseline lab-a export serve docker mlflow clean

PYTHON ?= python
RUN = $(PYTHON) -m
CONFIGS = lab-a-predictive/configs

setup:  ## Install everything in editable mode and enable git hooks
	$(PYTHON) -m pip install -e ".[dev,lab-a,tracking]"
	pre-commit install

lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .

typecheck:
	mypy common lab-a-predictive/src

test:
	pytest --cov=airlab --cov=lab_a --cov-report=term-missing

smoke:  ## Fast end-to-end runs (offline data)
	AIRLAB_TRACKING=off $(RUN) lab_a.baseline --config $(CONFIGS)/baseline_smoke.yaml

baseline:
	$(RUN) lab_a.baseline --config $(CONFIGS)/baseline.yaml

lab-a:  ## All Lab A experiments (~1 h on a laptop); A4 last, it reads the others' results
	$(RUN) lab_a.supervised  --config $(CONFIGS)/a1_classification.yaml
	$(RUN) lab_a.supervised  --config $(CONFIGS)/a1_regression.yaml
	$(RUN) lab_a.forecasting --config $(CONFIGS)/a2_forecasting.yaml
	$(RUN) lab_a.clustering  --config $(CONFIGS)/a3_clustering.yaml
	$(RUN) lab_a.anomaly     --config $(CONFIGS)/a3_anomaly.yaml
	$(RUN) lab_a.benchmark   --config $(CONFIGS)/a4_benchmark.yaml

export:  ## Train and export the served models to ./models
	$(RUN) lab_a.serving.export --config $(CONFIGS)/a5_export.yaml

serve:  ## API on http://localhost:8000/docs (after `make export`)
	uvicorn lab_a.serving.api:app --reload --port 8000

docker:  ## Build and run the API container
	docker compose up --build lab-a-api

mlflow:  ## MLflow UI at http://localhost:5000
	docker compose up -d mlflow

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov runs models-smoke
