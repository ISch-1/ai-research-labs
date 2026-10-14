.PHONY: setup lint format typecheck test smoke baseline mlflow clean

PYTHON ?= python

setup:  ## Install the toolkit + dev tools in editable mode and enable git hooks
	$(PYTHON) -m pip install -e ".[dev,tracking]"
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

smoke:  ## Fast end-to-end run used by CI
	AIRLAB_TRACKING=off $(PYTHON) -m lab_a.baseline --config lab-a-predictive/configs/baseline_smoke.yaml

baseline:  ## Full Lab A baseline; writes lab-a-predictive/reports/
	$(PYTHON) -m lab_a.baseline --config lab-a-predictive/configs/baseline.yaml

mlflow:  ## MLflow UI at http://localhost:5000
	docker compose up -d mlflow

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov runs
