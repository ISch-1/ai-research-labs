# ai-research-labs

[![CI](https://github.com/ISch-1/ai-research-labs/actions/workflows/ci.yml/badge.svg)](https://github.com/ISch-1/ai-research-labs/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.11%20|%203.12%20|%203.13-blue)](https://www.python.org/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-261230)](https://github.com/astral-sh/ruff)
[![License](https://img.shields.io/badge/License-MIT-blue)](LICENSE)

> Three reproducible research labs that compare classical ML, statistical methods and neural networks — on tabular data, text and images — and answer each question **with confidence intervals and significance tests, not single numbers**. Every experiment is a config file plus one command; every model that matters ships behind an API.

---

## Labs

| Lab | Questions it answers | Status |
|---|---|---|
| [**A — Predictive & Statistical Modeling**](lab-a-predictive/) | Do complex models really beat linear ones on tabular data? Classical vs. ML vs. neural forecasting? Which anomaly detector, and how stable are the clusters? Served via a typed FastAPI + Docker service. | ✅ Done |
| [**B — NLP & LLM Research**](lab-b-nlp-llm/) | TF-IDF vs. embeddings vs. fine-tuning vs. prompting. How good is a RAG system, really — and at what cost and latency? | ⚪ Planned |
| [**C — Computer Vision Research**](lab-c-vision/) | From a NumPy perceptron to transfer learning: what do augmentation and pre-training buy, and what is the model actually looking at? | ⚪ Planned |

---

## How an experiment works

```
configs/*.yaml ──> typed config (pydantic) ──> set_seed()
      │
      v
 repeated stratified k-fold CV ──> corrected t-test / Wilcoxon / Friedman + Nemenyi
      │
      v
 held-out test set ──> metrics with 95% bootstrap CI ──> McNemar / Diebold-Mariano
      │
      v
 MLflow (or JSON) run + Markdown report in <lab>/reports/
```

The shared toolkit lives in [`common/airlab`](common/airlab):

| Module | Purpose |
|---|---|
| [`stats.py`](common/airlab/stats.py) | Nadeau-Bengio corrected t-test, Wilcoxon, McNemar, Friedman + Nemenyi critical difference, Holm correction, Diebold-Mariano |
| [`plots.py`](common/airlab/plots.py) | Critical-difference diagrams and shared figure helpers |
| [`reporting.py`](common/airlab/reporting.py) | Markdown tables and the long-format results files every experiment writes |
| [`metrics.py`](common/airlab/metrics.py) | Metric reports with percentile bootstrap confidence intervals |
| [`config.py`](common/airlab/config.py) | YAML → pydantic configs that reject typos |
| [`tracking.py`](common/airlab/tracking.py) | MLflow tracking with a zero-dependency JSON fallback |
| [`data.py`](common/airlab/data.py) | Dataset download cache with SHA-256 verification |
| [`seed.py`](common/airlab/seed.py) | One call seeds Python, NumPy and PyTorch |

---

## Quickstart

Prerequisites: Python 3.11+. Docker is optional (model API container, MLflow UI).

```bash
git clone https://github.com/ISch-1/ai-research-labs.git
cd ai-research-labs
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e ".[dev,lab-a,tracking]"

pytest                               # unit + smoke tests
python -m lab_a.baseline --config lab-a-predictive/configs/baseline.yaml
```

Lab A's experiments, model API and Docker image are described in
[lab-a-predictive/README.md](lab-a-predictive/README.md). The baseline above prints a report like
[`lab-a-predictive/reports/baseline_breast_cancer.md`](lab-a-predictive/reports/baseline_breast_cancer.md)
and logs the run to `./mlruns`. Browse runs with `mlflow ui` or `docker compose up -d mlflow`
(<http://localhost:5000>, then set `MLFLOW_TRACKING_URI=http://localhost:5000`).

With `make` available: `make setup`, `make test`, `make lab-a` (all experiments), `make export && make serve` (API), `make docker`.

---

## Engineering standards

- **Reproducible:** fixed seeds, versioned configs, data downloaded by code (never committed).
- **Honest evaluation:** test set touched once; all comparisons reported with CIs and the appropriate paired test — see [docs/conventions.md](docs/conventions.md).
- **Quality gates:** ruff, mypy and pytest on every push (CI matrix 3.11–3.13), every experiment's offline smoke config, and a Docker build that starts the API and calls it; pre-commit strips notebook outputs and blocks secrets.

## Related work

- [db-sentinel](https://github.com/ISch-1/db-sentinel) — Isolation Forest anomaly detection running in production on database metrics (the applied counterpart of Lab A's anomaly study).

## Roadmap

- [x] **Phase 0** — repo, shared toolkit, CI, end-to-end baseline
- [x] **Phase 1** — Lab A: regression/classification, forecasting, clustering & anomalies, statistical comparison, model-serving API
- [ ] **Phase 2** — Lab B: representations, text classification, semantic search & RAG, LLM evaluation, cost/latency/quality
- [ ] **Phase 3** — Lab C: perceptron & MLP from scratch, CNNs & transfer learning, augmentation, error analysis, interpretability
- [ ] **Phase 4** — reports, model cards, hosted demos

## License

[MIT](LICENSE)
