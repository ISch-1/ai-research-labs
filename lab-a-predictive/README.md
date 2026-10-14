# Lab A — Predictive & Statistical Modeling

Compares statistical models, classical ML and neural networks on tabular and time-series data,
and serves the winners behind an API.

| Module | Topic | Status |
|---|---|---|
| A0 | End-to-end baseline (`lab_a.baseline`) | ✅ |
| A1 | Regression & classification (California Housing, Adult/Bank Marketing, Wine Quality) | ⚪ |
| A2 | Time-series forecasting with rolling-origin backtesting | ⚪ |
| A3 | Clustering & anomaly detection (extends the Wholesale-customers K-Means study) | ⚪ |
| A4 | Statistical comparison across datasets (Friedman + Nemenyi CD diagram) | ⚪ |
| A5 | FastAPI model-serving API + Docker | ⚪ |

## Run

```bash
python -m lab_a.baseline --config lab-a-predictive/configs/baseline.yaml        # full
python -m lab_a.baseline --config lab-a-predictive/configs/baseline_smoke.yaml  # seconds
```

## Layout

```
configs/    YAML experiment configs (+ *_smoke.yaml for CI)
src/lab_a/  experiment code (importable as `lab_a`)
notebooks/  exploration only, outputs stripped
reports/    generated Markdown reports (committed)
```
