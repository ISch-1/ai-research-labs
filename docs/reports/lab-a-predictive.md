# Do complex models win? Statistical tests across tabular prediction, forecasting, clustering and anomaly detection

*Lab A technical report · [ai-research-labs](../../README.md) · code in [`lab-a-predictive/`](../../lab-a-predictive/) ·
detailed per-module reports in [`lab-a-predictive/reports/`](../../lab-a-predictive/reports/)*

## Abstract

We compare statistical models, classical machine learning and neural networks on four kinds of
problem — tabular classification and regression, time-series forecasting, clustering and
unsupervised anomaly detection — using one protocol: a held-out test set touched once, repeated
cross-validation or rolling-origin backtesting, and a significance test for every comparison
(corrected t-test, McNemar, Diebold-Mariano, Friedman with Nemenyi and Wilcoxon-Holm post-hoc
tests). Gradient boosting wins clearly on large tabular data (ROC-AUC 0.927 on Adult), but on
small datasets linear models are statistically tied with the best, and over 12 OpenML-CC18
datasets Random Forest, LightGBM and logistic regression cannot be separated (Holm p = 0.18).
Classical ETS and SARIMA beat ML and neural forecasters on short seasonal series; a seasonal
naive forecast wins when a structural break occurs. A widely used customer segmentation turns out
to be unstable under bootstrap resampling, and the best anomaly detector depends entirely on the
anomaly type. The best models are served through a typed FastAPI service in Docker.

## 1. Questions

1. **A1.** On tabular data, do trees, boosting and neural networks beat linear and statistical
   models — and does hyper-parameter tuning change the answer?
2. **A2.** For forecasting, are classical statistical models still competitive with ML and
   neural models?
3. **A3a.** Are customer segments found by clustering real, i.e. stable and externally valid?
4. **A3b.** Which unsupervised anomaly detector should be used when no labels are available?
5. **A4.** Across many datasets, does any model family win *in general*?
6. **A5.** How should such models be served so that their inputs and provenance are checked?

## 2. Setup

| | |
|---|---|
| Data | 6 tabular datasets (Adult, Bank Marketing, Breast Cancer, California Housing, Diabetes, Wine Quality), 6 public time series (Rdatasets), Wholesale Customers (UCI), 4 anomaly datasets (KDD Cup 1999 HTTP and SA, Mammography, synthetic), 12 binary OpenML-CC18 datasets — see the [data cards](../data-cards.md) |
| Models | OLS/Ridge/Lasso, logistic regression (+ `statsmodels` inference), k-NN, SVM, decision tree, Random Forest, LightGBM, MLP; naive, seasonal naive, drift, ETS, SARIMA, LightGBM and MLP on lags; K-Means, GMM, Ward, DBSCAN, HDBSCAN; robust z-score, Mahalanobis, Isolation Forest, LOF, One-Class SVM, autoencoder |
| Protocol (A1) | stratified 80/20 split; 2×5-fold CV on the training split with default hyper-parameters; Optuna (TPE, 30 trials, inner 3-fold CV) for selected models; one evaluation on the test split |
| Protocol (A2) | rolling-origin backtesting with 4 origins, horizon = one season; log transform when the seasonal amplitude grows with the level |
| Protocol (A3) | clustering judged by bootstrap stability (ARI between the full fit and 100 refits on 80% subsamples) and agreement with a known label; anomaly detectors fitted without labels, scored by PR-AUC, randomised ones over 5 seeds |
| Statistics | Nadeau-Bengio corrected t-test (CV), exact McNemar (test set), Diebold-Mariano with HLN correction (forecasts), Friedman + Nemenyi critical difference and Wilcoxon signed-rank with Holm correction (across datasets), 95% bootstrap CIs everywhere |

## 3. Results

### 3.1 Tabular prediction (A1)

| Dataset (rows) | Best model | Test metric (95% CI) | Linear/statistical model | Significant? |
|---|---|---|---|---|
| Adult (48,842) | LightGBM, tuned | ROC-AUC 0.927 [0.922, 0.932] | logistic regression 0.905 [0.898, 0.911] | yes — corrected t-test p < 0.0001 in CV |
| Breast Cancer (569) | logistic regression | ROC-AUC 0.996 [0.986, 1.000] | — (it is the best) | no model differs significantly |
| California Housing (20,640) | LightGBM, tuned | RMSE 0.430 [0.410, 0.449], R² 0.86 | Ridge (tuned) R² 0.58 | yes |
| Diabetes (442) | LightGBM, tuned | RMSE 51.9 [44.6, 59.0] | Ridge 53.8 [46.6, 61.2] | no — tied |

Tuning mattered only for LightGBM on Adult (ROC-AUC 0.925 → 0.927, McNemar p = 0.001). The MLP
was never ranked first on any of the six datasets with default settings.

### 3.2 Forecasting (A2)

Mean MASE over rolling origins (< 1 beats the in-sample seasonal naive forecast):

| Model | Air passengers | CO₂ | J&J | Nottingham | UK driver deaths | UK gas | Avg rank |
|---|---|---|---|---|---|---|---|
| ETS | **0.62** | 0.51 | **1.27** | **0.58** | 1.38 | 1.73 | **2.0** |
| SARIMA | 0.64 | 0.47 | 1.45 | 0.61 | 1.25 | **1.65** | **2.0** |
| LightGBM (lags) | 1.09 | 0.45 | 2.30 | 0.93 | 2.04 | 2.19 | 3.5 |
| seasonal naive | 1.25 | 1.38 | 4.03 | 0.84 | **1.16** | 2.33 | 3.8 |
| MLP (lags) | 3.32 | **0.41** | 7.52 | 0.82 | 1.64 | 5.55 | 4.5 |

Friedman χ² = 22.1, p = 0.001. ML and neural forecasters win only on the long, smooth CO₂ series
(503 points); on series of 84–144 points the MLP can be worse than seasonal naive. On UK driver
deaths the 1983 seat-belt law breaks the series and the model that does not extrapolate the
trend wins.

### 3.3 Clustering (A3a)

| Solution (Wholesale Customers) | k | Silhouette | Bootstrap ARI (95% interval) | ARI vs. sales channel |
|---|---|---|---|---|
| K-Means, standardised spend (original study) | 3 | 0.548 | 0.73 [0.35, 1.00] | 0.23 |
| K-Means, log-standardised spend | 2 | — | 0.96 [0.83, 1.00] | 0.51 |
| Ward (highest silhouette) | — | 0.79 | 0.29 | — |

The original K = 3 segmentation is reproduced exactly but is unstable; log-transforming the
skewed spend gives a stable K = 2 solution that recovers the Horeca/Retail channel. The solution
with the best silhouette is the least stable: silhouette alone is a poor selection criterion.
The procedure is first validated on synthetic data with planted clusters (ARI 1.0).

### 3.4 Anomaly detection (A3b)

| Detector | Synthetic | KDD HTTP | KDD SA | Mammography | Avg rank |
|---|---|---|---|---|---|
| Isolation Forest | 0.988 | 0.830 | 0.500 | **0.203** | **2.0** |
| One-Class SVM | 0.972 | 0.488 | 0.293 | 0.148 | 3.0 |
| robust z-score (median/MAD) | 0.489 | **0.989** | **0.759** | 0.038 | 3.5 |
| LOF | **0.997** | 0.102 | 0.071 | 0.137 | 3.8 |
| Mahalanobis | 0.531 | 0.988 | 0.094 | 0.077 | 4.0 |
| autoencoder | 0.958 | 0.319 | 0.060 | 0.123 | 4.8 |

PR-AUC (a random detector scores the contamination rate, 2–5%). When attacks are extreme values
of a few features (KDD), a robust z-score is best; when they form dense clusters, local-density
methods (LOF) fail by design. Isolation Forest is the safest default — never the worst — but
Friedman p = 0.42: four datasets cannot establish a general winner.

### 3.5 Across datasets (A4)

7 classifiers on 12 binary OpenML-CC18 datasets (CV ROC-AUC), Friedman χ² = 36.9, p < 0.0001:

| Model | Avg rank | Mean ROC-AUC | Wins–losses vs. RF | Holm-adjusted p vs. RF |
|---|---|---|---|---|
| **Random Forest** | 2.00 | 0.879 | — | — |
| LightGBM | 2.83 | 0.869 | 5–7 | 0.18 |
| logistic regression | 3.17 | 0.864 | 3–9 | 0.18 |
| SVM | 3.83 | 0.840 | 2–10 | 0.010 |
| k-NN | 4.67 | 0.851 | 1–11 | 0.006 |
| MLP | 4.83 | 0.821 | 1–11 | 0.005 |
| decision tree | 6.67 | 0.797 | 0–12 | 0.003 |

![Critical-difference diagram](../../lab-a-predictive/reports/figures/a4_benchmark/cc18_cd.png)

The top three cannot be separated with default hyper-parameters; the Nemenyi critical difference
(2.60 ranks) is more conservative still and also groups SVM with the leaders.

### 3.6 Serving (A5)

The API serves `adult_income` (LightGBM, hold-out ROC-AUC 0.929), `california_housing`
(LightGBM, R² 0.85), `air_passengers` (ETS) and `kdd_http` (Isolation Forest). Requests are
validated against a pydantic model generated from each model's own feature schema, and
`GET /models/{name}` returns hold-out metrics, library versions and the git commit — see the
[model cards](../model-cards/lab-a-api-models.md).

## 4. Discussion

- **"Which model is best?" has no dataset-free answer.** In every module the winner changed
  with the dataset, and with 3–6 datasets no post-hoc difference reached significance. Claims of
  general superiority need many datasets and rank-based tests.
- **Size matters more than model class.** Boosting's advantage is real on tens of thousands of
  rows and disappears on a few hundred, where an interpretable linear model is the better choice.
- **Validate unsupervised results externally.** Silhouette rewarded the least stable clustering;
  bootstrap stability and an external label were the useful criteria.
- **Inspect models, not only metrics.** The served Adult model assigns 10% income probability to
  a profile with a capital gain of 5,000 and 89% to the same profile with 0 — because all 117
  people with a gain of exactly 5,013 earn ≤ 50K. Monotonic constraints would be the fix.

## 5. Limitations

- Default hyper-parameters in the cross-dataset comparisons: a fully tuned benchmark could
  reorder the top models (it would not change the conclusion that they are close).
- Small benchmarks (3–12 datasets) limit statistical power; non-significant differences are not
  evidence of equality.
- Forecasting covers six classic monthly/quarterly series; no exogenous variables, no
  hierarchical or intermittent demand.
- Adult contains sensitive attributes (sex, race); this lab measures predictive performance
  only, not fairness — the served model must not be used for decisions about people
  (see the [model card](../model-cards/lab-a-api-models.md)).

## 6. Reproducibility

```bash
pip install -e ".[dev,lab-a]"
python -m lab_a.supervised  --config lab-a-predictive/configs/a1_classification.yaml
python -m lab_a.supervised  --config lab-a-predictive/configs/a1_regression.yaml
python -m lab_a.forecasting --config lab-a-predictive/configs/a2_forecasting.yaml
python -m lab_a.clustering  --config lab-a-predictive/configs/a3_clustering.yaml
python -m lab_a.anomaly     --config lab-a-predictive/configs/a3_anomaly.yaml
python -m lab_a.benchmark   --config lab-a-predictive/configs/a4_benchmark.yaml
docker compose up --build lab-a-api
```

## References

- Nadeau, Bengio (2003). *Inference for the generalization error.* Machine Learning.
- Dietterich (1998). *Approximate statistical tests for comparing supervised classification learning algorithms.* Neural Computation.
- Demšar (2006). *Statistical comparisons of classifiers over multiple data sets.* JMLR.
- Benavoli, Corani, Mangili (2016). *Should we really use post-hoc tests based on mean-ranks?* JMLR.
- Diebold, Mariano (1995). *Comparing predictive accuracy.* JBES · Harvey, Leybourne, Newbold (1997). *Testing the equality of prediction mean squared errors.* IJF.
- Hyndman, Koehler (2006). *Another look at measures of forecast accuracy.* IJF · Hyndman, Athanasopoulos (2021). *Forecasting: principles and practice*, 3rd ed.
- Grinsztajn, Oyallon, Varoquaux (2022). *Why do tree-based models still outperform deep learning on tabular data?* NeurIPS Datasets & Benchmarks.
- Bischl et al. (2021). *OpenML benchmarking suites.* NeurIPS Datasets & Benchmarks.
- Ke et al. (2017). *LightGBM.* NeurIPS · Akiba et al. (2019). *Optuna.* KDD · Lundberg, Lee (2017). *SHAP.* NeurIPS.
- Hennig (2007). *Cluster-wise assessment of cluster stability.* CSDA · Campello, Moulavi, Sander (2013). *HDBSCAN.* PAKDD.
- Liu, Ting, Zhou (2008). *Isolation forest.* ICDM · Breunig et al. (2000). *LOF.* SIGMOD.
