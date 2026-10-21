# Model cards — Lab A model API

*Format after Mitchell et al. (2019). The four models served by
[`lab_a.serving`](../../lab-a-predictive/src/lab_a/serving/) (`GET /models/{name}` returns the
same information as JSON). Report: [Lab A](../reports/lab-a-predictive.md).*

All four are exported by `python -m lab_a.serving.export --config lab-a-predictive/configs/a5_export.yaml`
(seed 42). Supervised models are first scored on a stratified 20% hold-out split, then **refit on
all rows** before export, so the hold-out numbers describe the procedure, not the exact object
served. Every response carries `X-Process-Time-Ms`; inputs are validated against each model's
own schema (types, allowed categories; unknown fields rejected). Code under MIT.

## `adult_income` — income above $50K (LightGBM classifier)

| | |
|---|---|
| Data | UCI Adult via OpenML #1590 (1994 US Census extract), 48,842 rows, 14 features |
| Output | probability that annual income > $50K (positive rate 23.9%) |
| Hold-out | ROC-AUC 0.929, accuracy 0.875 |
| Intended use | demonstrating a typed, versioned prediction API on a standard benchmark |
| **Out of scope** | **any decision about a real person** — credit, hiring, insurance, pricing, benefits |

**Ethical considerations.** Inputs include `sex`, `race`, `native-country`, `marital-status` and
`relationship`. The model was evaluated for accuracy only — no fairness audit (error rates by
group, calibration by group) was performed, and dropping the sensitive columns would not remove
proxies such as `relationship`. The data are 30 years old and describe the 1994 US labour market.

**Caveat found by inspection.** The model gives 10% probability to a profile with
`capital-gain = 5000` and 89% to the same profile with 0: all 117 people in the data with a gain
of exactly 5,013 earn ≤ $50K and the trees learned that split. Monotonic constraints or binning
would be required before any real use.

## `california_housing` — median house value (LightGBM regressor)

| | |
|---|---|
| Data | 1990 US Census block groups in California (StatLib, via scikit-learn), 20,640 rows, 8 numeric features |
| Output | median house value in units of $100,000 (the target is capped at 5.0 in the data) |
| Hold-out | RMSE 0.450, MAE 0.297, R² 0.846 |
| Out of scope | valuing a real property; prices today (the data are from 1990) |

Values above the $500,000 cap cannot be predicted. Block-group aggregates hide within-group
variation; the model has no location data finer than latitude/longitude.

## `air_passengers` — monthly airline passengers (ETS)

| | |
|---|---|
| Data | Box & Jenkins international airline passengers, 1949–1960 (Rdatasets `AirPassengers`), 144 months |
| Model | ETS(A, A, M) on log(passengers) — additive errors, additive (undamped) trend, multiplicative seasonality — chosen by AIC among 6 state-space specifications |
| Output | point forecast and prediction interval (`level`) for `horizon` months after December 1960 |
| Backtest (A2) | same procedure, h = 12, 4 rolling origins (1957–1960): MASE 0.622, sMAPE 4.5%, MAE 18.6, RMSE 22.9, 90% interval coverage 0.85 (slightly under-covering) |
| Out of scope | forecasting any current series; it is a teaching example of a fitted state-space model |

**Specification.** The export fits ETS exactly as A2 does: the seasonal spread of the series
grows with its level (`needs_log`), so the model is selected and fitted on log(passengers), and
the API exponentiates the point forecast and the interval bounds (`"transform": "log"` in the
metadata). The back-transformed point forecast is therefore the median, not the mean, of the
forecast distribution. The same ETS(A, A, M) specification is selected at every backtest origin
and on the full series, so the A2 scores describe this model's procedure; forecasts continue the
upward trend (mean of the second forecast year 594 vs. 529 for the first).

## `kdd_http` — network-connection anomaly score (Isolation Forest)

| | |
|---|---|
| Data | KDD Cup 1999 (10%), HTTP subset, 50,000-row subsample, 3 log-transformed features (duration, bytes sent, bytes received); contamination 3.8% |
| Output | anomaly score and a flag (threshold 0.631, fixed at export from the contamination rate) |
| In-sample | ROC-AUC 0.993, PR-AUC 0.811, precision 0.84 / recall 1.00 at the threshold |
| Out of scope | intrusion detection on real traffic: KDD'99 is a 1998 simulated network with known artefacts and attack types that no longer exist |

The metrics are in-sample (labels are used only for scoring an unsupervised model). In A3b a
robust z-score scored higher on this subset (PR-AUC 0.989 vs. 0.830); Isolation Forest is served
because it was the most consistent detector across datasets.
