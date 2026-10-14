# Experiment & repository conventions

## Evaluation protocol

1. **Split once.** A stratified (or, for time series, chronological) test set is carved out
   first with the experiment seed and is used exactly once, at the end.
2. **Compare on CV.** Model selection and tuning use repeated k-fold CV on the training split
   only. Every preprocessing step lives inside a scikit-learn `Pipeline` so it is refit per fold
   (no leakage).
3. **Report uncertainty.** Test metrics are reported as `value [low, high]` with a 95% percentile
   bootstrap CI (`airlab.metrics.report`).
4. **Test differences with the right tool** (`airlab.stats`):

   | Situation | Test |
   |---|---|
   | 2 models, same CV folds | `corrected_resampled_ttest` (Nadeau-Bengio) or `wilcoxon_signed_rank` |
   | 2 classifiers, one test set | `mcnemar` |
   | ≥3 models over several datasets | `friedman` + `nemenyi_cd` |
   | 2 forecasters | `diebold_mariano` |

5. **Always include a baseline** (majority class, mean, seasonal naive…). A model that cannot
   beat it significantly is reported as such.

## Experiments

- One experiment = one module with a `run(config)` function + YAML configs in `<lab>/configs/`.
- Configs subclass `airlab.config.ExperimentConfig` (unknown keys are rejected).
- Every config has a `*_smoke.yaml` sibling that runs in seconds; CI executes it.
- Runs are tracked with `airlab.tracking.track_run`; Markdown reports go to `<lab>/reports/`
  and are committed (they are the "paper" of the lab). Raw data, models and `mlruns/` are not.
- Notebooks are for exploration only and are stored without outputs (pre-commit `nbstripout`).
  Anything a report depends on must be reproducible from a script.

## Git

- Branch per feature: `lab-a/forecasting-backtest`, `common/friedman-plot`.
- [Conventional Commits](https://www.conventionalcommits.org/): `feat(lab-a): add SARIMA baseline`,
  `fix(stats): HLN correction for h>1`, `docs(lab-b): RAG results`.
- One GitHub milestone per phase; issues per sub-topic (A1…C5). Tag a release when a lab ships
  (`v0.1-lab-a`, `v0.2-lab-b`, `v0.3-lab-c`, `v1.0`).
