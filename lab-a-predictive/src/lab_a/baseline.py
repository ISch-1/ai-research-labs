"""End-to-end baseline experiment that exercises the whole toolkit.

    python -m lab_a.baseline --config lab-a-predictive/configs/baseline.yaml

Protocol: hold out a stratified test set, compare models with repeated
stratified k-fold CV on the rest (corrected t-test vs. the CV winner), then
refit on the full training split and report test metrics with bootstrap CIs
plus McNemar against the runner-up.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field
from sklearn.base import BaseEstimator
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from airlab.config import ExperimentConfig, load_config
from airlab.metrics import CLASSIFICATION_METRICS, MetricCI, report
from airlab.seed import set_seed
from airlab.stats import TestResult, corrected_resampled_ttest, mcnemar
from airlab.tracking import track_run

DATASETS: dict[str, Callable[..., tuple[np.ndarray, np.ndarray]]] = {
    "breast_cancer": load_breast_cancer,
    "wine": load_wine,
}

ModelName = Literal["dummy", "logreg", "random_forest"]
DEFAULT_MODELS: list[ModelName] = ["dummy", "logreg", "random_forest"]


def build_model(name: ModelName, seed: int) -> BaseEstimator:
    if name == "dummy":
        return DummyClassifier(strategy="most_frequent")
    if name == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
    raise ValueError(f"unknown model {name!r}")


class CVConfig(BaseModel):
    n_splits: int = 5
    n_repeats: int = 3


class BaselineConfig(ExperimentConfig):
    dataset: Literal["breast_cancer", "wine"] = "breast_cancer"
    test_size: float = 0.2
    scoring: str = "accuracy"
    cv: CVConfig = Field(default_factory=CVConfig)
    models: list[ModelName] = Field(default_factory=lambda: list(DEFAULT_MODELS), min_length=2)
    bootstrap_resamples: int = 1000
    report_path: Path | None = None


def run(config: BaselineConfig) -> str:
    set_seed(config.seed)
    X, y = DATASETS[config.dataset](return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.test_size, stratify=y, random_state=config.seed
    )
    cv = RepeatedStratifiedKFold(
        n_splits=config.cv.n_splits, n_repeats=config.cv.n_repeats, random_state=config.seed
    )
    n_val = len(y_train) // config.cv.n_splits
    n_fit = len(y_train) - n_val

    with track_run(config.experiment, config.model_dump(mode="json"), config.tags) as tracker:
        cv_scores: dict[str, np.ndarray] = {
            name: cross_val_score(
                build_model(name, config.seed), X_train, y_train, cv=cv, scoring=config.scoring
            )
            for name in config.models
        }
        ranking: list[str] = sorted(cv_scores, key=lambda m: cv_scores[m].mean(), reverse=True)
        best, runner_up = ranking[0], ranking[1]
        cv_tests: dict[str, TestResult] = {
            name: corrected_resampled_ttest(
                cv_scores[best], cv_scores[name], n_train=n_fit, n_test=n_val
            )
            for name in ranking[1:]
        }

        test_reports: dict[str, dict[str, MetricCI]] = {}
        predictions: dict[str, np.ndarray] = {}
        for name in config.models:
            model = build_model(name, config.seed).fit(X_train, y_train)
            predictions[name] = model.predict(X_test)
            test_reports[name] = report(
                y_test,
                predictions[name],
                CLASSIFICATION_METRICS,
                n_resamples=config.bootstrap_resamples,
                seed=config.seed,
            )
            tracker.log_metrics(
                {f"{name}.cv_{config.scoring}": cv_scores[name].mean()}
                | {f"{name}.test_{k}": v.value for k, v in test_reports[name].items()}
            )
        test_mcnemar = mcnemar(y_test, predictions[best], predictions[runner_up])

        text = render_report(config, cv_scores, ranking, cv_tests, test_reports, test_mcnemar)
        if config.report_path:
            config.report_path.parent.mkdir(parents=True, exist_ok=True)
            config.report_path.write_text(text, encoding="utf-8")
            tracker.log_artifact(config.report_path)
    return text


def render_report(
    config: BaselineConfig,
    cv_scores: dict[str, np.ndarray],
    ranking: list[str],
    cv_tests: dict[str, TestResult],
    test_reports: dict[str, dict[str, MetricCI]],
    test_mcnemar: TestResult,
) -> str:
    best = ranking[0]
    folds = config.cv.n_splits * config.cv.n_repeats
    lines = [
        f"# Baseline — `{config.dataset}`",
        "",
        f"Seed {config.seed} · {config.cv.n_repeats}×{config.cv.n_splits}-fold CV "
        f"({folds} folds) · test size {config.test_size:.0%}",
        "",
        f"## Cross-validation ({config.scoring})",
        "",
        f"| Model | Mean ± std | vs `{best}` (corrected t-test p) |",
        "|---|---|---|",
    ]
    for name in ranking:
        scores = cv_scores[name]
        p = "—" if name == best else f"{cv_tests[name].pvalue:.4f}"
        lines.append(f"| {name} | {scores.mean():.4f} ± {scores.std(ddof=1):.4f} | {p} |")

    metric_names = list(next(iter(test_reports.values())))
    lines += [
        "",
        "## Held-out test set (95% bootstrap CI)",
        "",
        "| Model | " + " | ".join(metric_names) + " |",
        "|---|" + "---|" * len(metric_names),
    ]
    for name in ranking:
        cells = " | ".join(str(test_reports[name][m]) for m in metric_names)
        lines.append(f"| {name} | {cells} |")

    verdict = "significant" if test_mcnemar.significant() else "not significant"
    lines += [
        "",
        f"McNemar `{ranking[0]}` vs `{ranking[1]}` on the test set: "
        f"p = {test_mcnemar.pvalue:.4f} ({verdict} at α = 0.05).",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)
    text = run(load_config(args.config, BaselineConfig))
    # Windows consoles default to cp1252, which cannot print the report's Unicode.
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    print(text)


if __name__ == "__main__":
    main()
