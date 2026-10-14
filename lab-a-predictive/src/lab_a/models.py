"""Model zoo for tabular experiments.

Every model is a full ``Pipeline`` (imputation + encoding + scaling + estimator)
so that preprocessing is refit inside each CV fold and nothing leaks from the
validation data.

Families compared (the question Lab A asks):

* statistical / linear — ``linear``, ``ridge``, ``lasso``, ``logreg``
* classical ML         — ``tree``, ``random_forest``, ``lightgbm``, ``svm``, ``knn``
* neural network       — ``mlp``
* reference            — ``dummy`` (the baseline everything must beat)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import optuna
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from lab_a.datasets import Task

FAMILY = {
    "dummy": "baseline",
    "linear": "statistical",
    "ridge": "statistical",
    "lasso": "statistical",
    "logreg": "statistical",
    "tree": "classical ML",
    "random_forest": "classical ML",
    "lightgbm": "classical ML",
    "svm": "classical ML",
    "knn": "classical ML",
    "mlp": "neural network",
}

TREE_MODELS = {"tree", "random_forest", "lightgbm"}


def preprocessor(*, scale: bool = True, drop_first: bool = False) -> ColumnTransformer:
    """Numeric: median impute (+ standardize). Categorical: mode impute + one-hot."""
    numeric_steps: list[Any] = [SimpleImputer(strategy="median")]
    if scale:
        numeric_steps.append(StandardScaler())
    categorical = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            drop="first" if drop_first else None,
            min_frequency=0.01,  # categories under 1% are pooled into "infrequent"
            sparse_output=False,
        ),
    )
    return ColumnTransformer(
        [
            ("num", make_pipeline(*numeric_steps), make_column_selector(dtype_include="number")),
            ("cat", categorical, make_column_selector(dtype_exclude="number")),
        ],
        verbose_feature_names_out=False,
    )


def _estimator(name: str, task: Task, seed: int, params: dict[str, Any]) -> Any:
    clf = task == "classification"
    factories: dict[str, Callable[[], Any]] = {
        "dummy": lambda: (
            DummyClassifier(strategy="prior") if clf else DummyRegressor(strategy="mean")
        ),
        "linear": lambda: LinearRegression(),
        "ridge": lambda: Ridge(**{"alpha": 1.0} | params),
        "lasso": lambda: Lasso(**{"alpha": 0.01, "max_iter": 5000} | params),
        "logreg": lambda: LogisticRegression(**{"C": 1.0, "max_iter": 2000} | params),
        "tree": lambda: (DecisionTreeClassifier if clf else DecisionTreeRegressor)(
            **{"max_depth": 8, "min_samples_leaf": 5, "random_state": seed} | params
        ),
        "random_forest": lambda: (RandomForestClassifier if clf else RandomForestRegressor)(
            **{"n_estimators": 300, "min_samples_leaf": 2, "n_jobs": -1, "random_state": seed}
            | params
        ),
        "lightgbm": lambda: (LGBMClassifier if clf else LGBMRegressor)(
            **{"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31, "random_state": seed}
            | {"verbose": -1, "n_jobs": 1}
            | params
        ),
        "svm": lambda: (
            SVC(**{"C": 1.0, "probability": True, "random_state": seed} | params)
            if clf
            else SVR(**{"C": 1.0} | params)
        ),
        "knn": lambda: (KNeighborsClassifier if clf else KNeighborsRegressor)(
            **{"n_neighbors": 15, "weights": "distance"} | params
        ),
        "mlp": lambda: (MLPClassifier if clf else MLPRegressor)(
            **{
                "hidden_layer_sizes": (64, 32),
                "alpha": 1e-3,
                "early_stopping": True,
                "max_iter": 500,
                "random_state": seed,
            }
            | params
        ),
    }
    if name not in factories:
        raise ValueError(f"unknown model {name!r}; choose from {sorted(factories)}")
    if (name == "logreg" and not clf) or (name in {"linear", "ridge", "lasso"} and clf):
        raise ValueError(f"model {name!r} does not support task {task!r}")
    return factories[name]()


def build(name: str, task: Task, seed: int = 42, params: dict[str, Any] | None = None) -> Pipeline:
    """Preprocessing + estimator. Trees skip scaling (it changes nothing for them)."""
    return Pipeline(
        [
            ("prep", preprocessor(scale=name not in TREE_MODELS)),
            ("model", _estimator(name, task, seed, params or {})),
        ]
    )


def default_models(task: Task) -> list[str]:
    if task == "classification":
        return ["dummy", "logreg", "tree", "random_forest", "lightgbm", "svm", "knn", "mlp"]
    return [
        "dummy",
        "linear",
        "ridge",
        "lasso",
        "tree",
        "random_forest",
        "lightgbm",
        "svm",
        "knn",
        "mlp",
    ]


# ------------------------------------------------------------------ tuning spaces


def search_space(name: str, trial: optuna.trial.BaseTrial) -> dict[str, Any]:
    """Optuna search spaces for the models worth tuning."""
    if name == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 8, 128, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        }
    if name == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=100),
            "max_features": trial.suggest_float("max_features", 0.2, 1.0),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20, log=True),
        }
    if name == "logreg":
        return {"C": trial.suggest_float("C", 1e-3, 100, log=True)}
    if name in {"ridge", "lasso"}:
        return {"alpha": trial.suggest_float("alpha", 1e-4, 100, log=True)}
    if name == "svm":
        return {
            "C": trial.suggest_float("C", 1e-2, 100, log=True),
            "gamma": trial.suggest_float("gamma", 1e-4, 1, log=True),
        }
    if name == "mlp":
        width = trial.suggest_categorical("width", [32, 64, 128, 256])
        depth = trial.suggest_int("depth", 1, 3)
        return {
            "hidden_layer_sizes": tuple([width] * depth),
            "alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
            "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
        }
    if name == "knn":
        return {"n_neighbors": trial.suggest_int("n_neighbors", 3, 75, log=True)}
    raise ValueError(f"no search space for {name!r}")


def feature_names(pipeline: Pipeline, X: pd.DataFrame) -> list[str]:
    return list(pipeline.named_steps["prep"].get_feature_names_out(X.columns))
