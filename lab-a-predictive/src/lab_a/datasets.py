"""Dataset registry for Lab A.

Every loader returns tidy pandas objects and caches downloads under
``airlab.data.data_dir()``. Bundled datasets (``breast_cancer``, ``diabetes``,
``co2``, ``synthetic_*``) need no network and are what CI uses.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn import datasets as skd

from airlab.data import data_dir, fetch

Task = Literal["classification", "regression"]


# --------------------------------------------------------------------------- tabular


@dataclass
class TabularDataset:
    name: str
    X: pd.DataFrame
    y: pd.Series
    task: Task
    source: str

    @property
    def n_categorical(self) -> int:
        return sum(not pd.api.types.is_numeric_dtype(t) for t in self.X.dtypes)


def _binary_target(y: pd.Series) -> pd.Series:
    """Map a two-class target to 0/1 with the minority class as the positive class."""
    counts = y.value_counts()
    if len(counts) != 2:
        raise ValueError(f"expected a binary target, got {len(counts)} classes")
    positive = counts.idxmin()
    return (y == positive).astype(int).rename("target")


def _sklearn_home() -> str:
    return str(data_dir() / "sklearn")


def openml(
    data_id: int, name: str | None = None, *, task: Task = "classification"
) -> TabularDataset:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bunch = skd.fetch_openml(
            data_id=data_id, data_home=_sklearn_home(), as_frame=True, parser="auto"
        )
    X = bunch.data.copy()
    for col in X.select_dtypes("category"):
        X[col] = X[col].astype("object")
    y = _binary_target(bunch.target) if task == "classification" else bunch.target.astype(float)
    return TabularDataset(name or bunch.details["name"], X, y, task, f"OpenML #{data_id}")


def _breast_cancer() -> TabularDataset:
    X, y = skd.load_breast_cancer(return_X_y=True, as_frame=True)
    # sklearn encodes malignant as 0; make the clinically relevant class positive.
    return TabularDataset("breast_cancer", X, (1 - y).rename("target"), "classification", "sklearn")


def _diabetes() -> TabularDataset:
    X, y = skd.load_diabetes(return_X_y=True, as_frame=True)
    return TabularDataset("diabetes", X, y.astype(float), "regression", "sklearn")


def _california_housing() -> TabularDataset:
    X, y = skd.fetch_california_housing(return_X_y=True, as_frame=True, data_home=_sklearn_home())
    return TabularDataset("california_housing", X, y.astype(float), "regression", "StatLib/sklearn")


def _wine_quality() -> TabularDataset:
    return openml(40691, "wine_quality_red", task="regression")


def _synthetic_classification(kind: str) -> TabularDataset:
    """Offline binary datasets with known structure (used by tests and smoke runs)."""
    if kind == "linear":
        X, y = skd.make_classification(
            n_samples=600, n_features=10, n_informative=5, flip_y=0.05, random_state=1
        )
    else:  # Hastie et al. 10.2: nonlinear (chi-squared) decision boundary
        X, y = skd.make_hastie_10_2(n_samples=1000, random_state=1)
        y = (y > 0).astype(int)
    cols = [f"x{i}" for i in range(X.shape[1])]
    return TabularDataset(
        f"synthetic_{kind}",
        pd.DataFrame(X, columns=cols),
        _binary_target(pd.Series(y)),
        "classification",
        "synthetic",
    )


TABULAR: dict[str, Callable[[], TabularDataset]] = {
    "breast_cancer": _breast_cancer,
    "synthetic_linear": lambda: _synthetic_classification("linear"),
    "synthetic_hastie": lambda: _synthetic_classification("hastie"),
    "diabetes": _diabetes,
    "california_housing": _california_housing,
    "wine_quality_red": _wine_quality,
    "adult": lambda: openml(1590, "adult"),
    "bank_marketing": lambda: openml(1461, "bank_marketing"),
}

# Binary classification datasets from the OpenML-CC18 suite used for the
# cross-dataset comparison (A4). Small enough to run every model with CV.
BENCHMARK_CLASSIFICATION: dict[str, int] = {
    "credit_g": 31,
    "pima_diabetes": 37,
    "spambase": 44,
    "kc1": 1067,
    "pc1": 1068,
    "banknote": 1462,
    "blood_transfusion": 1464,
    "ilpd": 1480,
    "ozone_level": 1487,
    "phoneme": 1489,
    "qsar_biodeg": 1494,
    "wdbc": 1510,
}


def load_tabular(name: str) -> TabularDataset:
    if name in TABULAR:
        return TABULAR[name]()
    if name in BENCHMARK_CLASSIFICATION:
        return openml(BENCHMARK_CLASSIFICATION[name], name)
    raise KeyError(f"unknown dataset {name!r}")


# ----------------------------------------------------------------------- time series


@dataclass
class TimeSeries:
    name: str
    y: pd.Series  # DatetimeIndex with a fixed frequency
    season_length: int
    source: str


_RDATASETS = "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/{}.csv"
_RDATASETS_SHA256 = {
    "AirPassengers": "bdb98adbd418a6de6842a742e0602f363c3b841c26677e7e61a1e055e9509bd8",
    "nottem": "634e2e374c5d59c1332e6e6de207d691cb4ebdcd007f7f0e2993ac2a19c90473",
    "UKDriverDeaths": "5ea8754082b2a4649966f6f639dc8a06fef2a1e2f7e3a41a372501986369179f",
    "UKgas": "69ec013fbeeaf956a42603aaf8f68d943a3aab5f90b6ed89c5de2ed02eba59ff",
    "JohnsonJohnson": "8d84846f400a24fd6bed71d80c5a89b9cda2e47702ef58c03610a538ca9157e7",
}


def _rdataset(name: str, label: str, season_length: int = 12) -> TimeSeries:
    """Rdatasets ``ts`` objects: decimal-year ``time`` column, monthly or quarterly."""
    path = fetch(_RDATASETS.format(name), f"{name}.csv", sha256=_RDATASETS_SHA256[name])
    frame = pd.read_csv(path)
    first = float(frame["time"].iloc[0])
    start = pd.Timestamp(year=int(first), month=1 + round((first % 1) * 12), day=1)
    freq = "MS" if season_length == 12 else "QS"
    index = pd.date_range(start, periods=len(frame), freq=freq)
    values = pd.Series(frame["value"].to_numpy(float), index, name=label)
    return TimeSeries(label, values, season_length, f"Rdatasets `{name}`")


def _co2() -> TimeSeries:
    from statsmodels.datasets import co2

    weekly = co2.load_pandas().data["co2"]
    monthly = weekly.resample("MS").mean().interpolate()
    monthly = monthly["1960":"2001-11"]
    return TimeSeries("co2_mauna_loa", monthly.rename("co2"), 12, "statsmodels (Mauna Loa)")


def _synthetic_seasonal(n: int = 120, seed: int = 0) -> TimeSeries:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    values = 50 + 0.3 * t + 8 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 1.5, n)
    index = pd.date_range("2010-01-01", periods=n, freq="MS")
    return TimeSeries("synthetic_seasonal", pd.Series(values, index, name="y"), 12, "synthetic")


TIME_SERIES: dict[str, Callable[[], TimeSeries]] = {
    "air_passengers": lambda: _rdataset("AirPassengers", "air_passengers"),
    "nottingham_temperature": lambda: _rdataset("nottem", "nottingham_temperature"),
    "uk_driver_deaths": lambda: _rdataset("UKDriverDeaths", "uk_driver_deaths"),
    "uk_gas": lambda: _rdataset("UKgas", "uk_gas", season_length=4),
    "johnson_johnson": lambda: _rdataset("JohnsonJohnson", "johnson_johnson", season_length=4),
    "co2_mauna_loa": _co2,
    "synthetic_seasonal": _synthetic_seasonal,
}


def load_series(name: str) -> TimeSeries:
    return TIME_SERIES[name]()


# ------------------------------------------------------------ clustering / anomalies


_WHOLESALE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00292/Wholesale%20customers%20data.csv"
_WHOLESALE_SHA256 = "c3d018c643565b85cee733c4a2ac76dd76e080e857cb23f0ccfcc2e15a6c17ef"
WHOLESALE_SPEND = ["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"]


def load_wholesale() -> pd.DataFrame:
    """UCI Wholesale customers: annual spend in 6 categories + Channel / Region."""
    return pd.read_csv(fetch(_WHOLESALE_URL, "wholesale_customers.csv", sha256=_WHOLESALE_SHA256))


@dataclass
class AnomalyDataset:
    name: str
    X: np.ndarray
    y: np.ndarray  # 1 = anomaly
    source: str

    @property
    def contamination(self) -> float:
        return float(self.y.mean())


def _kdd(subset: Literal["http", "SA"]) -> AnomalyDataset:
    bunch = skd.fetch_kddcup99(subset=subset, data_home=_sklearn_home(), percent10=True)
    frame = pd.DataFrame(bunch.data)
    y = (bunch.target != b"normal.").astype(int)
    if subset == "SA":
        # Columns 1-3 are categorical (protocol, service, flag): one-hot encode them.
        frame = pd.get_dummies(frame, columns=[1, 2, 3], dtype=float)
    # sklearn already log-transforms the heavy-tailed byte counts of "http".
    X = frame.to_numpy(dtype=float)
    return AnomalyDataset(f"kdd_{subset.lower()}", X, y, "KDD Cup 1999 (10%)")


def _mammography() -> AnomalyDataset:
    ds = openml(310, "mammography")
    return AnomalyDataset("mammography", ds.X.to_numpy(float), ds.y.to_numpy(), "OpenML #310")


def _synthetic_anomalies(
    n: int = 2000, contamination: float = 0.05, seed: int = 0
) -> AnomalyDataset:
    rng = np.random.default_rng(seed)
    n_out = int(n * contamination)
    inliers, _ = skd.make_blobs(
        n - n_out, n_features=4, centers=3, cluster_std=1.0, random_state=seed
    )
    outliers = rng.uniform(inliers.min(0) - 4, inliers.max(0) + 4, size=(n_out, 4))
    X = np.vstack([inliers, outliers])
    y = np.r_[np.zeros(n - n_out, int), np.ones(n_out, int)]
    return AnomalyDataset("synthetic_blobs", X, y, "synthetic")


ANOMALY: dict[str, Callable[[], AnomalyDataset]] = {
    "kdd_http": lambda: _kdd("http"),
    "kdd_sa": lambda: _kdd("SA"),
    "mammography": _mammography,
    "synthetic_blobs": _synthetic_anomalies,
}


def load_anomaly(name: str, *, max_rows: int | None = None, seed: int = 0) -> AnomalyDataset:
    """Load an anomaly dataset, optionally stratified-subsampled to ``max_rows``."""
    ds = ANOMALY[name]()
    if max_rows is None or len(ds.y) <= max_rows:
        return ds
    rng = np.random.default_rng(seed)
    keep = np.concatenate(
        [
            rng.choice(
                np.flatnonzero(ds.y == label),
                round(max_rows * np.mean(ds.y == label)),
                replace=False,
            )
            for label in (0, 1)
        ]
    )
    keep.sort()
    return AnomalyDataset(ds.name, ds.X[keep], ds.y[keep], ds.source + f" (subsample {len(keep)})")
