"""Config, seeding, data cache and tracking."""

import json

import numpy as np
import pytest
from pydantic import ValidationError

from airlab import data, tracking
from airlab.config import ExperimentConfig, flatten, load_config
from airlab.seed import set_seed


class DemoConfig(ExperimentConfig):
    alpha: float = 1.0


def test_load_config_with_override(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("experiment: demo\nalpha: 0.5\n")
    assert load_config(path, DemoConfig).alpha == 0.5
    assert load_config(path, DemoConfig, alpha=2.0).alpha == 2.0


def test_load_config_rejects_unknown_keys(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("experiment: demo\nalpah: 0.5\n")  # typo must fail loudly
    with pytest.raises(ValidationError):
        load_config(path, DemoConfig)


def test_flatten():
    assert flatten({"a": 1, "b": {"c": 2, "d": {"e": 3}}}) == {"a": 1, "b.c": 2, "b.d.e": 3}


def test_set_seed_makes_numpy_reproducible():
    set_seed(5)
    first = np.random.rand(3)
    set_seed(5)
    np.testing.assert_array_equal(first, np.random.rand(3))


def test_data_dir_respects_env(tmp_path):
    assert data.data_dir() == tmp_path / "data"


def test_fetch_uses_cache_and_verifies_checksum(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("x,y\n1,2\n")
    digest = data.sha256sum(source)

    path = data.fetch(source.as_uri(), "demo.csv", sha256=digest)
    assert path.read_text() == "x,y\n1,2\n"

    source.unlink()  # second call must hit the cache, not the network
    assert data.fetch(source.as_uri(), "demo.csv", sha256=digest) == path

    with pytest.raises(ValueError, match="Checksum mismatch"):
        data.fetch(source.as_uri(), "demo.csv", sha256="0" * 64)


def test_json_tracking_backend(monkeypatch, tmp_path):
    monkeypatch.setenv("AIRLAB_TRACKING", "json")
    monkeypatch.setattr(tracking, "REPO_ROOT", tmp_path)
    artifact = tmp_path / "report.md"
    artifact.write_text("hello")

    with tracking.track_run("exp", {"model": {"C": 1.0}}, {"phase": "0"}) as run:
        run.log_metrics({"acc": 0.9}, step=1)
        run.log_artifact(artifact)

    (run_dir,) = (tmp_path / "runs" / "exp").iterdir()
    record = json.loads((run_dir / "run.json").read_text())
    assert record["params"] == {"model.C": 1.0}
    assert record["metrics"] == [{"step": 1, "acc": 0.9}]
    assert record["tags"] == {"phase": "0"}
    assert (run_dir / "report.md").read_text() == "hello"


def test_unknown_tracking_backend(monkeypatch):
    monkeypatch.setenv("AIRLAB_TRACKING", "nope")
    with pytest.raises(ValueError), tracking.track_run("exp"):
        pass


def test_parse_overrides_reads_yaml_values():
    from airlab.experiment import parse_overrides

    assert parse_overrides(
        ["datasets=[sst2, emotion]", "max_test=50", "finetune=null", "name=x=y"]
    ) == {
        "datasets": ["sst2", "emotion"],
        "max_test": 50,
        "finetune": None,
        "name": "x=y",
    }
    with pytest.raises(ValueError):
        parse_overrides(["no-equals-sign"])


def test_host_description_names_cpu_count():
    import os

    from airlab.experiment import host_description

    assert f"{os.cpu_count()} logical CPUs" in host_description()
