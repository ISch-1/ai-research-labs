import pytest


@pytest.fixture(autouse=True)
def _isolate_side_effects(monkeypatch, tmp_path):
    """Keep tests from touching the real dataset cache or tracking store."""
    monkeypatch.setenv("AIRLAB_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("AIRLAB_TRACKING", "off")
