"""Typed experiment configuration loaded from YAML.

Each experiment defines a pydantic model that extends ``ExperimentConfig``;
typos and wrong types in a YAML file fail fast instead of silently using a
default.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, Field

C = TypeVar("C", bound="ExperimentConfig")


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment: str
    seed: int = 42
    tags: dict[str, str] = Field(default_factory=dict)


def load_config(path: str | Path, model: type[C], **overrides: Any) -> C:
    """Read a YAML file into ``model``; keyword overrides win over the file.

    Dotted keys reach into nested sections: ``{"pets.finetune": ["resnet18"]}``.
    """
    with Path(path).open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    for key, value in overrides.items():
        *parents, leaf = key.split(".")
        node = raw
        for part in parents:
            if not isinstance(node.get(part), dict):
                node[part] = {}
            node = node[part]
        node[leaf] = value
    return model.model_validate(raw)


def flatten(params: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested dicts to dotted keys (``model.C``) for experiment trackers."""
    flat: dict[str, Any] = {}
    for key, value in params.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten(value, f"{name}."))
        else:
            flat[name] = value
    return flat
