"""Reproducibility helpers: one call seeds every RNG the labs use."""

from __future__ import annotations

import os
import random

import numpy as np

DEFAULT_SEED = 42


def set_seed(seed: int = DEFAULT_SEED, *, deterministic_torch: bool = True) -> int:
    """Seed Python, NumPy and (if installed) PyTorch. Returns the seed for logging."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - third-party code still uses the global RNG

    try:
        import torch
    except ImportError:
        return seed

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed


def rng(seed: int = DEFAULT_SEED) -> np.random.Generator:
    """A fresh NumPy Generator; prefer this over the global RNG in new code."""
    return np.random.default_rng(seed)
