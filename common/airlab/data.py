"""Dataset download and caching.

Raw data never goes into git. Files are cached under ``$AIRLAB_DATA_DIR``
(default: ``<repo>/data``) and verified by SHA-256 when a checksum is given.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    """Root directory of the local dataset cache (created if missing)."""
    root = Path(os.environ.get("AIRLAB_DATA_DIR", REPO_ROOT / "data"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def sha256sum(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(url: str, filename: str, *, sha256: str | None = None, subdir: str = "raw") -> Path:
    """Download ``url`` into the cache once and return the local path.

    The download goes to a temp file first so an interrupted transfer never
    leaves a half-written file that looks cached.
    """
    target = data_dir() / subdir / filename
    if target.exists():
        if sha256 and sha256sum(target) != sha256:
            raise ValueError(f"Checksum mismatch for cached file {target}; delete it and retry.")
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        with urllib.request.urlopen(url, timeout=60) as response:
            shutil.copyfileobj(response, tmp)

    if sha256 and (actual := sha256sum(tmp_path)) != sha256:
        tmp_path.unlink(missing_ok=True)
        raise ValueError(f"Checksum mismatch for {url}: expected {sha256}, got {actual}")
    tmp_path.replace(target)
    return target
