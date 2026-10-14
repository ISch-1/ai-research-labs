"""Small helpers to turn results into Markdown reports and tidy CSV files."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import pandas as pd


def md_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    """Render a GitHub-flavoured Markdown table."""
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "---|" * len(headers),
    ]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def fmt_p(pvalue: float) -> str:
    """p-values the way papers print them."""
    return "< 0.0001" if pvalue < 1e-4 else f"{pvalue:.4f}"


def write_results(rows: list[dict[str, Any]], path: str | Path) -> pd.DataFrame:
    """Write long-format results (one row per dataset × model × fold × metric).

    These CSVs are the hand-off between experiments and the cross-dataset
    statistical comparison, so every experiment uses the same columns:
    ``experiment, dataset, model, fold, metric, value``.
    """
    frame = pd.DataFrame(
        rows, columns=["experiment", "dataset", "model", "fold", "metric", "value"]
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame
