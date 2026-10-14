"""Plots shared by the labs. All functions save to disk and never open windows."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def save(fig: plt.Figure, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def cd_diagram(
    average_ranks: dict[str, float], cd: float, path: str | Path, *, title: str = ""
) -> Path:
    """Critical-difference diagram (Demšar 2006).

    Models are placed on a rank axis (1 = best, on the left). Thick bars join
    groups of models whose average ranks differ by less than ``cd``: within a
    bar the data cannot tell the models apart.
    """
    names = sorted(average_ranks, key=average_ranks.get)  # type: ignore[arg-type]
    ranks = np.array([average_ranks[n] for n in names])
    k = len(names)
    low, high = 1, max(k, int(np.ceil(ranks.max())))

    # Cliques: maximal runs of models whose rank spread is below the CD.
    cliques: list[tuple[int, int]] = []
    for i in range(k):
        j = i
        while j + 1 < k and ranks[j + 1] - ranks[i] < cd:
            j += 1
        if j > i and not any(a <= i and j <= b for a, b in cliques):
            cliques.append((i, j))

    # Layout in data units: CD scale on top, rank axis, clique bars, then label rows.
    axis_y = 0.0
    row_gap = 0.35
    first_row = -0.35 - 0.18 * len(cliques)
    n_rows = (k + 1) // 2
    bottom = first_row - row_gap * (n_rows - 1) - 0.3

    fig, ax = plt.subplots(figsize=(8, 1.6 + 0.3 * n_rows + 0.15 * len(cliques)))
    ax.set_xlim(low - 0.3, high + 0.3)
    ax.set_ylim(bottom, 0.95)
    ax.axis("off")

    ax.hlines(axis_y, low, high, color="black", lw=1)
    for r in range(low, high + 1):
        ax.vlines(r, axis_y, axis_y + 0.08, color="black", lw=1)
        ax.text(r, axis_y + 0.12, str(r), ha="center", va="bottom", fontsize=9)

    ax.hlines(0.6, low, low + cd, color="black", lw=1.5)
    ax.vlines([low, low + cd], 0.55, 0.65, color="black", lw=1.5)
    ax.text(low + cd / 2, 0.7, f"CD = {cd:.2f}", ha="center", va="bottom", fontsize=9)

    for n, (i, j) in enumerate(cliques):
        y = axis_y - 0.15 - 0.18 * n
        ax.hlines(y, ranks[i] - 0.03, ranks[j] + 0.03, color="tab:red", lw=3)

    # Labels: best half on the left, worst half on the right.
    for i, (name, rank) in enumerate(zip(names, ranks, strict=True)):
        left = i < n_rows
        row = i if left else k - 1 - i
        y = first_row - row * row_gap
        x_text = low - 0.25 if left else high + 0.25
        ax.plot([rank, rank, x_text], [axis_y, y, y], color="black", lw=0.8)
        ax.text(
            x_text,
            y,
            f"{name} ({rank:.2f})",
            ha="right" if left else "left",
            va="center",
            fontsize=9,
        )

    if title:
        ax.set_title(title, fontsize=10)
    return save(fig, path)
