"""
Consistent plot style for all figures.
Saves every figure as both PDF (vector, for LaTeX) and PNG.
Colour-blind-safe palette (Wong 2011).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,   # embed fonts as Type 42 (TrueType) — required by ACM
    "ps.fonttype": 42,
})

# Wong (2011) colour-blind-safe palette
PALETTE = {
    "Beauty": "#0072B2",   # blue
    "Sports": "#E69F00",   # orange/amber
    "positive": "#009E73", # green
    "negative": "#D55E00", # vermillion
    "neutral":  "#CC79A7", # pink
    "accent":   "#56B4E9", # sky blue
}

CATEGORY_COLORS = [PALETTE["Beauty"], PALETTE["Sports"]]


def set_style():
    sns.set_theme(style="whitegrid", palette=list(PALETTE.values()))


def save_figure(fig: plt.Figure, figures_dir: str | Path, stem: str) -> None:
    """Save fig as <stem>.pdf and <stem>.png inside figures_dir."""
    out = Path(figures_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{stem}.pdf")
    fig.savefig(out / f"{stem}.png")
    plt.close(fig)


def category_palette() -> dict[str, str]:
    return {"Beauty": PALETTE["Beauty"], "Sports": PALETTE["Sports"]}
