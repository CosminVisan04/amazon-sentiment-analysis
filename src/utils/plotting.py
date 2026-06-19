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
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

PALETTE = {
    "Beauty": "#0072B2",
    "Sports": "#E69F00",
    "positive": "#009E73",
    "negative": "#D55E00",
    "neutral":  "#CC79A7",
    "accent":   "#56B4E9",
}

CATEGORY_COLORS = [PALETTE["Beauty"], PALETTE["Sports"]]


# applies the shared seaborn theme
def set_style():
    sns.set_theme(style="whitegrid", palette=list(PALETTE.values()))


# saves a figure as pdf and png
def save_figure(fig: plt.Figure, figures_dir: str | Path, stem: str) -> None:
    out = Path(figures_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{stem}.pdf")
    fig.savefig(out / f"{stem}.png")
    plt.close(fig)


# returns the category-to-colour mapping
def category_palette() -> dict[str, str]:
    return {"Beauty": PALETTE["Beauty"], "Sports": PALETTE["Sports"]}
