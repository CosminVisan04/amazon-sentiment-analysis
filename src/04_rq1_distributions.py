"""
Phase 4 — RQ1: How does the distribution of sentiment scores differ between
Beauty and Sports & Outdoors reviews?

Tests
-----
- Mann-Whitney U (two-sided) for location difference
- Kolmogorov-Smirnov for distribution shape difference
Effect sizes: Cliff's delta, rank-biserial r
Bootstrap 95% CI for difference in means and medians (fixed seed)

Figures
-------
outputs/figures/rq1_kde.{pdf,png}
outputs/figures/rq1_violin.{pdf,png}
outputs/figures/rq1_ecdf.{pdf,png}

Outputs
-------
outputs/stats/rq1.json
outputs/stats/rq1.csv
outputs/tables/rq1_descriptives.{csv,tex}
"""
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.io import load_config, load_parquet, parquet_exists, ensure_dir, setup_logging, project_root
from utils.stats import (
    bootstrap_ci, bootstrap_diff_ci, cliffs_delta_from_u, rank_biserial_r
)
from utils.plotting import set_style, save_figure, PALETTE, category_palette

setup_logging()
logger = logging.getLogger("amazon_sentiment.04_rq1")


# ── Descriptive statistics ────────────────────────────────────────────────────

def descriptives(arr: np.ndarray, label_counts: dict) -> dict:
    total = len(arr)
    return {
        "n": total,
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "iqr": float(np.percentile(arr, 75) - np.percentile(arr, 25)),
        "pct_positive": label_counts.get("positive", 0) / total * 100,
        "pct_neutral":  label_counts.get("neutral",  0) / total * 100,
        "pct_negative": label_counts.get("negative", 0) / total * 100,
    }


# ── Figures ───────────────────────────────────────────────────────────────────

def plot_kde(beauty: np.ndarray, sports: np.ndarray, figures_dir: Path):
    import matplotlib.pyplot as plt
    import seaborn as sns

    set_style()
    fig, ax = plt.subplots(figsize=(7, 4))

    palette = category_palette()
    sns.kdeplot(beauty, ax=ax, color=palette["Beauty"], label="Beauty", linewidth=2, fill=True, alpha=0.25)
    sns.kdeplot(sports, ax=ax, color=palette["Sports"], label="Sports & Outdoors", linewidth=2, fill=True, alpha=0.25)

    ax.axvline(0.05,  color="gray", linestyle="--", linewidth=0.8, alpha=0.7, label="VADER ±0.05 thresholds")
    ax.axvline(-0.05, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)

    ax.set_xlabel("VADER Compound Score")
    ax.set_ylabel("Density")
    ax.set_title("Sentiment Score Distribution by Category")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, figures_dir, "rq1_kde")
    logger.info("Saved rq1_kde")


def plot_violin(df: pd.DataFrame, figures_dir: Path):
    import matplotlib.pyplot as plt
    import seaborn as sns

    set_style()
    fig, ax = plt.subplots(figsize=(6, 5))

    palette = category_palette()
    sns.violinplot(
        data=df, x="category", y="vader_compound",
        hue="category", palette=palette, legend=False,
        inner="box", ax=ax, cut=0,
        order=["Beauty", "Sports"],
    )
    ax.axhline(0.05,  color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.axhline(-0.05, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_xlabel("Category")
    ax.set_ylabel("VADER Compound Score")
    ax.set_title("Sentiment Distribution (Violin + Box)")
    fig.tight_layout()
    save_figure(fig, figures_dir, "rq1_violin")
    logger.info("Saved rq1_violin")


def plot_ecdf(beauty: np.ndarray, sports: np.ndarray, figures_dir: Path):
    import matplotlib.pyplot as plt

    set_style()
    fig, ax = plt.subplots(figsize=(7, 4))

    palette = category_palette()
    for arr, label, color in [
        (beauty, "Beauty", palette["Beauty"]),
        (sports, "Sports & Outdoors", palette["Sports"]),
    ]:
        sorted_arr = np.sort(arr)
        ecdf = np.arange(1, len(sorted_arr) + 1) / len(sorted_arr)
        ax.plot(sorted_arr, ecdf, color=color, label=label, linewidth=1.8)

    ax.axvline(0.05,  color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.axvline(-0.05, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_xlabel("VADER Compound Score")
    ax.set_ylabel("Cumulative Proportion")
    ax.set_title("Empirical CDF of Sentiment Scores by Category")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, figures_dir, "rq1_ecdf")
    logger.info("Saved rq1_ecdf")


# ── LaTeX helpers ─────────────────────────────────────────────────────────────

def descriptives_to_latex(desc_df: pd.DataFrame) -> str:
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{RQ1 descriptive statistics for VADER compound scores by category.}",
        r"\label{tab:rq1_descriptives}",
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"Category & $n$ & Mean & Median & SD & IQR & \%Pos & \%Neu & \%Neg \\",
        r"\midrule",
    ]
    for _, row in desc_df.iterrows():
        lines.append(
            f"{row['category']} & {int(row['n']):,} & {row['mean']:.3f} & "
            f"{row['median']:.3f} & {row['std']:.3f} & {row['iqr']:.3f} & "
            f"{row['pct_positive']:.1f} & {row['pct_neutral']:.1f} & {row['pct_negative']:.1f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    cfg = load_config()

    stats_dir  = project_root() / cfg["paths"]["stats_dir"]
    tables_dir = project_root() / cfg["paths"]["tables_dir"]
    figures_dir = project_root() / cfg["paths"]["figures_dir"]
    for d in [stats_dir, tables_dir, figures_dir]:
        ensure_dir(d)

    df = load_parquet("data/sample_scored.parquet")
    logger.info(f"Loaded {len(df):,} rows")

    beauty = df[df["category"] == "Beauty"]["vader_compound"].values
    sports = df[df["category"] == "Sports"]["vader_compound"].values

    b_labels = df[df["category"] == "Beauty"]["vader_label"].value_counts().to_dict()
    s_labels = df[df["category"] == "Sports"]["vader_label"].value_counts().to_dict()

    boot_cfg = cfg["bootstrap"]

    # ── Descriptives ──────────────────────────────────────────────────────────
    desc = {
        "Beauty": descriptives(beauty, b_labels),
        "Sports": descriptives(sports, s_labels),
    }

    # ── Mann-Whitney U ────────────────────────────────────────────────────────
    u_stat, p_mw = sp_stats.mannwhitneyu(beauty, sports, alternative="two-sided")
    rbr = rank_biserial_r(u_stat, len(beauty), len(sports))
    logger.info(f"Mann-Whitney U={u_stat:.1f}, p={p_mw:.4e}, rank-biserial r={rbr:.4f}")

    # ── Kolmogorov-Smirnov ────────────────────────────────────────────────────
    ks_stat, p_ks = sp_stats.ks_2samp(beauty, sports)
    logger.info(f"KS stat={ks_stat:.4f}, p={p_ks:.4e}")

    # ── Cliff's delta (derived from U — mathematically identical to pairwise) ─
    cd = cliffs_delta_from_u(u_stat, len(beauty), len(sports))
    logger.info(f"Cliff's delta={cd:.4f}")

    # ── Bootstrap CIs ─────────────────────────────────────────────────────────
    mean_ci_b = bootstrap_ci(beauty, np.mean, boot_cfg["n_iterations"], boot_cfg["ci_level"], boot_cfg["seed"])
    mean_ci_s = bootstrap_ci(sports, np.mean, boot_cfg["n_iterations"], boot_cfg["ci_level"], boot_cfg["seed"])
    med_ci_b  = bootstrap_ci(beauty, np.median, boot_cfg["n_iterations"], boot_cfg["ci_level"], boot_cfg["seed"])
    med_ci_s  = bootstrap_ci(sports, np.median, boot_cfg["n_iterations"], boot_cfg["ci_level"], boot_cfg["seed"])
    diff_mean_ci = bootstrap_diff_ci(beauty, sports, np.mean, boot_cfg["n_iterations"], boot_cfg["ci_level"], boot_cfg["seed"])
    diff_med_ci  = bootstrap_diff_ci(beauty, sports, np.median, boot_cfg["n_iterations"], boot_cfg["ci_level"], boot_cfg["seed"])

    # ── Compile results ───────────────────────────────────────────────────────
    results = {
        "rq": "RQ1",
        "question": "How does the distribution of sentiment scores differ between Beauty and Sports & Outdoors reviews?",
        "n_beauty": int(len(beauty)),
        "n_sports": int(len(sports)),
        "descriptives": desc,
        "mean_ci_beauty_95": mean_ci_b,
        "mean_ci_sports_95": mean_ci_s,
        "median_ci_beauty_95": med_ci_b,
        "median_ci_sports_95": med_ci_s,
        "diff_mean_beauty_minus_sports_95ci": diff_mean_ci,
        "diff_median_beauty_minus_sports_95ci": diff_med_ci,
        "mann_whitney": {
            "U": float(u_stat),
            "p": float(p_mw),
            "rank_biserial_r": float(rbr),
        },
        "kolmogorov_smirnov": {
            "D": float(ks_stat),
            "p": float(p_ks),
        },
        "cliffs_delta": float(cd),
    }

    with open(stats_dir / "rq1.json", "w") as f:
        json.dump(results, f, indent=2)

    stats_rows = [
        {"test": "Mann-Whitney U", "stat": u_stat, "p": p_mw, "effect_size": rbr, "effect_name": "rank-biserial r"},
        {"test": "Kolmogorov-Smirnov", "stat": ks_stat, "p": p_ks, "effect_size": cd, "effect_name": "Cliff's delta"},
    ]
    pd.DataFrame(stats_rows).to_csv(stats_dir / "rq1.csv", index=False)
    logger.info(f"Stats saved → {stats_dir / 'rq1.json'}")

    # ── Descriptives table ────────────────────────────────────────────────────
    desc_rows = [{"category": cat, **vals} for cat, vals in desc.items()]
    desc_df = pd.DataFrame(desc_rows)
    desc_df.to_csv(tables_dir / "rq1_descriptives.csv", index=False)
    (tables_dir / "rq1_descriptives.tex").write_text(descriptives_to_latex(desc_df), encoding="utf-8")

    # ── Figures ───────────────────────────────────────────────────────────────
    plot_kde(beauty, sports, figures_dir)
    plot_violin(df, figures_dir)
    plot_ecdf(beauty, sports, figures_dir)

    # ── Console summary ───────────────────────────────────────────────────────
    print("\n=== RQ1 Results ===")
    print(f"\nDescriptives:")
    print(desc_df.round(4).to_string(index=False))

    print(f"\nMann-Whitney U={u_stat:.1f}, p={p_mw:.4e}, rank-biserial r={rbr:.4f}")
    print(f"Kolmogorov-Smirnov D={ks_stat:.4f}, p={p_ks:.4e}")
    print(f"Cliff's delta={cd:.4f}")

    print(f"\nBootstrap 95% CI — Beauty mean: [{mean_ci_b[0]:.4f}, {mean_ci_b[1]:.4f}]")
    print(f"Bootstrap 95% CI — Sports mean: [{mean_ci_s[0]:.4f}, {mean_ci_s[1]:.4f}]")
    print(f"Bootstrap 95% CI — diff (Beauty−Sports) mean: [{diff_mean_ci[0]:.4f}, {diff_mean_ci[1]:.4f}]")
    print(f"Bootstrap 95% CI — diff (Beauty−Sports) median: [{diff_med_ci[0]:.4f}, {diff_med_ci[1]:.4f}]")

    # Plain-language answer for FINDINGS.md
    sig = "statistically significant" if p_mw < 0.05 else "not statistically significant"
    print(f"\nAnswer: The difference in compound scores between Beauty and Sports is {sig} "
          f"(Mann-Whitney p={p_mw:.2e}, Cliff's delta={cd:.3f}).")


if __name__ == "__main__":
    main()
