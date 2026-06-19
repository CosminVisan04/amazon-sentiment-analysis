import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.metrics import (
    confusion_matrix, accuracy_score, f1_score,
    precision_recall_fscore_support, cohen_kappa_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.io import load_config, load_parquet, ensure_dir, setup_logging, project_root
from utils.stats import spearman_ci, bootstrap_ci, cohens_kappa_weighted
from utils.plotting import set_style, save_figure, PALETTE

setup_logging()
logger = logging.getLogger("amazon_sentiment.06_rq3")

LABEL_ORDER = ["negative", "neutral", "positive"]


# spearman correlation with a confidence interval
def spearman_with_ci(x, y, ci: float = 0.95) -> dict:
    rho, p = sp_stats.spearmanr(x, y)
    lo, hi = spearman_ci(rho, len(x), ci)
    return {"rho": float(rho), "p": float(p), "ci_lo": lo, "ci_hi": hi, "n": len(x)}


# accuracy of always predicting the majority class
def majority_class_baseline(y_true) -> dict:
    counts = pd.Series(y_true).value_counts(normalize=True)
    return {"majority_class": str(counts.index[0]), "baseline_accuracy": float(counts.iloc[0])}


# accuracy, macro-f1, kappa, and per-class metrics
def classification_metrics(y_true, y_pred) -> dict:
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", labels=LABEL_ORDER, zero_division=0)
    kappa = cohens_kappa_weighted(y_true, y_pred, weights="quadratic")
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=LABEL_ORDER, zero_division=0
    )
    per_class = {
        label: {
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, label in enumerate(LABEL_ORDER)
    }
    return {
        "accuracy": float(acc),
        "macro_f1": float(f1_macro),
        "kappa_quadratic": float(kappa),
        "per_class": per_class,
        "n": len(y_true),
    }


# plots a confusion matrix heatmap for one category
def plot_confusion(y_true, y_pred, category: str, figures_dir: Path):
    import matplotlib.pyplot as plt
    import seaborn as sns

    set_style()
    cm = confusion_matrix(y_true, y_pred, labels=LABEL_ORDER)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(
        cm_norm, annot=cm, fmt="d", cmap="Blues",
        xticklabels=LABEL_ORDER, yticklabels=LABEL_ORDER,
        ax=ax, cbar_kws={"label": "Row proportion"},
        linewidths=0.5,
    )
    ax.set_xlabel("VADER Label (predicted)")
    ax.set_ylabel("Star Label (true)")
    ax.set_title(f"Confusion Matrix - {category}")
    fig.tight_layout()
    save_figure(fig, figures_dir, f"rq3_confusion_{category.lower()}")
    logger.info(f"Saved rq3_confusion_{category.lower()}")


# accuracy by review length bucket with bootstrap ci
def agreement_by_length(df: pd.DataFrame, boot_cfg: dict) -> pd.DataFrame:
    bins   = [0, 5, 20, 50, np.inf]
    labels = ["≤5", "6–20", "21–50", ">50"]
    df = df.copy()
    df["len_bucket"] = pd.cut(df["token_len"], bins=bins, labels=labels, right=True)

    rows = []
    for bucket in labels:
        sub = df[df["len_bucket"] == bucket]
        if len(sub) == 0:
            continue
        correct = (sub["vader_label"] == sub["star_label"]).astype(int).values
        acc = float(correct.mean())
        lo, hi = bootstrap_ci(correct, np.mean, boot_cfg["n_iterations"], boot_cfg["ci_level"], boot_cfg["seed"])
        rows.append({"bucket": bucket, "n": len(sub), "accuracy": acc, "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)


# plots agreement accuracy across length buckets
def plot_agreement_by_length(length_df: pd.DataFrame, figures_dir: Path):
    import matplotlib.pyplot as plt

    set_style()
    fig, ax = plt.subplots(figsize=(6.5, 4))

    x = range(len(length_df))
    bars = ax.bar(x, length_df["accuracy"], color=PALETTE["accent"],
                  yerr=[length_df["accuracy"] - length_df["ci_lo"],
                        length_df["ci_hi"] - length_df["accuracy"]],
                  capsize=4, error_kw={"linewidth": 1.2})

    for bar, row in zip(bars, length_df.itertuples()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"n={row.n:,}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(list(x))
    ax.set_xticklabels(length_df["bucket"])
    ax.set_xlabel("Review Length (tokens)")
    ax.set_ylabel("VADER–Star Agreement (accuracy)")
    ax.set_title("VADER Agreement with Star Ratings by Review Length")
    ax.set_ylim(0, 1.0)
    fig.tight_layout()
    save_figure(fig, figures_dir, "rq3_agreement_by_length")
    logger.info("Saved rq3_agreement_by_length")


# samples and saves rating-vader disagreement cases
def mine_mismatches(df: pd.DataFrame, cfg: dict, category: str, qual_dir: Path):
    mm_cfg = cfg["mismatch"]
    seed = cfg["sampling"]["seed"]
    n = mm_cfg["sample_per_direction"]

    cols = ["category", "year", "rating", "vader_compound", "token_len", "review_text"]

    high_star_low_vader = df[
        (df["rating"] >= mm_cfg["high_star_low_vader"]["min_rating"]) &
        (df["vader_compound"] <= mm_cfg["high_star_low_vader"]["max_compound"])
    ][cols]

    low_star_high_vader = df[
        (df["rating"] <= mm_cfg["low_star_high_vader"]["max_rating"]) &
        (df["vader_compound"] >= mm_cfg["low_star_high_vader"]["min_compound"])
    ][cols]

    def _sample(sub, label):
        if len(sub) <= n:
            return sub.assign(mismatch_type=label)
        return sub.sample(n, random_state=seed).assign(mismatch_type=label)

    combined = pd.concat([
        _sample(high_star_low_vader, "high_star_low_vader"),
        _sample(low_star_high_vader, "low_star_high_vader"),
    ]).sort_values("mismatch_type")

    out = qual_dir / f"mismatches_{category}.csv"
    combined.to_csv(out, index=False)
    logger.info(
        f"Mismatches {category}: {len(high_star_low_vader)} high-star/low-VADER, "
        f"{len(low_star_high_vader)} low-star/high-VADER → saved {out}"
    )
    return len(high_star_low_vader), len(low_star_high_vader)


# runs phase 6 end to end
def main():
    cfg = load_config()

    stats_dir   = project_root() / cfg["paths"]["stats_dir"]
    tables_dir  = project_root() / cfg["paths"]["tables_dir"]
    figures_dir = project_root() / cfg["paths"]["figures_dir"]
    qual_dir    = project_root() / cfg["paths"]["qualitative_dir"]
    for d in [stats_dir, tables_dir, figures_dir, qual_dir]:
        ensure_dir(d)

    df = load_parquet("data/sample_scored.parquet")
    logger.info(f"Loaded {len(df):,} rows")

    boot_cfg = cfg["bootstrap"]
    results  = {"rq": "RQ3", "question": "How well do star ratings and VADER sentiment scores agree?"}

    spearman_overall = spearman_with_ci(df["rating"], df["vader_compound"])
    results["spearman_overall"] = spearman_overall
    logger.info(
        f"Spearman overall: rho={spearman_overall['rho']:.4f}, "
        f"p={spearman_overall['p']:.4e}, 95% CI=[{spearman_overall['ci_lo']:.4f}, {spearman_overall['ci_hi']:.4f}]"
    )

    spearman_per_cat = {}
    for cat in ["Beauty", "Sports"]:
        sub = df[df["category"] == cat]
        sp = spearman_with_ci(sub["rating"], sub["vader_compound"])
        spearman_per_cat[cat] = sp
        logger.info(f"Spearman {cat}: rho={sp['rho']:.4f}, p={sp['p']:.4e}, CI=[{sp['ci_lo']:.4f}, {sp['ci_hi']:.4f}]")
    results["spearman_per_category"] = spearman_per_cat

    metrics_overall = classification_metrics(df["star_label"], df["vader_label"])
    results["classification_overall"] = metrics_overall
    logger.info(
        f"Overall: accuracy={metrics_overall['accuracy']:.4f}, "
        f"macro-F1={metrics_overall['macro_f1']:.4f}, "
        f"kappa={metrics_overall['kappa_quadratic']:.4f}"
    )

    metrics_per_cat = {}
    for cat in ["Beauty", "Sports"]:
        sub = df[df["category"] == cat]
        m = classification_metrics(sub["star_label"], sub["vader_label"])
        metrics_per_cat[cat] = m
        logger.info(
            f"{cat}: accuracy={m['accuracy']:.4f}, macro-F1={m['macro_f1']:.4f}, kappa={m['kappa_quadratic']:.4f}"
        )
    results["classification_per_category"] = metrics_per_cat

    baseline_overall = majority_class_baseline(df["star_label"])
    baseline_per_cat = {
        cat: majority_class_baseline(df[df["category"] == cat]["star_label"])
        for cat in ["Beauty", "Sports"]
    }
    results["majority_class_baseline_overall"] = baseline_overall
    results["majority_class_baseline_per_category"] = baseline_per_cat
    logger.info(
        f"Majority-class baseline (overall): {baseline_overall['majority_class']} "
        f"= {baseline_overall['baseline_accuracy']:.4f} "
        f"(VADER accuracy {metrics_overall['accuracy']:.4f})"
    )
    for cat in ["Beauty", "Sports"]:
        b = baseline_per_cat[cat]
        logger.info(
            f"Majority-class baseline ({cat}): {b['majority_class']} = {b['baseline_accuracy']:.4f}"
        )

    agree_beauty = (df[df["category"] == "Beauty"]["vader_label"] ==
                    df[df["category"] == "Beauty"]["star_label"]).sum()
    disagree_beauty = len(df[df["category"] == "Beauty"]) - agree_beauty
    agree_sports = (df[df["category"] == "Sports"]["vader_label"] ==
                    df[df["category"] == "Sports"]["star_label"]).sum()
    disagree_sports = len(df[df["category"] == "Sports"]) - agree_sports

    chi2, p_chi2 = sp_stats.chi2_contingency(
        [[agree_beauty, disagree_beauty], [agree_sports, disagree_sports]]
    )[:2]
    results["chi2_agreement_by_category"] = {
        "chi2": float(chi2), "p": float(p_chi2),
        "beauty_accuracy": float(agree_beauty / len(df[df["category"] == "Beauty"])),
        "sports_accuracy": float(agree_sports / len(df[df["category"] == "Sports"])),
    }
    logger.info(
        f"Chi-square agreement Beauty vs Sports: chi2={chi2:.4f}, p={p_chi2:.4e}"
    )

    length_df = agreement_by_length(df, boot_cfg)
    results["agreement_by_length"] = length_df.to_dict(orient="records")
    logger.info(f"Agreement by length:\n{length_df.to_string(index=False)}")

    for cat in ["Beauty", "Sports"]:
        sub = df[df["category"] == cat]
        plot_confusion(sub["star_label"], sub["vader_label"], cat, figures_dir)

    plot_agreement_by_length(length_df, figures_dir)

    mismatch_counts = {}
    for cat in ["Beauty", "Sports"]:
        sub = df[df["category"] == cat]
        hi_lo, lo_hi = mine_mismatches(sub, cfg, cat, qual_dir)
        mismatch_counts[cat] = {"high_star_low_vader": hi_lo, "low_star_high_vader": lo_hi}
    results["mismatch_counts"] = mismatch_counts

    with open(stats_dir / "rq3.json", "w") as f:
        json.dump(results, f, indent=2)

    stat_rows = [
        {"metric": "Spearman rho (overall)", "value": spearman_overall["rho"],
         "p": spearman_overall["p"], "ci_lo": spearman_overall["ci_lo"], "ci_hi": spearman_overall["ci_hi"]},
        {"metric": "Accuracy (overall)", "value": metrics_overall["accuracy"], "p": None, "ci_lo": None, "ci_hi": None},
        {"metric": "Macro-F1 (overall)", "value": metrics_overall["macro_f1"], "p": None, "ci_lo": None, "ci_hi": None},
        {"metric": "Kappa quadratic (overall)", "value": metrics_overall["kappa_quadratic"], "p": None, "ci_lo": None, "ci_hi": None},
        {"metric": "Chi2 Beauty vs Sports agreement", "value": chi2, "p": p_chi2, "ci_lo": None, "ci_hi": None},
        {"metric": "Majority-class baseline accuracy (overall)", "value": baseline_overall["baseline_accuracy"], "p": None, "ci_lo": None, "ci_hi": None},
    ]
    pd.DataFrame(stat_rows).to_csv(stats_dir / "rq3.csv", index=False)
    length_df.to_csv(tables_dir / "rq3_agreement_by_length.csv", index=False)
    logger.info(f"Stats saved → {stats_dir / 'rq3.json'}")

    print("\n=== RQ3 Results ===")
    print(f"\nSpearman rho (overall): {spearman_overall['rho']:.4f}  "
          f"p={spearman_overall['p']:.4e}  "
          f"95% CI=[{spearman_overall['ci_lo']:.4f}, {spearman_overall['ci_hi']:.4f}]")
    for cat in ["Beauty", "Sports"]:
        sp = spearman_per_cat[cat]
        print(f"Spearman rho ({cat}):   {sp['rho']:.4f}  p={sp['p']:.4e}  CI=[{sp['ci_lo']:.4f}, {sp['ci_hi']:.4f}]")

    print(f"\nMajority-class baseline (overall): {baseline_overall['majority_class']} "
          f"= {baseline_overall['baseline_accuracy']:.4f}")
    print(f"Overall: accuracy={metrics_overall['accuracy']:.4f}  "
          f"macro-F1={metrics_overall['macro_f1']:.4f}  "
          f"kappa={metrics_overall['kappa_quadratic']:.4f}")
    for cat in ["Beauty", "Sports"]:
        m = metrics_per_cat[cat]
        print(f"{cat}:  accuracy={m['accuracy']:.4f}  macro-F1={m['macro_f1']:.4f}  kappa={m['kappa_quadratic']:.4f}")

    print(f"\nChi-square (category agreement difference): chi2={chi2:.4f}, p={p_chi2:.4e}")

    print("\nAgreement by length bucket:")
    print(length_df.round(4).to_string(index=False))

    print("\nMismatch counts:")
    for cat, counts in mismatch_counts.items():
        print(f"  {cat}: {counts['high_star_low_vader']} high-star/low-VADER, "
              f"{counts['low_star_high_vader']} low-star/high-VADER")


if __name__ == "__main__":
    main()
