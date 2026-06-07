"""
Phase 3 — Sentiment scoring with VADER.

For each review:
  - Score review_text with VADER → neg, neu, pos, compound
  - Derive vader_label:  compound >= 0.05 → positive
                         compound <= -0.05 → negative
                         else             → neutral
  - Derive star_label:   rating 4-5 → positive
                         rating 3   → neutral
                         rating 1-2 → negative

Both label mappings are modelling choices stated explicitly here and in FINDINGS.md.
Using star_label (not vader_label) to define positive/negative reviews in RQ4 keeps
that analysis independent of the tool being audited in RQ3.

Outputs
-------
data/sample_scored.parquet
outputs/tables/sentiment_summary.{csv,tex}
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.io import load_config, save_parquet, load_parquet, parquet_exists, ensure_dir, setup_logging, project_root

setup_logging()
logger = logging.getLogger("amazon_sentiment.03_sentiment")


def score_vader(texts, batch_log_every: int = 50_000):
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()

    neg, neu, pos, compound = [], [], [], []
    for i, text in enumerate(texts):
        scores = sia.polarity_scores(str(text))
        neg.append(scores["neg"])
        neu.append(scores["neu"])
        pos.append(scores["pos"])
        compound.append(scores["compound"])
        if (i + 1) % batch_log_every == 0:
            logger.info(f"  VADER scored {i + 1:,} / {len(texts):,} reviews")

    return neg, neu, pos, compound


def apply_vader_label(compound, pos_thresh: float, neg_thresh: float):
    labels = []
    for c in compound:
        if c >= pos_thresh:
            labels.append("positive")
        elif c <= neg_thresh:
            labels.append("negative")
        else:
            labels.append("neutral")
    return labels


def apply_star_label(ratings, positive_stars, neutral_stars, negative_stars):
    mapping = {}
    for s in positive_stars:
        mapping[s] = "positive"
    for s in neutral_stars:
        mapping[s] = "neutral"
    for s in negative_stars:
        mapping[s] = "negative"
    return [mapping.get(int(r), "neutral") for r in ratings]


def _to_latex(df) -> str:
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Sentiment label distribution by category after VADER scoring.}",
        r"\label{tab:sentiment_summary}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Category & Label source & Positive & Neutral & Negative & Total \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"{row['category']} & {row['label_source']} & "
            f"{int(row['positive']):,} & {int(row['neutral']):,} & "
            f"{int(row['negative']):,} & {int(row['total']):,} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main():
    import pandas as pd

    cfg = load_config()

    scored_out = "data/sample_scored.parquet"
    if parquet_exists(scored_out):
        logger.info(f"Cache hit: {scored_out} exists. Delete it to re-run Phase 3.")
        return

    df = load_parquet("data/sample_clean.parquet")
    logger.info(f"Loaded sample_clean.parquet: {len(df):,} rows")

    # ── VADER scoring ─────────────────────────────────────────────────────────
    logger.info("Running VADER …")
    neg, neu, pos, compound = score_vader(df["review_text"].tolist())
    df["vader_neg"] = neg
    df["vader_neu"] = neu
    df["vader_pos"] = pos
    df["vader_compound"] = compound
    logger.info("VADER scoring complete.")

    # ── Derive labels ─────────────────────────────────────────────────────────
    v = cfg["vader"]
    sl = cfg["star_labels"]

    df["vader_label"] = apply_vader_label(
        df["vader_compound"], v["positive_threshold"], v["negative_threshold"]
    )
    df["star_label"] = apply_star_label(
        df["rating"],
        positive_stars=sl["positive"],
        neutral_stars=sl["neutral"],
        negative_stars=sl["negative"],
    )

    logger.info(
        f"VADER label distribution:\n{df['vader_label'].value_counts().to_string()}"
    )
    logger.info(
        f"Star label distribution:\n{df['star_label'].value_counts().to_string()}"
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    save_parquet(df, scored_out)
    logger.info(f"Saved → {scored_out}")

    # ── Summary table ─────────────────────────────────────────────────────────
    rows = []
    for cat in ["Beauty", "Sports"]:
        sub = df[df["category"] == cat]
        for source, col in [("VADER", "vader_label"), ("Star rating", "star_label")]:
            counts = sub[col].value_counts()
            rows.append({
                "category": cat,
                "label_source": source,
                "positive": counts.get("positive", 0),
                "neutral": counts.get("neutral", 0),
                "negative": counts.get("negative", 0),
                "total": len(sub),
            })

    summary = pd.DataFrame(rows)
    tables_dir = project_root() / cfg["paths"]["tables_dir"]
    ensure_dir(tables_dir)
    summary.to_csv(tables_dir / "sentiment_summary.csv", index=False)
    (tables_dir / "sentiment_summary.tex").write_text(_to_latex(summary), encoding="utf-8")
    logger.info(f"Sentiment summary → {tables_dir / 'sentiment_summary.csv'}")

    print("\n=== Sentiment Label Distribution ===")
    print(summary.to_string(index=False))

    print("\n=== VADER Compound Stats by Category ===")
    print(df.groupby("category")["vader_compound"].describe().round(3).to_string())


if __name__ == "__main__":
    main()
