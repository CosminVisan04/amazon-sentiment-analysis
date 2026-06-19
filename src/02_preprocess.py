import html
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.io import load_config, save_parquet, load_parquet, parquet_exists, ensure_dir, setup_logging, project_root

setup_logging()
logger = logging.getLogger("amazon_sentiment.02_preprocess")


# strips html, markers, and normalises apostrophes
def clean_text(text: str) -> str:
    text = html.unescape(str(text))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\[\[ASIN:[^\]]+\]\]", " ", text)
    text = text.replace("â", "'").replace("’", "'").replace("‘", "'")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# seeds langdetect for deterministic results
def _setup_langdetect(seed: int = 42):
    from langdetect import DetectorFactory
    DetectorFactory.seed = seed


# fraction of ascii characters in a string
def _ascii_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for c in text if ord(c) < 128) / len(text)


# checks whether text is english
def _is_english(text: str) -> bool:
    if _ascii_ratio(text) < 0.85:
        return False
    try:
        from langdetect import detect, LangDetectException
        return detect(text) == "en"
    except Exception:
        return False


# runs language detection with progress reporting
def _is_english_with_progress(texts, log_every: int = 20_000):
    n = len(texts)
    results = [False] * n
    try:
        from tqdm import tqdm
        iterator = tqdm(texts, total=n, desc="  langdetect", unit="review")
        for i, text in enumerate(iterator):
            results[i] = _is_english(text)
        return results
    except ImportError:
        for i, text in enumerate(texts):
            results[i] = _is_english(text)
            if (i + 1) % log_every == 0 or (i + 1) == n:
                logger.info(f"    langdetect progress: {i + 1:,}/{n:,}")
        return results


# filters a dataframe down to english reviews
def language_filter(df, seed: int = 42):
    import pandas as pd

    _setup_langdetect(seed)

    ascii_ratios = df["review_text"].apply(_ascii_ratio)
    fast_reject = ascii_ratios < 0.85
    fast_reject_count = fast_reject.sum()
    logger.info(f"  Language fast-reject (ASCII heuristic): {fast_reject_count:,} rows dropped")

    survivors = df[~fast_reject].copy()

    logger.info(f"  Running langdetect on {len(survivors):,} survivors …")
    is_en = pd.Series(
        _is_english_with_progress(survivors["review_text"].tolist()),
        index=survivors.index,
    )
    slow_reject_count = (~is_en).sum()
    logger.info(f"  Language langdetect-reject: {slow_reject_count:,} rows dropped")

    kept = survivors[is_en]
    total_dropped = fast_reject_count + slow_reject_count
    return kept, total_dropped


# renders the preprocessing log as a latex table
def _to_latex(log_df) -> str:
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Preprocessing filter log: row counts at each stage.}",
        r"\label{tab:preprocessing_log}",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Step & Rows remaining & Rows dropped \\",
        r"\midrule",
    ]
    for _, row in log_df.iterrows():
        lines.append(
            f"{row['step']} & {int(row['rows_remaining']):,} & {int(row['rows_dropped']):,} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# runs phase 2 end to end
def main():
    import pandas as pd

    cfg = load_config()

    clean_out = "data/sample_clean.parquet"
    if parquet_exists(clean_out):
        logger.info(f"Cache hit: {clean_out} exists. Delete it to re-run Phase 2.")
        return

    df = load_parquet("data/sample_raw.parquet")
    logger.info(f"Loaded sample_raw.parquet: {len(df):,} rows")

    log_rows = []

    def checkpoint(step: str, before: int, after: int):
        dropped = before - after
        log_rows.append({"step": step, "rows_remaining": after, "rows_dropped": dropped})
        logger.info(f"[{step}] {before:,} → {after:,}  (dropped {dropped:,})")

    n0 = len(df)

    df["title"] = df["title"].fillna("").apply(clean_text)
    df["text"] = df["text"].fillna("").apply(clean_text)
    df["review_text"] = df.apply(
        lambda r: f"{r['title']}. {r['text']}" if r["title"] else r["text"],
        axis=1,
    )

    df["char_len"] = df["review_text"].str.len()
    df["token_len"] = df["review_text"].str.split().str.len()

    mask_empty = df["review_text"].str.strip().str.len() == 0
    df = df[~mask_empty].reset_index(drop=True)
    checkpoint("Drop empty/whitespace", n0, len(df))

    n_before = len(df)
    df = df.drop_duplicates(subset=["category", "year", "review_text"]).reset_index(drop=True)
    checkpoint("Drop duplicate texts", n_before, len(df))

    n_before = len(df)
    df, lang_dropped = language_filter(df, seed=cfg["sampling"]["seed"])
    df = df.reset_index(drop=True)
    checkpoint("Language filter (keep English)", n_before, len(df))

    logger.info(f"Final clean dataset: {len(df):,} rows")
    logger.info(f"Token length distribution:\n{df['token_len'].describe().to_string()}")
    logger.info(
        f"Category counts:\n{df['category'].value_counts().to_string()}"
    )

    cols = [
        "category", "year", "rating", "title", "text", "review_text",
        "char_len", "token_len", "helpful_vote", "verified_purchase", "timestamp",
    ]
    save_parquet(df[cols], clean_out)
    logger.info(f"Saved → {clean_out}")

    log_df = pd.DataFrame(log_rows)
    tables_dir = project_root() / cfg["paths"]["tables_dir"]
    ensure_dir(tables_dir)

    log_df.to_csv(tables_dir / "preprocessing_log.csv", index=False)
    (tables_dir / "preprocessing_log.tex").write_text(_to_latex(log_df), encoding="utf-8")
    logger.info(f"Preprocessing log → {tables_dir / 'preprocessing_log.csv'}")

    print("\n=== Preprocessing Log ===")
    print(log_df.to_string(index=False))

    print("\n=== Token Length Summary ===")
    print(df["token_len"].describe().round(1).to_string())

    print("\n=== Category Balance After Cleaning ===")
    print(df.groupby("category")["year"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
