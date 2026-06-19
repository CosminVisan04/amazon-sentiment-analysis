import json
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.io import load_config, load_parquet, ensure_dir, setup_logging, project_root
from utils.logodds import fighting_words_z
from utils.plotting import set_style, save_figure, PALETTE

setup_logging()
logger = logging.getLogger("amazon_sentiment.07_rq4")

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

NEGATION_SAFE_STOP_WORDS = list(ENGLISH_STOP_WORDS - {"no", "nor", "not"})
NEGATION_TOKEN_PATTERN = r"(?u)\b[a-zA-Z]+(?:'[a-zA-Z]+)?\b"


# builds a count vectorizer with negation preserved
def _build_vectorizer(cfg: dict, texts: list[str]):
    from sklearn.feature_extraction.text import CountVectorizer
    vec = CountVectorizer(
        lowercase=True,
        ngram_range=tuple(cfg["words"]["ngram_range"]),
        min_df=cfg["words"]["min_doc_freq"],
        stop_words=NEGATION_SAFE_STOP_WORDS,
        token_pattern=NEGATION_TOKEN_PATTERN,
        max_features=100_000,
    )
    vec.fit(texts)
    return vec


# converts a vectorized corpus to a word counter
def _texts_to_counter(vec, texts: list[str]) -> Counter:
    X = vec.transform(texts)
    vocab = vec.get_feature_names_out()
    counts = np.asarray(X.sum(axis=0)).flatten().astype(int)
    return Counter(dict(zip(vocab, counts)))


# top tf-idf terms per group as a cross-check
def _tfidf_top(texts_i: list[str], texts_j: list[str], label_i: str, label_j: str, cfg: dict, n: int) -> pd.DataFrame:
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf = TfidfVectorizer(
        lowercase=True,
        ngram_range=tuple(cfg["words"]["ngram_range"]),
        min_df=cfg["words"]["min_doc_freq"],
        stop_words=NEGATION_SAFE_STOP_WORDS,
        token_pattern=NEGATION_TOKEN_PATTERN,
        max_features=100_000,
    )
    all_texts = texts_i + texts_j
    X = tfidf.fit_transform(all_texts)
    vocab = tfidf.get_feature_names_out()
    X_i = X[: len(texts_i)]
    X_j = X[len(texts_i) :]

    mean_i = np.asarray(X_i.mean(axis=0)).flatten()
    mean_j = np.asarray(X_j.mean(axis=0)).flatten()

    rows = []
    top_i = np.argsort(mean_i)[::-1][:n]
    for idx in top_i:
        rows.append({"group": label_i, "word": vocab[idx], "mean_tfidf": float(mean_i[idx])})
    top_j = np.argsort(mean_j)[::-1][:n]
    for idx in top_j:
        rows.append({"group": label_j, "word": vocab[idx], "mean_tfidf": float(mean_j[idx])})
    return pd.DataFrame(rows)


# runs one fighting words comparison and saves outputs
def run_comparison(
    texts_i: list[str],
    texts_j: list[str],
    label_i: str,
    label_j: str,
    comparison_name: str,
    cfg: dict,
    tables_dir: Path,
    qual_dir: Path,
    figures_dir: Path,
) -> pd.DataFrame:
    logger.info(
        f"[{comparison_name}] {label_i} ({len(texts_i):,}) vs {label_j} ({len(texts_j):,})"
    )

    all_texts = texts_i + texts_j
    vec = _build_vectorizer(cfg, all_texts)

    count_i  = _texts_to_counter(vec, texts_i)
    count_j  = _texts_to_counter(vec, texts_j)
    background = _texts_to_counter(vec, all_texts)

    z_df = fighting_words_z(count_i, count_j, background=background, min_df=1)
    n_top = cfg["words"]["top_n"]

    top_i = z_df.head(n_top).copy()
    top_j = z_df.tail(n_top).copy()
    top_i["distinctive_of"] = label_i
    top_j["distinctive_of"] = label_j

    combined = pd.concat([top_i, top_j], ignore_index=True)
    combined.to_csv(tables_dir / f"rq4_words_{comparison_name}.csv", index=False)
    _save_words_latex(combined, label_i, label_j, comparison_name, n_top, tables_dir)
    logger.info(f"  Saved rq4_words_{comparison_name}.csv")

    tfidf_df = _tfidf_top(texts_i, texts_j, label_i, label_j, cfg, n_top)
    tfidf_df.to_csv(qual_dir / f"tfidf_{comparison_name}.csv", index=False)

    _plot_diverging(z_df, label_i, label_j, comparison_name, n_top, figures_dir)

    return combined


# renders the distinctive words table as latex
def _save_words_latex(df: pd.DataFrame, label_i: str, label_j: str,
                      name: str, n: int, tables_dir: Path):
    top_i = df[df["distinctive_of"] == label_i][["word", "z"]].head(n)
    top_j = df[df["distinctive_of"] == label_j][["word", "z"]].tail(n).iloc[::-1]

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{RQ4 - top " + str(n) + r" most distinctive words (log-odds z-score). "
        r"Comparison: " + label_i.replace("&", r"\&") + r" vs " + label_j.replace("&", r"\&") + r".}",
        r"\label{tab:rq4_" + name + r"}",
        r"\begin{tabular}{lr|lr}",
        r"\toprule",
        f"\\textbf{{{label_i}}} & z & \\textbf{{{label_j}}} & z \\\\",
        r"\midrule",
    ]
    for (_, ri), (_, rj) in zip(top_i.iterrows(), top_j.iterrows()):
        lines.append(
            f"{ri['word']} & {ri['z']:.2f} & {rj['word']} & {rj['z']:.2f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (tables_dir / f"rq4_words_{name}.tex").write_text("\n".join(lines), encoding="utf-8")


# plots a diverging bar chart of word z-scores
def _plot_diverging(z_df: pd.DataFrame, label_i: str, label_j: str,
                    name: str, n: int, figures_dir: Path):
    import matplotlib.pyplot as plt

    set_style()
    top_i = z_df.head(n)[["word", "z"]].copy()
    top_j = z_df.tail(n)[["word", "z"]].copy().iloc[::-1]

    plot_df = pd.concat([
        top_j.assign(side="j"),
        top_i.assign(side="i"),
    ]).reset_index(drop=True)
    plot_df = plot_df.sort_values("z")

    colors = [PALETTE["negative"] if s == "j" else PALETTE["positive"]
              for s in plot_df["side"]]

    fig, ax = plt.subplots(figsize=(7, max(6, len(plot_df) * 0.28)))
    bars = ax.barh(range(len(plot_df)), plot_df["z"], color=colors, edgecolor="none")
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df["word"], fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Log-odds z-score")
    ax.set_title(f"Distinctive Words: {label_i} vs {label_j}")

    from matplotlib.patches import Patch
    legend = [
        Patch(color=PALETTE["positive"], label=label_i),
        Patch(color=PALETTE["negative"], label=label_j),
    ]
    ax.legend(handles=legend, loc="lower right", fontsize=9)
    fig.tight_layout()
    save_figure(fig, figures_dir, f"rq4_words_{name}")
    logger.info(f"  Saved rq4_words_{name}")


# runs phase 7 end to end
def main():
    cfg = load_config()

    tables_dir  = project_root() / cfg["paths"]["tables_dir"]
    figures_dir = project_root() / cfg["paths"]["figures_dir"]
    qual_dir    = project_root() / cfg["paths"]["qualitative_dir"]
    stats_dir   = project_root() / cfg["paths"]["stats_dir"]
    for d in [tables_dir, figures_dir, qual_dir, stats_dir]:
        ensure_dir(d)

    df = load_parquet("data/sample_scored.parquet")
    logger.info(f"Loaded {len(df):,} rows")

    beauty = df[df["category"] == "Beauty"]
    run_comparison(
        texts_i=beauty[beauty["star_label"] == "positive"]["review_text"].tolist(),
        texts_j=beauty[beauty["star_label"] == "negative"]["review_text"].tolist(),
        label_i="Positive (4-5★)",
        label_j="Negative (1-2★)",
        comparison_name="beauty",
        cfg=cfg,
        tables_dir=tables_dir,
        qual_dir=qual_dir,
        figures_dir=figures_dir,
    )

    sports = df[df["category"] == "Sports"]
    run_comparison(
        texts_i=sports[sports["star_label"] == "positive"]["review_text"].tolist(),
        texts_j=sports[sports["star_label"] == "negative"]["review_text"].tolist(),
        label_i="Positive (4-5★)",
        label_j="Negative (1-2★)",
        comparison_name="sports",
        cfg=cfg,
        tables_dir=tables_dir,
        qual_dir=qual_dir,
        figures_dir=figures_dir,
    )

    run_comparison(
        texts_i=beauty["review_text"].tolist(),
        texts_j=sports["review_text"].tolist(),
        label_i="Beauty & Personal Care",
        label_j="Sports & Outdoors",
        comparison_name="crosscat",
        cfg=cfg,
        tables_dir=tables_dir,
        qual_dir=qual_dir,
        figures_dir=figures_dir,
    )

    print("\n=== RQ4 Results ===")
    for name, label_i, label_j in [
        ("beauty",   "Positive (4-5★)", "Negative (1-2★)"),
        ("sports",   "Positive (4-5★)", "Negative (1-2★)"),
        ("crosscat", "Beauty & Personal Care", "Sports & Outdoors"),
    ]:
        result_df = pd.read_csv(tables_dir / f"rq4_words_{name}.csv")
        top_i = result_df[result_df["distinctive_of"] == label_i]["word"].head(10).tolist()
        top_j = result_df[result_df["distinctive_of"] == label_j]["word"].head(10).tolist()
        print(f"\n[{name.upper()}]")
        print(f"  {label_i}: {', '.join(top_i)}")
        print(f"  {label_j}: {', '.join(top_j)}")

    logger.info("Phase 7 complete.")


if __name__ == "__main__":
    main()
