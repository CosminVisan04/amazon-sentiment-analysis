from __future__ import annotations

import numpy as np
import pandas as pd
from collections import Counter


# aligns two counters on a shared vocabulary
def _counts_to_array(
    count_i: Counter,
    count_j: Counter,
    background: Counter,
    min_df: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    vocab = sorted(w for w, c in background.items() if c >= min_df)
    y_i = np.array([count_i.get(w, 0) for w in vocab], dtype=float)
    y_j = np.array([count_j.get(w, 0) for w in vocab], dtype=float)
    alpha = np.array([background[w] for w in vocab], dtype=float)
    return y_i, y_j, alpha, vocab


# computes fighting words log-odds z-scores
def fighting_words_z(
    count_i: Counter,
    count_j: Counter,
    background: Counter | None = None,
    min_df: int = 10,
) -> pd.DataFrame:
    if background is None:
        background = count_i + count_j

    y_i, y_j, alpha, vocab = _counts_to_array(count_i, count_j, background, min_df)

    n_i = y_i.sum()
    n_j = y_j.sum()
    alpha_0 = alpha.sum()

    log_odds_i = np.log(
        (y_i + alpha) / (n_i + alpha_0 - y_i - alpha)
    )
    log_odds_j = np.log(
        (y_j + alpha) / (n_j + alpha_0 - y_j - alpha)
    )

    delta = log_odds_i - log_odds_j
    sigma2 = 1.0 / (y_i + alpha) + 1.0 / (y_j + alpha)
    z = delta / np.sqrt(sigma2)

    return (
        pd.DataFrame({
            "word": vocab,
            "y_i": y_i.astype(int),
            "y_j": y_j.astype(int),
            "alpha": alpha,
            "delta": delta,
            "sigma2": sigma2,
            "z": z,
        })
        .sort_values("z", ascending=False)
        .reset_index(drop=True)
    )


# sanity check against a known toy example
def _test_fighting_words():
    ci = Counter({"great": 100, "product": 50, "terrible": 2, "ok": 10})
    cj = Counter({"terrible": 100, "product": 50, "great": 2, "ok": 10})
    bg = ci + cj

    df = fighting_words_z(ci, cj, background=bg, min_df=1)

    z_great = df.loc[df["word"] == "great", "z"].iloc[0]
    z_terrible = df.loc[df["word"] == "terrible", "z"].iloc[0]

    assert z_great > 5.0, f"Expected z('great') >> 0, got {z_great:.2f}"
    assert z_terrible < -5.0, f"Expected z('terrible') << 0, got {z_terrible:.2f}"
    assert df.iloc[0]["word"] == "great", "Top word should be 'great'"
    assert df.iloc[-1]["word"] == "terrible", "Bottom word should be 'terrible'"

    print("logodds unit test passed.")
    print(df.head(4).to_string(index=False))


if __name__ == "__main__":
    _test_fighting_words()
