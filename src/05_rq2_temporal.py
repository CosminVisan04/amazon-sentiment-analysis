import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.io import load_config, load_parquet, ensure_dir, setup_logging, project_root
from utils.stats import bootstrap_ci, holm_bonferroni
from utils.plotting import set_style, save_figure, category_palette

setup_logging()
logger = logging.getLogger("amazon_sentiment.05_rq2")


# mann-kendall trend test plus sen's slope
def mann_kendall(series: np.ndarray):
    import pymannkendall as mk
    result = mk.original_test(series)
    return {
        "trend": result.trend,
        "p": float(result.p),
        "tau": float(result.Tau),
        "sens_slope": float(result.slope),
        "sens_intercept": float(result.intercept),
    }


# fits a simple ols model of y on x
def ols_simple(df: pd.DataFrame, y: str, x: str):
    import statsmodels.formula.api as smf
    formula = f"{y} ~ {x}"
    model = smf.ols(formula, data=df).fit()
    coef = model.params[x]
    pval = model.pvalues[x]
    ci   = model.conf_int().loc[x].tolist()
    return {
        "coef": float(coef),
        "p": float(pval),
        "ci_95": ci,
        "r_squared": float(model.rsquared),
        "n": int(model.nobs),
    }


# fits the year by category interaction ols model
def ols_interaction(df: pd.DataFrame):
    import statsmodels.formula.api as smf
    model = smf.ols("vader_compound ~ year * C(category)", data=df).fit()

    rows = []
    for term in model.params.index:
        rows.append({
            "term": term,
            "coef": float(model.params[term]),
            "p": float(model.pvalues[term]),
            "ci_lo": float(model.conf_int().loc[term, 0]),
            "ci_hi": float(model.conf_int().loc[term, 1]),
        })
    return pd.DataFrame(rows), float(model.rsquared)


# plots mean compound trend by year and category
def plot_trend(yearly: pd.DataFrame, figures_dir: Path):
    import matplotlib.pyplot as plt

    set_style()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    palette = category_palette()

    for cat, color in palette.items():
        sub = yearly[yearly["category"] == cat].sort_values("year")
        ax.plot(sub["year"], sub["mean_compound"], marker="o", color=color,
                label=cat, linewidth=2, markersize=5)
        ax.fill_between(sub["year"], sub["ci_lo"], sub["ci_hi"],
                        color=color, alpha=0.15)

    ax.set_xlabel("Year")
    ax.set_ylabel("Mean VADER Compound Score")
    ax.set_title("Average Sentiment Trend by Category (2015–2023)")
    ax.set_xticks(sorted(yearly["year"].unique()))
    ax.legend()
    fig.tight_layout()
    save_figure(fig, figures_dir, "rq2_trend")
    logger.info("Saved rq2_trend")


# renders the yearly means table as latex
def yearly_to_latex(df: pd.DataFrame) -> str:
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{RQ2: Mean VADER compound score per (category $\times$ year) with 95\% bootstrap CI.}",
        r"\label{tab:rq2_yearly_means}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Category & Year & $n$ & Mean & CI low & CI high \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"{row['category']} & {int(row['year'])} & {int(row['n']):,} & "
            f"{row['mean_compound']:.4f} & {row['ci_lo']:.4f} & {row['ci_hi']:.4f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# runs phase 5 end to end
def main():
    cfg = load_config()

    stats_dir   = project_root() / cfg["paths"]["stats_dir"]
    tables_dir  = project_root() / cfg["paths"]["tables_dir"]
    figures_dir = project_root() / cfg["paths"]["figures_dir"]
    for d in [stats_dir, tables_dir, figures_dir]:
        ensure_dir(d)

    df = load_parquet("data/sample_scored.parquet")
    logger.info(f"Loaded {len(df):,} rows")

    boot_cfg = cfg["bootstrap"]

    yearly_rows = []
    for cat in ["Beauty", "Sports"]:
        for yr in sorted(df["year"].unique()):
            vals = df[(df["category"] == cat) & (df["year"] == yr)]["vader_compound"].values
            lo, hi = bootstrap_ci(vals, np.mean, boot_cfg["n_iterations"], boot_cfg["ci_level"], boot_cfg["seed"])
            yearly_rows.append({
                "category": cat,
                "year": int(yr),
                "n": len(vals),
                "mean_compound": float(np.mean(vals)),
                "ci_lo": lo,
                "ci_hi": hi,
            })
    yearly = pd.DataFrame(yearly_rows)
    logger.info(f"Yearly means computed for {len(yearly)} cells")

    mk_results = {}
    for cat in ["Beauty", "Sports"]:
        series = yearly[yearly["category"] == cat].sort_values("year")["mean_compound"].values
        mk_results[cat] = mann_kendall(series)
        r = mk_results[cat]
        logger.info(
            f"Mann-Kendall {cat}: trend={r['trend']}, p={r['p']:.4f}, "
            f"tau={r['tau']:.4f}, Sen's slope={r['sens_slope']:.6f}/yr"
        )

    ols_results = {}
    for cat in ["Beauty", "Sports"]:
        sub = df[df["category"] == cat].copy()
        ols_results[cat] = ols_simple(sub, "vader_compound", "year")
        r = ols_results[cat]
        logger.info(
            f"OLS {cat}: coef(year)={r['coef']:.6f}, p={r['p']:.4e}, "
            f"95% CI=[{r['ci_95'][0]:.6f}, {r['ci_95'][1]:.6f}], R²={r['r_squared']:.5f}"
        )

    interaction_df, r2_int = ols_interaction(df)
    logger.info(f"Interaction OLS R²={r2_int:.5f}")
    logger.info(f"\n{interaction_df.to_string(index=False)}")

    interaction_term = interaction_df[interaction_df["term"].str.contains("year.*Sports|Sports.*year", regex=True)]
    if not interaction_term.empty:
        row = interaction_term.iloc[0]
        sig = "SIGNIFICANT" if row["p"] < 0.05 else "not significant"
        logger.info(
            f"Interaction term year×Sports: coef={row['coef']:.6f}, "
            f"p={row['p']:.4e} → {sig}"
        )

    pw_rows = []
    for yr in sorted(df["year"].unique()):
        b = df[(df["category"] == "Beauty") & (df["year"] == yr)]["vader_compound"].values
        s = df[(df["category"] == "Sports") & (df["year"] == yr)]["vader_compound"].values
        u, p = sp_stats.mannwhitneyu(b, s, alternative="two-sided")
        pw_rows.append({"year": yr, "U": u, "p_raw": p})

    pw_df = pd.DataFrame(pw_rows)
    pw_df["p_adj"] = holm_bonferroni(pw_df["p_raw"].tolist())
    pw_df["significant_adj"] = pw_df["p_adj"] < 0.05
    logger.info(f"Per-year pairwise tests (Holm-corrected):\n{pw_df.to_string(index=False)}")

    results = {
        "rq": "RQ2",
        "question": "How does average review sentiment in each category change over 2015-2023, and do the two categories trend differently?",
        "yearly_means": yearly.to_dict(orient="records"),
        "mann_kendall": mk_results,
        "ols_per_category": ols_results,
        "ols_interaction": {
            "terms": interaction_df.to_dict(orient="records"),
            "r_squared": r2_int,
        },
        "pairwise_per_year": pw_df.to_dict(orient="records"),
    }
    with open(stats_dir / "rq2.json", "w") as f:
        json.dump(results, f, indent=2)

    yearly.to_csv(tables_dir / "rq2_yearly_means.csv", index=False)
    (tables_dir / "rq2_yearly_means.tex").write_text(yearly_to_latex(yearly), encoding="utf-8")
    interaction_df.to_csv(stats_dir / "rq2_interaction_ols.csv", index=False)
    pw_df.to_csv(stats_dir / "rq2_pairwise.csv", index=False)

    stat_rows = []
    for cat in ["Beauty", "Sports"]:
        r = mk_results[cat]
        o = ols_results[cat]
        stat_rows.append({
            "category": cat,
            "mk_trend": r["trend"], "mk_p": r["p"], "mk_tau": r["tau"],
            "sens_slope_per_year": r["sens_slope"],
            "ols_coef_year": o["coef"], "ols_p": o["p"],
            "ols_ci_lo": o["ci_95"][0], "ols_ci_hi": o["ci_95"][1],
        })
    pd.DataFrame(stat_rows).to_csv(stats_dir / "rq2.csv", index=False)
    logger.info(f"Stats saved → {stats_dir / 'rq2.json'}")

    plot_trend(yearly, figures_dir)

    print("\n=== RQ2 Results ===")
    print("\nYearly means:")
    print(yearly.round(4).to_string(index=False))

    print("\nMann-Kendall trends:")
    for cat, r in mk_results.items():
        print(f"  {cat}: trend={r['trend']}, p={r['p']:.4f}, Sen's slope={r['sens_slope']:.6f}/yr")

    print("\nInteraction OLS (year × category):")
    print(interaction_df.round(6).to_string(index=False))

    print("\nPer-year pairwise (Holm-corrected):")
    print(pw_df.round(6).to_string(index=False))


if __name__ == "__main__":
    main()
