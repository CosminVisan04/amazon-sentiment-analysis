# Amazon Review Sentiment Analysis

Computational pipeline for the paper:
*Sentiment Trends in Amazon Consumer Reviews: A Comparative Analysis of
Beauty and Sports & Outdoors Products (2015–2023)*

---

## Quick start

```bash
# 1. Install dependencies (Python 3.10+)
pip install -r requirements.txt

# 2. Run the full pipeline (cache-aware — skips completed phases)
python run_all.py

# 3. Re-run a single phase
python src/03_sentiment.py

# 4. Force re-run everything from scratch
python run_all.py --force   # then manually delete data/*.parquet
```

---

## Pipeline phases

| Script | Input | Output | What it does |
|---|---|---|---|
| `src/01_sample.py` | HuggingFace stream | `data/sample_raw.parquet` | Stratified reservoir sampling, 11,112 reviews/(category × year) |
| `src/02_preprocess.py` | `sample_raw.parquet` | `data/sample_clean.parquet` | Dedup, language filter, length recording |
| `src/03_sentiment.py` | `sample_clean.parquet` | `data/sample_scored.parquet` | VADER scoring, vader_label, star_label |
| `src/04_rq1_distributions.py` | `sample_scored.parquet` | figures + stats | Mann-Whitney U, KS, Cliff's delta, bootstrap CI |
| `src/05_rq2_temporal.py` | `sample_scored.parquet` | figures + stats | Mann-Kendall, Sen's slope, interaction OLS |
| `src/06_rq3_agreement.py` | `sample_scored.parquet` | figures + stats + qualitative | VADER validation: Spearman, kappa, length buckets, mismatches |
| `src/07_rq4_words.py` | `sample_scored.parquet` | figures + tables + qualitative | Fightin' Words log-odds, TF-IDF |

Each phase is **cache-aware**: it checks for its output file and exits immediately if
it already exists. Delete the output to force a re-run of that phase only.

---

## Configuration

All parameters live in `config.yaml` — no magic numbers in scripts.

Key settings:
```yaml
sampling:
  samples_per_cell: 11112   # reviews per (category × year) cell
  seed: 42                  # global random seed

vader:
  positive_threshold: 0.05
  negative_threshold: -0.05

words:
  min_doc_freq: 10
  top_n: 25
  ngram_range: [1, 2]

bootstrap:
  n_iterations: 10000
  ci_level: 0.95
```

---

## Output structure

```
outputs/
  figures/    *.pdf (vector, for LaTeX) + *.png
  tables/     *.csv + *.tex (LaTeX snippets, drop straight into paper)
  stats/      rq1-4.json + .csv (all test statistics)
  qualitative/ mismatches_*.csv, tfidf_*.csv
FINDINGS.md   — plain-language answers to all four RQs with key numbers
run.log       — full execution log, reconstructs the methodology
```

---

## Reproducibility

- Fixed seed = 42 propagates through sampling, bootstrap, and all random operations
- `langdetect` seeded via `DetectorFactory.seed = 42`
- Parquet intermediate files are gitignored; re-running `python run_all.py` from
  scratch regenerates identical outputs given the same HuggingFace dataset version

---

## Utilities (`src/utils/`)

| Module | Contents |
|---|---|
| `io.py` | Config loading, parquet I/O, logging setup |
| `stats.py` | Bootstrap CI, Cliff's delta, Holm-Bonferroni, Spearman CI, Cohen's κ |
| `logodds.py` | Fightin' Words implementation + unit test |
| `plotting.py` | Consistent style, Wong colour-blind palette, PDF+PNG saver |
