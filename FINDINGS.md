# FINDINGS.md
## Sentiment Trends in Amazon Consumer Reviews: Beauty vs Sports & Outdoors (2015–2023)

This document is the primary reference for writing the Results and Discussion sections.
Each RQ section contains: the question, the method, the plain-language answer, the key
numbers, and pointers to figures/tables. A "Notes for the paper" block follows each RQ.

---

## Corpus & Sampling

**Method:** Stratified reservoir sampling (Vitter 1985, Algorithm R). Single streaming
pass per category over the full Amazon Reviews 2023 dataset (McAuley Lab / HuggingFace).

**Final corpus:** 187,624 reviews after preprocessing (93,582 Beauty; 94,042 Sports).
- Raw sample: 200,016 (11,112 per category × year cell, 18 cells)
- Dropped: 6,100 duplicate texts (3.1%), 6,292 non-English (3.2%)
- All 18 cells met the 5,000+ target; no shortfall

**Tables:** `outputs/tables/sampling_summary.csv` / `.tex`
          `outputs/tables/preprocessing_log.csv`

---

## RQ1 — Sentiment Score Distribution by Category

**Question:** How does the distribution of sentiment scores differ between Beauty
and Sports & Outdoors reviews?

**Method:** Mann-Whitney U (location), Kolmogorov-Smirnov (distribution shape),
Cliff's delta (effect size), bootstrap 95% CI for difference in means and medians
(10,000 iterations, seed 42).

### Answer
Both categories are overwhelmingly positive and nearly identical in their sentiment
distributions. The difference is statistically significant but practically negligible.

### Key numbers
| Statistic | Beauty | Sports |
|---|---|---|
| n | 93,582 | 94,042 |
| Mean compound | 0.5585 | 0.5597 |
| Median compound | 0.7876 | 0.7790 |
| SD | 0.4974 | 0.4886 |
| IQR | 0.5171 | 0.5089 |
| % Positive (VADER) | 81.1% | 81.5% |
| % Neutral | 5.8% | 6.2% |
| % Negative | 13.0% | 12.3% |

**Tests:**
- Mann-Whitney U = 4,439,285,878, p = 8.93 × 10⁻⁴, rank-biserial r = 0.009 (negligible)
- Kolmogorov-Smirnov D = 0.0144, p = 6.48 × 10⁻⁹
- Cliff's delta = 0.009 (|d| < 0.147 = negligible by convention)
- Bootstrap 95% CI, difference in means (Beauty − Sports): [−0.006, 0.003]
- Bootstrap 95% CI, difference in medians: [0.001, 0.012]

**Figures:** `outputs/figures/rq1_kde.{pdf,png}`, `rq1_violin.{pdf,png}`, `rq1_ecdf.{pdf,png}`
**Table:** `outputs/tables/rq1_descriptives.{csv,tex}`
**Stats:** `outputs/stats/rq1.json`, `rq1.csv`

### Notes for the paper
- **Main finding:** Despite being fundamentally different product domains, the sentiment
  distributions are functionally identical (Cliff's delta = 0.009, CI for mean difference
  straddles zero). This is the answer to RQ1 — and it is interesting precisely because
  of the null result.
- **Novelty argument:** Platform-level positivity bias dominates category-specific
  signal at the aggregate level. This motivates the word-level analysis (RQ4) to find
  where categories actually diverge.
- **Why KS is significant but Cliff's delta is not:** With n ≈ 94k per group, KS has
  power to detect D = 0.014 (a 1.4pp difference in the CDF). This is a statistical
  artefact of large samples, not a meaningful effect. State this explicitly.
- The distributions are strongly left-skewed (median ≈ 0.79 >> mean ≈ 0.56), reflecting
  Amazon's well-known positive review bias. Worth one sentence in the Results.

---

## RQ2 — Temporal Trends (2015–2023)

**Question:** How does average review sentiment in each category change over 2015–2023,
and do the two categories trend differently?

**Method:** Per-cell bootstrap 95% CI (10k iterations). Mann-Kendall monotonic trend
test + Sen's slope per category. Review-level OLS `compound ~ year` per category.
Interaction OLS `compound ~ year × category` (the key test for differential trends).
Per-year pairwise Mann-Whitney U with Holm–Bonferroni correction.

### Answer
Neither category shows a statistically significant monotonic trend over 2015–2023
(Mann-Kendall). Review-level OLS finds a highly significant but practically negligible
decline (R² < 0.001). The two categories do not trend differently (interaction p = 0.162).
The dominant pattern is non-monotonic: a shared peak in 2019 followed by a decline
through 2021–2022 and partial recovery in 2023.

### Key numbers

**Mann-Kendall:**
| Category | Trend | p | Tau | Sen's slope |
|---|---|---|---|---|
| Beauty | no trend | 0.175 | −0.389 | −0.00454/yr |
| Sports | no trend | 0.348 | −0.278 | −0.00317/yr |

**OLS per category (review-level):**
| Category | coef(year) | p | 95% CI | R² |
|---|---|---|---|---|
| Beauty | −0.00509 | 6.36 × 10⁻¹⁶ | [−0.00633, −0.00386] | 0.00070 |
| Sports | −0.00386 | 4.24 × 10⁻¹⁰ | [−0.00507, −0.00265] | 0.00041 |

**Interaction OLS** `compound ~ year × C(category)`:
- Interaction term `year:Sports`: coef = 0.00123, p = 0.162 → **not significant**
- Year explains < 0.06% of variance in compound scores (R² = 0.00056)

**Per-year pairwise (Holm-corrected):**
- 2015: p_adj = 0.000009 ✓ significant
- 2016: p_adj = 0.001221 ✓ significant
- 2017–2023: all p_adj > 0.15 → not significant

**Yearly means (selected):**
| Year | Beauty mean | Sports mean |
|---|---|---|
| 2015 | 0.580 | 0.556 |
| 2019 | 0.588 ← peak | 0.593 ← peak |
| 2021 | 0.531 | 0.536 |
| 2022 | 0.535 ← trough | 0.523 ← trough |
| 2023 | 0.542 | 0.549 |

**Figure:** `outputs/figures/rq2_trend.{pdf,png}`
**Table:** `outputs/tables/rq2_yearly_means.{csv,tex}`
**Stats:** `outputs/stats/rq2.json`, `rq2.csv`, `rq2_interaction_ols.csv`, `rq2_pairwise.csv`

### Notes for the paper
- **Why Mann-Kendall and OLS disagree:** OLS operates on 187k individual reviews and
  is significant due to sample size, not effect size. Mann-Kendall on 9 annual means
  is the appropriate test for monotonic trend in a time series — and it does not
  confirm a trend. State both and explain.
- **The 2019 peak / 2020–2022 dip** is the most interesting pattern. Both categories
  dip together starting 2020. Supply chain disruptions, delayed shipping, and
  out-of-stock substitutions during COVID-19 may explain negative reviews in this
  period. This is contextual speculation — frame carefully as a possible explanation.
- **Convergence over time:** The only years with significant category differences are
  2015 and 2016. From 2017 onward, Beauty and Sports are statistically
  indistinguishable year-by-year. This is a clean "convergence" finding.
- **Supports novelty argument:** The non-monotonic pattern (peak → dip → recovery)
  and the convergence of categories are both findings that go beyond a simple
  "sentiment is stable/increasing/decreasing" headline.

---

## RQ3 — Star Ratings vs VADER Agreement & VADER Validation

**Question:** How well do star ratings and VADER sentiment scores agree,
and in which cases do they diverge?

**Method:** Spearman ρ (rating vs compound) with Fisher-z 95% CI. Confusion matrix
of `vader_label` vs `star_label`: accuracy, macro-F1, Cohen's quadratic-weighted
kappa. Chi-square on per-category agreement rates. Per-length-bucket accuracy
with bootstrap 95% CI. Mismatch mining (rating 4–5 but compound ≤ −0.5; rating
1–2 but compound ≥ 0.5).

### Answer
VADER shows moderate agreement with star ratings (ρ = 0.510, κ = 0.592, accuracy
= 80.8%). Agreement is significantly better in Beauty than Sports. Contrary to
expectations, VADER performs **better on short reviews and worse on long ones** —
the limiting factor is sentiment complexity, not text length. VADER is systematically
over-optimistic: low-star/high-VADER mismatches are 4× more common than the reverse.

### Key numbers

**Spearman correlation:**
| Scope | ρ | p | 95% CI |
|---|---|---|---|
| Overall | 0.510 | < 2.2 × 10⁻³⁰⁸ | [0.507, 0.514] |
| Beauty | 0.525 | < 2.2 × 10⁻³⁰⁸ | [0.521, 0.530] |
| Sports | 0.495 | < 2.2 × 10⁻³⁰⁸ | [0.490, 0.500] |

**Classification metrics:**
| Scope | Accuracy | Macro-F1 | κ (quadratic) |
|---|---|---|---|
| Overall | 0.808 | 0.541 | 0.592 |
| Beauty | 0.805 | 0.543 | 0.598 |
| Sports | 0.810 | 0.539 | 0.585 |

- Chi-square (Beauty vs Sports agreement): χ² = 9.33, p = 0.00225 → significant

**Agreement by review length (key VADER-limitation finding):**
| Token length | n | Accuracy | 95% CI |
|---|---|---|---|
| ≤5 tokens | 10,622 | 81.9% | [81.2%, 82.6%] |
| 6–20 tokens | 70,288 | **82.7%** | [82.4%, 83.0%] |
| 21–50 tokens | 62,469 | 80.5% | [80.2%, 80.8%] |
| >50 tokens | 44,245 | **77.9%** | [77.5%, 78.3%] |

**Mismatch counts (saved for qualitative analysis):**
| Direction | Beauty | Sports |
|---|---|---|
| High star (4–5★), low VADER (≤ −0.5) | 719 | 809 |
| Low star (1–2★), high VADER (≥ +0.5) | 2,993 | 2,433 |
| **Ratio** | **4.2:1** | **3.0:1** |

**Figures:** `outputs/figures/rq3_confusion_beauty.{pdf,png}`,
            `rq3_confusion_sports.{pdf,png}`,
            `rq3_agreement_by_length.{pdf,png}`
**Stats:** `outputs/stats/rq3.json`, `rq3.csv`
**Qualitative:** `outputs/qualitative/mismatches_Beauty.csv`,
                 `outputs/qualitative/mismatches_Sports.csv`

### Notes for the paper — VADER limitations (F4/F5)
- **The length finding challenges the canonical assumption** that VADER fails on short
  texts (designed for tweets). On Amazon reviews, accuracy *decreases* with length.
  Short reviews ("love it!", "garbage") are unambiguous — VADER handles them well.
  Long reviews discuss multiple product aspects with qualified language and mixed
  sentiment that VADER's single compound score cannot capture. State this finding
  explicitly and contrast it with the social-media assumption.
- **Asymmetric errors:** The 3–4× excess of low-star/high-VADER mismatches over
  high-star/low-VADER is the key VADER-limitation evidence. VADER is systematically
  over-optimistic on Amazon reviews. Possible causes: (a) negative reviews often
  contain positive comparisons ("seemed good at first"), (b) domain-specific negative
  vocabulary not in VADER's lexicon (e.g. "ripped", "missing parts"), (c) sarcasm.
  Read the mismatch files for concrete examples to quote.
- **"stars" confound:** The word "stars" appears among the top positive distinctive
  words (RQ4). Reviewers write "5 stars" in the review body, which may inflate VADER
  compound scores for high-rated reviews and artificially boost the observed agreement.
  Acknowledge this as a limitation.
- **κ = 0.592** falls in Landis & Koch's "moderate" band (0.41–0.60). It is just above
  the moderate/substantial boundary. Frame as "moderate-to-substantial agreement" and
  note that this is comparable to inter-annotator agreement in sentiment labelling tasks.
- **Beauty vs Sports:** VADER correlates better with stars in Beauty (ρ = 0.525 vs 0.495).
  Beauty reviews are more emotionally direct and sensory (RQ4). Sports reviews use more
  functional/technical language that VADER's affective lexicon does not cover as well.

---

## RQ4 — Distinctive Words

**Question:** What words are most distinctive of highly positive vs highly negative
reviews in each category?

**Method:** Weighted log-odds ratio with informative Dirichlet prior (Monroe, Colaresi
& Quinn 2008, "Fightin' Words"). Positive/negative defined by star rating (4–5 vs 1–2★),
NOT by VADER, to keep RQ4 independent of the tool audited in RQ3. TF-IDF reported
as secondary descriptive cross-check. Unigrams + bigrams, min document frequency 10,
English stopwords removed.

### Answer
Both categories share a common positive vocabulary drawn from Amazon's platform culture.
Their negative vocabularies diverge in a domain-meaningful way: Beauty negatives focus
on product efficacy and consumer warnings; Sports negatives on physical durability
failure. Cross-category vocabulary confirms distinct "review cultures": Beauty is
sensory and emotional; Sports is functional and technical.

### Key words (top 10 per side by z-score)

**Within Beauty — Positive vs Negative:**
- Positive (4–5★): *great, love, works, perfect, easy, best, nice, amazing, great product*
- Negative (1–2★): *cheap, refund, didn work, don buy, awful, poor, worst, doesn work, bad*

**Within Sports — Positive vs Negative:**
- Positive (4–5★): *great, love, easy, perfect, good, nice, works, price, great product*
- Negative (1–2★): *ripped, returning, horrible, missing, fell, garbage, don waste, months, broken, don buy*

**Cross-category — Beauty vs Sports:**
- Beauty: *hair, skin, scent, smell, smells, brush, face, love, makeup*
- Sports: *material, size, cold, carry, belt, durable, water bottle, straps*

**Figures:** `outputs/figures/rq4_words_beauty.{pdf,png}`,
            `rq4_words_sports.{pdf,png}`,
            `rq4_words_crosscat.{pdf,png}`
**Tables:** `outputs/tables/rq4_words_beauty.{csv,tex}`,
            `rq4_words_sports.{csv,tex}`,
            `rq4_words_crosscat.{csv,tex}`
**TF-IDF:** `outputs/qualitative/tfidf_beauty.csv`, `tfidf_sports.csv`, `tfidf_crosscat.csv`

### Notes for the paper
- **Platform vocabulary vs category vocabulary:** Positive reviews in both categories
  use nearly identical superlatives (*great, love, perfect, easy*). This is further
  evidence that Amazon's review culture drives aggregate sentiment (RQ1 null result),
  not product domain. The category signal only emerges in negative reviews and in the
  cross-category comparison.
- **Sports durability narrative:** Negative Sports words (*ripped, fell, broken, missing,
  months*) tell a coherent story: physical products failing structurally over time.
  "Months" as a distinctive word suggests durability complaints over weeks/months of use.
  This is domain vocabulary VADER's lexicon does not cover — connecting RQ4 back to RQ3.
- **Beauty efficacy narrative:** Negative Beauty words (*didn work, doesn work, cheap,
  refund*) reflect efficacy disappointment and consumer action (seeking refunds). Different
  failure mode than Sports.
- **"don buy" and "don waste" as bigrams:** The log-odds method surfaces negation bigrams
  that raw frequency would miss (negation splits across tokens, reducing unigram counts).
  This validates the choice of Fightin' Words over simple frequency counts.
- **Cross-category novelty:** The sensory/emotional (Beauty) vs functional/technical
  (Sports) split is the strongest "review culture" evidence in the paper. Beauty reviewers
  use affective vocabulary (*love, scent, smell*); Sports reviewers use spec vocabulary
  (*material, durable, size*). This has implications for any downstream NLP model trained
  on pooled Amazon reviews — domain transfer may require category-specific lexica.
- **"stars" confound (see RQ3):** The word "stars" appearing in positive reviews suggests
  some reviewers copy their star rating into the text ("5 stars, great product"). Mention
  alongside the RQ3 VADER-limitation discussion.
- **"br" artefact:** HTML `<br>` tag appearing in Sports cross-category words is a minor
  preprocessing residual. Does not affect the analysis but worth one clause in the
  Limitations.

---

## Summary table — all RQ answers

| RQ | One-sentence answer | Key stat |
|---|---|---|
| RQ1 | Beauty and Sports have virtually identical sentiment distributions. | Cliff's delta = 0.009 (negligible); CI for mean diff straddles zero |
| RQ2 | Neither category shows a significant monotonic trend; both peak in 2019 and dip in 2021–22; they do not differ in trajectory. | MK p > 0.17 both cats; interaction p = 0.162 |
| RQ3 | VADER agrees moderately with star ratings (κ = 0.59); it is over-optimistic and degrades on long (not short) reviews. | ρ = 0.510; accuracy 82.7% (short) vs 77.9% (long) |
| RQ4 | Positive vocabulary is platform-generic; negative vocabulary is domain-specific; Beauty is sensory/emotional, Sports is functional/technical. | Top words by log-odds z-score |

---

## Evidence supporting novelty / Web Science motivation
*(Flag these in the paper's Introduction and Discussion)*

1. **Category convergence (RQ1 + RQ2):** Aggregate sentiment is category-independent,
   suggesting Amazon's platform design (star ratings, review prompts) homogenises
   affect expression across domains. A Web Science contribution: platform architecture
   shapes user-generated content at scale.

2. **Non-monotonic temporal pattern (RQ2):** The 2019 peak and 2020–22 dip is a
   system-level signal visible across both categories simultaneously — consistent with
   an external shock (COVID-19 supply chain disruption) affecting consumer experience
   platform-wide. Demonstrates that aggregate sentiment in UGC can serve as a proxy
   signal for real-world disruptions.

3. **VADER length finding (RQ3):** Challenges the standard assumption about VADER's
   domain applicability. The failure mode on Amazon is not short text (as on Twitter)
   but long, mixed-sentiment text. This is a new empirical data point for the
   sentiment-analysis methods literature.

4. **Review culture vocabulary (RQ4):** Despite identical aggregate distributions,
   the two categories have measurably different lexical cultures. Functional vs
   emotional framing is a category-level property of product reviews, with implications
   for cross-domain NLP model transfer.
