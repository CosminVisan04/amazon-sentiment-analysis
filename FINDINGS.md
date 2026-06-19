# FINDINGS.md
## Sentiment Trends in Amazon Consumer Reviews: Beauty vs Sports & Outdoors (2015–2023)

This document is the primary reference for writing the Results and Discussion sections.
Each RQ section contains: the question, the method, the plain-language answer, the key
numbers, and pointers to figures/tables. A "Notes for the paper" block follows each RQ.

---

## Revision notes (data-quality corrections applied)

This revision follows four corrections to the pipeline, applied after reviewer-style
feedback on the first pass. All numbers below are from the corrected, re-run pipeline.

1. **Duplicate detection was scoped to `(category, year, review_text)`** instead of
   `review_text` alone. The old global dedup could delete a review just because
   identical text existed in a *different* category or year, silently damaging the
   stratified balance. Effect: only 3,562 duplicates dropped (1.8%) vs. 6,100 (3.1%)
   originally - about 2,538 reviews were being wrongly discarded before.
2. **HTML/markup cleaning** (`clean_text()`) strips `<tags>`, unescapes HTML entities,
   and removes `[[ASIN:...]]` markers before any other processing. This removes the
   `br` artefact that previously polluted the RQ4 cross-category word list (confirmed
   gone from `rq4_words_crosscat.csv` below).
3. **Apostrophe normalisation.** The corpus mixes mojibake (`â`), curly (`’`/`‘`), and
   straight (`'`) apostrophes for the *same* contraction across different reviews. A
   tokenizer that only recognises one form splits the rest - `doesn't` → `doesn` + a
   stray `t` - destroying the negation signal. All three forms are now normalised to
   a straight `'` before tokenising, and the RQ4 vectorizer keeps contractions intact
   and no longer strips `no`/`nor`/`not` as stopwords. Concretely: the Beauty negative
   word list previously showed broken fragments `didn work`, `doesn work`; it now
   correctly shows `doesn't work`, `didn't`, `did not`, `don't waste`, `don't buy`.
4. **Majority-class baseline added to RQ3** so VADER's classification accuracy can be
   judged against chance-level performance, not in isolation.

Net effect on the corpus: 189,587 reviews after cleaning (94,261 Beauty; 95,326 Sports),
up slightly from 187,624 because the dedup fix preserved more legitimately-unique
reviews. The qualitative conclusions for RQ1 and RQ4 are essentially unchanged; **RQ2's
differential-trend conclusion changes** (see below) and **RQ3's Beauty-vs-Sports framing
is corrected** to match the actual per-metric pattern rather than a one-directional claim.

---

## Corpus & Sampling

**Method:** Stratified reservoir sampling (Vitter 1985, Algorithm R). Single streaming
pass per category over the full Amazon Reviews 2023 dataset (McAuley Lab / HuggingFace).

**Final corpus:** 189,587 reviews after preprocessing (94,261 Beauty; 95,326 Sports).
- Raw sample: 200,016 (11,112 per category × year cell, 18 cells)
- Dropped: 3,562 duplicate texts (1.8%, scoped to category × year), 6,867 non-English (3.5%)
- All 18 cells met the 11,112 target; no shortfall

**Tables:** `outputs/tables/sampling_summary.csv` / `.tex`
          `outputs/tables/preprocessing_log.csv`

---

## RQ1 - Sentiment Score Distribution by Category

**Question:** How does the distribution of sentiment scores differ between Beauty
and Sports & Outdoors reviews?

**Method:** Mann-Whitney U (location), Kolmogorov-Smirnov (distribution shape),
Cliff's delta (effect size), bootstrap 95% CI for difference in means and medians
(10,000 iterations, seed 42).

### Answer
Both categories are overwhelmingly positive and nearly identical in their sentiment
distributions. The difference is statistically detectable but practically negligible:
the bootstrap CI for the *mean* difference straddles zero, while the CI for the
*median* difference does not - Beauty's median is reliably a little higher, but the
two distributions otherwise overlap almost completely.

### Key numbers
| Statistic | Beauty | Sports |
|---|---|---|
| n | 94,261 | 95,326 |
| Mean compound | 0.5559 | 0.5580 |
| Median compound | 0.7845 | 0.7769 |
| SD | 0.4986 | 0.4878 |
| IQR | 0.5251 | 0.5094 |
| % Positive (VADER) | 81.0% | 81.4% |
| % Neutral | 5.9% | 6.3% |
| % Negative | 13.2% | 12.2% |

**Tests:**
- Mann-Whitney U = 4,535,211,928.5, p = 3.67 × 10⁻⁴, rank-biserial r = 0.0095 (negligible)
- Kolmogorov-Smirnov D = 0.0159, p = 8.72 × 10⁻¹¹
- Cliff's delta = 0.0095 (|d| < 0.147 = negligible by convention)
- Bootstrap 95% CI, difference in means (Beauty − Sports): [−0.0065, 0.0023] - **straddles zero**
- Bootstrap 95% CI, difference in medians (Beauty − Sports): [0.0058, 0.0135] - **excludes zero**

**Figures:** `outputs/figures/rq1_kde.{pdf,png}`, `rq1_violin.{pdf,png}`, `rq1_ecdf.{pdf,png}`
**Table:** `outputs/tables/rq1_descriptives.{csv,tex}`
**Stats:** `outputs/stats/rq1.json`, `rq1.csv`

### Notes for the paper
- **Main finding:** Despite being fundamentally different product domains, the sentiment
  distributions are functionally identical (Cliff's delta = 0.0095). This is the answer
  to RQ1 - and it is interesting precisely because of the near-null result.
- **Mean vs. median nuance:** Report both CIs. The median difference is the more
  reliable signal here (CI excludes zero); the mean difference is not distinguishable
  from zero at this sample size. This is a more honest statement than picking one
  summary statistic.
- **Novelty argument:** Platform-level positivity bias dominates category-specific
  signal at the aggregate level. This motivates the word-level analysis (RQ4) to find
  where categories actually diverge.
- **Why KS is significant but Cliff's delta is not:** With n ≈ 95k per group, KS has
  power to detect D = 0.016 (a 1.6pp difference in the CDF). This is a statistical
  artefact of large samples, not a meaningful effect. State this explicitly.
- The distributions are strongly left-skewed (median ≈ 0.78 >> mean ≈ 0.56), reflecting
  Amazon's well-known positive review bias. Worth one sentence in the Results.

---

## RQ2 - Temporal Trends (2015–2023)

**Question:** How does average review sentiment in each category change over 2015–2023,
and do the two categories trend differently?

**Method:** Per-cell bootstrap 95% CI (10k iterations). Mann-Kendall monotonic trend
test + Sen's slope per category. Review-level OLS `compound ~ year` per category.
Interaction OLS `compound ~ year × category` (the key test for differential trends).
Per-year pairwise Mann-Whitney U with Holm–Bonferroni correction.

### Answer
Neither category shows a statistically significant monotonic trend over 2015–2023 by
Mann-Kendall (which only has 9 annual points to work with). Review-level OLS - which
pools all 189,587 individual reviews - finds a highly significant but practically tiny
decline in both categories, **and this time the interaction term is also significant**
(p = 0.014): Beauty's fitted slope is steeper than Sports'. This should *not* be read as
a clean, steadily-widening gap, though - per-year pairwise tests find a significant
category difference only in 2015–2016, with no later year reaching significance after
Holm–Bonferroni correction. The interaction result is real but is driven by the early
years, not by a year-on-year divergence that keeps growing. The dominant visual pattern
is still non-monotonic: a shared peak in 2019, a dip starting 2020, with Beauty's low
point in 2021 and Sports' low point in 2022, and partial recovery in 2023.

### Key numbers

**Mann-Kendall:**
| Category | Trend | p | Tau | Sen's slope |
|---|---|---|---|---|
| Beauty | no trend | 0.118 | −0.444 | −0.00536/yr |
| Sports | no trend | 0.348 | −0.278 | −0.00291/yr |

**OLS per category (review-level):**
| Category | coef(year) | p | 95% CI | R² |
|---|---|---|---|---|
| Beauty | −0.00576 | 6.01 × 10⁻²⁰ | [−0.00699, −0.00453] | 0.00089 |
| Sports | −0.00359 | 4.39 × 10⁻⁹ | [−0.00479, −0.00239] | 0.00036 |

**Interaction OLS** `compound ~ year × C(category)`:
- Interaction term `year:Sports`: coef = +0.00217, p = **0.0136** → **significant**,
  95% CI [0.00045, 0.00389] (excludes zero)
- Beauty's slope is significantly steeper (more negative) than Sports'
- Whole-model R² = 0.00063 - the effect is real but explains a tiny fraction of variance

**Per-year pairwise (Holm-corrected):**
- 2015: p_adj = 2.71 × 10⁻⁸ ✓ significant
- 2016: p_adj = 3.49 × 10⁻⁵ ✓ significant
- 2017–2023: all p_adj > 0.48 → not significant

**Yearly means (selected):**
| Year | Beauty mean | Sports mean |
|---|---|---|
| 2015 | 0.580 | 0.555 |
| 2019 | 0.586 ← peak (both) | 0.592 ← peak (both) |
| 2021 | 0.526 ← Beauty trough | 0.535 |
| 2022 | 0.529 | 0.522 ← Sports trough |
| 2023 | 0.537 | 0.548 |

**Figure:** `outputs/figures/rq2_trend.{pdf,png}`
**Table:** `outputs/tables/rq2_yearly_means.{csv,tex}`
**Stats:** `outputs/stats/rq2.json`, `rq2.csv`, `rq2_interaction_ols.csv`, `rq2_pairwise.csv`

### Notes for the paper
- **This conclusion changed from the pre-correction analysis**, where the interaction
  term was not significant (p = 0.162) and the categories looked statistically
  indistinguishable in trajectory. After fixing the dedup/cleaning issues, the same
  test on the corrected data finds a significant interaction (p = 0.014). State both
  the result and the caveat below - don't headline this as a strong divergence.
- **Why this is a weak effect despite significance:** the interaction model explains
  only 0.063% of variance (R² = 0.00063), and the per-year pairwise tests find a
  significant gap only in 2015–2016. The most defensible statement is: *"Beauty's
  average sentiment declined modestly faster than Sports' over the full period, but
  this is driven by an early-years gap (2015–2016) rather than a trend that
  progressively widens - from 2017 onward, the two categories are statistically
  indistinguishable year-by-year."* Avoid claiming either a clean "convergence" or a
  clean "divergence" - the evidence supports neither extreme.
- **Why Mann-Kendall and OLS disagree:** OLS operates on 189k individual reviews and
  is significant due to sample size, not effect size. Mann-Kendall on 9 annual means
  is the appropriate test for monotonic trend in a time series - and it does not
  confirm a trend in either category. State both and explain the discrepancy.
- **The 2019 peak / 2020–2022 dip** is the most interesting pattern, and it's now
  slightly asymmetric: both categories peak together in 2019, but Beauty bottoms out
  in 2021 while Sports bottoms out in 2022 - a one-year lag between categories.
  Supply chain disruptions, delayed shipping, and out-of-stock substitutions during
  COVID-19 may explain the dip; the one-year offset is contextual speculation worth
  one cautious sentence, not a strong claim.
- **Supports novelty argument:** The non-monotonic pattern (peak → dip → recovery)
  is a system-level signal worth highlighting regardless of the trend-test outcome.

---

## RQ3 - Star Ratings vs VADER Agreement & VADER Validation

**Question:** How well do star ratings and VADER sentiment scores agree,
and in which cases do they diverge?

**Method:** Spearman ρ (rating vs compound) with Fisher-z 95% CI. Confusion matrix
of `vader_label` vs `star_label`: accuracy, macro-F1, Cohen's quadratic-weighted
kappa. Majority-class baseline accuracy. Chi-square on per-category agreement rates.
Per-length-bucket accuracy with bootstrap 95% CI. Mismatch mining (rating 4–5 but
compound ≤ −0.5; rating 1–2 but compound ≥ 0.5).

Star-rating-derived labels (`star_label`) are used throughout as **reference labels**,
not as manually-annotated ground truth - "accuracy" below means agreement with this
proxy, not objective correctness. This distinction matters because both signals come
from the same reviewer at the same time and could share systematic biases.

### Answer
VADER shows moderate agreement with star-rating-derived labels (ρ = 0.510,
κ = 0.597, accuracy = 80.9%) - only about 3.9 percentage points above the
majority-class baseline (77.0%, "always predict positive"), because the class
distribution is itself heavily skewed toward positive. VADER is strong on positive
reviews (F1 = 0.90) but performs poorly on neutral reviews (F1 = 0.12) - it essentially
cannot reliably separate neutral from positive/negative. The Beauty-vs-Sports
difference is small and not one-directional: exact-label accuracy is slightly higher
for Sports (81.1% vs 80.6%), while weighted kappa and rank correlation are slightly
higher for Beauty (κ = 0.604 vs 0.588; ρ = 0.528 vs 0.493) - the category difference
is practically negligible even though the chi-square test is significant (an artefact
of the large sample). Contrary to the standard assumption that VADER (designed for
short social-media text) should struggle with short reviews, accuracy here **decreases**
with review length - the longest reviews are the hardest, not the shortest. VADER is
also systematically over-optimistic: low-star/high-VADER mismatches outnumber the
reverse by roughly 3–4×.

### Key numbers

**Spearman correlation:**
| Scope | ρ | p | 95% CI |
|---|---|---|---|
| Overall | 0.510 | < 1 × 10⁻³⁰⁰ | [0.507, 0.514] |
| Beauty | 0.528 | < 1 × 10⁻³⁰⁰ | [0.523, 0.532] |
| Sports | 0.493 | < 1 × 10⁻³⁰⁰ | [0.488, 0.498] |

**Classification metrics:**
| Scope | Accuracy | Macro-F1 | κ (quadratic) | Majority-class baseline |
|---|---|---|---|---|
| Overall | 0.809 | 0.542 | 0.597 | 0.770 (+3.9pp) |
| Beauty | 0.806 | 0.545 | 0.604 | 0.756 (+5.1pp) |
| Sports | 0.811 | 0.540 | 0.588 | 0.784 (+2.7pp) |

- Chi-square (Beauty vs Sports agreement): χ² = 6.90, p = 0.0086 → significant,
  but the underlying accuracy gap is < 0.5 percentage points - a large-N artefact,
  not a meaningful category effect.

**Per-class performance (overall):**
| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Positive | 0.881 | 0.929 | 0.904 | 145,974 |
| Negative | 0.672 | 0.540 | 0.599 | 29,954 |
| Neutral | 0.135 | 0.115 | **0.124** | 13,659 |

**Agreement by review length (key VADER-limitation finding):**
| Token length | n | Accuracy | 95% CI |
|---|---|---|---|
| ≤5 tokens | 12,448 | 82.3% | [81.7%, 83.0%] |
| 6–20 tokens | 70,444 | **82.7%** | [82.4%, 83.0%] |
| 21–50 tokens | 62,507 | 80.6% | [80.3%, 80.9%] |
| >50 tokens | 44,188 | **78.0%** | [77.6%, 78.4%] |

**Mismatch counts (saved for qualitative analysis):**
| Direction | Beauty | Sports |
|---|---|---|
| High star (4–5★), low VADER (≤ −0.5) | 725 | 806 |
| Low star (1–2★), high VADER (≥ +0.5) | 2,896 | 2,383 |
| **Ratio** | **4.0:1** | **3.0:1** |

**Figures:** `outputs/figures/rq3_confusion_beauty.{pdf,png}`,
            `rq3_confusion_sports.{pdf,png}`,
            `rq3_agreement_by_length.{pdf,png}`
**Stats:** `outputs/stats/rq3.json`, `rq3.csv`
**Qualitative:** `outputs/qualitative/mismatches_Beauty.csv`,
                 `outputs/qualitative/mismatches_Sports.csv`

### Notes for the paper - VADER limitations (F4/F5)
- **Reference labels, not ground truth:** Make this explicit in the Methodology.
  Star ratings are themselves a noisy, single-dimension proxy for sentiment (e.g. a
  4★ review can contain harsh criticism of a minor flaw). Disagreement between VADER
  and stars is not automatically a VADER "error."
- **Baseline-relative framing:** VADER's headline 80.9% accuracy sounds strong in
  isolation but is only ~3.9pp above trivially guessing the majority class. This
  matters because the *useful* signal VADER adds is small in absolute terms - most of
  its apparent accuracy comes from the corpus being mostly positive, not from VADER
  correctly resolving hard cases.
- **The neutral class is where VADER actually fails:** F1 = 0.12 vs 0.90 for positive.
  This is a stronger and more precise version of "VADER struggles with subtlety" than
  a single overall accuracy number conveys - lead with this in the Discussion.
- **The length finding challenges the canonical assumption** that VADER fails on short
  texts (designed for tweets). On Amazon reviews, accuracy *decreases* with length.
  Short reviews ("love it!", "garbage") are unambiguous - VADER handles them well.
  Long reviews discuss multiple product aspects with qualified language and mixed
  sentiment that VADER's single compound score cannot capture.
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
- **κ = 0.597** falls in Landis & Koch's "moderate" band (0.41–0.60), just below the
  substantial boundary. Frame as "moderate agreement" - don't round up to "substantial."
- **Beauty vs Sports - do not claim one category agrees "better":** accuracy slightly
  favours Sports, kappa and rank correlation slightly favour Beauty. State both metrics
  and conclude the category effect is small and not one-directional, rather than
  picking the metric that tells a cleaner story.

---

## RQ4 - Distinctive Words

**Question:** What words are most distinctive of highly positive vs highly negative
reviews in each category?

**Method:** Weighted log-odds ratio with informative Dirichlet prior (Monroe, Colaresi
& Quinn 2008, "Fightin' Words"). Positive/negative defined by star rating (4–5 vs 1–2★),
NOT by VADER, to keep RQ4 independent of the tool audited in RQ3. TF-IDF reported
as secondary descriptive cross-check. Unigrams + bigrams, min document frequency 10,
stopwords removed *except* negations (no/nor/not), apostrophe-aware tokenisation so
contractions stay intact (see Revision notes above).

### Answer
Both categories share a common positive vocabulary drawn from Amazon's platform culture.
Their negative vocabularies diverge in a domain-meaningful way: Beauty negatives focus
on product efficacy, refunds, and now-correctly-preserved negation phrases
(*doesn't work, didn't, did not, don't buy*); Sports negatives focus on physical
durability failure (*broken, broke, garbage, useless, don't waste*). Cross-category
vocabulary confirms distinct "review cultures": Beauty is sensory and emotional;
Sports is functional and technical.

### Key words (top 10 per side by z-score)

**Within Beauty - Positive vs Negative:**
- Positive (4–5★): *great, love, works, perfect, easy, best, stars, nice, amazing, great product*
- Negative (1–2★): *not, waste, disappointed, waste money, money, star, return, did not, not worth, not buy*
  - note `doesn't work` and `didn't` now also appear intact in the full top-25 (see
  `rq4_words_beauty.csv`), replacing the broken `didn work` / `doesn work` fragments
  from before the apostrophe fix.

**Within Sports - Positive vs Negative:**
- Positive (4–5★): *great, love, easy, perfect, good, stars, nice, works, price, great product*
- Negative (1–2★): *not, broke, disappointed, return, waste, poor, waste money, cheap, not worth, poor quality*
  - `don't waste` and `don't buy` now appear intact (full list in `rq4_words_sports.csv`).

**Cross-category - Beauty vs Sports:**
- Beauty: *hair, skin, scent, smell, smells, brush, product, face, love, makeup*
- Sports: *bike, fit, bag, comfortable, water, fits, ball, tent, camping, weight*
  - the HTML `br` artefact present in the previous run's cross-category list is gone.

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
  evidence that Amazon's review culture drives aggregate sentiment (RQ1 near-null
  result), not product domain. The category signal only emerges in negative reviews
  and in the cross-category comparison.
- **Negation is now handled correctly - show the before/after as a methods point:**
  the pre-correction Beauty negative list contained `didn work`, `doesn work` - broken
  fragments caused by inconsistent apostrophe encoding in the source data, *not* a
  genuine linguistic pattern. Normalising apostrophes before tokenising fixed this; the
  corrected list shows the intact, more interpretable forms `doesn't work`, `didn't`,
  `don't buy`, `don't waste`. This is worth one sentence in the Methodology as a
  concrete illustration of why text-cleaning choices matter for word-level analysis.
- **Sports durability narrative:** Negative Sports words (*broken, broke, garbage,
  useless*) tell a coherent story: physical products failing structurally. This is
  domain vocabulary VADER's lexicon does not cover - connecting RQ4 back to RQ3.
- **Beauty efficacy narrative:** Negative Beauty words (*waste, waste money, return,
  disappointed, not worth, not buy*) reflect efficacy disappointment and consumer
  action (refunds/returns) rather than physical failure - a different failure mode
  than Sports.
- **Cross-category novelty:** The sensory/emotional (Beauty) vs functional/technical
  (Sports) split is the strongest "review culture" evidence in the paper. Beauty
  reviewers use affective vocabulary (*scent, smell, skin, hair*); Sports reviewers
  use functional/spec vocabulary (*fit, comfortable, durable, weight, camping*). This
  has implications for any downstream NLP model trained on pooled Amazon reviews -
  domain transfer may require category-specific lexica.
- **"stars" confound (see RQ3):** The word "stars" appearing in positive reviews
  suggests some reviewers copy their star rating into the text ("5 stars, great
  product"). Mention alongside the RQ3 VADER-limitation discussion.

---

## Summary table - all RQ answers

| RQ | One-sentence answer | Key stat |
|---|---|---|
| RQ1 | Beauty and Sports have nearly identical sentiment distributions; medians differ slightly, means don't. | Cliff's delta = 0.0095 (negligible); median-diff CI excludes zero, mean-diff CI doesn't |
| RQ2 | Neither category shows a significant monotonic trend (Mann-Kendall); Beauty's review-level slope is significantly steeper than Sports' (interaction p = 0.014), but the gap is concentrated in 2015–16, not a progressively widening divergence. | MK p > 0.11 both cats; interaction p = 0.014, R² = 0.0006 |
| RQ3 | VADER agrees moderately with star-rating labels (κ = 0.60) but barely beats the majority-class baseline (+3.9pp) and is far weaker on neutral reviews (F1 = 0.12) than positive (F1 = 0.90); it degrades on long (not short) reviews. | ρ = 0.510; accuracy 80.9% vs baseline 77.0% |
| RQ4 | Positive vocabulary is platform-generic; negative vocabulary is domain-specific and now correctly preserves negation (doesn't, didn't, don't); Beauty is sensory/emotional, Sports is functional/technical. | Top words by log-odds z-score |

---

## Evidence supporting novelty / Web Science motivation
*(Flag these in the paper's Introduction and Discussion)*

1. **Near-identical aggregate distributions, but a small and significant
   differential trend (RQ1 + RQ2):** Aggregate sentiment level is largely
   category-independent, but the *trajectory* over time is not quite - Beauty
   declined modestly faster than Sports, concentrated in an early-years (2015–16)
   gap. A Web Science contribution: platform-level positivity dominates absolute
   sentiment, but category-specific dynamics still show up in finer-grained temporal
   analysis. Avoid overstating this as a clean "divergence" story - it is a real but
   small effect, not visible year-by-year after 2016.

2. **Non-monotonic temporal pattern (RQ2):** The 2019 peak and 2020–22 dip is a
   system-level signal visible across both categories, with a one-year lag between
   Beauty's trough (2021) and Sports' trough (2022) - consistent with an external
   shock (COVID-19 supply chain disruption) propagating slightly differently across
   product categories. Demonstrates that aggregate sentiment in UGC can serve as a
   proxy signal for real-world disruptions.

3. **VADER length finding (RQ3):** Challenges the standard assumption about VADER's
   domain applicability. The failure mode on Amazon is not short text (as on Twitter)
   but long, mixed-sentiment text - and even more strikingly, VADER's accuracy
   advantage over a trivial majority-class baseline is small (+3.9pp) and concentrated
   entirely in the positive class. This is a new, more precise empirical data point
   for the sentiment-analysis methods literature than a single accuracy number.

4. **Review culture vocabulary (RQ4):** Despite near-identical aggregate distributions,
   the two categories have measurably different lexical cultures, and this signal
   survives a methodological correction (negation-preserving tokenisation) that
   changed the specific words but not the qualitative pattern. Functional vs emotional
   framing is a category-level property of product reviews, with implications for
   cross-domain NLP model transfer.
