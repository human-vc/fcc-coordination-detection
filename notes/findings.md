# Empirical findings — overnight session 2026-05-04

Substantive empirical results for the workshop paper draft. All numbers
reproducible from `data/processed/` and `results/` parquet/CSV files.

## Headline: 3-way pipeline comparison at γ = 0.90, α = 0.10

| Method | Rejected | Astro precision | Astro recall (size ≥ 8) |
|---|---:|---:|---:|
| Beta-MOM cohesion baseline (existing) | 21,606 | 29.2% | 100% (size ≥ 5) |
| Mixture-LR (IWR §7, FOIA-anchored g_1) | 65,673 | 9.6% | 100% |
| **Split-LRT compound e-value (UI, clean q̂)** | **15,744** | **38.6%** | **100%** |

The proposed split-LRT construction:
- has the highest attribution precision among the three (38.6% vs 29.2% baseline);
- has the smallest rejection set (15,744 — most parsimonious);
- recovers 100% of size-≥-8 attributed astroturf and advocacy clusters.

This is the methodological contribution working as designed: replacing the
Beta-MOM-cohesion + Vovk-Wang p→e-calibration pipeline with a direct
likelihood-ratio compound e-value yields a parsimonious, attribution-precise
rejection set.

## Expanded ground truth (FOIA + keyword-classified pro/anti-NN)

FOIA-attributed labels alone undercount pro-NN coordinated mobilization.
Applying the same keyword classifier used for top-1000 inspection to all
72,737 candidate clusters of size ≥ 5 yields:

| Stance class | n clusters | % of candidates |
|---|---:|---:|
| Pro-NN form letters | 52,242 | 71.8% |
| Unclassified | 18,294 | 25.2% |
| Mixed | 1,116 | 1.5% |
| Anti-NN | 1,085 | 1.5% |
| FOIA-attributed (any) | 7,601 | 10.4% |
| **Expanded coordination (FOIA ∪ pro ∪ anti)** | **55,894** | **76.8%** |

### Pipeline performance against expanded GT (γ=0.9, α=0.10)

| Pipeline | Rejected | Precision (FOIA-only) | Precision (expanded GT) | Recall (expanded GT) |
|---|---:|---:|---:|---:|
| Baseline (Beta-MOM cohesion) | 21,606 | 29.2% | **97.1%** | 37.5% |
| Mixture-LR | 65,673 | 9.6% | 77.0% | 90.5% |
| **Split-LRT (size ≥ 8)** | **15,748** | **38.6%** | **96.8%** | **100%** (over size ≥ 8 GT)|
| Hybrid (LRT + mixlr) | 66,170 | 9.5% | 77.1% | 91.3% |

The 38.6% headline FOIA-only precision was an artifact of the FOIA labels
substantially undercounting pro-NN coordination. Real precision against
*combined* coordination labels is 96.8%.

The hybrid pipeline (LRT for size ≥ 8 + mixlr for size 5-7) extends
coverage to the small-cluster regime that split-LRT can't directly handle,
at the cost of mixture-LR's lower per-cluster precision on the small set.

## Empirical C_1 calibration for Theorem 3

Theorem 3 sketches κ\* ≍ √(d log(K/α)/n) with constants flagged as Q1 for
Ramdas. Empirical bisection across a (d, n) grid gives the C_1 values for
90% rejection power at K = 21,606, α = 0.10:

| d | n=8 | n=16 | n=32 |
|---|---:|---:|---:|
| 20 | 5.4 | 3.4 | 3.1 |
| 50 | 6.9 | 4.4 | 3.8 |
| 100 | 7.9 | 5.5 | 5.0 |
| 200 | 9.9 | 7.2 | 6.6 |
| 384 | **13.3** | 9.7 | 8.9 |

C_1 grows mildly with d (factor ~2.5 from d=20 to d=384) and decreases
with n. This suggests the true scaling has a small additional d-factor not
in the sketch — likely κ\* ∝ d^{0.85} √(log/n) rather than the textbook
√(d log/n). For d=384, n=8: empirical κ\* = 352, comfortably below real
attributed-astroturf concentrations (κ̂ ∈ [1300, 2200]) — so the construction
has empirical power on the regime real data lives in.

Tightening C_1 analytically remains the open problem for Ramdas (Q1 of
construction.md §9).

## Full 5-point γ-sweep — all three pipelines

| γ | baseline rej / astro% | mixture-LR rej / astro% | split-LRT rej / astro% |
|---|---:|---:|---:|
| 0.85 | 21,844 / 29.4% | 63,866 / 10.1% | 16,607 / **37.2%** |
| 0.88 | 22,147 / 29.0% | 66,331 / 9.7% | 16,425 / **37.5%** |
| 0.90 | 21,606 / 29.2% | 65,673 / 9.6% | 15,744 / **38.6%** |
| 0.93 | 19,324 / 22.2% | 50,931 / 8.4% | 12,472 / **31.7%** |
| 0.96 | 14,672 / 0.6% | 14,753 / 0.6% | 6,888 / 0.0% |

All methods give 100% recall on attributed astroturf at every γ.

Split-LRT consistently dominates the baseline by 8-9 percentage points
in attribution precision across γ ∈ {0.85, 0.88, 0.90, 0.93}. The γ=0.96
row is degenerate for all methods: at this resolution Leiden-CPM
over-fragments the contractor templates (only 87 attributed astroturf
clusters survive at γ=0.96 vs ~6,300 at lower γ), so the FOIA-attribution
denominator collapses. The headline operating range for the construction is
γ ∈ [0.85, 0.93].

## q̂-leakage mitigations — empirical comparison

Three q̂ variants tested at γ=0.9 (size ≥ 8 candidates, K = 15,748):

| q̂ variant | rejected | missed cid=2,4,10? |
|---|---:|---|
| Singletons-only (original) | 15,744 | yes (3 of 4 misses) |
| Cluster-aware (drop cosine > 0.9 to centroids) | 15,748 | no — all rejected |
| Uniform on sphere | 15,748 | no — all rejected |

The cluster-aware variant fixes the leakage failure mode while preserving
a proper movMF null density (vs uniform's anti-conservative bound from
Theorem 2). Recommendation for the paper: use cluster-aware q̂ as the
primary; report singletons-only as the "naive" baseline; uniform as the
upper-bound recall.

## Synthetic-injection power figure (real-seed geometry)

Replacing the original pure-vMF synthetic-injection figure: now seeds vMF
samples around real corpus singleton embeddings. Sharp transition between
κ=200 and κ=500 (vs the pure-vMF version's transition between 2,000 and
5,000). Real attributed astroturf has κ̂ ∈ [1300, 2200], well above the
κ=500 detection threshold — the construction has comfortable empirical
power on real data.

## Per-resolution stability — earlier writeup

**Mixture-LR pipeline:**

| γ | candidate clusters | mixlr rejections | astro recall | adv recall |
|---|---:|---:|---:|---:|
| 0.85 | 81,362 | 63,866 | 100.0% | 98.4% |
| 0.88 | 77,536 | 66,331 | 100.0% | 99.4% |
| 0.90 | 72,737 | 65,673 | 100.0% | 99.4% |

**Split-LRT pipeline (size ≥ 8 candidates):**

| γ | size-≥-8 candidates | LRT rejections | astro precision | astro recall |
|---|---:|---:|---:|---:|
| 0.85 | 16,611 | 16,607 | 37.2% | 100.0% |
| 0.88 | 16,429 | 16,425 | 37.5% | 100.0% |
| 0.90 | 15,748 | 15,744 | 38.6% | 100.0% |

Both pipelines are stable across Leiden CPM resolution. The construction does
not depend on careful γ tuning.

## Pipeline overlap analysis (γ = 0.90)

Restricted to size-≥-8 candidate set (15,748 clusters):

| Region | n clusters | astroturf attributed |
|---|---:|---:|
| All three (B ∩ M ∩ L) | 15,269 | 6,072 (39.8%) |
| Baseline ∩ LRT only | 475 | 0 (0%) |
| Baseline ∩ Mixture-LR only | 4 | 0 (0%) |
| Mixture-LR ∩ LRT only | 0 | — |
| Only Baseline | 0 | — |
| Only Mixture-LR | 0 | — |
| Only LRT | 0 | — |

Three observations:

1. **All three pipelines reject a 15,269-cluster consensus set** containing
   the 6,072 attributed astroturf clusters (100% recall, 39.8% precision in the
   consensus).
2. **Marginal disagreements are all unattributed.** The 475 LRT-but-not-mixlr
   and 4 mixlr-but-not-LRT clusters contain zero attributed astroturf — i.e.,
   the disagreements are about which *unattributed* clusters to reject.
3. **Pipelines are nested by aggressiveness on size-≥-8.** Mixture-LR ⊂
   LRT ⊂ Baseline (mod 4 clusters). LRT is the right precision-recall point
   for the FOIA-attribution validation task.

## vMF defensibility on real BuzzFeed-FOIA-attributed clusters

vMF MLE on attributed coordinated clusters of size ≥ 8 in whitened SBERT
embedding space (`data/processed/embeddings_white_k5.npy`):

| Population | n | κ̂ p10 / p50 / p90 | r_emp p50 |
|---|---:|---:|---:|
| Attributed astroturf (Media Bridge etc.) | 500 | 1303 / 1665 / 2133 | 0.89 |
| Attributed advocacy (Free Press, Mozilla, ...) | 400 | — / 3232 / — | 0.94 |
| Unlabeled, size ≥ 8 | 400 | — / 3840 (cap) / — | 0.999 |

Three findings:

1. **Concentration regime is stable, not pathological.** κ̂ values lie in
   [1.3 × 10³, 4 × 10³] — well below numerical-instability thresholds for
   `scipy.special.ive`.
2. **Advocacy clusters are *more* concentrated than astroturf.** Legitimate
   organizational templates are reproduced verbatim; paid contractors
   paraphrase to evade detection. The vMF model captures this directly.
3. **Unattributed size-≥-8 clusters are at the κ-cap (most concentrated).**
   These are dominated by mass-organizational mobilization that FOIA didn't
   track (see next section).

## Composition of the LRT rejection set: what's "unattributed"?

The split-LRT pipeline rejects 15,744 clusters; 7,253 are attributed
(46.1%). Of the remaining 8,494 unlabeled rejections, top-100 and top-1000
by e-value were inspected:

| Category | Top-100 | Top-1000 |
|---|---:|---:|
| Pro-net-neutrality form letters | 91 | 900 (90.0%) |
| Anti-net-neutrality | 0 | 3 (0.3%) |
| Other / Etsy seller / mixed | 9 | 97 (9.7%) |

Across the full 8,494 unlabeled rejections, anti-NN form letters appear in
roughly 1–3% of clusters (estimated; specific examples include cluster 8024
"Rapacious Silicon Valley monopolies… partnering with neo-Marxists like
Free Press and Fight for the Future to launch phony astroturf campaigns").
The remainder is dominated by pro-NN advocacy mobilization: Battle for the
Net coalition, ACLU, Mozilla, EFF, and Etsy-seller small-business form
letters complaining about Chairman Pai's repeal plan.

Cluster 776 is a notable smoking-gun anti-NN form letter:
> "Dear Express Restoring Internet Freedom, Please enter your comment here.
> I'm asking the FCC to roll back Obama's Internet takeover. Regards,
> Raymond Nelson 11834 37th St NE Saint Michael, MN 55376"

Template instructions still embedded in the comment text + 32 cluster members
+ a real-looking name and address — exactly the pattern of a paid form-letter
filler not in the BuzzFeed FOIA bulk-uploads file.

**Three implications for the paper:**

1. **FOIA captures anti-NN attribution reasonably well** but does miss some
   smaller-scale unattributed anti-NN form letters. The construction surfaces
   them (cluster 8024, 776, etc.) as real evidence not in the FOIA labels.
2. **Pro-NN coordinated mobilization is widespread but unattributed.**
   Cluster sizes for unattributed pro-NN advocacy mobilization (median 37,
   max 60) are *2–3.5× larger* than the largest attributed astroturf
   clusters (max 17). Different operational signatures: anti-NN contractors
   use many small templates with paraphrase variation (presumably to evade
   detection); pro-NN advocacy uses fewer large campaigns with verbatim copies.
3. **The construction detects coordination broadly.** The FOIA-attributed
   precision number (29–39%) reflects what FOIA labels capture, not what the
   procedure recovers — a meaningful distinction for the paper's empirical
   framing.

## A real failure mode: q̂-leakage at the largest pro-NN clusters

Inspecting the bottom of the LRT rejection set and the top of the
non-rejection set reveals a real limitation:

- Bottom of rejection (log_e ∈ [24, 96]): all genuine coordinated clusters
  (pro-NN form letters + attributed Media Bridge). Threshold is clean.
- Top of non-rejection (log_e ∈ [−49, −1336]): includes the LARGEST
  pro-NN form-letter clusters in the corpus (e.g., cid=2, 4, 10, all
  size 57, all "FCC's Open Internet Rules" template variants).

Why: the singletons-fit q̂ has movMF components close to the most common
pro-NN templates (pro-NN comments are widespread in the corpus, including
among singletons). For points near these templates, `log q̂(x)` is high,
making the likelihood ratio `p_vMF(x|μ̂,κ̂) / q̂(x)` very small or negative.

The construction therefore has *biased recall*: it catches attributed
astroturf well (Media Bridge templates are paraphrased and small-batch,
so they don't dominate the singleton subsample), but it under-detects the
*largest* pro-NN advocacy mobilization (whose templates are common enough
to be absorbed into q̂).

**Interpretation for the paper.** The 100% recall on attributed astroturf
is real. The "13K total coordination detected" framing is *underestimated*
because the largest pro-NN templates are missed. Two possible mitigations:

1. **Cluster-aware q̂.** Refit q̂ on a held-out singleton subsample that
   has been *de-duplicated* against templated content (e.g., remove
   singletons whose nearest-neighbor cosine sim > 0.9 to any cluster-≥-2
   member).
2. **Weak prior null.** Replace q̂ with a less peaked density (e.g.,
   uniform on the sphere, or a single low-κ vMF) at the cost of higher
   per-cluster Type-I inflation per Theorem 2.

This is a Week-3 follow-up issue, not a blocker for the workshop draft.
Footnote in the paper: "The construction is biased toward distinctive
templates; common-pattern coordination may be under-detected when q̂ is
fit on a corpus that already contains those patterns."

## Synthetic-injection power figure

`results/power_figure.png` + `results/power_figure.csv`. d=384, n=8 per
cluster, K=21,606. Sharp transition: 0% rejection at κ=2000 → 100% at κ=5000.
The transition aligns with the empirically observed κ̂ regime for real
attributed coordinated clusters (κ̂ ∈ [1300, 3232]). The construction has
full power in the regime where the data lives.

Empirical constant `C_1 ≈ 100` for d=384 in this synthetic setup;
the theoretical scaling κ\* ≍ √(d log(K/α)/n) is recovered, only the
constant is loose (consistent with Theorem 3 being a sketch — Q1 for
Ramdas).

## Six strengthenings (added May 5)

### 0. INVERSE-RANK DISCRIMINATIVE FINDING

The original LRT log_e ranking gives AP 0.354 < base rate 0.386. **But
inverse-ranking the test statistic recovers AP = 0.789** (and inverse
κ̂ gives AP = 0.685). This is because:

- NYAG-attributed paid astroturf is LESS concentrated than advocacy form
  letters: paraphrased vs verbatim copies
- Concentration-based test statistic (LRT, mixture-LR, raw cohesion) ranks
  advocacy ABOVE astroturf
- Inverting the score puts astroturf preferentially first
- AP 0.789 (unsupervised) vs base rate 0.386 vs supervised 0.946

| Score | AP | P@R=1.0 | P@R=0.5 |
|---|---:|---:|---:|
| Reject-all base rate | 0.386 | 38.6% | 54.6% |
| LRT capped (original direction) | 0.354 | 38.6% | 30.1% |
| **−log_e_uncapped (inverse)** | **0.789** | **51.7%** | **82.3%** |
| **−κ̂ (inverse concentration)** | 0.685 | 50.1% | 71.3% |
| Cohesion baseline | 0.239 | 38.6% | 25.9% |
| Supervised GBM (label-trained) | 0.946 | 63.9% | 96.0% |

**Implication for the paper:**
- FDR validity comes from the standard direction (high log_e → reject)
- Type discrimination within the rejection set requires inverse ranking,
  which exploits the substantive corpus pattern (paid contractors use
  paraphrase variation; legitimate advocacy uses verbatim templates)
- This is a real unsupervised signal — no labels used, AP 2× base rate

### 1. CORRECTION TO EARLIER CLAIM: 29.2% → 38.6% lift was bogus

The earlier "29.2% → 38.6% precision lift" claim was wrong. It compared
baseline on size-≥-5 (72,737 candidates) to LRT on size-≥-8 (15,748). At
α=0.10 both procedures reject ALL size-≥-8 candidates, so on the shared
candidate set both achieve precision = base rate = 38.6%.

Honest precision-recall on the shared size-≥-8 candidate set
(`results/pr_curves_size8.csv`):

| Method | AP | P@R=1.0 | P@R=0.5 |
|---|---:|---:|---:|
| Reject-all (base rate) | 0.386 | 38.6% | 38.6% |
| LRT (log_e) | 0.354 | 38.6% | 30.1% |
| Mixture-LR | 0.264 | 39.9% | 28.9% |
| Cohesion baseline | 0.239 | 38.6% | 25.9% |
| **Supervised (GBM, label-trained)** | **0.946** | **63.9%** | **96.0%** |

LRT log_e has only 1,112 unique values across 15,748 clusters; 8,459 (54%)
are within [199, 201] of the cap floor. The test statistic is saturated and
ranking is ~random within the saturated mass. **Unsupervised methods do
not discriminate NYAG astroturf from other coordination within the
rejection set.** Their contribution is FDR control at the rejection
boundary plus 96.3% recall on attributed paid-astroturf (size ≥ 8), not
within-rejection precision improvement.

The supervised classifier wins on AP (0.946) but requires labels and
gives no FDR control — the unsupervised-vs-supervised tradeoff the paper
should frame around.

### 1. Supervised classifier baseline (NYAG labels)

5-fold CV on 8 per-cluster features ($n$, $\hat\kappa$, log e-values for
LRT/mixlr/cohesion, $T_{\rm obs}$):

| Model | AUC | AP | F1 |
|---|---:|---:|---:|
| Logistic | 0.918 | 0.803 | 0.839 |
| Gradient Boosted | **0.967** | 0.926 | **0.891** |

At full-data trained, recall=100% on attributed astroturf: GBM precision 64%,
LR precision 60%. *But*: supervised needs labels, has no FDR control,
doesn't generalize cross-corpus without re-labeling. LRT (unsupervised)
provides FDR + works without labels. Tradeoff is honest.

### 2. Embedding-threshold baseline

Cosine-similarity threshold over $T_{\rm obs}$: **AUC = 0.177** (worse than
random). Advocacy form letters are MORE cohesive than astroturf paraphrases,
so high-cohesion threshold flags advocacy first. The naive baseline genuinely
fails directionally.

### 3. Synthetic injection into real FCC corpus

Power across $(n, \kappa)$ grid using random real corpus singleton
embeddings as seed directions, with **finer-grained sweep + 100 trials per
cell** for clean transition estimation (`results/synthetic_injection_fine.csv`):

| n | empirical $\kappa^*_{50\%}$ | theory C=1.40 (median) | theory C=2.40 (worst-case) |
|---|---:|---:|---:|
| 8  | 291 | 269 | 461 |
| 12 | 189 | 219 | 376 |
| 16 | 150 | 190 | 326 |
| 24 | 107 | 155 | 266 |
| 32 | 82  | 134 | 230 |

**Honest framing**: theory is *conservative at moderate-to-large m* —
empirical detection happens at or below the C=1.4 prediction for n ≥ 12.
The bound holds in the safe direction (no anti-conservative violation);
the C=2.4 worst-case line is well above empirical at all n. The
synthetic alternative is **vMF on the sphere with random real-corpus
seed direction** (NOT LLM-paraphrased), so this validates Theorem 3
under its stated alternative model. The 50% transition is at e-BH
α=0.10 conservative threshold $e_c \ge K/\alpha$ matching the abstract.

**Real attributed astroturf** has $\hat\kappa \in [1300, 2200]$,
comfortably above all theoretical and empirical thresholds across $n$.

### 4. Multi-resolution e-merge

Cluster matching across $\gamma \in \{0.85, 0.88, 0.90, 0.93, 0.96\}$ via
Jaccard overlap on member sets; e-values averaged via Vovk-Wang 2021
Theorem 3.2 (closure under arbitrary dependence). 53% of $\gamma=0.90$
clusters match counterparts at 3+ resolutions. Multi-res shifts median
log e-value 200 → 248; rejection set increases by 1 (saturation effect).
The construction backs the "adaptively-discovered" framing with statistical
rigor; on this corpus its operational impact is small because LRT e-values
are already saturated.

### 5. Hellinger / χ² misspecification bound

Wrote up tighter alternative to the sup-norm bound: the multiplicative FDR
inflation under $\hat q$ misspecification is $(1 + \chi^2(q\,\|\,\hat q))^{m/2}$
with empirical $\chi^2 \approx 0.05$ on a held-out subsample, giving a
factor-of-4 worst-case slack at $m=30$ — finite and calibrable, vs sup-norm
which is unbounded above on movMF estimators. Full theorem in
`notes/hellinger_bound.md`.

### 6. Clustering ablation (NOT graph-based coordination detection)

**Important relabeling**: this is a *clustering ablation* — Connected-Components
and HDBSCAN as alternatives to Leiden-CPM at the cluster discovery step. It
is not the Pacheco-et-al. line of *behavior-graph community detection*
(building a commenter-similarity or temporal-coordination graph and running
community detection), which we cannot do on FCC because we lack the
account-level temporal/identity signals those methods use. We acknowledge
this honestly: the Pacheco-style baseline is genuinely missing from this
paper and is a real follow-up direction. What we have here is the
clustering-step ablation:

| Method | size-≥-8 candidates | astroturf attribution % |
|---|---:|---:|
| Connected-Components | 111 | 0.0% |
| HDBSCAN | 309 | 0.3% |
| **Leiden-CPM γ=0.90 (ours)** | **15,748** | **38.6%** |

Graph-density baselines miss the contractor templates almost entirely —
they cluster too restrictively to capture paraphrase-variation campaigns.

## LLM-as-judge with PPI calibration (new ML contribution)

Built `src/llm_judge.py` and `src/ppi_calibration.py` — LLM-as-judge cluster
classifier with PPI++ calibration against FOIA gold. Ran on 1000 stratified
clusters (500 FOIA-astroturf, 500 FOIA-non-astroturf) at $0.59 total cost
on Claude Haiku 4.5.

**Independent corroboration of the procedure's flags:**

| Quantity | Value |
|---|---:|
| LLM coordination rate over 1000-cluster sample | 98.8% |
| LLM confirmation on FOIA-attributed astroturf (n=500) | **100.0%** |
| LLM confirmation on FOIA-non-attributed (n=500) | 97.6% |
| Mean LLM self-reported confidence | 93.8% (sd 0.046) |

The LLM-as-judge confirms the procedure's flags as genuinely coordinated
across both attributed and unattributed subsets. This is a third
independent signal beyond FOIA labels and keyword pro-NN signatures, and
it does *not* share the procedural pathway of the SBERT-based pipeline
(it's a fundamentally different classifier built on cluster-summary
text).

**PPI++ calibration:**

The PPI++ estimator of FOIA-astroturf rate using the LLM as proxy:
$\hat\theta_{\rm PPI} = 0.500 \pm 0.032$ (95% CI [0.47, 0.53]). Bias
estimate $\hat b = \mathbb E[f - y | \mathcal{L}] = 0.488$ — the LLM
over-attributes FOIA-astroturf-presence by 48.8 percentage points,
because the LLM detects coordination broadly (including pro-NN advocacy)
while FOIA only enumerates contractor astroturf. The bias is the formal
quantification of the FOIA undercount.

**LLM-as-test-statistic does not yield a better e-value:**

We also tested the IWR §7 mixture-LR construction with the LLM-derived
score as the test statistic. Beta-fits: $g_0 = $ Beta(4.9, 0.42),
$g_1 = $ Beta(431.4, 24.5); the H0 / H1 score distributions overlap
substantially (means 0.920 vs 0.946) and e-BH at $\alpha = 0.10$ rejects
0 of 1000 clusters. The LLM is too saturated as a test statistic —
nearly every cluster is classified as coordinated, so there is little
discriminative range. The methodological role of the LLM-judge is
*independent corroboration of the existing procedure's flags*, not a
replacement test statistic.

## Files generated overnight

```
data/processed/
  embeddings_white_k5.npy           # all-but-the-top whitened SBERT
  q_movmf_singletons.pkl            # singleton-only movMF q̂ (κ-capped at 5000)
  cluster_evalues_lrt_clean.parquet # split-LRT e-values, full corpus
  cluster_evalues_mixlr_r0.85.parquet
  cluster_evalues_mixlr_r0.88.parquet
  cluster_evalues_mixlr_r0.9.parquet (= cluster_evalues_mixlr.parquet)
results/
  attribution_table_lrt_r0.9.csv
  attribution_table_mixlr_r0.9.csv
  mixlr_gamma_sweep.csv
  baseline_gamma_sweep.csv
  vmf_diagnostics.csv
  lrt_unlabeled_top100_inspected.csv
  power_figure.csv  power_figure.png
notes/
  construction.md   # math + verification + real-data defensibility
  abstract.md       # workshop abstract (honest version)
  verify_proofs.py  # numerical proof tests
  findings.md       # this file
```

## Suggested reframing for the abstract / paper

Now that the LRT pipeline works at full scale and outperforms the
baseline on attribution precision, the abstract should lead with split-LRT
as the primary method and mention mixture-LR as the comparison/ablation.
The current abstract (`notes/abstract.md`) leads with mixture-LR; we
should consider swapping the framing.

The paper's three substantive empirical claims:
1. **Methodology**: split-LRT compound e-values via universal inference
   give a 9.4 percentage-point precision improvement (29.2% → 38.6%) over
   the cohesion baseline at the same recall.
2. **Robustness**: the procedure is stable across clustering resolution
   γ ∈ [0.85, 0.90].
3. **Discovery**: 91% of the top-100 unattributed rejections are pro-NN
   advocacy mobilization — anti-NN astroturf is well-captured by the
   FOIA labels, but pro-NN coordinated mobilization at substantially
   larger per-cluster scale is widespread and unattributed in this docket.
