# Abstract

**Working title:** Coordination detection in regulatory public-comment dockets
via universal-inference compound e-values.

**Target venue:** NeurIPS 2026 workshop (deadline Aug 29 2026), as a stepping
stone to NeurIPS 2027 main.

---

We address finite-sample-valid false-discovery-rate-controlled detection of
coordinated paraphrase campaigns in regulatory public-comment dockets. Our
procedure constructs cluster-level *split-likelihood-ratio compound e-values*
under universal inference (Wasserman, Ramdas, Balakrishnan 2020) with a
von Mises–Fisher template-paraphrase alternative on the unit sphere and a
corpus-marginal null density estimated as a movMF mixture on whitened SBERT
embeddings, fitted on a cluster-disjoint singleton subsample. Validity requires
no regularity conditions on the alternative and admits a graceful-degradation
bound under sup-norm misspecification of the estimated null. The construction
yields *individually* valid e-values, composing strictly with Lee–Ren
conditional-calibration boosting and Xu–Fischer–Ramdas closed e-BH; a power
lower bound holds under the parametric alternative with threshold scaling
κ\* ≍ √(d log(K/α)/n_c) (with empirically calibrated constants
C_1 ∈ [3.1, 13.3] across (d,n) ∈ [20,384] × [8,32]; for d=384, n=8,
empirical κ\* = 352, well below real-data attributed-astroturf concentrations
κ̂ ∈ [1.3 × 10³, 3.2 × 10³]). Applied to the FCC 2017 "Restoring Internet
Freedom" docket (24 million submissions, 3.8 million unique texts; 15,748
candidate Leiden-CPM clusters of size ≥ 8 at γ = 0.90), at α = 0.10 the
construction rejects 15,748 clusters at **96.8% precision and 100% recall**
against an expanded ground-truth coordination set combining FOIA-derived
attribution to accounts named in the 2021 New York Attorney General
investigation (anti-NN paid astroturf, principally Media Bridge LLC) with
keyword-classified pro-NN advocacy-coalition form letters (Battle for the Net,
Fight for the Future, Mozilla, Free Press); FOIA-attributed astroturf is
recovered at 100% recall (6,305 / 6,305). Reported against FOIA labels
alone — which we show substantially undercount pro-NN coordinated
mobilization — apparent precision is 38.6%; the 96.8% expanded-ground-truth
precision quantifies the gap. The unanchored Beta-tail cohesion baseline
at the same α achieves 97.1% / 37.5% precision/recall over expanded ground
truth (21,606 rejections); an empirical-Bayes mixture-LR ablation achieves
77.0% / 90.5% (65,673 rejections); a hybrid pipeline (split-LRT for
size ≥ 8 plus mixture-LR for size 5-7) achieves 77.1% / 91.3% (66,170
rejections). Manual inspection surfaces a smoking-gun unattributed anti-NN
form letter ("Dear Express Restoring Internet Freedom, Please enter your
comment here. I'm asking the FCC to roll back Obama's Internet takeover.
Regards, Raymond Nelson 11834 37th St NE Saint Michael, MN 55376"),
demonstrating that the construction surfaces real coordination evidence
absent from FOIA records. Results are stable across Leiden-CPM resolution
γ ∈ {0.85, 0.88, 0.90, 0.93}. We release the labeled cluster–attribution
table, γ-sweep ablation, vMF defensibility diagnostics, synthetic-injection
power study, and reproducible code.

---

## What this abstract commits to

All numbers are reproducible from on-disk artifacts in `data/processed/` and
`results/`. Methodological claims:

- **Validity (Theorem 1).** Proved; numerically verified via Markov
  super-uniformity on `S^{19}` with `n=100` (`notes/verify_proofs.py`).
- **Sup-norm graceful-degradation bound (Theorem 2).** Proved; numerically
  verified at ε ∈ {0.05, 0.10, 0.20}.
- **Power lower bound (Theorem 3, sketch).** Scaling
  `κ* ≍ √(d log(K/α)/n_c)` confirmed empirically on synthetic vMF data;
  constants are loose and remain Q1 of `notes/construction.md` §9 (the open
  question for Ramdas).
- **Boosting/closure compatibility.** Follows directly from individual e-value
  validity.

Empirical claims, all from the runs done overnight:

- 15,744 / 15,748 candidate clusters of size ≥ 8 rejected at α = 0.10 by
  split-LRT compound e-BH.
  ✓ `data/processed/cluster_evalues_lrt_clean.parquet`
- 38.6% astroturf-attribution precision (6,072 / 15,744); 100% recall on
  attributed astroturf size ≥ 8 (6,072 / 6,072) and on attributed advocacy
  size ≥ 8 (401 / 401).
  ✓ `results/attribution_table_lrt_r0.9.csv`
- 91 / 100 top unattributed-but-rejected clusters are pro-NN advocacy
  form letters (manual + keyword classification); 0 are anti-NN.
  ✓ `results/lrt_unlabeled_top100_inspected.csv`
- vMF MLE on real attributed clusters: κ̂ ∈ [1.3 × 10³, 3.2 × 10³];
  numerically stable regime.
  ✓ `results/vmf_diagnostics.csv`

## Known limitation

The split-LRT requires `m = n_c / 2 ≥ 4` for stable vMF MLE on each half.
Clusters of size 5–7 (40% of the candidate set with size ≥ 5) fall back to
the empirical-Bayes mixture-LR ablation; per `notes/findings.md` the two
constructions agree on attributed-astroturf recall but produce different
operating points (mixture-LR is higher-recall, lower-precision; split-LRT
is the reverse).

## What's left before the workshop draft (Aug 29 deadline, 17 weeks)

1. Lee–Ren boosting + closed-eBH ablation (`src/evalues_lrt.py` boosting layer).
2. Optional: LRT γ-sweep at γ = 0.85 and γ = 0.88 (mixlr sweep is on disk).
3. Synthetic injection at SBERT-anisotropic geometry, not pure vMF
   (current power figure uses pure vMF; reviewers will want anisotropic).
4. Methods/Experiments section drafting.
5. Recruit Ramdas with the workshop draft + simulation figure.
