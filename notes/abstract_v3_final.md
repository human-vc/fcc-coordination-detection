# Abstract v3 — addresses all 7 issues from the critique

## Fixes applied (mapping to issues 1-7)

1. **Mega-sentence broken into 3 sentences.** Method named, then construction described, then null density.
2. **Explicit contribution pitch.** "We make three contributions. (1) **Theorem 1**: ... (2) **Theorem 2**: ... (3) **Theorem 3**: ..."
3. **Keyword-GT framed honestly.** Instead of "96.8% precision against expanded GT" (which mixed FOIA and self-constructed labels), now leads with FOIA-only recall (100%) + ablation precision-gain (38.6% vs 29.2%) + a separate independent finding ("91% of unattributed rejections match pro-NN form-letter signatures") — three distinct empirical claims, no label leakage.
4. **FOIA caveat removed entirely.** Replaced with the cleaner cohesion-baseline-comparison framing.
5. **Heavy ablation sentence cut.** Now mentions only the most relevant single comparison (vs cohesion baseline).
6. **Real closing.** "Our results suggest FOIA-derived attribution substantially undercounts true coordinated mobilization, motivating broader text-based audits."
7. **Theorems explicitly named.** "Theorem 1: finite-sample validity. Theorem 2: sup-norm graceful-degradation. Theorem 3 (sketched): a power lower bound..." — reader can immediately tell what's proven vs sketched.

---

## Final abstract text

```
Coordinated paraphrase campaigns in regulatory public-comment dockets pose a
multiple-testing problem that resists existing tools: candidate clusters are
discovered adaptively from millions of texts, and external attribution labels
cover only a fraction of true coordination. We propose split-likelihood-ratio
compound e-values via universal inference for cluster-level FDR control. Each
cluster's e-value is built by within-cluster sample-splitting under a von
Mises-Fisher template-paraphrase alternative on the unit sphere, against a
corpus-marginal null density fit as a movMF mixture on cluster-disjoint
singleton SBERT embeddings. We make three contributions. (1) Theorem 1:
finite-sample validity with no regularity conditions on the alternative. (2)
Theorem 2: a sup-norm graceful-degradation bound under null misspecification.
(3) Theorem 3 (sketched): a power lower bound with detection threshold
$\kappa^* \asymp \sqrt{d \log(K/\alpha)/n_c}$ and empirically calibrated
constants $C_1 \in [3.1, 13.3]$. The construction yields individually valid
e-values that compose strictly with Lee-Ren conditional-calibration boosting
and Xu-Fischer-Ramdas closed e-BH. Applied at $\alpha=0.10$ to the FCC 2017
"Restoring Internet Freedom" docket (3.8M unique comments; 15,748 size-$\geq$-8
Leiden-CPM clusters), the construction recovers 100% of the 6,305
BuzzFeed-FOIA-attributed astroturf clusters named in the 2021 New York
Attorney General investigation, improves attribution precision over the
cohesion baseline by 9 percentage points (38.6% vs 29.2%) at matched recall,
and surfaces 8,494 unattributed coordinated clusters of which 91% match
independent pro-NN advocacy form-letter signatures. Our results suggest
FOIA-derived attribution substantially undercounts true coordinated
mobilization, motivating broader text-based audits.
```

(Verify char count below before pasting.)
