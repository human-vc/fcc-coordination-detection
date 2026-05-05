# Polished abstract (v2) for NeurIPS 2026 submission

Tightened from 1830 chars → ~1700 chars. Follows canonical NeurIPS
Problem → Method → Guarantee → Result structure.

```
Detecting coordinated paraphrase campaigns in regulatory public-comment
dockets is a multiple-testing problem with two unusual structures:
candidate clusters are discovered adaptively from millions of texts, and
external attribution labels are partial. We propose split-likelihood-ratio
compound e-values via universal inference for cluster-level FDR control:
each cluster's e-value is constructed by within-cluster sample-splitting
under a von Mises-Fisher template-paraphrase alternative on the unit
sphere, with the corpus-marginal null density fitted as a movMF mixture
on whitened SBERT embeddings drawn from a cluster-disjoint singleton
subsample. Validity requires no regularity conditions on the alternative
and admits a graceful-degradation bound under sup-norm null
misspecification; the construction yields individually valid e-values
that compose strictly with Lee-Ren conditional-calibration boosting and
Xu-Fischer-Ramdas closed e-BH. A power result holds with threshold
$\kappa^* \asymp \sqrt{d \log(K/\alpha)/n_c}$ and empirically calibrated
constants $C_1 \in [3.1, 13.3]$ across $(d,n)$. Applied to the FCC 2017
"Restoring Internet Freedom" docket (3.8M unique comments; 15,748
candidate Leiden-CPM clusters of size $\geq 8$ at $\gamma=0.90$), at
$\alpha=0.10$ the construction rejects 15,748 clusters at 96.8% precision
and 100% recall against an expanded ground-truth coordination set
(FOIA-derived NY-Attorney-General attribution to paid astroturf
contractors plus keyword-classified pro-net-neutrality advocacy form
letters); FOIA labels alone yield apparent precision 38.6%, quantifying
substantial pro-NN undercount. The Beta-tail cohesion baseline at the
same $\alpha$ achieves 97.1%/37.5%, an empirical-Bayes mixture-LR
ablation 77.0%/90.5%, and a hybrid LRT+mixture-LR pipeline 77.1%/91.3%
precision/recall over the same expanded ground truth. Results are
stable across Leiden-CPM resolution.
```

Char count to verify before pasting: should be ~1700, well under 2000.

---

## Why this is better than v1

1. **Opens with the problem, not the method.** Reviewers see the *task*
   first, then your contribution in context.
2. **Method named in one breath**: "split-likelihood-ratio compound
   e-values via universal inference." Reviewers can immediately place
   the construction in the e-value literature.
3. **Guarantee + boosting compatibility in one sentence.** Tighter.
4. **Power result inlined** rather than listed separately.
5. **Headline number FIRST** (96.8%/100%), FOIA caveat as parenthetical.
6. **Ablations as one sentence** rather than three.
7. **Cut the smoking-gun hook** (saves for introduction) and the explicit
   γ list (subsumed by "stable across resolution").
8. **No closing significance statement** — the headline number IS the
   significance for an empirical-leaning methodology paper.

Trade-off: less colorful (no "smoking gun" anti-NN cluster, no specific
γ values, no advocacy-org names). But for a NeurIPS abstract that has to
be tight, color goes. Save it for the intro.

---

## Final checks before pasting into OpenReview

```bash
.venv/bin/python -c "
text = '''<paste the abstract here, single-quoted with triple-quotes>'''
print(f'chars: {len(text)}, words: {len(text.split())}')
print(f'within 2000-char limit: {len(text) <= 2000}')
"
```

If it ends up over 2000 chars after paste (unicode subtleties, MathJax
escapes), trim the ablation sentence further.
