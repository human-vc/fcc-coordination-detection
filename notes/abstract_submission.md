# NeurIPS 2026 main — submission package

**Deadline:** May 4, 2026 (AOE) = **May 5, 11:59 AM PT / 2:59 PM ET**.
Full paper due May 6 AOE (~48 hours after abstract).

**Portal:** https://openreview.net/group?id=NeurIPS.cc/2026/Conference

---

## Title (≤ ~250 chars typical)

> Coordination Detection in Regulatory Public-Comment Dockets via Universal-Inference Compound E-Values

(74 chars — fine.)

---

## Abstract (max **2000 chars** plain unicode text, mathjax allowed)

This is the version that fits the 2000-char limit:

```
We address finite-sample-valid false-discovery-rate-controlled detection
of coordinated paraphrase campaigns in regulatory public-comment dockets.
Our procedure constructs cluster-level split-likelihood-ratio compound
e-values under universal inference (Wasserman-Ramdas-Balakrishnan 2020)
with a von Mises-Fisher template-paraphrase alternative on the unit
sphere and a corpus-marginal null density estimated as a movMF mixture
on whitened SBERT embeddings, fitted on a cluster-disjoint singleton
subsample. Validity requires no regularity conditions on the alternative
and admits a graceful-degradation bound under sup-norm null
misspecification. The construction yields individually valid e-values,
composing strictly with Lee-Ren conditional-calibration boosting and
Xu-Fischer-Ramdas closed e-BH; a power result follows for a vMF
alternative with threshold $\kappa^* \asymp \sqrt{d \log(K/\alpha)/n_c}$
and empirically calibrated constants $C_1 \in [3.1, 13.3]$. Applied to
the FCC 2017 "Restoring Internet Freedom" docket (24M submissions, 3.8M
unique texts), at $\alpha=0.10$ the construction rejects 15,748 size-$\geq$-8
clusters at 96.8% precision and 100% recall against an expanded
ground-truth coordination set combining FOIA-derived NY-Attorney-General
attribution to paid astroturf contractors (Media Bridge LLC) with
keyword-classified pro-net-neutrality coalition form letters; FOIA
labels alone yield 38.6% precision, demonstrating substantial pro-NN
undercount. The cohesion-based baseline achieves 97.1%/37.5%
precision/recall; an empirical-Bayes mixture-LR ablation 77.0%/90.5%; a
hybrid pipeline 77.1%/91.3%. Manual inspection surfaces an unattributed
anti-NN form letter with template instructions still in the comment
text. Results are stable across Leiden-CPM resolution
$\gamma\in\{0.85,0.88,0.90,0.93\}$.
```

Char count: ~1,990 (including newlines). Within limit. Verify with
`.venv/bin/python -c "print(len(open('...').read()))"` before pasting.

**Critical reminder from NeurIPS guidance:**
> "Submissions with placeholder abstracts that are rewritten for the
> full submission risk being removed without consideration."

This abstract reflects the actual claimed contribution and matches what
will be in the full paper, so it's not a placeholder.

---

## Keywords (descriptive, not hierarchical)

Suggested:
- false discovery rate
- e-values
- universal inference
- coordination detection
- compound e-values
- regulatory text
- von Mises-Fisher
- network analysis (or: text clustering)

OpenReview lets you add as many as helpful for routing to area chairs.

---

## Contribution Type — pick ONE

| Option | Fit |
|---|---|
| **General** | ✓ Best fit — methodology paper with theory + empirics |
| Theory | Less fit — Theorem 3 is a sketch; theory isn't the primary contribution |
| Use-Inspired | Possible — application to FCC docket motivates the methodology |
| Concept & Feasibility | No — we have full empirics, not preliminary |
| Negative Results | No |

**Recommend: General.** "Use-Inspired" would also be defensible if you
want to lean into the FCC application as the framing.

---

## Author list

NeurIPS 2026 rule: **all authors entered by FULL paper deadline (May 6).
No changes after. No additions/removals for accepted papers.**

For tonight, enter:
- Jacob Crainic (you)
- Add coauthor placeholders if you intend to recruit before May 6
  (e.g., if you've already approached Hall — add as author. Ramdas
  not yet contacted — do NOT add without consent)

**Critical:** if you add an author tonight, get explicit consent first.
Adding without consent is a rule violation.

If submitting solo tonight, you can email Hall first thing tomorrow
to confirm before May 6.

---

## OpenReview profile checklist (must be done by submission)

- [ ] Profile exists (create at https://openreview.net/signup)
- [ ] Profile up to date with current affiliation
- [ ] All your prior publications listed
- [ ] Email confirmed
- [ ] Suggested by Apr 21 — but you can still register; just do it now

---

## Other typical OpenReview submission fields

Based on NeurIPS 2025/2024 pattern (2026 likely similar):

- **Primary subject area** — pick from a long list. Likely:
  - "Theory > Statistics" or
  - "Applications > Society"
- **Conflicts of interest** — auto-populated from your OpenReview
  profile + entered email domains. List your advisor + collaborators.
- **TLDR** (one-line summary, optional but recommended) — try:
  > "Cluster-level FDR control for coordinated comment campaigns via
  > universal-inference compound e-values, yielding 96.8% precision /
  > 100% recall against expanded ground truth on the FCC 2017 docket."
- **Anonymized PDF** — for the **full paper** on May 6. Not required
  tonight for the abstract step (verify on OpenReview portal).
- **Code/data availability statement** — likely required at full paper
  stage; you have artifacts on disk to release.

---

## What can change between May 4 abstract and May 6 full paper

- Title, keywords, abstract: editable, but cannot **substantively
  change** the claimed contribution.
- Authors: editable up to May 6 (must have OpenReview profiles); cannot
  change after.
- TLDR / subject area: editable.
- Full PDF: required by May 6, not May 4.

---

## Action checklist for tonight

1. [ ] Create / update OpenReview profile (10 min)
2. [ ] Decide on coauthor list (with consent if added)
3. [ ] Visit OpenReview NeurIPS 2026 portal → "Submit"
4. [ ] Paste title, abstract (≤2000 chars), keywords, contribution type, TLDR
5. [ ] Submit
6. [ ] Receive confirmation email — KEEP THIS
7. [ ] Sleep
8. [ ] Tomorrow: write/format the full paper for May 6 deadline (~36 hours away)

---

## Sources

- [NeurIPS 2026 Call for Papers](https://neurips.cc/Conferences/2026/CallForPapers)
- [NeurIPS 2026 Dates and Deadlines](https://neurips.cc/Conferences/2026/Dates)
- [NeurIPS 2026 Main Track Handbook](https://neurips.cc/Conferences/2026/MainTrackHandbook)
- [NeurIPS 2026 OpenReview](https://openreview.net/group?id=NeurIPS.cc/2026/Conference)
- [NeurIPS 2025 FAQ for Authors](https://neurips.cc/Conferences/2025/PaperInformation/NeurIPS-FAQ) — character limit + placeholder rule confirmed here
- [NeurIPS 2026 Tweet on dates](https://x.com/NeurIPSConf/status/2041303457554940336)
