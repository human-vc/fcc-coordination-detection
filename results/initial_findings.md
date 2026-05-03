# Initial findings — FCC 2017 corpus

Date: 2026-05-03
Commit: see `git log` at time of writing

## Corpus scale

| Table | Rows |
|---|---|
| submissions | 23,951,967 |
| filers | 24,112,988 |
| comments (express texts) | 3,800,693 |
| documents (PDFs) | 13,519 |
| near_duplicates (PDF-level) | 372,283 |
| exact_duplicates (PDF-level) | 2,299 |
| interest_groups (labels) | 710 |
| docs_cited (in final order) | 2,060 |

99.9% of submissions are express format (23.92M of 23.95M). Only 29,717 are standard PDF submissions.

## Template reuse — the coordination signal

The submissions table links to a 3.8M-row deduplicated comment-text table. Counting how many submissions share each unique `comment_id`:

| Template size | Count |
|---|---|
| singletons (n=1) | 2,985,780 |
| small (n=2–10) | 760,848 |
| medium (n=11–100) | 15,776 |
| large (n=101–10K) | 38,212 |
| massive (n=10K–100K) | 74 |
| **huge (n>100K)** | **31** |

**Top 5 templates by copy count:**
1. ae893cffcb… — **1,096,832** copies
2. 5df92c1778… — 818,331 copies
3. 50124415b6… — 550,000 copies
4. 8e5c7ddcb4… — 546,251 copies
5. 64f058fe2d… — 469,202 copies

**80% of all 24M submissions (19.17M)** belong to templates with ≥10 copies — almost certainly coordinated. The remaining 4.78M submissions (singletons + small) are the population where the *interesting* methodological problem lives: detecting coordination that has been deliberately varied to evade exact-template matching.

## Implications for the paper

1. **Easy regime — exact templates.** ~31 templates account for >50% of all submissions. Detection is trivial; the methodological question is why agencies haven't deployed these defenses.
2. **Hard regime — soft coordination.** Among the ~3M singleton submissions, an unknown fraction are paraphrased versions of coordinated content designed to evade detection. This is where embedding-based similarity + e-BH FDR control becomes load-bearing.
3. **Validation strategy.** Use the heavy-template clusters as gold-positive labels (we *know* they're coordinated) and a stratified sample of true singletons as gold-negative labels. The methodological challenge is the borderline cases.
4. **Next step in the pipeline.** Embed the 3M singleton express comments with MiniLM, build a kNN similarity graph at varying thresholds, run cluster detection, compute cluster-level e-values calibrated against the gold-positive heavy templates, apply e-BH for FDR-controlled detection in the long tail.

## Cluster detection on the precomputed PDF graph (sanity check)

Running Leiden community detection on the 372K near-duplicate edges among 4,278 documents:
- 537 connected clusters total
- 106 clusters with ≥5 docs (= candidate coordination campaigns)
- Largest cluster: 494 docs

These are PDF-level, not the main coordination signal (which lives in express templates). They serve as a sanity check that the graph approach works.
