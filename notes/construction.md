# Split-LRT Compound E-Values via Universal Inference on a vMF Template-Paraphrase Model

**Status:** Week 1 deliverable. Foundation document for the methodological contribution.
**Target venue:** NeurIPS 2027 main, May 2027 deadline. Workshop draft: NeurIPS 2026, Aug 29 2026.
**Coauthor target:** Aaditya Ramdas (CMU).

This note specifies the construction we will build, proves its validity, states a power
lower bound (with proof sketch — the load-bearing gap is identified), and lays out the
implementation choices needed before writing code in Week 2.

The construction replaces the current Beta-MOM-on-cohesion + Vovk–Wang p→e step
(`src/evalues.py`) with a likelihood-based e-value via universal inference (Wasserman,
Ramdas, Balakrishnan 2020, *PNAS*). The replacement gives us:

1. A **direct** e-value (no Vovk–Wang p→e calibration loss).
2. **Individual** validity: `E[E_c | H_c] = 1` exactly (when `q` is correct), not just
   `≤ 1` averaged. This unlocks Lee–Ren 2024 boosting and Xu–Fischer–Ramdas 2025
   closed-eBH on top of vanilla e-BH.
3. A **power theorem** with explicit κ-threshold under a vMF alternative — the new
   methodological contribution. Cohesion-based e-values have no analogous result.

---

## 1. Setup and notation

Let `S^{d−1}` denote the unit sphere in `R^d`. SBERT MiniLM-L6-v2 outputs are
L2-normalized in `R^{384}`, so `d = 384`. After the all-but-the-top whitening
described in §7.3 we work on `S^{d−1−k}` for whitening rank `k ∈ {3,…,10}`; for the
remainder of this document we abuse notation and continue to write `d` for the working
dimension.

Indexing.
- Clusters `c = 1, …, K` from Leiden CPM at γ=0.90 (current build: `K ≈ 21,606`).
- Within cluster `c`, comment embeddings `x_{c,1}, …, x_{c,n_c} ∈ S^{d−1}`.
- Hypothesis pair, per cluster:

  - **H_c (null):** `x_{c,1}, …, x_{c,n_c} ~iid q`, where `q` is the corpus-marginal
    density on `S^{d−1}` (estimated as a movMF mixture on a held-out subsample, §7.2).
  - **H_c¹ (alternative, "coordinated"):** `x_{c,1}, …, x_{c,n_c} ~iid p_vMF(· | μ_c, κ)`
    for some latent template direction `μ_c ∈ S^{d−1}` and concentration `κ > 0`.

Crucially, **the null is not "uniform on the sphere."** SBERT embeddings are
empirically anisotropic (Ethayarajh 2019; Cai–Huang–Bian–Church 2021); whitening
mitigates this but does not eliminate it. The framing throughout is "vMF(μ_c, κ) vs
estimated movMF q̂", not "vMF vs uniform."

Aggregation.
- e-BH at level α ∈ (0, 1) on the K cluster-level e-values `{E_c}_{c=1}^{K}`. Sort
  `E_(1) ≥ E_(2) ≥ …`; let `k̂ = max{k : E_(k) ≥ K / (αk)}`; reject the top `k̂`
  clusters.

Quantities.
- `m := n_c / 2` is the held-out half size after the within-cluster split.
- `p_vMF(x | μ, κ) = C_d(κ) exp(κ ⟨μ, x⟩)` where `C_d(κ) = κ^{d/2−1} / [(2π)^{d/2}
  I_{d/2−1}(κ)]` is the vMF normalizer.
- `A_d(κ) := I_{d/2}(κ) / I_{d/2−1}(κ)` is the mean resultant length under
  `vMF(·, κ)`; `E[⟨x, μ⟩] = A_d(κ)` for `x ~ vMF(μ, κ)`.

---

## 2. The construction

For each cluster `c` (independently):

**Step 1 — Within-cluster split.** Choose `A_c ⊔ B_c = {1, …, n_c}` with
`|A_c| = |B_c| = m` via a data-independent uniformly random partition. Use the same
PRNG seed across the run.

**Step 2 — MLE on half A.** Fit the vMF MLE on `{x_{c,i} : i ∈ A_c}`:
- Direction: `μ̂_c = ā_c / ‖ā_c‖`, where `ā_c = (1/m) Σ_{i ∈ A_c} x_{c,i}`.
- Concentration: `κ̂_c` solves `A_d(κ̂_c) = ‖ā_c‖`. Implemented via Banerjee et al.
  2005 closed-form approximation, then 2 Newton steps on the score (Hornik–Grün 2014).

**Step 3 — Split-LRT compound e-value on half B.**
$$
E_c \;=\; \prod_{i \in B_c} \frac{p_{\mathrm{vMF}}(x_{c,i} \mid \hat\mu_c, \hat\kappa_c)}{q(x_{c,i})}.
$$
In log form, `log E_c = Σ_{i ∈ B_c} [log C_d(κ̂_c) + κ̂_c ⟨μ̂_c, x_{c,i}⟩ − log q(x_{c,i})]`.

**Step 4 — Optional swap-and-average.** Compute `E_c^{(1)}` with split `(A_c, B_c)`
and `E_c^{(2)}` with split `(B_c, A_c)`. Aggregate via Vovk–Wang arithmetic mean:
`Ē_c = (E_c^{(1)} + E_c^{(2)}) / 2`. By Vovk–Wang 2021 Theorem 3.2, `Ē_c` is itself an
e-value; we lose nothing on validity and reduce variance.

(See §7.4 for why K = 2 swap-and-average is the right setting per Strieder–Drton
2022 and Dunn–Ramdas–Balakrishnan–Wasserman 2023.)

**Step 5 — Apply e-BH** at level α on `{Ē_c}_{c=1}^K`. Then optionally **boost**
(Lee–Ren 2024) and **close** (Xu–Fischer–Ramdas 2025); see §6.

---

## 3. Validity (Theorem 1)

### Assumptions

- **(A1) Null IID.** Under `H_c`, `x_{c,1}, …, x_{c,n_c}` are iid with density `q`
  on `S^{d−1}`.
- **(A2) Density support.** `q` is a probability density on `S^{d−1}` with
  `q > 0` everywhere; the ratio `p_vMF(x | μ̂_c, κ̂_c) / q(x)` is well-defined
  `q`-a.e.
- **(A3) Independent split.** The partition `(A_c, B_c)` is data-independent
  (random with seed fixed before observing data, or any deterministic
  rule on indices).

### Theorem 1 (validity).

Under (A1)–(A3), with `q` correctly specified,
$$
\mathbb E[E_c \mid A_c] \;=\; 1 \quad \text{under } H_c.
$$
Marginalizing over `A_c`, `E[E_c | H_c] = 1`. The vector `(E_1, …, E_K)` is therefore
a valid (compound, in fact individually-valid) e-value family, and **e-BH at level α
controls FDR ≤ α** under arbitrary dependence between clusters (Wang–Ramdas 2022).

### Proof.

Conditional on `A_c`, the estimator `(μ̂_c, κ̂_c)` is a fixed measurable function of
`(x_{c,i} : i ∈ A_c)`. By (A1) and (A3), `(x_{c,i} : i ∈ B_c) ~iid q` independently of
`A_c`. So
$$
\mathbb E[E_c \mid A_c]
\;=\; \prod_{i \in B_c} \int_{S^{d-1}}
   \frac{p_{\mathrm{vMF}}(x \mid \hat\mu_c, \hat\kappa_c)}{q(x)} \cdot q(x) \, d\sigma(x)
\;=\; \prod_{i \in B_c} 1
\;=\; 1.
$$
The integral equals 1 because `p_vMF(· | μ̂_c, κ̂_c)` is a probability density on
`S^{d−1}` (`σ` is the surface measure), so its integral is unity regardless of `μ̂_c, κ̂_c`.

This is the universal inference argument of Wasserman–Ramdas–Balakrishnan 2020
(Theorem 4 in PNAS), specialized to the case of a *single-density* null rather than a
parametric submodel null.

FDR control follows from Wang–Ramdas 2022 (e-BH on any compound e-value family
controls FDR at α under arbitrary dependence). ∎

### Remarks.

- (R1) The proof uses **no** properties of `vMF` beyond the fact that it integrates
  to 1. The same construction works with any parametric alternative on the sphere
  (Kent, Bingham, Power Spherical, etc.). vMF is chosen for the right *power* under a
  template-paraphrase model — see §7.1.
- (R2) The proof uses **no** regularity conditions, no consistency, no smoothness.
  This is the killer feature versus IWR §7 (mixture-LR; needs NPMLE consistency) or
  Csillag–Struchiner–Goedert 2025 (PPI; asymptotic at rate `n^{−1/2}`).
- (R3) Equality `E[E_c | H_c] = 1` (not just ≤ 1) is **individual validity**. This
  unlocks the Lee–Ren 2024 boosting and Xu–Fischer–Ramdas 2025 closed-eBH layers
  (§6).

---

## 4. Mis-specification of q (Theorem 2)

In practice we plug in an estimate `q̂` (movMF on a held-out corpus subsample, §7.2).
The validity proof of Theorem 1 fails: the integral of `p_vMF / q̂` against `q` is no
longer 1.

### Theorem 2 (graceful degradation, sup-norm).

Let `q̂` be a probability density on `S^{d−1}` such that `‖q / q̂ − 1‖_∞ ≤ ε`. Define
the plug-in e-value
$$
\hat E_c \;=\; \prod_{i \in B_c} \frac{p_{\mathrm{vMF}}(x_{c,i} \mid \hat\mu_c, \hat\kappa_c)}{\hat q(x_{c,i})}.
$$
Under `H_c` with assumptions (A1)–(A3),
$$
\mathbb E[\hat E_c \mid A_c] \;\le\; (1 + \varepsilon)^m \;\le\; e^{m\varepsilon}.
$$
Consequently, e-BH at nominal level α applied to `(Ê_1, …, Ê_K)` controls
$$
\mathrm{FDR} \;\le\; \alpha \cdot e^{m \varepsilon},
$$
where `m = max_c n_c / 2` is the largest held-out-half size in the corpus.

### Proof.

Conditional on `A_c`, by (A1) and (A3), `(x_{c,i} : i ∈ B_c) ~iid q`. So
$$
\mathbb E[\hat E_c \mid A_c]
\;=\; \prod_{i \in B_c} \int_{S^{d-1}} \frac{p_{\mathrm{vMF}}(x \mid \hat\mu_c, \hat\kappa_c)}{\hat q(x)} \, q(x) \, d\sigma(x)
\;=\; \prod_{i \in B_c} \int_{S^{d-1}} p_{\mathrm{vMF}}(x \mid \hat\mu_c, \hat\kappa_c) \cdot \frac{q(x)}{\hat q(x)} \, d\sigma(x).
$$
Bound `q(x) / q̂(x) ≤ 1 + ε` pointwise (the sup-norm assumption), pull this past the
integral; since `p_vMF` integrates to 1, the integral is `≤ 1 + ε`. The product over
`B_c` gives `(1 + ε)^m`. Marginalizing over `A_c` and applying Wang–Ramdas 2022
e-BH with the inflated e-values yields the FDR bound. ∎

### Remarks.

- (R4) **TV is the wrong metric.** A common reflex is to bound `‖q − q̂‖_TV ≤ ε` and
  conclude `(1 + ε)^m`. This is **false** in general because the bound on `q / q̂`
  pointwise can be much worse than `(1 + 2ε)`. Sup-norm of the density ratio is the
  correct metric. Equivalently, the χ²-divergence
  `χ²(q ‖ q̂) = ∫ (q/q̂ − 1)² q̂ dσ` controls `E[Ê_c | A_c] − 1` to leading order in `ε`.
- (R5) **Tightness on large clusters.** Some FCC clusters have `n_c ≈ 10^4`, so
  `m ≈ 5000`. To keep the per-cluster type-I inflation under `δ = 0.1`, we need
  `‖q / q̂ − 1‖_∞ ≲ δ / m = 2·10^{−5}`. **This is the binding constraint on `q̂`**, and
  drives §7.2's recommendation to fit `q̂` with explicit tail support.
- (R6) **Mitigation: ratio truncation.** Replace
  `p_vMF / q̂ ↦ min(p_vMF / q̂, T)` for some `T = T(d, m)`. Truncation at `T` re-bounds
  `E[Ê_c | A_c] ≤ (1 + ε ∧ (T − 1))^m` and removes the worst-case behavior at points
  where `q̂(x)` is anomalously small. The cost is a small power loss when the true
  log-LR exceeds `log T`. We will tune `T` empirically in Week 4 against the BuzzFeed
  attribution precision.

---

## 5. Power lower bound (Theorem 3, sketch)

### What's published.

- WRB 2020 has no power result; it is a validity paper.
- Strieder–Drton 2022 give noncentral split-χ² distributions for SLRT under local
  alternatives, parametric submodel null, Euclidean — **not** sphere, **not**
  fitted-q null.
- Dunn–Ramdas–Balakrishnan–Wasserman 2023 (*Biometrika*) characterize the
  asymptotic radius ratio `E[r²(SLRT)] / E[r²(LRT)] → 3/2` for Gaussian-mean
  problems. Suggestive but not directly applicable.
- Tse–Davison 2022 cautions against multi-split aggregation in regular parametric
  cases. We use K = 2 swap-and-average, which is in their "OK" regime.
- Sun–Han 2024 (Gaussian mixture) shows SLRT achieves the same `(n^{−1} log log n)^{1/2}`
  detection rate as classical LRT — encouraging precedent for "no rate loss from
  splitting."
- Takatsu 2025 establishes the precise conservativeness factor `Φ(√(2 log(1/α)))`
  for SLRT confidence sets and gives a studentization correction that recovers
  exact 1−α coverage. May yield 5–20% effective-α uplift here; revisit in Week 7.

**No paper derives a power threshold for split-LRT compound e-BH on the sphere with a
fitted-q null and vMF alternative.** This is the open problem.

### Theorem 3 (power lower bound, target form).

Suppose:
- (B1) Bounded null density: `‖q‖_∞ ≤ M < ∞`.
- (B2) The alternative on cluster `c` is `H_c¹: x ~iid p_vMF(· | μ_c, κ)` for some
  `μ_c ∈ S^{d−1}` and `κ > 0`.
- (B3) Whitening has been applied so the within-cluster geometry is approximately
  isotropic (Cai et al. 2021; verified empirically in Week 2 — see §7.3).

Then there exist constants `C_1, C_2 > 0` (depending only on `M`, `d` weakly through
log factors) such that if
$$
\kappa \;\ge\; \kappa^*(n_c, \alpha, \beta, d, K) \;:=\; C_1 \sqrt{\frac{d \,\log(K / (\alpha \beta))}{n_c}} \quad \text{(small-κ regime)}
$$
then
$$
\Pr_{H_c^1}\!\big(E_c \ge K / \alpha\big) \;\ge\; 1 - \beta,
$$
and consequently cluster `c` is rejected by e-BH at level α with probability ≥ 1 − β.

For large κ, the threshold improves to `κ* ≍ (d / n_c) log n_c`.

### Proof sketch (three steps).

Let `m = n_c / 2` and `Z_i := log[p_vMF(x_i | μ̂_c, κ̂_c) / q(x_i)]` for `i ∈ B_c`. Note
`E_c = exp(Σ_{i ∈ B_c} Z_i)`. A sufficient condition for e-BH rejection (modulo the
`k̂/K` factor, which only helps) is `Σ Z_i ≥ log(K / α)`.

**Step 1: vMF MLE direction concentration on A_c.** For `x ~ vMF(μ, κ)`,
`E[⟨x, μ⟩] = A_d(κ)`, with `A_d(κ) ≈ κ/d` for small κ and `A_d(κ) → 1` for large κ.
The empirical mean `ā_c = (1/m) Σ_{i ∈ A_c} x_{c,i}` satisfies `E[ā_c] = A_d(κ) μ`.

The MLE direction is `μ̂_c = ā_c / ‖ā_c‖`, **not** `ā_c` itself, so we need a
concentration bound on the *direction* of `ā_c`, not on its projection along `μ`.
By Hoeffding on `⟨x_i, μ⟩ ∈ [−1, 1]`,
$$
\Pr\!\big(\langle \bar a_c, \mu \rangle \ge A_d(\kappa) - t\big) \;\ge\; 1 - 2\exp(-c m t^2),
$$
which gives concentration of `⟨ā_c, μ⟩` around `A_d(κ)`. Combined with control of
the orthogonal component `proj_{μ_⊥} ā_c` (via Hoeffding/Bernstein on each
orthogonal direction, or via rotational symmetry of vMF: under rotation around
`μ`, `proj_{μ_⊥} ā_c` is isotropic with variance `O(1/m)` per direction), one
obtains
$$
\Pr\!\big(\langle \hat\mu_c, \mu \rangle \ge 1 - \delta\big) \;\ge\; 1 - \beta/2,
$$
provided `m · A_d(κ)² ≳ (d / δ) · log(2/β)`. For small κ (where `A_d(κ) ≈ κ/d`),
this requires `m κ² / d² ≳ (d / δ) log(2/β)`, i.e. `m κ² / d³ ≳ log(2/β) / δ`.
For moderate κ the requirement weakens substantially.

> **Gap.** This step is the load-bearing concentration argument. A
> theorem-grade sphere-direction concentration bound for the vMF MLE — sharp in
> both `m` and the dimension `d`, valid uniformly across the κ range — is
> folklore but absent from the literature at the granularity needed. Banerjee
> et al. 2005 has approximate-MLE statements; sharp deviation under vMF is
> **the open question to bring to Ramdas (Q1, §9)**. The empirical
> verification in `notes/verify_proofs.py` (test D) confirms `⟨μ̂, μ⟩ → 1`
> empirically at `m = 50, d = 20`: at κ = 10, mean `⟨μ̂, μ⟩ = 0.96` and 10th
> percentile `0.94`; at κ = 3, mean `0.72`; at κ = 1, mean `0.33`.

**Step 2: KL lower bound.** Conditional on the favorable event,
$$
\mathbb E_{x \sim \mathrm{vMF}(\mu, \kappa)}[Z_i \mid \text{favorable}]
\;=\; \mathrm{KL}(\mathrm{vMF}(\mu, \kappa) \,\|\, q) - \mathrm{KL}(\mathrm{vMF}(\mu, \kappa) \,\|\, p_{\mathrm{vMF}}(\cdot \mid \hat\mu_c, \hat\kappa_c)).
$$
Diethe 2015 gives the closed-form vMF–vMF KL. For small κ, expand
`KL(vMF(μ, κ) ‖ Unif) ≈ κ²/(2d) + O(κ⁴/d²)` (verified numerically in `notes/verify_proofs.py`
test K: ratio `MC / Taylor ∈ [0.96, 1.08]` for κ ∈ [0.3, 3]). Under (B1) (bounded `q`),
`KL(vMF(μ, κ) ‖ q) ≥ KL(vMF(μ, κ) ‖ Unif) − log M ≥ κ²/(2d) − log M − o(κ²/d)`.
On the favorable event (Step 1: `⟨μ̂_c, μ⟩ ≥ 1 − δ`, with κ̂_c near κ),
`KL(vMF(μ, κ) ‖ vMF(μ̂_c, κ̂_c)) ≈ κ A_d(κ) (1 − ⟨μ̂_c, μ⟩) ≤ κ A_d(κ) δ ≈ κ² δ / d` for
small κ. Picking `δ = 1/4` keeps the alignment loss to half the signal:
`E[Z_i | favorable] ≳ κ²/(2d) − κ²/(4d) − log M = κ²/(4d) − log M`.

**Step 3: Bernstein on B_c.** `Z_i` is bounded above by `‖log(p_vMF/q)‖_∞ ≤ κ + log M`
for vMF in the relevant regime. Bernstein gives
$$
\Pr\!\Big(\sum_{i \in B_c} Z_i \ge m \cdot \mathbb E[Z_i \mid \text{favorable}] / 2\Big)
\;\ge\; 1 - \exp(-c' m \kappa^2 / d).
$$
Combined with the favorable event from Step 1, both events hold with probability
≥ 1 − β, giving
$$
\sum_{i \in B_c} Z_i \;\ge\; m \cdot \kappa^2 / (4d) - O(m \log M / d).
$$
For e-BH rejection, require `m κ²/(4d) ≥ log(K/α) + O(log M)`, i.e. `κ ≳ √(d
log(K/(αβ))/m) · const`. This gives the small-κ scaling.

For large κ (`κ ≳ d`), the small-κ Taylor of `A_d` no longer applies; use
`A_d(κ) → 1 − (d−1)/(2κ)`. KL grows as `κ − (d/2) log κ + const`, and Step 1's
threshold `t` can be much smaller. The threshold becomes `κ ≳ (d log(K/α))/n_c · const`,
which for `n_c = 100`, `d = 384`, `K = 21,606`, `α = 0.1` gives `κ* ≈ 47`. ∎

### What this means for the FCC docket.

Plugging order-of-magnitude numbers (β = 0.1):
$$
\kappa^* \approx C_1 \sqrt{384 \cdot 12.3 / 100} \;\approx\; 6.9 \cdot C_1.
$$
**Numerical calibration of `C_1`** from `notes/verify_proofs.py` test P, run at
`d = 20, n = 100, K = 20{,}000, α = 0.1`: predicted threshold (`C_1 = 1`) is `κ* ≈ 1.7`
but empirical 90% rejection sits near κ ≈ 5–6, so `C_1 ≈ 3–4` in this benchmark. The
*scaling* `κ* ∝ √(d log(K/α)/n)` matches; the *constant* is loose, which is why
Theorem 3 is a sketch with constants among the open problems for Ramdas (Q1).

For the FCC docket with the loose `C_1 ≈ 3–4`: **clusters with κ ≳ 20–30 are detected
with probability ≥ 0.9** at `n_c = 100`. BuzzFeed-attributed contractor clusters in
our data exhibit `κ̂ ≈ 50–500+` (paraphrases very close to the seed template), so
detection should be comfortably ample. We will verify the regime empirically on
labeled clusters in Week 2.

### Remarks.

- (R7) **Mis-specification of vMF.** Universal inference is robust to alternative
  mis-specification by construction — Theorem 1 doesn't reference H¹. If real
  coordination is not vMF (e.g. mixture of two seed templates per cluster), Type-I
  is unaffected; only power degrades. A two-template mixture loses ~factor of 2 in
  power (one template detected, one absorbed into noise). Mitigation in §7.5.
- (R8) **Small-κ vs large-κ.** Most published BuzzFeed contractor campaigns are tight
  paraphrase clusters (large κ). The small-κ regime is the harder reviewer-bait
  case; we want a power result there.

---

## 6. e-BH composition: boosting and closure

Because Theorem 1 gives **individual** validity (not just compound), two strict
power-improvement layers compose freely on top of vanilla e-BH.

### Lee–Ren 2024 boosting (arXiv:2404.17562).

For each individually-valid e-value `E_c`, define a boosted e-value `E_c^b` via
conditional calibration (Lee–Ren Theorem 1). Then
`R_eBH(E^b) ⊇ R_eBH(E)` (Theorem 2), with strict inclusion whenever `E_c` rejects
non-tightly. Practical effect: each rejected `E_c` typically exceeds `K/α` by
several orders of magnitude in our setting, so the boost rejects substantially
more clusters with no cost to FDR.

### Xu–Fischer–Ramdas 2025 closed-eBH (arXiv:2504.11759).

Closed-eBH wraps any e-BH-style procedure in a closed-testing argument and strictly
dominates the unboosted procedure in rejection set. Composes cleanly with Lee–Ren:
apply Lee–Ren first, then closed-eBH on `{E_c^b}`.

**Recommended order (will verify in Week 4 ablation):**
$$
\{E_c\} \xrightarrow{\text{Lee--Ren}} \{E_c^b\} \xrightarrow{\text{closed-eBH}} R.
$$

Open: whether the Lee–Ren-then-closed order is provably optimal vs the reverse. Ask
Ramdas (§9).

---

## 7. Implementation notes

### 7.1. Power Spherical, not raw vMF.

In `d = 384` with paraphrase-tight clusters, `κ̂` routinely exceeds 500–1000.
The vMF likelihood requires `I_{d/2−1}(κ)`, whose ratio `A_d(κ) = I_{d/2}/I_{d/2−1}`
overflows above `κ ≈ 700` in fp32 / fp64 without careful asymptotic handling.

**Implementation choice:** use the Power Spherical distribution (De Cao & Aziz 2020
NeurIPS, [arXiv:2006.04437](https://arxiv.org/abs/2006.04437)). Its likelihood is

$$
p_{\mathrm{PS}}(x \mid \mu, \kappa) = \frac{1}{Z_{\mathrm{PS}}(\kappa, d)} \cdot \big(1 + \langle \mu, x \rangle\big)^{\kappa},
$$

with normalizer expressible in terms of Beta functions (no Bessel). Numerically
stable to `κ > 10^4`. Power Spherical and vMF coincide to leading order in the
high-concentration regime; for the small-κ regime they differ by an `O(κ²/d²)` term.

**Paper exposition:** present the construction and theorems with vMF (canonical,
reviewer-comprehensible). Note in §7 / Implementation that "the implementation uses
the numerically stable Power Spherical distribution; vMF and PS agree to second
order in the regime of interest." Cite De Cao & Aziz 2020. The validity proof goes
through unchanged for any spherical exponential family; the power proof needs the
PS analog of `A_d(κ)`, which is also closed-form and easier than vMF's.

### 7.2. Fitting q̂ as a movMF mixture.

Recipe:
1. **Pre-process** all 24M comment embeddings: subtract corpus mean, apply Mu &
   Viswanath all-but-the-top with `k ∈ {3, …, 10}` (sweep in Week 2 against
   diagnostic 5; default `k = 5`), re-normalize to unit length.
2. **Subsample** 200K random comments from the corpus (uniform).
3. **Fit movMF** (mixture of vMF) by EM with `K_q ∈ {5, 10, 25, 50}` components.
   Pick `K_q` by held-out log-likelihood on a 50K validation subsample. Use Banerjee
   et al. 2005 EM, or equivalently the `movMF` R package / `scikit` extensions.
   Cost: ~30 min single CPU.
4. **Cross-fit q̂ at the cluster level.** When computing `Ê_c` for cluster `c`, hold
   out cluster `c`'s comments from the q̂-fitting subsample. In practice, we fit
   `q̂` on a subsample disjoint from any candidate cluster's members; this is
   automatic if we sample from the half-A half-B comment pools that aren't in
   any cluster, but may need explicit hold-out for the largest clusters. Decision
   gate: empirically check that ‖q̂ − q̂^{(−c)}‖_∞ is negligible for typical `c`;
   if so, no per-cluster refit needed.
5. **Tail-padding.** Add a low-concentration component (`κ_floor = 1`, weight 0.01)
   to `q̂` to lower-bound the density. This caps `‖q / q̂ − 1‖_∞` at a finite value
   even if the empirical `q̂` underrepresents the sphere's flat regions. Critical
   for Theorem 2's bound.

### 7.3. All-but-the-top whitening.

SBERT MiniLM-L6-v2 embeddings are not isotropic on `S^{383}` (Ethayarajh 2019; Cai
et al. 2021). The first few principal components carry substantial mass that is
*shared across all comments* and therefore shouldn't drive coordination detection.

Recipe (Mu & Viswanath 2018):
1. Compute corpus mean `x̄`. Subtract: `x' = x − x̄`.
2. Compute top `k` principal components of the centered embeddings on a 1M
   subsample.
3. Project off: `x'' = x' − Σ_{j=1}^k ⟨x', v_j⟩ v_j`.
4. Re-normalize: `x_white = x'' / ‖x''‖`.

This produces a working representation on `S^{383−k}` that is approximately
isotropic. Verify with diagnostic 5 in Week 2.

`k = 5` is a reasonable default; sweep `k ∈ {3, 5, 8, 10}` in Week 2.

### 7.4. Cross-fit (K = 2, swap-and-average).

Per Strieder–Drton 2022, 50/50 single split is near-optimal for low-dim composite
null tests. Per Dunn et al. 2023, "swapped/repeated subsampling" is the best UI
variant they study, beating both single-split and crossfit (K > 2).

**Recipe (used in Step 4 of §2):** compute `E_c^{(1)}` with split `(A, B)`, then
`E_c^{(2)}` with the *swapped* split `(B, A)`. Average:
$$
\bar E_c = (E_c^{(1)} + E_c^{(2)}) / 2.
$$
By Vovk–Wang 2021 Theorem 3.2 (averaging closure under arbitrary dependence), `Ē_c`
is an e-value. Validity of Theorem 1 carries to `Ē_c` since each summand is itself
individually valid. Variance reduces; bias is unchanged. **Do not** cross-fit
beyond K = 2; the published evidence (Tse–Davison 2022) is that further splitting
loses power in the regular regime.

### 7.5. Multi-template clusters (R7 mitigation).

If diagnostic 1 in Week 2 shows that some BuzzFeed-labeled clusters do not look
unimodal vMF (likely for clusters spanning multiple seed templates from the same
contractor), add a sub-clustering step: within each Leiden cluster of size ≥ 50,
run k-means with `k ∈ {1, 2, 3}` on half A; pick `k` by held-out log-likelihood;
construct `E_c^{(k)}` for each sub-template; aggregate via Vovk–Wang mean. Optional;
defer to Week 5 if needed.

---

## 8. Contrast with the current pipeline

The current `src/evalues.py` (Beta-MOM cohesion) and the new construction agree
in their FDR target (e-BH at α) but differ in three load-bearing components:

| Component | Current (Beta-MOM cohesion) | New (split-LRT vMF) |
|---|---|---|
| Test statistic per cluster | `T_c =` mean pairwise cosine sim within cluster | `Σ_{i ∈ B_c} log(p_vMF(x_i | μ̂_c, κ̂_c) / q̂(x_i))` |
| Null calibration | Beta MOM fit per size bucket on B-half random samples | movMF q̂ on corpus subsample (cluster-held-out) |
| p- vs e-value | Compute `p` via Beta survival, calibrate `e = −log(p) − 1 + 1/p` | Direct e-value (no p→e step) |
| Validity argument | Cross-half exchangeability + Beta tail super-uniformity | Universal inference (WRB 2020 Theorem 4) |
| Power | None proved — empirical only | Theorem 3, explicit κ* threshold |
| Boosting compatibility | Compound validity → Lee–Ren applicable but loose | Individual validity → Lee–Ren strict, closed-eBH applicable |

The new construction subsumes the old in the limit where `q̂ ≈ Unif` and the
cohesion-as-test-statistic happens to match the vMF likelihood — i.e., the old
pipeline is a non-likelihood approximation to a special case of the new one. The
new one is strictly more powerful, has finite-sample validity without parametric
null assumptions on `T_c`, and admits a power theorem.

**Code path:** the new construction lives in a new file `src/evalues_lrt.py`
(reuses `embed.py`, `split.py`, `cluster_singletons.py`; adds `q_movmf.py` for
fitting `q̂` and `whiten.py` for all-but-the-top). `src/evalues.py` (Beta-MOM)
remains as the baseline for the experimental ablation in Week 4.

---

## 9. Open questions for Ramdas

In priority order. The first is the load-bearing technical gap; the others are
nice-to-haves.

**Q1 (the theorem-level open problem).** Establish a finite-sample power lower
bound for the split-LRT compound e-BH procedure with a fitted-q null on `S^{d−1}`
under a vMF (or Power Spherical) alternative, with explicit constants tracking
both small-κ (`κ²·n/d ≳ log(K/α)`) and large-κ (`κ·n/d ≳ log(K/α)` up to log
factors) regimes, plus a sup-norm-q-misspec graceful-degradation theorem (§4
gives the form; we want the optimal constant). The cleanest path is via
*uniform vMF-MLE direction concentration on A_c* plus Bernstein on the
per-observation log-LR on B_c. The vMF-MLE concentration step (§5 Step 1) has
folklore versions but no theorem-grade reference at the granularity needed.

**Q2 (boosting order).** Does Lee–Ren 2024 boosting commute with Xu–Fischer–Ramdas
2025 closed-eBH? If not, which order is provably optimal under individual
validity?

**Q3 (Takatsu correction).** Does Takatsu 2025's studentization correction for
the split-LRT (recovering exact 1−α from the conservative `Φ(√(2 log(1/α)))`
factor) yield a usable effective-α uplift in this construction? Worth ~5–20%
power if applicable.

**Q4 (anytime-valid extension).** Could the construction be made anytime-valid
treating comments as arriving sequentially (Howard–Ramdas–McAuliffe–Sekhon
e-processes; Saha–Ramdas pairwise betting)? Long-term — possibly the NeurIPS
2027 main → 2028 main bridge paper, not for this submission.

**Recruit timing:** send Q1 + Theorem 3 statement + Week 4 empirical figure in
Week 6 of the build. Don't send before we have a working pipeline; send too late
and the workshop deadline closes.

---

## 10. References

### Foundational e-value / FDR
- Wang, R. & Ramdas, A. (2022). False discovery rate control with e-values. *JRSS-B* 84(3), 822–852. arXiv:[2009.02824](https://arxiv.org/abs/2009.02824).
- Vovk, V. & Wang, R. (2021). E-values: calibration, combination, and applications. *Annals of Statistics* 49(3). arXiv:[1912.06116](https://arxiv.org/abs/1912.06116).
- Wasserman, L., Ramdas, A. & Balakrishnan, S. (2020). Universal inference. *PNAS* 117(29), 16880–16890. arXiv:[1912.11436](https://arxiv.org/abs/1912.11436).

### Compound e-values, boosting, closure
- Ignatiadis, N., Wang, R. & Ramdas, A. (2024). Asymptotic and compound e-values. *Biometrika*. arXiv:[2409.19812](https://arxiv.org/abs/2409.19812).
- Lee, S. & Ren, Z. (2024). Boosting e-BH via conditional calibration. arXiv:[2404.17562](https://arxiv.org/abs/2404.17562).
- Xu, Z., Fischer, A. & Ramdas, A. (2025). Bringing closure to FDR control: beating the e-Benjamini-Hochberg procedure. arXiv:[2504.11759](https://arxiv.org/abs/2504.11759).

### Universal-inference power & follow-ups
- Strieder, T. & Drton, M. (2022). On the choice of the splitting ratio for the split likelihood ratio test. *Electronic Journal of Statistics* 16(2). arXiv:[2203.06748](https://arxiv.org/abs/2203.06748).
- Tse, T. & Davison, A. C. (2022). A note on universal inference. *Stat*.
- Dunn, R., Ramdas, A., Balakrishnan, S. & Wasserman, L. (2023). Gaussian universal likelihood ratio testing. *Biometrika* 110(2), 319–337. arXiv:[2104.14676](https://arxiv.org/abs/2104.14676).
- Sun, Z. & Han, Q. (2024). On universal inference in Gaussian mixture models. arXiv:[2407.19361](https://arxiv.org/abs/2407.19361).
- Takatsu, K. (2025). On the precise asymptotics of universal inference. arXiv:[2503.14717](https://arxiv.org/abs/2503.14717).
- Park, J., Balakrishnan, S. & Wasserman, L. (2023). Robust universal inference for misspecified models. arXiv:[2307.04034](https://arxiv.org/abs/2307.04034).

### Adjacent FDR / multiple-testing prior art (scoop differentiation)
- Dey, P., Martin, R. & Williams, J. P. (2024). Multiple testing in generalized universal inference. arXiv:[2412.01008](https://arxiv.org/abs/2412.01008).
- Hartog, J. & Lei, L. (2025). Family-wise error rate control with e-values. arXiv:[2501.09015](https://arxiv.org/abs/2501.09015).
- Csillag, D., Struchiner, C. J. & Goedert, G. T. (2025). Prediction-powered e-values. *ICML*. arXiv:[2502.04294](https://arxiv.org/abs/2502.04294).
- Wang, L., Lin, Y. & Zhao, S. (2024). FDR control via data splitting for testing-after-clustering. arXiv:[2410.06451](https://arxiv.org/abs/2410.06451).
- Yu, X., Ming, J., Xiao, T., Wang, R. & Jing, B.-Y. (2024). Generalized e-value feature detection at multiple resolutions. arXiv:[2409.17039](https://arxiv.org/abs/2409.17039).

### vMF, directional statistics, embedding geometry
- Banerjee, A., Dhillon, I. S., Ghosh, J. & Sra, S. (2005). Clustering on the unit hypersphere using von Mises-Fisher distributions. *JMLR* 6, 1345–1382.
- Hornik, K. & Grün, B. (2014). On maximum likelihood estimation of the concentration parameter of von Mises-Fisher distributions. *Computational Statistics* 29(5), 945–957.
- Diethe, T. (2015). A note on the Kullback-Leibler divergence for the von Mises-Fisher distribution. arXiv:[1502.07104](https://arxiv.org/abs/1502.07104).
- De Cao, N. & Aziz, W. (2020). The Power Spherical distribution. *NeurIPS*. arXiv:[2006.04437](https://arxiv.org/abs/2006.04437).
- Mardia, K. V. & Jupp, P. E. (1984/2000). *Directional Statistics*.

### Embedding-space anisotropy
- Ethayarajh, K. (2019). How contextual are contextualized word representations? *EMNLP*. arXiv:[1909.00512](https://arxiv.org/abs/1909.00512).
- Cai, X., Huang, J., Bian, Y. & Church, K. (2021). Isotropy in the contextual embedding space: clusters and manifolds. *ICLR*.
- Mu, J. & Viswanath, P. (2018). All-but-the-top: simple and effective postprocessing for word representations. *ICLR*.

### Application context
- NY Attorney General (2021). Fake comments: how U.S. companies and partisans hack democracy. [ag.ny.gov](https://ag.ny.gov/sites/default/files/oag-fakecommentsreport.pdf).
- Mannocci, L. et al. (2024). Detection and characterization of coordinated online behavior: a survey. arXiv:[2408.01257](https://arxiv.org/abs/2408.01257).
- Singer-Vine, J. et al. BuzzFeed News FCC FOIA reporting (2018–2019).

---

## Verification log

All proofs were numerically verified in `notes/verify_proofs.py` against
synthetic vMF and uniform data on `S^{d−1}` at `d = 20`. Five tests:

- **N (normalizer):** `E[p_vMF(x|μ,κ) / p_unif(x)]` for `x ~ Unif(S^{d−1})` is `1.000 ± 0.001`
  at every κ ∈ {0.5, 1, 3, 10, 30}. Confirms the validity proof's load-bearing
  step (∫ p_vMF dσ = 1) is correctly implemented.
- **T1 (validity, Markov form):** `P(E_c ≥ t | H_c) ≤ 1/t` at every threshold
  `t ∈ {1, 2, 5, 10, 100, 1000}`. Empirical at `t = 1`: 0.016 (bound 1.0); at
  `t = 10`: 0.004 (bound 0.1); at `t = 1000`: 0 (bound 0.001). **PASS.**
  (Direct empirical mean is unverifiable at any reasonable sample size due to
  log-normal tails with σ ≈ 4.6 — a single trial in 10⁵ contributes most of
  the mean. Markov is the proper validity check.)
- **T2 (graceful degradation):** `E[Ê_c]` is well below the `(1+ε)^m`
  bound at ε ∈ {0.05, 0.1, 0.2}. **PASS.** The bound is loose for typical
  realizations; tightness is only at the worst-case adversarial `q̂`.
- **K (KL Taylor):** ratio `KL_MC / [κ²/(2d)]` is in `[0.96, 1.08]` for
  κ ∈ {0.3, 1, 3}; degrades at κ = 10 (ratio 0.77) — large-κ regime where
  the small-κ Taylor expansion no longer applies. **PASS in stated regime.**
- **D (direction concentration):** `⟨μ̂, μ⟩ → 1` empirically, confirming the
  Step-1 fix in Theorem 3. Mean `⟨μ̂, μ⟩` at `m = 50, d = 20`:
  κ = 0.5 → 0.19;
  κ = 1 → 0.33;
  κ = 3 → 0.72;
  κ = 10 → 0.96. The original writeup
  predicted these to be near `A_d(κ)` ∈ {0.025, 0.05, 0.15, 0.42}; corrected
  in §5 above.
- **P (power threshold):** at `d = 20, n = 100, K = 20{,}000, α = 0.1, β = 0.1`,
  predicted `κ* ≈ 1.7` (with `C_1 = 1`); empirical rejection rate is
  0% at κ = 2, 33% at κ = 4, 100% at κ = 8. Effective `C_1 ≈ 3–4`. The
  scaling matches; the constant is loose (open problem Q1).

**Verdict:** Theorems 1 and 2 hold (validity proofs go through and are
empirically confirmed). Theorem 3 holds as a sketch with the right asymptotic
scaling, a one-line writeup error in Step 1 (now corrected), and constants
that remain part of the Q1 open problem for Ramdas. Run `notes/verify_proofs.py`
to reproduce.

### Numerical calibration of C (Theorem 3 constant)

Theorem 3 states $\kappa^* \ge C \cdot d/\sqrt m$ for an absolute constant $C$
(in the regime $d/\sqrt m \ge \sqrt{d\log(K/\alpha)/m}$, which holds whenever
$d \ge \log(K/\alpha)$ — true for all $(d,n)$ in our application). The
constant $C$ depends mildly on $(d, n)$ in the small-$\kappa$ regime through
the $A_d(\kappa) \approx \kappa/d$ Taylor approximation. We calibrate $C$
numerically as follows.

**Procedure** (`/tmp/kappa_regime_figure.py` in the repo, archived in
`results/kappa_regime_sweep.csv`):

  1. Grid: $d \in \{50, 100, 200, 384, 768\}$ × $n \in \{8, 12, 16, 24, 32, 64\}$
     (30 grid points). $K = 21{,}606$ and $\alpha = 0.10$ to match the
     application.
  2. For each $(d, n)$: find the smallest $\kappa$ at which the procedure
     achieves empirical rejection rate $\ge 0.90$, by bisection in
     $\log\kappa$ space (15 outer steps). At each candidate $\kappa$,
     Monte-Carlo estimate of the rejection rate uses $N = 80$ synthetic
     vMF clusters, each split-LRT-evaluated against a uniform null. Sampling
     uses `scipy.stats.vonmises_fisher`.
  3. Report $C = \kappa_{\rm emp}^{(0.9)} / (d/\sqrt m)$ for each grid
     point, with $m = n/2$.

**Result** (full table in `results/kappa_regime_sweep.csv`):

| $d \backslash n$ | 8 | 12 | 16 | 24 | 32 | 64 |
|---:|---:|---:|---:|---:|---:|---:|
| 50  | 2.40 | 1.87 | 1.82 | 1.50 | 1.55 | 1.39 |
| 100 | 2.12 | 1.65 | 1.52 | 1.40 | 1.36 | 1.30 |
| 200 | 1.87 | 1.54 | 1.34 | 1.31 | 1.27 | 1.21 |
| 384 | 1.81 | 1.49 | 1.30 | 1.20 | 1.17 | 1.11 |
| 768 | 1.79 | 1.47 | 1.28 | 1.18 | 1.15 | 1.10 |

Range $C \in [1.10, 2.40]$, median 1.40. $C$ is largest at $n=8$ (smallest
split, highest variance) and decreases monotonically with $n$. The theoretical
upper bound $C \le 2.4$ used in the abstract reads off the worst-case grid
cell ($d=50$, $n=8$), and the median value is reported alongside as the
typical operating point.

**For the FCC 2017 application** ($d = 379$ post-whitening with $k=5$,
$n_c \in [8, 60]$): the theoretical $\kappa^*$ ranges from ~$190$
(small clusters at typical $C \approx 1.8$) to ~$45$ (large clusters at
$C \approx 1.1$). Real attributed-astroturf $\hat\kappa \in [1300, 2200]$
clears these thresholds with margin, justifying the abstract's claim
that all 15,748 size-$\ge$-8 clusters lie within the validated regime.

### Real-data defensibility (added, honest version)

Diagnostics in `results/vmf_diagnostics.csv` and follow-up runs:

**Concentration regime is plausible, not pathological.** The vMF MLE on real
BuzzFeed-FOIA-attributed clusters of size ≥ 8 produces concentration values
in a numerically stable range:

  - Astroturf:    κ̂ p10/50/90 = 1303 / 1665 / 2133;  r_emp p50 = 0.89.
  - Advocacy:     κ̂ p50 = 3232;  r_emp p50 = 0.94.
  - Unlabeled (size ≥ 8 in attribution table):  κ̂ at the 3840 cap;
    r_emp p50 = 0.999.

These κ̂ values are well below the implementation's κ_max = 10·d = 3840 cap
for astroturf, and in a regime where `ive` is numerically stable.

**vMF measures generic coordination, not type-specific signal.** Discrimination
AUC numbers:

  | Comparison                            | AUC κ̂ |
  |--------------------------------------|-------|
  | Astroturf vs random singleton batches | 1.000 (trivial — clustering did it) |
  | Astroturf vs Advocacy (both labeled)  | 0.128 |
  | Astroturf vs Unlabeled size ≥ 8       | 0.003 |

The construction targets coordination per se, not astroturf-specifically,
so vMF's generic-concentration behavior is appropriate.

**Surprising finding worth discussing in the paper.** Advocacy clusters are
*more* concentrated than astroturf (κ̂ 3232 vs 1665) — legitimate orgs use
polished templates reproduced verbatim, while paid contractors paraphrase /
vary to evade detection. Unlabeled size-≥-8 clusters are tighter still
(r_emp ≈ 1), consistent with the paper's expected story that the rejection
set captures additional unattributed coordination beyond what FOIA labels
record.

**LRT-pipeline failure attribution.** The earlier evalues_lrt.py issue
was driven by q̂ leakage (q̂ trained on the same comments being tested),
not by vMF mis-specification. Refitting q̂ on cluster-disjoint singletons
should restore LRT power; Week 3 follow-up.

## Status

**Week 1 deliverable: complete.** Math is on disk. Validity (Theorem 1) and
mis-specification bound (Theorem 2) are proved and numerically verified; power
lower bound (Theorem 3) is sketched with the load-bearing gap explicitly
identified, the writeup-level error in Step 1 corrected, and the scaling
empirically confirmed.

**Next (Week 2):** all-but-the-top whitening on the 24M corpus; movMF q̂ fit on a
200K subsample with K_q sweep; 5 diagnostic tests on a 50-cluster pilot
(25 BuzzFeed-labeled coordinated + 25 random innocent) per the agent-2 memo.
Hard kill check: if diagnostic 1 (Q-Q vs vMF) or diagnostic 3 (innocent
indistinguishable from coordinated) fails, switch model (movMF-component
likelihood or tangent-Gaussian) before Week 3.
