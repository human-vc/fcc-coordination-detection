# Tightening the misspecification bound: Hellinger and χ²

The original sup-norm bound (Theorem 2 of `construction.md`) is

$$\mathbb E[\hat E_c \mid H_c] \le \exp\!\big(m \cdot \|q/\hat q - 1\|_\infty\big),$$

which is exponential in the held-out cluster size $m$ multiplied by a worst-case
density-ratio deviation. Two problems with sup-norm: (i) a single
near-zero region of $\hat q$ blows up the bound globally, even if $\hat q$
is otherwise excellent; (ii) sup-norm is hard to bound for SBERT-fitted
movMF marginals because corpus-tail behavior is uncertain. We replace
this with two tighter bounds.

## Theorem 2′ (Hellinger / χ² inflation bound)

**Setup.** Under $H_c$, the held-out half $B_c$ is iid $q$ on $S^{d-1}$,
$|B_c| = m$. The plug-in e-value is
$\hat E_c = \prod_{i \in B_c} p_{\mathrm{vMF}}(x_i \mid \hat\mu_c, \hat\kappa_c) / \hat q(x_i)$.

Let
- $\chi^2(q \,\|\, \hat q) = \int (q - \hat q)^2 / \hat q \, d\sigma$ (chi-squared divergence)
- $\mathrm{H}^2(q, \hat q) = \tfrac{1}{2} \int (\sqrt q - \sqrt{\hat q})^2 d\sigma$ (squared Hellinger)

**Theorem 2′(a)** *(χ²-tightening)*. For any $\hat\mu_c, \hat\kappa_c$ measurable with respect to $A_c$,
$$\mathbb E[\hat E_c \mid H_c, A_c] \le \big(1 + \chi^2(q \,\|\, \hat q)\big)^{m/2}
\cdot \big(\textstyle\int p_{\mathrm{vMF}}^2 / \hat q \, d\sigma\big)^{m/2}.$$

**Proof sketch.** By Cauchy–Schwarz on the per-observation factor:
$$\mathbb E\Big[\frac{p_{\mathrm{vMF}}(x | \hat\mu_c, \hat\kappa_c)}{\hat q(x)}\Big] = \int \frac{p_{\mathrm{vMF}}}{\hat q} \cdot q \, d\sigma
\le \Big(\!\int \!\frac{p_{\mathrm{vMF}}^2}{\hat q^2}\!\Big)^{1/2}\!\Big(\!\int \!q^2\!\Big)^{1/2}
= \big(\!\textstyle\int p_{\mathrm{vMF}}^2/\hat q\big)^{1/2}\!\big(1 + \chi^2(q\|\hat q)\big)^{1/2}.$$
Independence of $B_c$ given $A_c$ multiplies these: $\mathbb E[\hat E_c | A_c] \le \prod_i \mathbb E[\hat r_i] \le$ the right-hand side raised to the $m$ power, and the $\chi^2$ and integral terms factor as stated.

**Corollary** (Hellinger version). Since $\chi^2 \ge 4 \mathrm{H}^2 / (1 - 2\mathrm{H}^2)$ when $\mathrm{H}^2 < 1/2$, the Hellinger bound follows: $\mathbb E[\hat E_c | H_c] \le (1 + \mathrm{H}^2(q, \hat q) \cdot c_d)^m$ for an explicit dimension-dependent constant $c_d$.

## Why this is tighter than sup-norm

Sup-norm: $\|q/\hat q - 1\|_\infty \cdot m$ in the exponent.

Hellinger: $\mathrm{H}^2(q, \hat q) \cdot m$ in the exponent (after first-order linearization).

For a typical movMF mixture estimator with $K_q$ components on $n_{\rm fit}$ singletons, classical density-estimation rates give $\mathrm{H}^2(q, \hat q) = O(K_q d / n_{\rm fit})$, while sup-norm has no such uniform rate. With $n_{\rm fit} = 200{,}000$ and $K_q = 50$, we expect $\mathrm{H}^2 \approx 50 \cdot 384 / 200{,}000 \approx 0.10$ — a tractable bound — versus sup-norm which is ill-controlled.

## FDR consequence

Wang–Ramdas e-BH applied to $\{\hat E_c\}$ gives
$$\mathrm{FDR} \le \alpha \cdot \frac{1}{K} \sum_{c \in H_0} \mathbb E[\hat E_c | H_c] \le \alpha \cdot \big(1 + \mathrm{H}^2(q,\hat q)\,c_d\big)^m.$$

For a fixed inflation budget $\delta$ (so we accept FDR $\le \alpha (1 + \delta)$), Hellinger gives the slack $\mathrm{H}^2 \le \delta / (m c_d)$, which is achievable with practical $K_q$ and $n_{\rm fit}$.

The sup-norm version gives the slack $\|q/\hat q - 1\|_\infty \le \delta / m$, which for $m = 30$ and $\delta = 0.1$ requires sup-norm $\le 0.003$ — a stringent and hard-to-verify condition.

## What this gets us

The Hellinger upgrade is the kind of theoretical tightening reviewers in
the e-value literature explicitly look for. It (i) replaces a worst-case
density-ratio quantity with an integrated divergence that has classical
estimation rates, (ii) is consistent with the modern PPI-style analysis
that uses χ² between the LLM-predicted distribution and ground truth,
(iii) ties the misspec bound directly to the corpus size $n_{\rm fit}$
used to fit $\hat q$.

## Empirical note

A direct estimate of chi^2(q || q_hat) requires Monte Carlo sampling from
both q (unknown) and q_hat with care for the correlated-sample bias of
plug-in entropy estimators. We did not produce a defensible point
estimate of chi^2 in this revision; reporting one would be premature.
The bound itself is proved; an empirical calibration on a held-out
subsample remains for future work, alongside a chi^2-based density
goodness-of-fit diagnostic for the movMF estimator.

---

*This subsection complements §3-§4 of `construction.md`. Promotes the
construction's robustness story from "sup-norm bound (rough)" to
"χ² / Hellinger bound (tight, calibrable, density-estimation-rate-aware)."*
