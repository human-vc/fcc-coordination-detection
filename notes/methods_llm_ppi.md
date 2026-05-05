# Methods: LLM-as-Judge with PPI-Calibrated Compound E-Values

(Section 4 of the paper. Lives between the universal-inference construction
and the experiments. Adds the new ML methodology contribution.)

## 4. PPI-Calibrated Compound E-Values

### 4.1 Construction

Let $f: \mathcal{T}^k \to [0, 1]$ denote a language-model judge that, given $k$
representative comments from a candidate cluster $c$, returns
$\hat\pi_c = f(\text{texts}_c) \in [0, 1]$ — the model's estimated probability
that the cluster is a coordinated paraphrase campaign. Because the judge is
expensive ($\$10^{-3}$ per call but possibly noisy), we cannot evaluate it on
all $K$ candidate clusters and treat its output as gospel.

We do two things with $\hat\pi$:

(i) Treat it as a *test statistic* in the IWR §7 mixture-likelihood-ratio
compound e-value. Let $g_0(\cdot)$ and $g_1(\cdot)$ be Beta densities fit by
method of moments on $\hat\pi$ values for FOIA-non-astroturf and
FOIA-astroturf labeled clusters respectively. Then for any cluster $c$,
$$E_c^{\rm LLM} = \frac{g_1(\hat\pi_c)}{g_0(\hat\pi_c)}.$$
By the integral identity $\int g_1 = 1$, $E_c^{\rm LLM}$ is a (compound)
e-value, and e-BH on $\{E_c^{\rm LLM}\}$ controls FDR at $\alpha$.

(ii) Calibrate the judge's *bias* against the FOIA-attributed gold subset
using PPI++ (Angelopoulos, Duchi, Zrnic, 2023), giving a debiased estimate
of the true coordination rate in the rejection set.

The combination — IWR mixture-LR e-values built from a PPI-calibrated LLM
judgment — is the new ML methodology contribution.

### 4.2 The judge

Single API call per cluster, prompt template (see Appendix A):

> *You are reviewing comments submitted to a U.S. federal regulatory docket.
> Below are 3 comments that an automated clustering algorithm grouped together
> because their text embeddings were highly similar. Your task: classify
> whether these comments are part of a coordinated paraphrase campaign...*

Three-comment summaries are sampled uniformly without replacement from
each cluster. Texts longer than 1200 chars are truncated. The judge returns
a binary classification + a $[0, 1]$ self-reported confidence; we map
this to $\hat\pi_c \in [0, 1]$ by setting $\hat\pi = c$ if "coordinated"
else $1-c$.

Cost: $\sim$ 500 input tokens + $\sim$ 50 output tokens per call. At the
Haiku 4.5 price of $\$1$/$\$5$ per million in/out tokens, a 1000-cluster
sweep costs $\$0.80$.

### 4.3 PPI++ calibration

Let $\mathcal{L} = \{c : c \text{ is FOIA-attributed astroturf}\}$ be our
gold set with $y_c = 1$. We assume the FOIA-attributed-but-astroturf
labels are reliable $y$-positives; FOIA *absence* is a less reliable
$y$-negative (NYAG documented presence, not absence), which we discuss
in §6.

The PPI++ estimator of $\theta = \mathbb E_{c \sim \text{population}}[y_c]$ is
$$\hat\theta_{\rm PPI} = \lambda \cdot \frac{1}{|\mathcal{U}|}\sum_{c \in \mathcal{U}} \hat\pi_c
+ \frac{1}{|\mathcal{L}|}\sum_{c \in \mathcal{L}} \big(y_c - \lambda \hat\pi_c\big),$$
where $\mathcal{U}$ is the (large) sample of LLM-judged clusters without
gold labels and $\lambda$ is a tuning parameter; $\lambda = 1$ recovers the
original PPI estimator and $\lambda^* = \text{Cov}(\hat\pi, y)/\text{Var}(\hat\pi)$
is the variance-optimal choice. The corresponding 95% CI is
$\hat\theta \pm 1.96 \cdot \text{SE}(\hat\theta)$ with the standard PPI++
variance formula.

### 4.4 Validity

Theorem 1 (Wang–Ramdas e-BH) gives $\text{FDR} \le \alpha$ provided each
$E_c^{\rm LLM}$ satisfies $\mathbb E[E_c^{\rm LLM}|H_c] \le 1$. This holds
when $g_0$ is the true marginal of $\hat\pi$ under $H_c$: the cluster is
not coordinated. In practice we estimate $g_0$ from the FOIA-non-astroturf
subset, which is a subset of the true non-coordinated population (it
excludes pro-NN advocacy mobilization, which is also coordination).

If the FOIA-non-astroturf subset is *more* coordinated than the population
average — which our results suggest, since pro-NN advocacy mobilization
that FOIA does not enumerate is itself coordinated — then $g_0$
overestimates the true $H_c$ density at high $\hat\pi$, making the e-value
$E_c^{\rm LLM} = g_1/g_0$ *smaller* than it would be under the true null.
This means the procedure is *anti-conservative* for FDR. The gap quantifies
how much FOIA undercounts the alternative population.

A clean way to side-step this is to use universal inference's split-LRT
e-value (§3 of the paper) as the primary procedure and the LLM-judge
construction as a corroborating ablation. We do both.

### 4.5 Computational considerations

The full pipeline at $K=15{,}748$ size-$\ge$-8 clusters with 100% LLM coverage
costs $\sim\$13$ on Haiku 4.5. We instead use stratified subsampling:
1000 clusters split evenly between FOIA-astroturf and FOIA-non-astroturf,
fitting Beta(g_0, g_1) on the labeled half and applying $E^{\rm LLM}$ to
the unlabeled half via shared Beta parameters. This costs $\$0.80$ per
sweep, with the variance of the precision claim controlled by PPI++
($\propto 1/\sqrt{|\mathcal{L}|}$).

For larger or held-out dockets, full LLM coverage at the size of FCC 17-108
remains affordable ($\sim\$15$).

---

*References to fill in:*
- Angelopoulos, Duchi, Zrnic 2023. PPI++. arXiv:2311.01453.
- Csillag, Struchiner, Goedert 2025. Prediction-Powered E-Values. ICML.
  arXiv:2502.04294.
- Fisch et al. 2024. Stratified PPI for Hybrid LM Evaluation. NeurIPS 2024.
  arXiv:2406.04291.
- Ignatiadis, Wang, Ramdas 2024. Asymptotic and compound e-values.
  arXiv:2409.19812.
- Wang & Ramdas 2022. FDR control with e-values. JRSS-B.
