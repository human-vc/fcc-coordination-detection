"""Methodology bake-off baselines for AICE comparison.

Implements canonical FDR-controlling procedures from the literature.
Each baseline takes p-values (or e-values) and a covariate (where applicable)
and returns rejection indices at level alpha.

References
----------
- Benjamini-Hochberg (1995). JRSS-B 57(1).
- Storey (2002). JRSS-B 64(3).
- Storey-Tibshirani (2003). PNAS 100(16).
- Wang-Ramdas (2022). FDR control with e-values. JRSS-B 84(3).
- Lee-Ren (2024). Boosting e-BH via conditional calibration. arXiv:2404.17562.
- Ignatiadis-Klaus-Zaugg-Huber (2016). IHW. Nat Methods 13(7).
"""
from __future__ import annotations
import numpy as np
from typing import Optional


# ---------- BH and Storey ----------

def bh(p: np.ndarray, alpha: float) -> tuple[np.ndarray, int]:
    """Standard Benjamini-Hochberg procedure.

    Parameters
    ----------
    p : ndarray, p-values in [0, 1].
    alpha : float, nominal FDR level.

    Returns
    -------
    rej, k_hat : indices of rejected hypotheses, count of rejections.
    """
    p = np.asarray(p, dtype=np.float64)
    K = len(p)
    if K == 0:
        return np.array([], dtype=int), 0
    order = np.argsort(p)
    sorted_p = p[order]
    threshold = alpha * np.arange(1, K + 1) / K
    valid = np.where(sorted_p <= threshold)[0]
    k_hat = int(valid.max() + 1) if valid.size else 0
    return order[:k_hat], k_hat


def by(p: np.ndarray, alpha: float) -> tuple[np.ndarray, int]:
    """Benjamini-Yekutieli (2001), valid under arbitrary dependence."""
    p = np.asarray(p, dtype=np.float64)
    K = len(p)
    if K == 0:
        return np.array([], dtype=int), 0
    H_K = float(np.sum(1.0 / np.arange(1, K + 1)))
    return bh(p, alpha / H_K)


def storey_qvalue(p: np.ndarray, alpha: float, lam_grid: Optional[np.ndarray] = None,
                  lambda_fixed: Optional[float] = None) -> tuple[np.ndarray, int, float]:
    """Storey-Tibshirani (2003) q-value procedure.

    Estimates pi_0 via cubic-spline smoother over lambda grid (default
    {0.05, 0.10, ..., 0.95}); applies BH at adjusted level.

    For small K (< 200) or unstable estimates, use lambda_fixed=0.5.

    Returns
    -------
    rej, k_hat, pi_0_hat
    """
    from scipy.interpolate import UnivariateSpline
    p = np.asarray(p, dtype=np.float64)
    K = len(p)
    if K == 0:
        return np.array([], dtype=int), 0, 1.0

    if lambda_fixed is not None:
        pi_0 = float((p > lambda_fixed).sum() / (K * (1 - lambda_fixed)))
        pi_0 = min(max(pi_0, 1e-8), 1.0)
    else:
        if lam_grid is None:
            lam_grid = np.arange(0.05, 0.96, 0.05)
        pi_0_grid = np.array([(p > lam).sum() / (K * (1 - lam)) for lam in lam_grid])
        pi_0_grid = np.clip(pi_0_grid, 1e-8, 1.0)
        try:
            spline = UnivariateSpline(lam_grid, pi_0_grid, k=3, s=0)
            pi_0 = float(spline(lam_grid.max()))
            pi_0 = min(max(pi_0, 1e-8), 1.0)
        except Exception:
            pi_0 = float(pi_0_grid[-1])

    return (*bh(p, alpha / pi_0), pi_0)


# ---------- e-BH and variants ----------

def ebh(log_e: np.ndarray, alpha: float) -> tuple[np.ndarray, int]:
    """e-BH (Wang-Ramdas 2022). Reject E_(k) >= K / (alpha * k)."""
    log_e = np.asarray(log_e, dtype=np.float64)
    K = len(log_e)
    if K == 0:
        return np.array([], dtype=int), 0
    e = np.exp(np.clip(log_e, -700, 700))
    order = np.argsort(-e)
    threshold = K / (alpha * np.arange(1, K + 1))
    rej_idx = np.where(e[order] >= threshold)[0]
    k_hat = int(rej_idx.max() + 1) if rej_idx.size else 0
    return order[:k_hat], k_hat


def ebh_lr_naive(stat: np.ndarray, a0: float, b0: float, a1: float, b1: float,
                 alpha: float) -> tuple[np.ndarray, int]:
    """Vanilla e-BH using naive likelihood ratio with KNOWN g_0, g_1.

    This is the e-BH baseline that does NOT use anchor identification or
    cross-fit. Assumes oracle access to (g_0, g_1) parameters. Provides
    an upper bound on e-BH performance with parametric e-values.
    """
    from scipy.stats import beta as beta_dist
    stat_c = np.clip(stat, 1e-6, 1 - 1e-6)
    log_e = beta_dist.logpdf(stat_c, a1, b1) - beta_dist.logpdf(stat_c, a0, b0)
    return ebh(log_e, alpha)


# ---------- Lee-Ren conditional calibration boost ----------

def ebh_cc(stat: np.ndarray, anchor_mask: np.ndarray, alpha: float,
           a0_est: tuple[float, float], a1_est: tuple[float, float],
           n_mc: int = 200, seed: int = 0) -> tuple[np.ndarray, int]:
    """Lee-Ren (2024) conditional calibration boost of e-BH.

    Boosts each e-value by Monte Carlo over the conditional null
    distribution given the sufficient statistic. Algorithm 1 of
    arXiv:2404.17562, simplified: for each j, find the largest c such that
    the conditional rejection probability under H_0 stays below alpha.

    Conservative implementation; does not implement the full conditional-
    calibration with e-process refinement, which requires knowing the
    sufficient statistic structure of the test problem.
    """
    from scipy.stats import beta as beta_dist
    rng = np.random.default_rng(seed)
    K = len(stat)
    a0, b0 = a0_est
    a1, b1 = a1_est

    stat_c = np.clip(stat, 1e-6, 1 - 1e-6)
    log_e = beta_dist.logpdf(stat_c, a1, b1) - beta_dist.logpdf(stat_c, a0, b0)
    e = np.exp(np.clip(log_e, -700, 700))

    # Boost factor: rough approximation of the conditional calibration.
    # Resample under H_0, count fraction with e >= threshold, compute boost.
    boost_factors = np.ones(K)
    for trial in range(n_mc):
        x_null = beta_dist.rvs(a0, b0, size=K, random_state=rng)
        e_null = np.exp(beta_dist.logpdf(np.clip(x_null, 1e-6, 1 - 1e-6), a1, b1)
                        - beta_dist.logpdf(np.clip(x_null, 1e-6, 1 - 1e-6), a0, b0))
        boost_factors += (e_null < e).astype(float)
    boost_factors /= (n_mc + 1)
    log_e_boosted = log_e + np.log(np.maximum(boost_factors, 1e-8))
    return ebh(log_e_boosted, alpha)


# ---------- Group-BH (simplified IHW substitute when rpy2 unavailable) ----------

def group_bh(p: np.ndarray, covariate: np.ndarray, alpha: float, n_groups: int = 4
             ) -> tuple[np.ndarray, int]:
    """Group-wise BH using quantile bins of covariate.

    A simplified, pure-Python substitute for IHW (Ignatiadis et al. 2016)
    when rpy2 is unavailable. Bins hypotheses by covariate quantiles and
    applies BH within each group. Less powerful than full IHW but provides
    a covariate-adaptive baseline.
    """
    p = np.asarray(p, dtype=np.float64)
    cov = np.asarray(covariate, dtype=np.float64)
    K = len(p)
    if K == 0:
        return np.array([], dtype=int), 0

    quantiles = np.quantile(cov, np.linspace(0, 1, n_groups + 1))
    quantiles[0] -= 1e-9
    bins = np.digitize(cov, quantiles[1:-1])

    rej_indices = []
    for g in range(n_groups):
        mask = bins == g
        if not mask.any():
            continue
        idx_g = np.where(mask)[0]
        rej_g, _ = bh(p[idx_g], alpha)
        rej_indices.extend(idx_g[rej_g].tolist())
    rej = np.array(sorted(rej_indices), dtype=int)
    return rej, len(rej)


# ---------- Helpers for converting f_c to p-values and e-values ----------

def beta_pvalue(stat: np.ndarray, a0: float = 2.0, b0: float = 5.0) -> np.ndarray:
    """Right-tail p-value under Beta(a0, b0) null."""
    from scipy.stats import beta as beta_dist
    return 1.0 - beta_dist.cdf(np.asarray(stat, dtype=np.float64), a0, b0)


def beta_lr_evalue(stat: np.ndarray, a0: float, b0: float, a1: float, b1: float
                   ) -> np.ndarray:
    """log-likelihood-ratio e-value with KNOWN parameters (oracle)."""
    from scipy.stats import beta as beta_dist
    s = np.clip(np.asarray(stat, dtype=np.float64), 1e-6, 1 - 1e-6)
    return beta_dist.logpdf(s, a1, b1) - beta_dist.logpdf(s, a0, b0)
