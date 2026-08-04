"""Empirical validation of Theorem 3 (Bayes-oracle attainment, parametric rate).

Theorem 3 (informal):
    Under (A1) Beta well-specification on the anchor stratum, (A2) anchor purity,
    (A3) cross-fit independence, (A4) no e-BH ties at the oracle cutoff, (A5)
    n_anchor → ∞ with L folds fixed, the AICE compound e-value satisfies:
        (i)   E_c^AICE - E_c^* = O_p(1/√n_anchor)   [parametric e-value approximation]
        (ii)  |FDP(R_n) - FDP(R*)| + |Pow(R_n) - Pow(R*)| = O_p(1/√n_anchor)  [regret]
        (iii) Pow(R_n) → Pow(R*) within simple-separable class

CORRECT comparison: against the IWR §7 oracle e-value, which uses the TRUE
non-anchor marginal density, NOT the alternative-component density.

The IWR §7 oracle is:
    E^IWR(x) = g_non_anchor(x) / g_0(x)
where g_non_anchor is the marginal density of non-anchor data (a mixture of
null and alternative).

Under our simulator, non-anchor data is a mixture:
    P(c is null | c is non-anchor) = pi_0*(1-anchor_rate) / (1 - pi_0*anchor_rate)
so g_non_anchor = w * Beta(a0,b0) + (1-w) * Beta(a1,b1).

We compare AICE log-e-value (fit on data) to log E^IWR (analytic marginal)
and verify O_p(1/√n_anchor) convergence as n_anchor → ∞.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist

from aice.core import fit_evalues, ebh
from aice.simulate import simulate_two_component_mixture


def iwr_oracle_log_e(
    stat: np.ndarray, pi_0: float, anchor_rate: float,
    a0: float, b0: float, a1: float, b1: float,
) -> np.ndarray:
    """Analytic IWR §7 oracle log e-value: log g_non_anchor(x) - log g_0(x).

    g_non_anchor = w * Beta(a0,b0) + (1 - w) * Beta(a1,b1) where
    w = P(null | non-anchor) = pi_0 * (1 - anchor_rate) / (1 - pi_0 * anchor_rate).
    """
    s = np.clip(np.asarray(stat, dtype=np.float64), 1e-6, 1 - 1e-6)
    w = pi_0 * (1 - anchor_rate) / (1 - pi_0 * anchor_rate)
    log_g0 = beta_dist.logpdf(s, a0, b0)
    log_g1 = beta_dist.logpdf(s, a1, b1)
    log_g_marg = np.logaddexp(np.log(w) + log_g0, np.log(1 - w) + log_g1)
    return log_g_marg - log_g0


def regret_at_n(
    K: int,
    pi_0: float,
    anchor_rate: float,
    a0: float = 2.0,
    b0: float = 5.0,
    a1: float = 8.0,
    b1: float = 2.0,
    alpha: float = 0.10,
    n_reps: int = 100,
    seed: int = 42,
) -> dict:
    """Estimate (i) e-value RMSE vs IWR oracle, (ii) FDP/power regret vs IWR oracle e-BH."""
    e_gap_sum_sq = 0.0
    e_gap_count = 0
    fdp_gaps = []
    power_gaps = []

    for rep in range(n_reps):
        rep_seed = seed + rep
        stat, is_null, anchor_mask = simulate_two_component_mixture(
            K=K, pi_0=pi_0, a0=a0, b0=b0, a1=a1, b1=b1,
            anchor_rate=anchor_rate, seed=rep_seed,
        )
        log_e_aice = fit_evalues(stat, anchor_mask, n_folds=5, seed=rep_seed)
        log_e_oracle = iwr_oracle_log_e(
            stat, pi_0=pi_0, anchor_rate=anchor_rate,
            a0=a0, b0=b0, a1=a1, b1=b1,
        )

        gap_sq = (log_e_aice - log_e_oracle) ** 2
        e_gap_sum_sq += float(gap_sq.mean())
        e_gap_count += 1

        rej_aice, k_aice = ebh(log_e_aice, alpha)
        rej_oracle, k_oracle = ebh(log_e_oracle, alpha)

        n_alt = max(int((~is_null).sum()), 1)
        fdp_aice = int(is_null[rej_aice].sum()) / max(k_aice, 1) if k_aice else 0.0
        fdp_oracle = int(is_null[rej_oracle].sum()) / max(k_oracle, 1) if k_oracle else 0.0
        pow_aice = int((~is_null)[rej_aice].sum()) / n_alt
        pow_oracle = int((~is_null)[rej_oracle].sum()) / n_alt
        fdp_gaps.append(abs(fdp_aice - fdp_oracle))
        power_gaps.append(abs(pow_aice - pow_oracle))

    n_anchor_avg = int(np.round(K * pi_0 * anchor_rate))
    return {
        "K": K,
        "pi_0": pi_0,
        "n_anchor_avg": n_anchor_avg,
        "alpha": alpha,
        "n_reps": n_reps,
        "evalue_rmse": float(np.sqrt(e_gap_sum_sq / e_gap_count)),
        "fdp_regret_mean": float(np.mean(fdp_gaps)),
        "fdp_regret_se": float(np.std(fdp_gaps) / np.sqrt(n_reps)),
        "power_regret_mean": float(np.mean(power_gaps)),
        "power_regret_se": float(np.std(power_gaps) / np.sqrt(n_reps)),
    }


def regret_sweep(
    K_list: list[int],
    pi_0: float = 0.7,
    anchor_rate: float = 0.10,
    n_reps: int = 100,
    alpha: float = 0.10,
    seed: int = 42,
) -> pd.DataFrame:
    """Sweep K to verify the 1/√n_anchor scaling of regret."""
    rows = []
    for K in K_list:
        rows.append(regret_at_n(
            K=K, pi_0=pi_0, anchor_rate=anchor_rate,
            alpha=alpha, n_reps=n_reps, seed=seed,
        ))
        print(f"  K={K} (n_anchor≈{rows[-1]['n_anchor_avg']}): "
              f"evalue_rmse={rows[-1]['evalue_rmse']:.4f}, "
              f"fdp_regret={rows[-1]['fdp_regret_mean']:.4f}±{rows[-1]['fdp_regret_se']:.4f}, "
              f"power_regret={rows[-1]['power_regret_mean']:.4f}±{rows[-1]['power_regret_se']:.4f}")
    return pd.DataFrame(rows)
