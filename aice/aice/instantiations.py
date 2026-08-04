"""Non-coordination instantiations of AICE for the methodology paper.

Demonstrates that AICE is a general framework, not bespoke to coordination
detection on regulatory dockets. Currently:

    differential_expression(...) -- RNA-seq DE testing with housekeeping
        gene anchors (HRT Atlas, Eisenberg-Levanon 2013). The structural
        analog of tau=1 byte-identical clusters: housekeeping gene lists
        are publicly curated and identified a priori, not from data.

This module produces the same kind of (stat, anchor_mask, is_alt) tuple
that the simulator produces, so AICE / baselines can be applied without
modification.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import nbinom, norm
from scipy.special import digamma


def simulate_de_experiment(
    G: int = 10000,
    pi_hk: float = 0.05,
    pi_de: float = 0.10,
    delta_min: float = 1.5,
    delta_sigma: float = 1.5,
    seed: int = 42,
) -> dict:
    """Simulate a differential expression experiment in z-score form.

    Two-groups model on standardized test statistics, following Efron (2008,
    Statist. Sci.) "Microarrays, Empirical Bayes and the Two-Groups Model."
    Housekeeping genes are structurally null (anchor); other null genes are
    non-anchor null; DE genes have z ~ N(delta, 1) with delta ~ truncated normal.

    Setup:
        is_hk[g] = True     if g is housekeeping (pi_hk fraction; structural null)
        is_de[g] = True     if g is differential (pi_de fraction)
        otherwise           non-anchor null

        z[g] ~ N(0, 1)              if not is_de
        z[g] ~ N(delta_g, 1)        if is_de, delta_g ~ N(0, delta_sigma^2)
                                    truncated to |delta| > delta_min

    Test statistic for AICE: stat = 1 - 2 * Phi(-|z|) in (0, 1).
        Under H_0: stat ~ Uniform(0, 1) since |z| has half-normal CDF.
        Under H_1: stat concentrates near 1.

    Returns dict with stat, anchor_mask, is_alt, z, delta.
    """
    rng = np.random.default_rng(seed)

    pi_0 = 1.0 - pi_de
    if pi_hk > pi_0:
        raise ValueError(f"pi_hk={pi_hk} cannot exceed pi_0={pi_0}")

    is_de = rng.random(G) < pi_de
    is_hk = np.zeros(G, dtype=bool)
    nonde_idx = np.where(~is_de)[0]
    n_hk = int(np.round(pi_hk * G))
    n_hk = min(n_hk, len(nonde_idx))
    if n_hk > 0:
        chosen_hk = rng.choice(nonde_idx, size=n_hk, replace=False)
        is_hk[chosen_hk] = True

    # delta: 0 for nulls, truncated N(0, sigma) for DE
    delta = np.zeros(G, dtype=np.float64)
    n_de = int(is_de.sum())
    if n_de > 0:
        de_deltas = rng.normal(0, delta_sigma, size=n_de)
        too_small = np.abs(de_deltas) < delta_min
        while too_small.any():
            de_deltas[too_small] = rng.normal(0, delta_sigma, size=int(too_small.sum()))
            too_small = np.abs(de_deltas) < delta_min
        delta[is_de] = de_deltas

    # Direct z-score sampling: z ~ N(delta, 1)
    z = rng.normal(delta, 1.0)

    # Map |z| to (0, 1): stat = 2 * Phi(|z|) - 1 = P(|Z'| <= |z|) for Z' ~ N(0,1)
    # Equivalent: stat = 1 - 2 * Phi(-|z|)
    stat = 1 - 2 * norm.cdf(-np.abs(z))
    stat = np.clip(stat, 1e-6, 1 - 1e-6)

    return {
        "stat": stat,
        "anchor_mask": is_hk,
        "is_alt": is_de,
        "z": z,
        "delta": delta,
    }


def run_de_bakeoff(seed: int = 42, alpha: float = 0.10) -> dict:
    """Run AICE + baselines on a simulated DE experiment, return metrics."""
    from aice.core import fit_evalues, ebh as aice_ebh, evaluate
    from aice.baselines import bh, by, storey_qvalue
    from sklearn.metrics import average_precision_score

    sim = simulate_de_experiment(seed=seed)
    stat = sim["stat"]
    anchor_mask = sim["anchor_mask"]
    is_alt = sim["is_alt"]
    G = len(stat)

    print(f"DE simulation: G = {G}, anchors = {anchor_mask.sum()}, "
          f"true DE = {is_alt.sum()}")

    # AICE on the (0, 1)-mapped z-scores with HK anchors
    log_e_aice = fit_evalues(stat, anchor_mask, n_folds=5, seed=seed)
    rej_aice, k_aice = aice_ebh(log_e_aice, alpha)

    # BH on |z|-derived p-values (standard normal null)
    p = 2 * norm.cdf(-np.abs(sim["z"]))
    rej_bh, k_bh = bh(p, alpha)
    rej_by, k_by = by(p, alpha)
    rej_storey, k_storey, _ = storey_qvalue(p, alpha, lambda_fixed=0.5)

    def score_metrics(rej, k, name):
        if k == 0:
            return {"method": name, "k": 0, "fdp": 0.0, "power": 0.0,
                    "ap": float(average_precision_score(is_alt.astype(int), stat))}
        n_false = int((~is_alt[rej]).sum())
        n_true = k - n_false
        fdp = n_false / k
        power = n_true / max(int(is_alt.sum()), 1)
        ap = float(average_precision_score(is_alt.astype(int), stat))
        return {"method": name, "k": k, "fdp": fdp, "power": power, "ap": ap}

    metrics = [
        score_metrics(rej_aice, k_aice, "AICE"),
        score_metrics(rej_bh, k_bh, "BH"),
        score_metrics(rej_by, k_by, "BY"),
        score_metrics(rej_storey, k_storey, "Storey"),
    ]
    return {"metrics": metrics, "G": G, "alpha": alpha}
