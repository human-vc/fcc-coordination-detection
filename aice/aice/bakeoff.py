"""Methodology bake-off harness for AICE.

Runs synthetic 2-component mixture simulations comparing AICE against the
canonical FDR baselines (BH, BY, Storey, e-BH oracle, Lee-Ren CC, group-BH).
Computes realized FDR, realized power (TPP), and average precision across
the full (K × pi_0 × dependence × alpha) grid.
"""
from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from aice.core import (
    fit_evalues, ebh as aice_ebh,
    conformal_p_from_evalues, aice_cc_boost, aice_storey_ebh,
)
from aice.baselines import (
    bh, by, storey_qvalue, ebh_lr_naive, ebh_cc,
    group_bh, beta_pvalue,
)
from aice.simulate import simulate_two_component_mixture


@dataclass
class TrialResult:
    method: str
    K: int
    pi_0: float
    dependence: str
    rho: float
    alpha: float
    rep: int
    k_rejected: int
    n_false: int
    n_true: int
    fdp: float
    power: float
    ap: float


def run_methods(
    stat: np.ndarray,
    is_null: np.ndarray,
    anchor_mask: np.ndarray,
    a0: float,
    b0: float,
    a1: float,
    b1: float,
    alpha: float,
    seed: int,
) -> dict[str, dict]:
    """Run all baselines + AICE on a single replication, return per-method metrics."""
    K = len(stat)
    n_alt = int((~is_null).sum())
    p = beta_pvalue(stat, a0, b0)

    methods: dict[str, dict] = {}

    # 1) BH on Beta-null p-values
    rej, k = bh(p, alpha)
    methods["BH"] = {"rej": rej, "k": k, "score": -p}

    # 2) BY (arbitrary dependence)
    rej, k = by(p, alpha)
    methods["BY"] = {"rej": rej, "k": k, "score": -p}

    # 3) Storey's q-value (lambda fixed at 0.5 for stability across all K)
    rej, k, _ = storey_qvalue(p, alpha, lambda_fixed=0.5)
    methods["Storey"] = {"rej": rej, "k": k, "score": -p}

    # 4) Vanilla e-BH with ORACLE likelihood ratio (knows true a0,b0,a1,b1)
    rej, k = ebh_lr_naive(stat, a0, b0, a1, b1, alpha)
    from aice.baselines import beta_lr_evalue
    log_e_oracle = beta_lr_evalue(stat, a0, b0, a1, b1)
    methods["e-BH (oracle LR)"] = {"rej": rej, "k": k, "score": log_e_oracle}

    # 5) Lee-Ren CC e-BH (boosts oracle LR via MC conditional calibration)
    rej, k = ebh_cc(
        stat, anchor_mask, alpha,
        a0_est=(a0, b0), a1_est=(a1, b1), n_mc=100, seed=seed,
    )
    methods["Lee-Ren CC"] = {"rej": rej, "k": k, "score": log_e_oracle}

    # 6) Group-BH (covariate = anchor mask + cluster index)
    cov = anchor_mask.astype(float) + np.linspace(0, 1, K) * 0.01
    rej, k = group_bh(p, cov, alpha, n_groups=4)
    methods["Group-BH"] = {"rej": rej, "k": k, "score": -p}

    # 7) AICE (cross-fit anchor-restricted Beta MLE) — vanilla baseline
    log_e_aice = fit_evalues(stat, anchor_mask, n_folds=5, seed=seed)
    rej, k = aice_ebh(log_e_aice, alpha)
    methods["AICE"] = {"rej": rej, "k": k, "score": log_e_aice}

    # 8) AICE+ — EM-decontaminated alternative density (the headline method)
    log_e_aiceplus = fit_evalues(
        stat, anchor_mask, n_folds=5, seed=seed, decontaminate=True,
    )
    rej, k = aice_ebh(log_e_aiceplus, alpha)
    methods["AICE+"] = {"rej": rej, "k": k, "score": log_e_aiceplus}

    # 9) AICE-BH — conformal-p calibration of AICE+ e-values, then BH
    p_aice = conformal_p_from_evalues(log_e_aiceplus, anchor_mask)
    rej, k = bh(p_aice, alpha)
    methods["AICE-BH"] = {"rej": rej, "k": k, "score": -p_aice}

    # 10) AICE+Storey — anchor-empirical Storey-e adaptive boost on AICE+
    rej, k, _pi = aice_storey_ebh(log_e_aiceplus, anchor_mask, alpha)
    methods["AICE+Storey"] = {"rej": rej, "k": k, "score": log_e_aiceplus}

    # Compute metrics per method
    out: dict[str, dict] = {}
    for name, m in methods.items():
        rej, k = m["rej"], m["k"]
        if k > 0:
            n_false = int(is_null[rej].sum())
            n_true = k - n_false
            fdp = n_false / k
            power = n_true / max(n_alt, 1)
        else:
            n_false = n_true = 0
            fdp = 0.0
            power = 0.0
        try:
            ap = float(average_precision_score(~is_null, m["score"])) if n_alt > 0 else float("nan")
        except Exception:
            ap = float("nan")
        out[name] = {
            "k_rejected": k,
            "n_false": n_false,
            "n_true": n_true,
            "fdp": fdp,
            "power": power,
            "ap": ap,
        }
    return out


def run_grid(
    K_list: list[int],
    pi_0_list: list[float],
    dependence_list: list[tuple[str, float]],
    alpha_list: list[float],
    n_reps: int = 500,
    a0: float = 2.0,
    b0: float = 5.0,
    a1: float = 8.0,
    b1: float = 2.0,
    anchor_rate: float = 0.10,
    seed: int = 0,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run the full simulation grid; return long-form DataFrame of results."""
    rows: list[dict] = []
    total = len(K_list) * len(pi_0_list) * len(dependence_list) * n_reps
    done = 0
    t0 = time.time()
    for K in K_list:
        for pi_0 in pi_0_list:
            for (dep, rho) in dependence_list:
                for rep in range(n_reps):
                    rep_seed = seed + rep + 10007 * (K + int(pi_0 * 100) + hash(dep) % 100)
                    rep_seed = abs(rep_seed) % (2**31)
                    stat, is_null, anchor_mask = simulate_two_component_mixture(
                        K=K, pi_0=pi_0, a0=a0, b0=b0, a1=a1, b1=b1,
                        anchor_rate=anchor_rate,
                        dependence=dep, rho=rho, seed=rep_seed,
                    )
                    for alpha in alpha_list:
                        results = run_methods(
                            stat, is_null, anchor_mask,
                            a0, b0, a1, b1, alpha, seed=rep_seed,
                        )
                        for method_name, metrics in results.items():
                            rows.append({
                                "method": method_name,
                                "K": K,
                                "pi_0": pi_0,
                                "dependence": dep,
                                "rho": rho,
                                "alpha": alpha,
                                "rep": rep,
                                **metrics,
                            })
                    done += 1
                    if verbose and done % max(1, total // 20) == 0:
                        elapsed = time.time() - t0
                        eta = elapsed * (total - done) / max(done, 1)
                        print(f"  {done}/{total} ({100*done/total:.1f}%, "
                              f"elapsed={elapsed:.0f}s, eta={eta:.0f}s)")
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate trial-level results to method × (K, pi_0, dependence, alpha)."""
    agg = df.groupby(
        ["method", "K", "pi_0", "dependence", "rho", "alpha"]
    ).agg(
        mean_fdr=("fdp", "mean"),
        sd_fdr=("fdp", "std"),
        mean_power=("power", "mean"),
        sd_power=("power", "std"),
        mean_k=("k_rejected", "mean"),
        mean_ap=("ap", "mean"),
        frac_fdr_excess=("fdp", lambda s: float((s > s.name[5] if False else (s > 0)).mean())),
        n_reps=("rep", "count"),
    ).reset_index()
    return agg
