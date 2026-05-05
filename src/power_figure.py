"""Synthetic-injection power figure: rejection rate vs κ for the LRT and
mixture-LR e-value constructions, at the FCC docket scale.

Generates synthetic vMF clusters at controlled concentration κ in the
SBERT-MiniLM dimension d=384, computes both (i) the cohesion-based
mixture-LR e-value using the fitted (g_0, g_1) from src/evalues_mixlr.py
and (ii) the split-LRT e-value via universal inference, then reports
the empirical rejection rate at the e-BH threshold.

Output:
  results/power_figure.csv  — kappa, n, rejection_rate_mixlr, rejection_rate_lrt
  results/power_figure.png  — plot

Compare to Theorem-3 sketch: predicted κ* ≍ √(d log(K/α)/n_c).
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import beta as beta_dist, vonmises_fisher

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"


# ---------- helpers (mirror src/evalues_lrt.py) ----------

from scipy.special import gammaln, ive, logsumexp


def log_unit_sphere_area(d: int) -> float:
    return float(np.log(2.0) + (d / 2) * np.log(np.pi) - gammaln(d / 2))


def log_vmf_norm(d: int, kappa: float) -> float:
    if kappa < 1e-12:
        return -log_unit_sphere_area(d)
    nu = d / 2 - 1
    log_I = float(np.log(ive(nu, kappa)) + kappa)
    return float(nu * np.log(kappa) - (d / 2) * np.log(2 * np.pi) - log_I)


def kappa_mle(r: float, d: int, *, n_newton: int = 3,
              kappa_max: float | None = None) -> float:
    if kappa_max is None:
        kappa_max = 10.0 * d
    r = float(min(max(r, 1e-9), 1 - 1e-9))
    kappa = r * (d - r * r) / (1.0 - r * r)
    kappa = min(kappa, kappa_max)
    nu = d / 2 - 1
    for _ in range(n_newton):
        try:
            A = float(ive(nu + 1, kappa) / ive(nu, kappa))
        except Exception:
            break
        A_prime = 1.0 - A * A - (d - 1) / max(kappa, 1e-9) * A
        step = (A - r) / max(A_prime, 1e-9)
        step = float(np.clip(step, -kappa / 2, kappa / 2))
        kappa = max(kappa - step, 1e-3)
        kappa = min(kappa, kappa_max)
    return float(kappa)


def vmf_mle(x: np.ndarray) -> tuple[np.ndarray, float]:
    d = x.shape[1]
    mean = x.mean(axis=0)
    r = float(np.linalg.norm(mean))
    if r < 1e-9:
        return mean / max(r, 1e-12), 1e-3
    return mean / r, kappa_mle(r, d)


def cohesion(x: np.ndarray) -> float:
    """Mean pairwise cosine similarity within x (n, d)."""
    n = x.shape[0]
    sims = x @ x.T
    if n < 2:
        return 0.0
    return float(sims[np.triu_indices(n, k=1)].mean())


# ---------- main ----------

def main(*, d: int = 384, n_per_cluster: int = 8,
         kappas: tuple[float, ...] = (1.0, 5.0, 20.0, 50.0, 100.0,
                                      200.0, 500.0, 1000.0, 2000.0),
         n_trials: int = 200, K_sim: int = 21_606, alpha: float = 0.10,
         seed: int = 0,
         g0_a: float = 27.80, g0_b: float = 17.46,
         g1_a: float = 415.18, g1_b: float = 25.71,
         skip_lrt: bool = True) -> None:
    """g0/g1 Beta params default to the pooled fits from evalues_mixlr.py."""
    rng = np.random.default_rng(seed)

    # e-BH conservative threshold: e ≥ K/α (sufficient condition for rejection)
    log_threshold = np.log(K_sim / alpha)
    print(f"d={d}, n_per_cluster={n_per_cluster}, K_sim={K_sim}, α={alpha}")
    print(f"e-BH conservative threshold: log e ≥ {log_threshold:.2f}  "
          f"(e ≥ {K_sim/alpha:.0f})")
    print(f"g_0 ~ Beta({g0_a:.1f}, {g0_b:.1f}), mean={g0_a/(g0_a+g0_b):.3f}")
    print(f"g_1 ~ Beta({g1_a:.1f}, {g1_b:.1f}), mean={g1_a/(g1_a+g1_b):.3f}")
    print(f"theory κ*  =  C_1 √(d log(K/(αβ))/n) "
          f"≈ √({d} · {np.log(K_sim/(alpha*0.1)):.1f} / {n_per_cluster}) "
          f"= {np.sqrt(d * np.log(K_sim/(alpha*0.1)) / n_per_cluster):.2f}  "
          f"(C_1=1; empirical C_1 ≈ 3-5)")

    rows = []
    for kappa in kappas:
        rej_mixlr = 0
        rej_lrt = 0
        log_e_mixlr_list = []
        log_e_lrt_list = []
        cohesion_list = []
        for t in range(n_trials):
            mu = rng.standard_normal(d)
            mu /= np.linalg.norm(mu)
            x = vonmises_fisher(mu, kappa, seed=rng).rvs(n_per_cluster)
            T = cohesion(x)
            cohesion_list.append(T)
            # mixture-LR e-value
            T_c = float(np.clip(T, 1e-6, 1 - 1e-6))
            log_e_m = (beta_dist.logpdf(T_c, g1_a, g1_b)
                       - beta_dist.logpdf(T_c, g0_a, g0_b))
            log_e_mixlr_list.append(log_e_m)
            if log_e_m >= log_threshold:
                rej_mixlr += 1
            # split-LRT e-value (optional; expensive in d=384)
            if not skip_lrt:
                perm = rng.permutation(n_per_cluster)
                half = n_per_cluster // 2
                A, B = x[perm[:half]], x[perm[half:half*2]]
                mu1, kap1 = vmf_mle(A)
                lp = log_vmf_norm(d, kap1) + kap1 * (B @ mu1)
                lq = -log_unit_sphere_area(d)  # reference: uniform null
                log_e_l = float(np.clip(lp - lq, -50, 50).sum())
                log_e_lrt_list.append(log_e_l)
                if log_e_l >= log_threshold:
                    rej_lrt += 1
        rows.append({
            "kappa": float(kappa),
            "d": d, "n": n_per_cluster, "n_trials": n_trials,
            "rejection_rate_mixlr": rej_mixlr / n_trials,
            "log_e_mixlr_p50": float(np.median(log_e_mixlr_list)),
            "log_e_mixlr_p10": float(np.quantile(log_e_mixlr_list, 0.10)),
            "log_e_mixlr_p90": float(np.quantile(log_e_mixlr_list, 0.90)),
            "rejection_rate_lrt": (rej_lrt / n_trials) if not skip_lrt else None,
            "log_e_lrt_p50": (float(np.median(log_e_lrt_list))
                              if not skip_lrt else None),
            "T_cohesion_p50": float(np.median(cohesion_list)),
        })
        print(f"  κ={kappa:8.1f}  T_p50={rows[-1]['T_cohesion_p50']:.3f}  "
              f"mixlr_rej={rows[-1]['rejection_rate_mixlr']:.3f}  "
              f"log_e_mixlr p50={rows[-1]['log_e_mixlr_p50']:.2f}")

    out = pd.DataFrame(rows)
    out_csv = RES / "power_figure.csv"
    out.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}")

    # plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=150)
        ax.plot(out["kappa"], out["rejection_rate_mixlr"],
                marker="o", label=f"Mixture-LR (cohesion, n={n_per_cluster})")
        ax.set_xscale("log")
        ax.set_xlabel("vMF concentration κ")
        ax.set_ylabel("Rejection rate at e-BH α=0.10")
        ax.axhline(0.5, color="gray", lw=0.5, ls=":")
        ax.axhline(0.9, color="gray", lw=0.5, ls=":")
        kstar = np.sqrt(d * np.log(K_sim/(alpha*0.1)) / n_per_cluster)
        ax.axvline(kstar, color="red", lw=0.5, ls="--",
                   label=f"theory κ*≈{kstar:.1f} (C_1=1)")
        ax.set_title(f"Power vs κ on synthetic vMF clusters  "
                     f"(d={d}, n={n_per_cluster}, K={K_sim:,})")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.3)
        out_png = RES / "power_figure.png"
        fig.tight_layout()
        fig.savefig(out_png)
        print(f"wrote {out_png}")
    except Exception as e:
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--d", type=int, default=384)
    p.add_argument("--n-per-cluster", type=int, default=8)
    p.add_argument("--kappas", type=float, nargs="+",
                   default=[1.0, 5.0, 20.0, 50.0, 100.0, 200.0, 500.0,
                            1000.0, 2000.0])
    p.add_argument("--n-trials", type=int, default=200)
    p.add_argument("--K", type=int, default=21_606,
                   help="docket-scale K for e-BH threshold")
    p.add_argument("--alpha", type=float, default=0.10)
    p.add_argument("--include-lrt", action="store_true",
                   help="also compute split-LRT e-values (slower)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    main(d=args.d, n_per_cluster=args.n_per_cluster,
         kappas=tuple(args.kappas), n_trials=args.n_trials,
         K_sim=args.K, alpha=args.alpha, seed=args.seed,
         skip_lrt=not args.include_lrt)
