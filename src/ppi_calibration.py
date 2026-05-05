"""PPI++ calibration of LLM-judge classifications against FOIA gold.

Implements the prediction-powered inference (PPI++) estimator of
Angelopoulos, Duchi, Zrnic (2023) for binary outcomes, applied to:
  - Estimating the true coordination rate of the rejection set
  - Producing PPI-corrected per-cluster coordination probabilities
  - Building compound e-values per Ignatiadis-Wang-Ramdas (2024) §7

Pipeline:
  1. Load LLM-judge classifications and FOIA labels.
  2. Compute the bias of the LLM judge against the FOIA gold subset.
  3. PPI-corrected estimate: θ̂_PPI = (1/N) Σ_unlab f(x) - (1/n) Σ_lab (f(x) - y),
     where f is the LLM, y is the gold label, lab is FOIA-attributed clusters.
  4. Variance: standard PPI++ formula, gives confidence intervals.
  5. Compound e-value: per-cluster ratio of (PPI-corrected coordination rate
     under H1 fitted on attributed) to (under H0 fitted on non-attributed).

Reference:
  Angelopoulos, Duchi, Zrnic (2023). "PPI++: Efficient prediction-powered
  inference." arXiv:2311.01453.
  Ignatiadis, Wang, Ramdas (2024). "Asymptotic and compound e-values."
  arXiv:2409.19812.
  Csillag, Struchiner, Goedert (2025). "Prediction-powered e-values."
  ICML 2025. arXiv:2502.04294.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import beta as beta_dist, norm

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"


def ppi_mean(*, f_lab: np.ndarray, y_lab: np.ndarray, f_unlab: np.ndarray,
             tuning_lambda: float = 1.0) -> dict:
    """PPI++ estimator for E[Y] (binary outcome).

    f_lab: model predictions on labeled set (n,)
    y_lab: gold labels on labeled set (n,)
    f_unlab: model predictions on unlabeled set (N,)
    tuning_lambda: PPI++ power tuning parameter; lambda=1 is original PPI.

    Returns dict with theta_hat, se, ci_lo, ci_hi (95%), bias_estimate.
    """
    n = len(f_lab)
    N = len(f_unlab)
    f_lab = f_lab.astype(np.float64)
    y_lab = y_lab.astype(np.float64)
    f_unlab = f_unlab.astype(np.float64)

    # PPI++ point estimate
    theta_hat = tuning_lambda * f_unlab.mean() + (y_lab - tuning_lambda * f_lab).mean()
    bias = (tuning_lambda * f_lab - y_lab).mean()

    # Variance estimate (PPI++ formula)
    # var(theta_hat) = (lambda^2 * Var(f_unlab))/N + Var(y - lambda*f)/n
    var_unlab = float(np.var(f_unlab, ddof=1)) if N > 1 else 0.0
    rectifier = y_lab - tuning_lambda * f_lab
    var_lab = float(np.var(rectifier, ddof=1)) if n > 1 else 0.0
    se = float(np.sqrt(tuning_lambda ** 2 * var_unlab / max(N, 1) + var_lab / max(n, 1)))

    # Optimal lambda* = Cov(f_lab, y_lab) / Var(f_lab) — variance reduction
    # see PPI++ paper Eq 5
    if tuning_lambda == "auto":
        cov_fy = float(np.cov(f_lab, y_lab, ddof=1)[0, 1]) if n > 1 else 0.0
        var_f = float(np.var(f_lab, ddof=1)) if n > 1 else 1.0
        lambda_star = cov_fy / var_f
        # recurse with optimal
        return ppi_mean(f_lab=f_lab, y_lab=y_lab, f_unlab=f_unlab,
                        tuning_lambda=lambda_star)

    return {
        "theta_hat": float(theta_hat),
        "se": se,
        "ci_lo": float(theta_hat - 1.96 * se),
        "ci_hi": float(theta_hat + 1.96 * se),
        "bias_estimate": float(bias),
        "n_labeled": n,
        "N_unlabeled": N,
        "tuning_lambda": float(tuning_lambda),
        "naive_f_mean": float(f_unlab.mean()),
        "naive_y_mean": float(y_lab.mean()) if n else None,
    }


def main(*, judge_path: Path, attr_path: Path, output_path: Path | None = None) -> None:
    print(f"loading LLM judgments: {judge_path}")
    j = pq.read_table(judge_path).to_pandas()
    j = j.dropna(subset=["coordinated"]).copy()
    j["f"] = j["coordinated"].astype(int)
    print(f"  {len(j):,} clusters with valid LLM classification")

    print(f"loading attribution: {attr_path}")
    att = pd.read_csv(attr_path)
    j = j.merge(att[["cluster_id", "frac_astroturf", "top_label"]],
                 on="cluster_id", how="left")
    j["foia_astro"] = (j["frac_astroturf"].fillna(0) >= 0.5).astype(int)

    print()
    print("=== Naive (LLM-only) coordination rate ===")
    print(f"  P(coordinated by LLM) = {j['f'].mean():.4f}  "
          f"(n={len(j):,})")

    print()
    print("=== FOIA-attributed astroturf rate (gold) ===")
    n_astro_in_sample = int(j["foia_astro"].sum())
    print(f"  P(FOIA astroturf | sampled) = {n_astro_in_sample}/{len(j):,} "
          f"= {n_astro_in_sample/len(j):.4f}")

    print()
    print("=== LLM precision on FOIA-attributed astroturf ===")
    foia_subset = j[j["foia_astro"] == 1]
    print(f"  P(LLM=coord | FOIA=astroturf) = "
          f"{foia_subset['f'].mean():.4f}  (n={len(foia_subset)})")
    nonfoia_subset = j[j["foia_astro"] == 0]
    print(f"  P(LLM=coord | FOIA=not astroturf) = "
          f"{nonfoia_subset['f'].mean():.4f}  (n={len(nonfoia_subset)})")

    # PPI++ for "true astroturf rate in the rejection set"
    # NOTE: The "labeled" set here is FOIA-astroturf-confirmed clusters.
    # FOIA negative is harder to interpret (NYAG has confirmed astroturf
    # presence, NOT confirmed absence on non-flagged clusters).
    # So this PPI estimate is for: if FOIA labeling were the truth,
    # what's the rate, and what's our LLM bias against it?
    print()
    print("=== PPI++ estimate of FOIA-astroturf rate (using LLM as proxy) ===")
    f_lab = foia_subset["f"].to_numpy()
    y_lab = np.ones(len(foia_subset))  # FOIA-attributed are y=1 (astroturf)
    nonfoia_in_lab = nonfoia_subset["f"].to_numpy()
    y_lab_full = np.concatenate([y_lab, np.zeros(len(nonfoia_subset))])
    f_lab_full = np.concatenate([f_lab, nonfoia_in_lab])
    f_unlab = j["f"].to_numpy()  # naive (use the full sample as 'unlabeled' from population)
    result = ppi_mean(f_lab=f_lab_full, y_lab=y_lab_full, f_unlab=f_unlab)
    print(f"  PPI θ̂ = {result['theta_hat']:.4f} ± {1.96*result['se']:.4f}")
    print(f"  95% CI = [{result['ci_lo']:.4f}, {result['ci_hi']:.4f}]")
    print(f"  Naive LLM = {result['naive_f_mean']:.4f}; bias estimate = {result['bias_estimate']:.4f}")

    print()
    print("=== Compound e-value via IWR §7 mixture-LR with LLM as statistic ===")
    # Fit Beta(g_0) on LLM confidence under FOIA-non-astroturf
    # Fit Beta(g_1) on LLM confidence under FOIA-astroturf
    # Per cluster: e = g_1(conf_c) / g_0(conf_c)
    if "confidence" in j.columns and j["confidence"].notna().sum() > 10:
        g0_conf = j.loc[j["foia_astro"] == 0, "confidence"].dropna()
        g1_conf = j.loc[j["foia_astro"] == 1, "confidence"].dropna()
        # Fold sign of confidence into "P(coord)": for "coord=no", convert
        # confidence into P(coord) = 1 - conf
        # For uniform handling: treat the LLM-derived score as
        #   score = conf if coord else (1 - conf)
        j["score"] = np.where(j["coordinated"] == 1,
                              j["confidence"], 1.0 - j["confidence"])
        s0 = j.loc[j["foia_astro"] == 0, "score"].dropna()
        s1 = j.loc[j["foia_astro"] == 1, "score"].dropna()
        print(f"  H0 (non-astroturf) score:  mean={s0.mean():.3f} std={s0.std():.3f} n={len(s0)}")
        print(f"  H1 (astroturf) score:      mean={s1.mean():.3f} std={s1.std():.3f} n={len(s1)}")

        # Fit Beta to each
        def fit_beta(s):
            s = np.clip(s, 1e-6, 1 - 1e-6)
            mu, var = float(s.mean()), float(s.var())
            if var <= 1e-9:
                return 1.0, 1.0
            common = mu * (1 - mu) / var - 1
            return max(mu * common, 1e-3), max((1 - mu) * common, 1e-3)
        a0, b0 = fit_beta(s0)
        a1, b1 = fit_beta(s1)
        print(f"  Beta_0 = ({a0:.2f}, {b0:.2f}), Beta_1 = ({a1:.2f}, {b1:.2f})")

        # Per-cluster compound e-value
        s = np.clip(j["score"].fillna(0.5).to_numpy(), 1e-6, 1 - 1e-6)
        log_g0 = beta_dist.logpdf(s, a0, b0)
        log_g1 = beta_dist.logpdf(s, a1, b1)
        log_e = log_g1 - log_g0
        j["log_e_llm"] = log_e
        print(f"  log_e_llm distribution: p10={np.quantile(log_e, 0.1):.2f} "
              f"p50={np.quantile(log_e, 0.5):.2f} p90={np.quantile(log_e, 0.9):.2f}")

        # e-BH at alpha=0.10
        df = j[["cluster_id", "log_e_llm", "foia_astro"]].copy().sort_values(
            "log_e_llm", ascending=False).reset_index(drop=True)
        df["e"] = np.exp(np.clip(df["log_e_llm"], -700, 700))
        K = len(df)
        df["rank"] = np.arange(1, K + 1)
        df["thr"] = K / (0.10 * df["rank"])
        df["rej"] = df["e"] >= df["thr"]
        if df["rej"].any():
            k_hat = df.index[df["rej"]].max() + 1
            rej = df.iloc[:k_hat]
        else:
            rej = df.iloc[:0]
        n_rej = len(rej)
        n_rej_astro = int(rej["foia_astro"].sum())
        print(f"  e-BH at α=0.10: {n_rej:,} rejected of {K:,}; "
              f"of these {n_rej_astro} are FOIA-astro "
              f"(precision {100*n_rej_astro/max(n_rej,1):.1f}%)")

    out_path = output_path or PROC / "llm_judge_ppi_results.parquet"
    j.to_parquet(out_path, compression="zstd", index=False)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--judge-path", type=Path,
                   default=PROC / "llm_judge_n1000.parquet")
    p.add_argument("--attr-path", type=Path,
                   default=RES / "attribution_table_r0.9.csv")
    p.add_argument("--output-path", type=Path, default=None)
    args = p.parse_args()
    main(judge_path=args.judge_path,
         attr_path=args.attr_path,
         output_path=args.output_path)
