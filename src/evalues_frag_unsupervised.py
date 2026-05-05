"""Unsupervised fragmentation compound e-value + closed-eBH.

Two improvements on src/evalues_fragmentation.py:

1. UNSUPERVISED g_0, g_1 via 2-component Beta mixture EM on the
   fragmentation distribution. No FOIA labels used for calibration.
   The mixture identifies (g_0 = low-fragmentation = verbatim) and
   (g_1 = high-fragmentation = paraphrase) automatically.

2. CLOSED-eBH (Xu, Fischer, Ramdas 2025): a strict improvement over
   vanilla e-BH that uses sum-based closed testing. For sorted
   e-values e_(1) ≥ e_(2) ≥ ..., reject the top-k where
       mean(e_(1), ..., e_(k)) ≥ 1/α
   for the largest such k. This is provably superset of e-BH
   under arbitrary dependence.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import beta as beta_dist
from sklearn.mixture import GaussianMixture
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"


def fit_beta_2mix_em(f: np.ndarray, n_iter: int = 100,
                     tol: float = 1e-5) -> tuple[dict, dict, float]:
    """Fit 2-component Beta mixture via EM.

    Returns (component_low, component_high, weight_low) where each
    component is dict(a=., b=., mean=.). Lower-mean is g_0 (verbatim);
    higher-mean is g_1 (paraphrase).

    Initialization: K-means on logit(f) -> 2 clusters; fit Beta MOM to
    each.
    """
    eps = 1e-6
    f = np.clip(f.astype(np.float64), eps, 1 - eps)

    def beta_mom(x):
        mu = float(x.mean()); var = float(x.var())
        if var <= eps:
            return 1.0, 1.0
        common = mu * (1 - mu) / var - 1
        if common <= 0:
            return 1.0, 1.0
        return max(mu * common, 1e-3), max((1 - mu) * common, 1e-3)

    # Init via GMM on logit(f) — gives 2-cluster assignment
    logit = np.log(f / (1 - f)).reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, random_state=0, n_init=5)
    gmm.fit(logit)
    z = gmm.predict(logit)
    # ensure z=0 is lower-mean component (verbatim)
    if f[z == 0].mean() > f[z == 1].mean():
        z = 1 - z

    # Beta MOM init per component
    a0, b0 = beta_mom(f[z == 0])
    a1, b1 = beta_mom(f[z == 1])
    w0 = (z == 0).mean()
    w1 = 1 - w0
    log_lik_prev = -np.inf

    # EM loop
    for it in range(n_iter):
        # E-step: posterior responsibilities
        log_p0 = beta_dist.logpdf(f, a0, b0) + np.log(max(w0, 1e-9))
        log_p1 = beta_dist.logpdf(f, a1, b1) + np.log(max(w1, 1e-9))
        log_total = np.logaddexp(log_p0, log_p1)
        gamma0 = np.exp(log_p0 - log_total)
        gamma1 = 1 - gamma0
        log_lik = log_total.sum()
        if abs(log_lik - log_lik_prev) < tol * abs(log_lik_prev):
            break
        log_lik_prev = log_lik

        # M-step: weighted Beta MOM
        n0 = gamma0.sum()
        n1 = gamma1.sum()
        if n0 < 1 or n1 < 1:
            break
        mu0 = (gamma0 * f).sum() / n0
        mu1 = (gamma1 * f).sum() / n1
        var0 = (gamma0 * (f - mu0) ** 2).sum() / n0
        var1 = (gamma1 * (f - mu1) ** 2).sum() / n1

        def fit(mu, var):
            if var <= eps:
                return 1.0, 1.0
            c = mu * (1 - mu) / var - 1
            if c <= 0:
                return 1.0, 1.0
            return max(mu * c, 1e-3), max((1 - mu) * c, 1e-3)

        a0, b0 = fit(mu0, var0)
        a1, b1 = fit(mu1, var1)
        w0 = n0 / (n0 + n1)
        w1 = 1 - w0

    if a0 / (a0 + b0) > a1 / (a1 + b1):
        a0, b0, a1, b1, w0, w1 = a1, b1, a0, b0, w1, w0

    return (
        {"a": a0, "b": b0, "mean": a0 / (a0 + b0), "weight": w0},
        {"a": a1, "b": b1, "mean": a1 / (a1 + b1), "weight": w1},
        float(log_lik_prev),
    )


def closed_ebh(scores: np.ndarray, alpha: float = 0.10) -> int:
    """Closed e-BH (Xu, Fischer, Ramdas 2025): reject top-k where
    mean(top-k e-values) >= 1/α. Returns k_hat."""
    e = np.exp(np.clip(scores, -700, 700))
    e_desc = np.sort(e)[::-1]
    cumsum = np.cumsum(e_desc)
    means = cumsum / np.arange(1, len(e_desc) + 1)
    threshold = 1.0 / alpha
    rej = means >= threshold
    if rej.any():
        return int(np.where(rej)[0].max() + 1)
    return 0


def vanilla_ebh(scores: np.ndarray, alpha: float = 0.10) -> int:
    e = np.exp(np.clip(scores, -700, 700))
    K = len(e)
    e_desc_idx = np.argsort(-e)
    e_desc = e[e_desc_idx]
    threshold = K / (alpha * np.arange(1, K + 1))
    rej = e_desc >= threshold
    if rej.any():
        return int(np.where(rej)[0].max() + 1)
    return 0


def main(*, alpha: float = 0.10) -> None:
    print("loading fragmentation scores...")
    frag = pd.read_csv(RES / "fragmentation_scores.csv")
    f = frag["fragmentation_rate"].to_numpy()
    print(f"  {len(frag):,} clusters")

    # === ITEM 1: UNSUPERVISED g_0, g_1 via 2-component Beta mixture EM ===
    print()
    print("=== UNSUPERVISED CALIBRATION via 2-component Beta mixture EM ===")
    g0, g1, ll = fit_beta_2mix_em(f)
    print(f"  g_0 (verbatim, low-f):    Beta({g0['a']:.2f}, {g0['b']:.2f}), "
          f"mean = {g0['mean']:.3f}, weight = {g0['weight']:.3f}")
    print(f"  g_1 (paraphrase, high-f): Beta({g1['a']:.2f}, {g1['b']:.2f}), "
          f"mean = {g1['mean']:.3f}, weight = {g1['weight']:.3f}")
    print(f"  log-likelihood: {ll:.2f}")

    print()
    print("compare to FOIA-supervised g_0/g_1:")
    print(f"  supervised advocacy mean (g_0): 0.467")
    print(f"  supervised astroturf mean (g_1): 0.881")
    print(f"  unsupervised match: {abs(g0['mean'] - 0.467):.3f} & "
          f"{abs(g1['mean'] - 0.881):.3f} away")

    # Compute compound e-values per cluster
    f_clipped = np.clip(f, 1e-6, 1 - 1e-6)
    log_g0 = beta_dist.logpdf(f_clipped, g0["a"], g0["b"])
    log_g1 = beta_dist.logpdf(f_clipped, g1["a"], g1["b"])
    log_e = log_g1 - log_g0
    frag["log_e_unsup"] = log_e

    # === ITEM 2: CLOSED-eBH ===
    print()
    print(f"=== e-BH variants at α={alpha} ===")
    k_van = vanilla_ebh(log_e, alpha)
    k_cls = closed_ebh(log_e, alpha)
    print(f"  vanilla e-BH:    rejects {k_van:,}")
    print(f"  closed-eBH:      rejects {k_cls:,}  "
          f"({'gain' if k_cls > k_van else 'tie'} of {k_cls - k_van:+,})")

    # Attribution + AP
    df = frag.sort_values("log_e_unsup", ascending=False).reset_index(drop=True)
    rej_van = df.iloc[:k_van]
    rej_cls = df.iloc[:k_cls]

    print()
    print(f"{'method':<22}{'k':>8}{'astro':>8}{'astro%':>10}{'adv':>6}{'recall_astro':>14}")
    print('-' * 70)
    for name, rej in [("vanilla e-BH", rej_van),
                       ("closed-eBH", rej_cls)]:
        n = len(rej)
        n_astro = int(rej["y_astro"].sum())
        n_adv = int(rej["y_adv"].sum())
        rec = n_astro / max(int(frag["y_astro"].sum()), 1)
        print(f"{name:<22}{n:>8,}{n_astro:>8,}{100*n_astro/max(n,1):>9.1f}%"
              f"{n_adv:>6,}{100*rec:>13.1f}%")

    ap = average_precision_score(frag["y_astro"], log_e)
    print(f"\nAP of unsupervised fragmentation e-value: {ap:.3f}")
    print(f"  (vs supervised-calibration AP 0.815, base rate 0.386)")

    out = frag.copy()
    out.to_parquet(PROC / "cluster_evalues_fragmentation_unsup.parquet",
                    compression="zstd", index=False)
    print(f"\nwrote {PROC}/cluster_evalues_fragmentation_unsup.parquet")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--alpha", type=float, default=0.10)
    args = p.parse_args()
    main(alpha=args.alpha)
