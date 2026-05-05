"""Fit a mixture of von Mises-Fisher (movMF) to the corpus-marginal q on S^{d-1}.

Used as the null density q̂ in the split-LRT e-value (see notes/construction.md
§3, §7.2). EM with Banerjee–Dhillon–Ghosh–Sra 2005 algorithm.

Output `q_movmf.pkl`:
  {
    "K_q":      int,
    "mus":      (K_q, d) float64 — component mean directions on S^{d-1}
    "kappas":   (K_q,)   float64 — concentrations
    "weights":  (K_q,)   float64 — mixture weights, sum to 1
    "log_likelihood_train": float
    "log_likelihood_heldout": float
    "tail_pad":  {"weight": 0.01, "kappa": 1.0, "mu": <arbitrary unit vec>}
                 -- added externally; see add_tail_pad() below
  }

The tail-pad component is REQUIRED for Theorem 2's sup-norm bound to be finite:
without it, q̂ can underrepresent regions with no training mass, and
‖q/q̂ − 1‖_∞ blows up. With the pad, q̂ has a uniform-on-sphere floor of weight
0.01 (more or less), giving a finite sup-norm bound.

Reference: Banerjee et al. 2005 "Clustering on the unit hypersphere using
vMF distributions" JMLR 6:1345; Hornik & Grün 2014 on κ MLE.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from time import time

import numpy as np
from scipy.special import gammaln, ive, logsumexp

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
DEFAULT_INPUT = PROC / "embeddings_white_k5.npy"
DEFAULT_OUTPUT = PROC / "q_movmf.pkl"


# ---------- vMF helpers (single source of truth; mirror verify_proofs.py) ----------

def log_unit_sphere_area(d: int) -> float:
    return float(np.log(2.0) + (d / 2) * np.log(np.pi) - gammaln(d / 2))


def log_vmf_norm(d: int, kappa: float) -> float:
    """log C_d(κ) = log[κ^{d/2-1} / ((2π)^{d/2} I_{d/2-1}(κ))]."""
    if kappa < 1e-12:
        return -log_unit_sphere_area(d)
    nu = d / 2 - 1
    log_I = float(np.log(ive(nu, kappa)) + kappa)  # ive = I_ν · e^{-κ}
    return float(nu * np.log(kappa) - (d / 2) * np.log(2 * np.pi) - log_I)


def kappa_mle(r: float, d: int, n_newton: int = 3) -> float:
    """Banerjee approx + Newton on A_d(κ) = r,  where r ∈ [0, 1)."""
    r = float(min(max(r, 1e-9), 1 - 1e-9))
    kappa = r * (d - r * r) / (1.0 - r * r)
    for _ in range(n_newton):
        nu = d / 2 - 1
        A = float(ive(nu + 1, kappa) / ive(nu, kappa))
        # A'(κ) = 1 - A² - (d-1)/κ · A
        A_prime = 1.0 - A * A - (d - 1) / max(kappa, 1e-9) * A
        kappa = max(kappa - (A - r) / max(A_prime, 1e-9), 1e-3)
    return float(kappa)


def log_vmf_pdf(x: np.ndarray, mu: np.ndarray, kappa: float) -> np.ndarray:
    """log p_vMF(x | μ, κ).  x: (n, d), μ: (d,)  ->  (n,)."""
    d = x.shape[1]
    return log_vmf_norm(d, kappa) + kappa * (x @ mu)


# ---------- EM ----------

def _log_responsibilities(
    x: np.ndarray, mus: np.ndarray, kappas: np.ndarray, weights: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Return (log γ_{ik},  total log-likelihood). x: (n, d), mus: (K, d)."""
    d = x.shape[1]
    # log p(x_i | k) = log C_d(κ_k) + κ_k <μ_k, x_i>
    log_norms = np.array([log_vmf_norm(d, k) for k in kappas])     # (K,)
    inner = x @ mus.T                                              # (n, K)
    log_p = log_norms[None, :] + kappas[None, :] * inner            # (n, K)
    log_w = np.log(weights + 1e-300)                                # (K,)
    log_joint = log_p + log_w[None, :]                             # (n, K)
    log_total = logsumexp(log_joint, axis=1)                       # (n,)
    log_gamma = log_joint - log_total[:, None]
    return log_gamma, float(log_total.sum())


def _initialize(x: np.ndarray, K: int, rng: np.random.Generator,
                init_kappa: float = 50.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random K data points as means, uniform weights, fixed initial kappa."""
    n, d = x.shape
    idx = rng.choice(n, size=K, replace=False)
    mus = x[idx].astype(np.float64).copy()
    mus /= np.linalg.norm(mus, axis=1, keepdims=True)
    kappas = np.full(K, init_kappa)
    weights = np.full(K, 1.0 / K)
    return mus, kappas, weights


def fit_movmf(
    x: np.ndarray, K: int, *, n_iter: int = 50, tol: float = 1e-3,
    rng: np.random.Generator | None = None, verbose: bool = True,
) -> dict:
    """EM for movMF on x ∈ S^{d-1}.  x: (n, d) float."""
    rng = rng or np.random.default_rng(0)
    n, d = x.shape
    mus, kappas, weights = _initialize(x, K, rng)
    prev_ll = -np.inf
    for it in range(n_iter):
        log_gamma, ll = _log_responsibilities(x, mus, kappas, weights)
        gamma = np.exp(log_gamma)                                  # (n, K)
        # M-step
        Nk = gamma.sum(axis=0)                                     # (K,)
        weights = Nk / n
        # weighted sums
        r_vec = gamma.T @ x                                        # (K, d)
        r_norm = np.linalg.norm(r_vec, axis=1)                     # (K,)
        # protect against empty components
        active = Nk > 1e-6
        for k in range(K):
            if not active[k]:
                # re-seed from a random data point
                mus[k] = x[rng.integers(0, n)]
                mus[k] /= np.linalg.norm(mus[k]) + 1e-12
                kappas[k] = 50.0
                weights[k] = 1.0 / (n * 100)
                continue
            mus[k] = r_vec[k] / max(r_norm[k], 1e-12)
            r_bar = float(r_norm[k] / Nk[k])
            kappas[k] = kappa_mle(r_bar, d=d)
        if verbose and (it < 3 or it % 10 == 0):
            print(f"    iter {it:3d}  ll={ll:.2f}  active={int(active.sum())}/{K}")
        if abs(ll - prev_ll) < tol * abs(prev_ll):
            if verbose:
                print(f"    converged at iter {it} (Δll={ll - prev_ll:.4f})")
            break
        prev_ll = ll
    return {"mus": mus, "kappas": kappas, "weights": weights,
            "log_likelihood_train": float(ll), "K": K, "n_iter_done": it + 1}


def heldout_loglik(
    x_held: np.ndarray, mus: np.ndarray, kappas: np.ndarray, weights: np.ndarray,
) -> float:
    _, ll = _log_responsibilities(x_held, mus, kappas, weights)
    return float(ll / x_held.shape[0])


# ---------- log q̂ ----------

def log_q_movmf(
    x: np.ndarray, mus: np.ndarray, kappas: np.ndarray, weights: np.ndarray,
) -> np.ndarray:
    """log q̂(x_i) = log Σ_k w_k p_vMF(x_i | μ_k, κ_k).  Returns (n,)."""
    d = x.shape[1]
    log_norms = np.array([log_vmf_norm(d, k) for k in kappas])
    inner = x @ mus.T
    log_p = log_norms[None, :] + kappas[None, :] * inner
    log_w = np.log(weights + 1e-300)
    return logsumexp(log_p + log_w[None, :], axis=1)


# ---------- tail-pad ----------

def add_tail_pad(model: dict, *, pad_weight: float = 0.01,
                 pad_kappa: float = 1.0, d: int | None = None,
                 rng: np.random.Generator | None = None) -> dict:
    """Append a low-concentration component to lower-bound q̂ on the sphere."""
    rng = rng or np.random.default_rng(0)
    mus = model["mus"]
    if d is None:
        d = mus.shape[1]
    pad_mu = rng.standard_normal(d).astype(np.float64)
    pad_mu /= np.linalg.norm(pad_mu)
    new_mus = np.vstack([mus, pad_mu])
    new_kappas = np.concatenate([model["kappas"], [pad_kappa]])
    new_weights = np.concatenate([
        model["weights"] * (1 - pad_weight),
        [pad_weight],
    ])
    out = dict(model)
    out["mus"] = new_mus
    out["kappas"] = new_kappas
    out["weights"] = new_weights
    out["tail_pad"] = {"weight": pad_weight, "kappa": pad_kappa, "mu": pad_mu}
    out["K"] = len(new_kappas)
    return out


# ---------- main: K-sweep ----------

def main(*, input_path: Path | None = None, output_path: Path | None = None,
         n_train: int = 200_000, n_held: int = 50_000,
         Ks: tuple[int, ...] = (5, 10, 25, 50),
         n_iter: int = 50, seed: int = 0,
         pad_weight: float = 0.01, pad_kappa: float = 1.0) -> None:
    in_path = input_path or DEFAULT_INPUT
    out_path = output_path or DEFAULT_OUTPUT
    rng = np.random.default_rng(seed)

    print(f"loading {in_path} (mmap)...")
    emb = np.load(in_path, mmap_mode="r")
    n_total, d = emb.shape
    print(f"  shape {emb.shape}  dtype {emb.dtype}")

    n_sample = n_train + n_held
    if n_total < n_sample:
        raise SystemExit(f"need {n_sample:,} rows, have {n_total:,}")
    idx = rng.choice(n_total, size=n_sample, replace=False); idx.sort()
    x_full = emb[idx].astype(np.float32)
    # ensure unit norm (whitening should preserve, but guard)
    x_full /= np.linalg.norm(x_full, axis=1, keepdims=True) + 1e-12
    x_train = x_full[:n_train]
    x_held = x_full[n_train:]
    print(f"  train: {x_train.shape}  held-out: {x_held.shape}")

    results = []
    for K in Ks:
        print(f"\nfitting K_q = {K}...")
        t0 = time()
        model = fit_movmf(x_train, K, n_iter=n_iter, rng=rng)
        ll_held = heldout_loglik(x_held, model["mus"], model["kappas"], model["weights"])
        elapsed = time() - t0
        print(f"  K={K:3d}  ll_train/n={model['log_likelihood_train']/n_train:.4f}  "
              f"ll_held/n={ll_held:.4f}  ({elapsed:.1f}s)")
        results.append({
            "K": K, "model": model, "ll_held_per_obs": ll_held,
            "elapsed_s": elapsed,
        })

    # pick best K by held-out
    best = max(results, key=lambda r: r["ll_held_per_obs"])
    print(f"\nbest K_q = {best['K']}  (held-out ll/n = {best['ll_held_per_obs']:.4f})")

    # add tail-pad
    final = add_tail_pad(
        best["model"], pad_weight=pad_weight, pad_kappa=pad_kappa,
        d=d, rng=rng
    )
    final["log_likelihood_heldout"] = best["ll_held_per_obs"]
    final["K_sweep_results"] = [
        {"K": r["K"], "ll_held_per_obs": r["ll_held_per_obs"]} for r in results
    ]
    final["dim"] = d
    final["n_train"] = n_train
    final["n_held"] = n_held
    final["seed"] = seed

    print(f"\nfinal model: K_q+pad = {final['K']}  (incl tail-pad κ={pad_kappa}, w={pad_weight})")
    print(f"writing {out_path}")
    with open(out_path, "wb") as f:
        pickle.dump(final, f)
    print("done.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input-path", type=Path, default=None,
                   help="default: data/processed/embeddings_white_k5.npy")
    p.add_argument("--output-path", type=Path, default=None,
                   help="default: data/processed/q_movmf.pkl")
    p.add_argument("--n-train", type=int, default=200_000)
    p.add_argument("--n-held", type=int, default=50_000)
    p.add_argument("--Ks", type=int, nargs="+", default=[5, 10, 25, 50])
    p.add_argument("--n-iter", type=int, default=50)
    p.add_argument("--pad-weight", type=float, default=0.01)
    p.add_argument("--pad-kappa", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    main(input_path=args.input_path, output_path=args.output_path,
         n_train=args.n_train, n_held=args.n_held, Ks=tuple(args.Ks),
         n_iter=args.n_iter, pad_weight=args.pad_weight,
         pad_kappa=args.pad_kappa, seed=args.seed)
