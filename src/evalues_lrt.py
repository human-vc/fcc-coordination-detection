from __future__ import annotations
import argparse
import pickle
from pathlib import Path
from time import time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.special import gammaln, ive, logsumexp
ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / 'data' / 'processed'
DEFAULT_EMB = PROC / 'embeddings_white_k5.npy'
DEFAULT_CLUST = PROC / 'clusters_leiden_r0.9.parquet'
DEFAULT_Q = PROC / 'q_movmf.pkl'
DEFAULT_OUT = PROC / 'cluster_evalues_lrt.parquet'

def log_unit_sphere_area(d: int) -> float:
    return float(np.log(2.0) + d / 2 * np.log(np.pi) - gammaln(d / 2))

def log_vmf_norm(d: int, kappa: float) -> float:
    if kappa < 1e-12:
        return -log_unit_sphere_area(d)
    nu = d / 2 - 1
    ive_val = float(ive(nu, kappa))
    if ive_val > 0.0:
        log_I = float(np.log(ive_val) + kappa)
    else:
        log_I = float(nu * np.log(kappa / 2.0) - gammaln(nu + 1))
    return float(nu * np.log(kappa) - d / 2 * np.log(2 * np.pi) - log_I)

def kappa_mle(r: float, d: int, *, n_newton: int=3, kappa_max: float | None=None) -> float:
    if kappa_max is None:
        kappa_max = 10.0 * d
    r = float(min(max(r, 1e-09), 1 - 1e-09))
    kappa = r * (d - r * r) / (1.0 - r * r)
    kappa = min(kappa, kappa_max)
    nu = d / 2 - 1
    for _ in range(n_newton):
        try:
            A = float(ive(nu + 1, kappa) / ive(nu, kappa))
        except Exception:
            break
        A_prime = 1.0 - A * A - (d - 1) / max(kappa, 1e-09) * A
        step = (A - r) / max(A_prime, 1e-09)
        step = float(np.clip(step, -kappa / 2, kappa / 2))
        kappa = max(kappa - step, 0.001)
        kappa = min(kappa, kappa_max)
    return float(kappa)

def vmf_mle(x: np.ndarray) -> tuple[np.ndarray, float]:
    d = x.shape[1]
    mean = x.mean(axis=0)
    r = float(np.linalg.norm(mean))
    if r < 1e-09:
        return (mean / max(r, 1e-12), 0.001)
    return (mean / r, kappa_mle(r, d))

class QMovMF:

    def __init__(self, model: dict):
        self.mus = np.asarray(model['mus'], dtype=np.float32)
        self.kappas = np.asarray(model['kappas'], dtype=np.float64)
        self.weights = np.asarray(model['weights'], dtype=np.float64)
        d = self.mus.shape[1]
        self.log_norms = np.array([log_vmf_norm(d, k) for k in self.kappas])
        self.log_w = np.log(self.weights + 1e-300)
        self.d = d

    def log_q(self, x: np.ndarray) -> np.ndarray:
        inner = x @ self.mus.T
        log_p = self.log_norms[None, :] + self.kappas[None, :] * inner
        return logsumexp(log_p + self.log_w[None, :], axis=1)

def split_lrt_log_e(x: np.ndarray, q: QMovMF, rng: np.random.Generator, *, swap_and_average: bool=True, ratio_log_cap: float=50.0) -> tuple[float, float, float]:
    n = x.shape[0]
    perm = rng.permutation(n)
    half = n // 2
    A_idx = perm[:half]
    B_idx = perm[half:half * 2]
    (A, B) = (x[A_idx], x[B_idx])
    (mu1, kap1) = vmf_mle(A)
    log_p1 = log_vmf_norm(B.shape[1], kap1) + kap1 * (B @ mu1)
    log_q1 = q.log_q(B)
    log_ratio_1 = np.clip(log_p1 - log_q1, -ratio_log_cap, ratio_log_cap)
    log_e1 = float(log_ratio_1.sum())
    if not swap_and_average:
        return (log_e1, kap1, half)
    (mu2, kap2) = vmf_mle(B)
    log_p2 = log_vmf_norm(A.shape[1], kap2) + kap2 * (A @ mu2)
    log_q2 = q.log_q(A)
    log_ratio_2 = np.clip(log_p2 - log_q2, -ratio_log_cap, ratio_log_cap)
    log_e2 = float(log_ratio_2.sum())
    log_e_avg = float(logsumexp([log_e1, log_e2]) - np.log(2.0))
    return (log_e_avg, 0.5 * (kap1 + kap2), half)

def main(*, embedding_path: Path | None=None, cluster_path: Path | None=None, q_path: Path | None=None, output_path: Path | None=None, min_cluster_size: int=8, swap_and_average: bool=True, ratio_log_cap: float=50.0, seed: int=0, max_clusters: int | None=None) -> None:
    emb_path = embedding_path or DEFAULT_EMB
    cl_path = cluster_path or DEFAULT_CLUST
    q_path = q_path or DEFAULT_Q
    out_path = output_path or DEFAULT_OUT
    print(f'loading embeddings (mmap): {emb_path}')
    emb = np.load(emb_path, mmap_mode='r')
    print(f'  shape {emb.shape}  dtype {emb.dtype}')
    print(f'loading q̂ from {q_path}')
    with open(q_path, 'rb') as f:
        q_model = pickle.load(f)
    q = QMovMF(q_model)
    print(f'  q̂: K_q = {len(q.kappas)} components (incl tail-pad), dim {q.d}')
    if q.d != emb.shape[1]:
        raise SystemExit(f'dim mismatch: q̂ has dim {q.d}, embeddings have dim {emb.shape[1]}')
    print(f'loading clusters: {cl_path}')
    cl = pq.read_table(cl_path).to_pandas()
    cand = cl[(cl['cluster_id'] >= 0) & (cl['cluster_size'] >= min_cluster_size)]
    groups = cand.groupby('cluster_id')
    n_clusters = groups.ngroups
    print(f'  {n_clusters:,} clusters with size >= {min_cluster_size}')
    if max_clusters is not None and n_clusters > max_clusters:
        keep_ids = list(groups.groups.keys())[:max_clusters]
        cand = cand[cand['cluster_id'].isin(keep_ids)]
        groups = cand.groupby('cluster_id')
        n_clusters = groups.ngroups
        print(f'  capped to {n_clusters:,} for this run')
    rng = np.random.default_rng(seed)
    rows = []
    t0 = time()
    print_every = max(1, n_clusters // 20)
    for (i, (cid, group)) in enumerate(groups):
        member_rows = group['row_id'].to_numpy(dtype=np.int64)
        x = emb[member_rows].astype(np.float32, copy=False)
        norms = np.linalg.norm(x, axis=1)
        ok = norms > 1e-09
        if ok.sum() < min_cluster_size:
            continue
        x = x[ok] / norms[ok, None]
        (log_e, kappa_hat_avg, m_eff) = split_lrt_log_e(x, q, rng, swap_and_average=swap_and_average, ratio_log_cap=ratio_log_cap)
        log_e = float(np.clip(log_e, -1000000000.0, 1000000000.0))
        e = float(np.exp(min(log_e, 700.0)))
        rows.append({'cluster_id': int(cid), 'n': int(x.shape[0]), 'm': int(m_eff), 'kappa_hat': float(kappa_hat_avg), 'log_e': log_e, 'e': e})
        if (i + 1) % print_every == 0:
            print(f'  {i + 1:,}/{n_clusters:,} clusters processed   ({time() - t0:.1f}s)')
    out = pd.DataFrame(rows)
    out.to_parquet(out_path, compression='zstd', index=False)
    print(f'\nwrote {out_path}  ({len(out):,} clusters)')
    print()
    print('e-value distribution (log):')
    for q_ in [0.5, 0.9, 0.99]:
        print(f"  log e q={q_:.2f}: {out['log_e'].quantile(q_):.2f}")
    print(f"  count e>1:    {(out['log_e'] > 0).sum():,}")
    print(f"  count e>20:   {(out['log_e'] > np.log(20)).sum():,}")
    print(f"  count e>1000: {(out['log_e'] > np.log(1000)).sum():,}")
    print()
    print('kappa_hat distribution:')
    for q_ in [0.5, 0.9, 0.99]:
        print(f"  q={q_:.2f}: kappa_hat={out['kappa_hat'].quantile(q_):.1f}")
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--embedding-path', type=Path, default=None)
    p.add_argument('--cluster-path', type=Path, default=None)
    p.add_argument('--q-path', type=Path, default=None)
    p.add_argument('--output-path', type=Path, default=None)
    p.add_argument('--min-cluster-size', type=int, default=8)
    p.add_argument('--no-swap', action='store_true', help='single split (no swap-and-average); default is K=2 swap')
    p.add_argument('--ratio-log-cap', type=float, default=50.0, help='per-observation log-ratio cap (Theorem 2 / R6 mitigation)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--max-clusters', type=int, default=None, help='for testing: cap the number of clusters processed')
    args = p.parse_args()
    main(embedding_path=args.embedding_path, cluster_path=args.cluster_path, q_path=args.q_path, output_path=args.output_path, min_cluster_size=args.min_cluster_size, swap_and_average=not args.no_swap, ratio_log_cap=args.ratio_log_cap, seed=args.seed, max_clusters=args.max_clusters)
