from __future__ import annotations
import argparse
import math
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pickle
from scipy.stats import vonmises_fisher
import sys
sys.path.insert(0, 'src')
from evalues_lrt import vmf_mle, log_vmf_norm, log_unit_sphere_area, QMovMF
ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / 'data' / 'processed'
RES = ROOT / 'results'
K = 21606
ALPHA = 0.1
LOG_THRESHOLD = math.log(K / ALPHA)
RATIO_CAP = 50.0

def power_at(d, n, kappa, q, real_seeds, n_trials, rng):
    rejected = 0
    for t in range(n_trials):
        seed = real_seeds[rng.integers(0, len(real_seeds))].astype(np.float32)
        norm = float(np.linalg.norm(seed))
        if norm < 1e-09:
            continue
        mu_seed = seed / norm
        x = vonmises_fisher(mu_seed, kappa, seed=rng).rvs(n).astype(np.float32)
        perm = rng.permutation(n)
        half = n // 2
        (A, B) = (x[perm[:half]], x[perm[half:half * 2]])
        (mu1, kap1) = vmf_mle(A)
        log_p1 = log_vmf_norm(d, kap1) + kap1 * (B @ mu1)
        log_q1 = q.log_q(B)
        log_e1 = float(np.clip(log_p1 - log_q1, -RATIO_CAP, RATIO_CAP).sum())
        (mu2, kap2) = vmf_mle(B)
        log_p2 = log_vmf_norm(d, kap2) + kap2 * (A @ mu2)
        log_q2 = q.log_q(A)
        log_e2 = float(np.clip(log_p2 - log_q2, -RATIO_CAP, RATIO_CAP).sum())
        log_e = float(np.logaddexp(log_e1, log_e2) - math.log(2))
        if log_e >= LOG_THRESHOLD:
            rejected += 1
    return rejected / max(n_trials, 1)

def main(*, n_seeds: int=5000, n_trials: int=80) -> None:
    print('loading real-corpus embeddings + cluster info + q̂...')
    emb = np.load(PROC / 'embeddings_white_k5.npy', mmap_mode='r')
    cl = pq.read_table(PROC / 'clusters_leiden_r0.9.parquet').to_pandas()
    with open(PROC / 'q_movmf_cluster_aware.pkl', 'rb') as f:
        q_model = pickle.load(f)
    q = QMovMF(q_model)
    d = emb.shape[1]
    rng = np.random.default_rng(42)
    sing_rows = cl[cl['cluster_size'] <= 1]['row_id'].to_numpy()
    seed_idx = rng.choice(sing_rows, size=n_seeds, replace=False)
    real_seeds = emb[seed_idx].astype(np.float32)
    print(f'  d={d}, n_seeds={n_seeds}')
    ns = [8, 12, 16, 24, 32]
    kappas = [50, 100, 200, 500, 1000, 2000, 3000, 5000]
    print(f'\nempirical power (real-corpus geometry)')
    print(f"{'n':>4}  " + '  '.join((f'κ={k}' for k in kappas)))
    print('-' * (6 + len(kappas) * 9))
    rows = []
    for n in ns:
        line = f'{n:>4}'
        for kappa in kappas:
            p = power_at(d, n, kappa, q, real_seeds, n_trials, rng)
            line += f'  {p:>5.2f}'
            rows.append({'n': n, 'kappa': kappa, 'power': p, 'theory_threshold_C1.4': 1.4 * d / math.sqrt(n / 2), 'theory_threshold_C2.4': 2.4 * d / math.sqrt(n / 2)})
        print(line)
    df = pd.DataFrame(rows)
    df.to_csv(RES / 'synthetic_injection_real.csv', index=False)
    print(f'\nwrote {RES}/synthetic_injection_real.csv')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        (fig, ax) = plt.subplots(1, 1, figsize=(7, 4.5), dpi=150)
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(ns)))
        for (i, n) in enumerate(ns):
            sub = df[df['n'] == n].sort_values('kappa')
            ax.plot(sub['kappa'], sub['power'], marker='o', color=colors[i], label=f'n={n}', linewidth=2)
            ax.axvline(1.4 * d / math.sqrt(n / 2), color=colors[i], ls=':', alpha=0.4)
        ax.set_xscale('log')
        ax.set_xlabel('injected $\\kappa$', fontsize=11)
        ax.set_ylabel('empirical rejection rate (real-corpus seeds)', fontsize=11)
        ax.set_title('Synthetic injection in real SBERT geometry: vertical lines = theoretical $\\kappa^*$ at $C=1.4$', fontsize=10)
        ax.axhline(0.5, color='gray', lw=0.5, ls=':')
        ax.axhline(0.9, color='gray', lw=0.5, ls=':')
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        fig.tight_layout()
        fig.savefig(RES / 'synthetic_injection_real.png', dpi=150)
        print(f'wrote {RES}/synthetic_injection_real.png')
    except Exception as e:
        print(f'(plot skipped: {e})')
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--n-seeds', type=int, default=5000)
    p.add_argument('--n-trials', type=int, default=80)
    args = p.parse_args()
    main(n_seeds=args.n_seeds, n_trials=args.n_trials)
