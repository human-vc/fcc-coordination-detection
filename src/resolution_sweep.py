from __future__ import annotations
import argparse
import shutil
import subprocess
import sys
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / 'data' / 'processed'
RES = ROOT / 'results'
RES.mkdir(exist_ok=True)
PY = sys.executable

def run(cmd: list[str]) -> None:
    print(f"  $ {' '.join(map(str, cmd))}")
    subprocess.run(cmd, check=True)

def main(*, resolutions: list[float], alpha: float, n_null_draws: int, min_cluster_size: int, partition: str, n_iterations: int=2) -> None:
    summary = []
    for r in resolutions:
        tag = f'leiden_r{r}'
        clust_out = PROC / f'clusters_{tag}.parquet'
        ev_out = PROC / f'cluster_evalues_{tag}.parquet'
        fdr_default = PROC / 'fdr_rejections.parquet'
        print(f'\n=== resolution {r} ({partition}) ===')
        run([PY, str(ROOT / 'src' / 'cluster_singletons.py'), '--resolution', str(r), '--partition', partition, '--min-cluster-size', str(min_cluster_size), '--n-iterations', str(n_iterations), '--out-path', str(clust_out)])
        run([PY, str(ROOT / 'src' / 'evalues.py'), '--n-null-draws', str(n_null_draws), '--min-cluster-size', str(min_cluster_size), '--cluster-path', str(clust_out), '--out-path', str(ev_out)])
        backup_ev = PROC / 'cluster_evalues.parquet.bak'
        if (PROC / 'cluster_evalues.parquet').exists():
            shutil.move(PROC / 'cluster_evalues.parquet', backup_ev)
        shutil.copy(ev_out, PROC / 'cluster_evalues.parquet')
        try:
            run([PY, str(ROOT / 'src' / 'ebh.py'), '--alpha', str(alpha)])
        finally:
            (PROC / 'cluster_evalues.parquet').unlink(missing_ok=True)
            if backup_ev.exists():
                shutil.move(backup_ev, PROC / 'cluster_evalues.parquet')
        fdr_snap = RES / f'fdr_rejections_{tag}.parquet'
        shutil.copy(fdr_default, fdr_snap)
        rj = pq.read_table(fdr_snap).to_pandas()
        cl = pq.read_table(clust_out).to_pandas()
        n_unique_clusters = int(cl[cl['cluster_id'] >= 0]['cluster_id'].nunique())
        n_rej = int(rj['rejected_ebh'].sum())
        rej_rows = int(rj.loc[rj['rejected_ebh'], 'n'].sum())
        max_size = int(cl[cl['cluster_id'] >= 0].groupby('cluster_id').size().max())
        summary.append({'resolution': r, 'candidate_clusters': int(len(rj)), 'unique_clusters_total': n_unique_clusters, 'largest_cluster_size': max_size, 'rej_ebh': n_rej, 'rows_in_rej': rej_rows})
    out = pd.DataFrame(summary)
    out_path = RES / f'resolution_sweep_{partition}.csv'
    out.to_csv(out_path, index=False)
    print(f'\n=== resolution sweep summary ({partition}) ===')
    print(out.to_string(index=False))
    print(f'\nwrote {out_path}')
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--resolutions', nargs='+', type=float, default=[0.85, 0.88, 0.9, 0.93, 0.96])
    p.add_argument('--partition', choices=['cpm', 'rb'], default='cpm')
    p.add_argument('--alpha', type=float, default=0.1)
    p.add_argument('--n-null-draws', type=int, default=5000)
    p.add_argument('--min-cluster-size', type=int, default=5)
    p.add_argument('--n-iterations', type=int, default=2, help='Leiden iterations per resolution (2=fast, -1=converge)')
    args = p.parse_args()
    main(resolutions=args.resolutions, partition=args.partition, alpha=args.alpha, n_null_draws=args.n_null_draws, min_cluster_size=args.min_cluster_size, n_iterations=args.n_iterations)
