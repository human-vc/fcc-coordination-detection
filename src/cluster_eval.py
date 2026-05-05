from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / 'data' / 'processed'
RES = ROOT / 'results'
RES.mkdir(exist_ok=True)

def load_method(method: str, *, cluster_path: Path | None=None, rejections_path: Path | None=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    if cluster_path is not None:
        cl = pq.read_table(cluster_path).to_pandas()
    elif method == 'leiden':
        cl = pq.read_table(PROC / 'clusters.parquet').to_pandas()
    else:
        cl = pq.read_table(PROC / f'clusters_{method}.parquet').to_pandas()
    rj_path = rejections_path or RES / f'fdr_rejections_{method}.parquet'
    rj = pq.read_table(rj_path).to_pandas()
    return (cl, rj)

def evaluate(method: str, gold_thresholds: list[int], majority_fracs: list[float], *, cluster_path: Path | None=None, rejections_path: Path | None=None) -> list[dict]:
    (cl, rj) = load_method(method, cluster_path=cluster_path, rejections_path=rejections_path)
    rejected_ids = set(rj.loc[rj['rejected_ebh'], 'cluster_id'].astype(int).tolist())
    rejected_n = len(rejected_ids)
    members = cl[cl['cluster_id'] >= 0].copy()
    if not len(members):
        return []
    grp = members.groupby('cluster_id')
    rows = []
    for gt in gold_thresholds:
        members[f'gold_{gt}'] = members['template_size'] >= gt
        gold_per_cluster = members.groupby('cluster_id')[f'gold_{gt}'].sum()
        size_per_cluster = grp.size()
        gold_frac = (gold_per_cluster / size_per_cluster).fillna(0.0)
        for mf in majority_fracs:
            tp_clusters = sum((1 for cid in rejected_ids if gold_frac.get(cid, 0.0) >= mf))
            fp_clusters = rejected_n - tp_clusters
            all_majority_clusters = set(gold_frac[gold_frac >= mf].index.astype(int))
            recovered = len(all_majority_clusters & rejected_ids)
            recall = recovered / max(1, len(all_majority_clusters))
            precision = tp_clusters / max(1, rejected_n)
            rows.append({'method': method, 'gold_threshold': gt, 'majority_frac': mf, 'rejected_clusters': rejected_n, 'TP_clusters': tp_clusters, 'FP_clusters': fp_clusters, 'majority_clusters_total': len(all_majority_clusters), 'cluster_precision': precision, 'cluster_recall': recall})
    return rows

def main(*, methods: list[str], gold_thresholds: list[int], majority_fracs: list[float], cluster_path: Path | None=None, rejections_path: Path | None=None, output: Path | None=None) -> None:
    all_rows = []
    for m in methods:
        try:
            all_rows.extend(evaluate(m, gold_thresholds, majority_fracs, cluster_path=cluster_path, rejections_path=rejections_path))
        except FileNotFoundError as e:
            print(f'  skip {m}: {e}')
    out = pd.DataFrame(all_rows)
    if out.empty:
        print('no methods evaluated')
        return
    path = output or RES / 'cluster_eval_table.csv'
    out.to_csv(path, index=False)
    focus = out[out['majority_frac'] == 0.5].drop(columns=['majority_frac'])
    print(focus.to_string(index=False))
    print(f'\nfull table → {path}')
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--methods', nargs='+', default=['leiden', 'connected_components', 'hdbscan_emb', 'minhash_lsh'])
    p.add_argument('--gold-thresholds', nargs='+', type=int, default=[10, 100, 1000])
    p.add_argument('--majority-fracs', nargs='+', type=float, default=[0.3, 0.5, 0.8])
    p.add_argument('--cluster-path', type=Path, default=None)
    p.add_argument('--rejections-path', type=Path, default=None)
    p.add_argument('--output', type=Path, default=None)
    args = p.parse_args()
    main(methods=args.methods, gold_thresholds=args.gold_thresholds, majority_fracs=args.majority_fracs, cluster_path=args.cluster_path, rejections_path=args.rejections_path, output=args.output)
