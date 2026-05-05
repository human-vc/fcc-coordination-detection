from __future__ import annotations
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / 'data' / 'processed'
RES = ROOT / 'results'

def main() -> None:
    coh = pq.read_table(PROC / 'cluster_evalues_leiden_r0.9.parquet').to_pandas()
    coh = coh[coh['n'] >= 8]
    att = pd.read_csv(RES / 'attribution_table_r0.9.csv')
    df = coh.merge(att[['cluster_id', 'frac_astroturf']], on='cluster_id', how='left')
    df['y'] = (df['frac_astroturf'].fillna(0) >= 0.5).astype(int)
    df['score'] = df['T_obs']
    print(f'size-≥-8 clusters: {len(df):,}')
    print(f"  astroturf positives: {int(df['y'].sum()):,} ({100 * df['y'].mean():.1f}%)")
    print(f"  cosine sim T_obs range: [{df['score'].min():.3f}, {df['score'].max():.3f}]")
    print()
    print(f"{'τ':>6}{'flagged':>10}{'astro':>8}{'precision':>11}{'recall':>9}{'F1':>8}")
    print('-' * 60)
    rows = []
    for tau in np.arange(0.9, 1.001, 0.005):
        flag = df['score'] >= tau
        n_flag = int(flag.sum())
        n_astro = int((flag & (df['y'] == 1)).sum())
        if n_flag == 0:
            (prec, rec, f1) = (0, 0, 0)
        else:
            prec = n_astro / n_flag
            rec = n_astro / max(int(df['y'].sum()), 1)
            f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0
        if n_flag:
            print(f'{tau:>6.3f}{n_flag:>10,}{n_astro:>8,}{100 * prec:>10.1f}%{100 * rec:>8.1f}%{f1:>8.3f}')
        rows.append({'tau': tau, 'flagged': n_flag, 'astro': n_astro, 'precision': prec, 'recall': rec, 'f1': f1})
    out = pd.DataFrame(rows)
    out.to_csv(RES / 'baseline_threshold_sweep.csv', index=False)
    from sklearn.metrics import roc_auc_score, average_precision_score
    auc = roc_auc_score(df['y'], df['score'])
    ap = average_precision_score(df['y'], df['score'])
    print()
    print(f'=== threshold baseline summary ===')
    print(f'  AUC: {auc:.3f}')
    print(f'  AP:  {ap:.3f}')
    print(f"  best F1: {out['f1'].max():.3f} at τ={out.loc[out['f1'].idxmax(), 'tau']:.3f}")
    target_threshold = df.loc[df['y'] == 1, 'score'].min()
    flag_at_recall1 = df['score'] >= target_threshold
    p_recall1 = (flag_at_recall1 & (df['y'] == 1)).sum() / max(flag_at_recall1.sum(), 1)
    print(f'  At recall=100% on astroturf (τ={target_threshold:.3f}): precision={100 * p_recall1:.1f}%, n_flagged={int(flag_at_recall1.sum()):,}')
    print()
    print(f'wrote {RES}/baseline_threshold_sweep.csv')
if __name__ == '__main__':
    main()
