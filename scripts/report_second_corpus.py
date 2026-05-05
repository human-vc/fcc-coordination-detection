from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

def main(*, evalues_path: Path, alpha: float, output_path: Path) -> None:
    df = pq.read_table(evalues_path).to_pandas()
    df = df.sort_values('log_e', ascending=False).reset_index(drop=True)
    df['e'] = np.exp(np.clip(df['log_e'], -700, 700))
    K = len(df)
    df['rank'] = np.arange(1, K + 1)
    df['threshold'] = K / (alpha * df['rank'])
    df['rej'] = df['e'] >= df['threshold']
    if df['rej'].any():
        k_hat = df.index[df['rej']].max() + 1
    else:
        k_hat = 0
    rejection = df.iloc[:k_hat] if k_hat > 0 else df.iloc[:0]
    report = {'evalues_path': str(evalues_path), 'alpha': alpha, 'K_candidates': int(K), 'k_hat': int(k_hat), 'rejection_fraction': float(k_hat / max(K, 1)), 'log_e_quantiles': {'p10': float(df['log_e'].quantile(0.1)), 'p50': float(df['log_e'].quantile(0.5)), 'p90': float(df['log_e'].quantile(0.9))}, 'kappa_hat_quantiles': {'p10': float(df['kappa_hat'].quantile(0.1)), 'p50': float(df['kappa_hat'].quantile(0.5)), 'p90': float(df['kappa_hat'].quantile(0.9))} if 'kappa_hat' in df.columns else None, 'cluster_size_quantiles': {'p10': float(df['n'].quantile(0.1)), 'p50': float(df['n'].quantile(0.5)), 'p90': float(df['n'].quantile(0.9))}}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w') as f:
        json.dump(report, f, indent=2)
    print(f'wrote {output_path}')
    print(json.dumps(report, indent=2))
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--evalues-path', type=Path, required=True)
    p.add_argument('--alpha', type=float, default=0.1)
    p.add_argument('--output-path', type=Path, required=True)
    args = p.parse_args()
    main(evalues_path=args.evalues_path, alpha=args.alpha, output_path=args.output_path)
