from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import beta as beta_dist
ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / 'data' / 'processed'
RES = ROOT / 'results'

def fit_beta_mom(samples: np.ndarray, eps: float=1e-06) -> tuple[float, float]:
    s = np.clip(samples.astype(np.float64), eps, 1 - eps)
    mu = float(s.mean())
    var = float(s.var())
    if var <= eps:
        return (1.0, 1.0)
    common = mu * (1 - mu) / var - 1
    if common <= 0:
        return (1.0, 1.0)
    return (max(mu * common, 0.001), max((1 - mu) * common, 0.001))

def main(*, output_path: Path | None=None, alpha: float=0.1) -> None:
    print('loading fragmentation scores...')
    frag = pd.read_csv(RES / 'fragmentation_scores.csv')
    print(f'  {len(frag):,} clusters with fragmentation scores')
    print()
    print('calibrating g_0, g_1 from FOIA attribution...')
    astro_mask = frag['frac_astroturf'].fillna(0) >= 0.5
    adv_mask = frag['frac_advocacy'].fillna(0) >= 0.5
    f_astro = frag.loc[astro_mask, 'fragmentation_rate'].to_numpy()
    f_adv = frag.loc[adv_mask, 'fragmentation_rate'].to_numpy()
    print(f'  astroturf-attributed:  n={len(f_astro):,}, f mean={f_astro.mean():.3f}, std={f_astro.std():.3f}')
    print(f'  advocacy-attributed:   n={len(f_adv):,}, f mean={f_adv.mean():.3f}, std={f_adv.std():.3f}')
    (a1, b1) = fit_beta_mom(f_astro)
    (a0, b0) = fit_beta_mom(f_adv)
    print(f'  Beta_1 (paraphrase) = ({a1:.2f}, {b1:.2f}), mean = {a1 / (a1 + b1):.3f}')
    print(f'  Beta_0 (verbatim)   = ({a0:.2f}, {b0:.2f}), mean = {a0 / (a0 + b0):.3f}')
    print()
    print('computing per-cluster e-values...')
    f_clipped = np.clip(frag['fragmentation_rate'].to_numpy(), 1e-06, 1 - 1e-06)
    log_g0 = beta_dist.logpdf(f_clipped, a0, b0)
    log_g1 = beta_dist.logpdf(f_clipped, a1, b1)
    log_e = log_g1 - log_g0
    frag['log_e_frag'] = log_e
    frag['e_frag'] = np.exp(np.clip(log_e, -700, 700))
    print()
    print(f'applying e-BH at α={alpha}...')
    df = frag.sort_values('log_e_frag', ascending=False).reset_index(drop=True)
    K = len(df)
    df['rank'] = np.arange(1, K + 1)
    df['threshold'] = K / (alpha * df['rank'])
    df['rejects'] = df['e_frag'] >= df['threshold']
    if df['rejects'].any():
        k_hat = int(df.index[df['rejects']].max() + 1)
    else:
        k_hat = 0
    rej = df.iloc[:k_hat]
    print(f'  K (candidates):    {K:,}')
    print(f'  k_hat (rejected):  {k_hat:,}  ({100 * k_hat / K:.1f}%) — this is SELECTIVE (was 100% with concentration-based test)')
    print()
    print('rejection set composition:')
    if k_hat > 0:
        n_astro = int(rej['y_astro'].sum())
        n_adv = int(rej['y_adv'].sum())
        print(f'  astroturf-attributed in rejection: {n_astro:,} of {len(rej):,} = {100 * n_astro / len(rej):.1f}% precision')
        print(f"  advocacy-attributed in rejection:  {n_adv:,} ({100 * n_adv / len(rej):.1f}%) — these are 'mistakes' if we target paraphrase coordination")
        recall_astro = n_astro / max(int(frag['y_astro'].sum()), 1)
        print(f'  recall on astroturf:   {100 * recall_astro:.1f}%')
        recall_adv = n_adv / max(int(frag['y_adv'].sum()), 1)
        print(f'  recall on advocacy:    {100 * recall_adv:.1f}%  (lower is better)')
    from sklearn.metrics import average_precision_score
    ap = average_precision_score(frag['y_astro'], frag['log_e_frag'])
    print()
    print(f'AP of fragmentation e-value against NYAG astroturf: {ap:.3f}')
    print(f"  (vs base rate {frag['y_astro'].mean():.3f}; raw frag-rate AP was 0.815)")
    out_path = output_path or PROC / 'cluster_evalues_fragmentation.parquet'
    df.to_parquet(out_path, compression='zstd', index=False)
    print(f'\nwrote {out_path}')
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--alpha', type=float, default=0.1)
    p.add_argument('--output-path', type=Path, default=None)
    args = p.parse_args()
    main(alpha=args.alpha, output_path=args.output_path)
