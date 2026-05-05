from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import beta as beta_dist, norm
ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / 'data' / 'processed'
RES = ROOT / 'results'

def ppi_mean(*, f_lab: np.ndarray, y_lab: np.ndarray, f_unlab: np.ndarray, tuning_lambda: float=1.0) -> dict:
    n = len(f_lab)
    N = len(f_unlab)
    f_lab = f_lab.astype(np.float64)
    y_lab = y_lab.astype(np.float64)
    f_unlab = f_unlab.astype(np.float64)
    theta_hat = tuning_lambda * f_unlab.mean() + (y_lab - tuning_lambda * f_lab).mean()
    bias = (tuning_lambda * f_lab - y_lab).mean()
    var_unlab = float(np.var(f_unlab, ddof=1)) if N > 1 else 0.0
    rectifier = y_lab - tuning_lambda * f_lab
    var_lab = float(np.var(rectifier, ddof=1)) if n > 1 else 0.0
    se = float(np.sqrt(tuning_lambda ** 2 * var_unlab / max(N, 1) + var_lab / max(n, 1)))
    if tuning_lambda == 'auto':
        cov_fy = float(np.cov(f_lab, y_lab, ddof=1)[0, 1]) if n > 1 else 0.0
        var_f = float(np.var(f_lab, ddof=1)) if n > 1 else 1.0
        lambda_star = cov_fy / var_f
        return ppi_mean(f_lab=f_lab, y_lab=y_lab, f_unlab=f_unlab, tuning_lambda=lambda_star)
    return {'theta_hat': float(theta_hat), 'se': se, 'ci_lo': float(theta_hat - 1.96 * se), 'ci_hi': float(theta_hat + 1.96 * se), 'bias_estimate': float(bias), 'n_labeled': n, 'N_unlabeled': N, 'tuning_lambda': float(tuning_lambda), 'naive_f_mean': float(f_unlab.mean()), 'naive_y_mean': float(y_lab.mean()) if n else None}

def main(*, judge_path: Path, attr_path: Path, output_path: Path | None=None) -> None:
    print(f'loading LLM judgments: {judge_path}')
    j = pq.read_table(judge_path).to_pandas()
    j = j.dropna(subset=['coordinated']).copy()
    j['f'] = j['coordinated'].astype(int)
    print(f'  {len(j):,} clusters with valid LLM classification')
    print(f'loading attribution: {attr_path}')
    att = pd.read_csv(attr_path)
    j = j.merge(att[['cluster_id', 'frac_astroturf', 'top_label']], on='cluster_id', how='left')
    j['foia_astro'] = (j['frac_astroturf'].fillna(0) >= 0.5).astype(int)
    print()
    print('=== Naive (LLM-only) coordination rate ===')
    print(f"  P(coordinated by LLM) = {j['f'].mean():.4f}  (n={len(j):,})")
    print()
    print('=== FOIA-attributed astroturf rate (gold) ===')
    n_astro_in_sample = int(j['foia_astro'].sum())
    print(f'  P(FOIA astroturf | sampled) = {n_astro_in_sample}/{len(j):,} = {n_astro_in_sample / len(j):.4f}')
    print()
    print('=== LLM precision on FOIA-attributed astroturf ===')
    foia_subset = j[j['foia_astro'] == 1]
    print(f"  P(LLM=coord | FOIA=astroturf) = {foia_subset['f'].mean():.4f}  (n={len(foia_subset)})")
    nonfoia_subset = j[j['foia_astro'] == 0]
    print(f"  P(LLM=coord | FOIA=not astroturf) = {nonfoia_subset['f'].mean():.4f}  (n={len(nonfoia_subset)})")
    print()
    print('=== PPI++ estimate of FOIA-astroturf rate (using LLM as proxy) ===')
    f_lab = foia_subset['f'].to_numpy()
    y_lab = np.ones(len(foia_subset))
    nonfoia_in_lab = nonfoia_subset['f'].to_numpy()
    y_lab_full = np.concatenate([y_lab, np.zeros(len(nonfoia_subset))])
    f_lab_full = np.concatenate([f_lab, nonfoia_in_lab])
    f_unlab = j['f'].to_numpy()
    result = ppi_mean(f_lab=f_lab_full, y_lab=y_lab_full, f_unlab=f_unlab)
    print(f"  PPI θ̂ = {result['theta_hat']:.4f} ± {1.96 * result['se']:.4f}")
    print(f"  95% CI = [{result['ci_lo']:.4f}, {result['ci_hi']:.4f}]")
    print(f"  Naive LLM = {result['naive_f_mean']:.4f}; bias estimate = {result['bias_estimate']:.4f}")
    print()
    print('=== Compound e-value via IWR §7 mixture-LR with LLM as statistic ===')
    if 'confidence' in j.columns and j['confidence'].notna().sum() > 10:
        g0_conf = j.loc[j['foia_astro'] == 0, 'confidence'].dropna()
        g1_conf = j.loc[j['foia_astro'] == 1, 'confidence'].dropna()
        j['score'] = np.where(j['coordinated'] == 1, j['confidence'], 1.0 - j['confidence'])
        s0 = j.loc[j['foia_astro'] == 0, 'score'].dropna()
        s1 = j.loc[j['foia_astro'] == 1, 'score'].dropna()
        print(f'  H0 (non-astroturf) score:  mean={s0.mean():.3f} std={s0.std():.3f} n={len(s0)}')
        print(f'  H1 (astroturf) score:      mean={s1.mean():.3f} std={s1.std():.3f} n={len(s1)}')

        def fit_beta(s):
            s = np.clip(s, 1e-06, 1 - 1e-06)
            (mu, var) = (float(s.mean()), float(s.var()))
            if var <= 1e-09:
                return (1.0, 1.0)
            common = mu * (1 - mu) / var - 1
            return (max(mu * common, 0.001), max((1 - mu) * common, 0.001))
        (a0, b0) = fit_beta(s0)
        (a1, b1) = fit_beta(s1)
        print(f'  Beta_0 = ({a0:.2f}, {b0:.2f}), Beta_1 = ({a1:.2f}, {b1:.2f})')
        s = np.clip(j['score'].fillna(0.5).to_numpy(), 1e-06, 1 - 1e-06)
        log_g0 = beta_dist.logpdf(s, a0, b0)
        log_g1 = beta_dist.logpdf(s, a1, b1)
        log_e = log_g1 - log_g0
        j['log_e_llm'] = log_e
        print(f'  log_e_llm distribution: p10={np.quantile(log_e, 0.1):.2f} p50={np.quantile(log_e, 0.5):.2f} p90={np.quantile(log_e, 0.9):.2f}')
        df = j[['cluster_id', 'log_e_llm', 'foia_astro']].copy().sort_values('log_e_llm', ascending=False).reset_index(drop=True)
        df['e'] = np.exp(np.clip(df['log_e_llm'], -700, 700))
        K = len(df)
        df['rank'] = np.arange(1, K + 1)
        df['thr'] = K / (0.1 * df['rank'])
        df['rej'] = df['e'] >= df['thr']
        if df['rej'].any():
            k_hat = df.index[df['rej']].max() + 1
            rej = df.iloc[:k_hat]
        else:
            rej = df.iloc[:0]
        n_rej = len(rej)
        n_rej_astro = int(rej['foia_astro'].sum())
        print(f'  e-BH at α=0.10: {n_rej:,} rejected of {K:,}; of these {n_rej_astro} are FOIA-astro (precision {100 * n_rej_astro / max(n_rej, 1):.1f}%)')
    out_path = output_path or PROC / 'llm_judge_ppi_results.parquet'
    j.to_parquet(out_path, compression='zstd', index=False)
    print(f'\nwrote {out_path}')
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--judge-path', type=Path, default=PROC / 'llm_judge_n1000.parquet')
    p.add_argument('--attr-path', type=Path, default=RES / 'attribution_table_r0.9.csv')
    p.add_argument('--output-path', type=Path, default=None)
    args = p.parse_args()
    main(judge_path=args.judge_path, attr_path=args.attr_path, output_path=args.output_path)
