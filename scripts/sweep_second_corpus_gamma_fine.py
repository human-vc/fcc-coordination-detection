"""Sweep γ_fine per second-corpus, find peak AP.

Loads coarse (γ=0.90) clusters once, then iterates over fine γ in
{0.93, 0.95, 0.96, 0.97, 0.98} (whichever are available on disk).
For each, computes fragmentation rate, fits Beta-mixture EM, computes AP
against the corpus-specific labels, and reports per-γ metrics.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import beta as beta_dist
from sklearn.metrics import average_precision_score
from sklearn.mixture import GaussianMixture


def fit_beta_2mix_em(f, n_iter=80, tol=1e-5):
    eps = 1e-6
    f = np.clip(f.astype(np.float64), eps, 1 - eps)
    if f.var() < 1e-10 or len(f) < 4:
        return 1.0, 1.0, 1.0, 1.0, 0.5
    logit = np.log(f / (1 - f)).reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, random_state=0, n_init=3)
    gmm.fit(logit)
    z = gmm.predict(logit)
    if (z == 0).sum() == 0 or (z == 1).sum() == 0:
        return 1.0, 1.0, 1.0, 1.0, 0.5
    if f[z == 0].mean() > f[z == 1].mean():
        z = 1 - z

    def fit(mu, var):
        if var <= eps:
            return 1.0, 1.0
        c = mu * (1 - mu) / var - 1
        if c <= 0:
            return 1.0, 1.0
        return max(mu * c, 1e-3), max((1 - mu) * c, 1e-3)

    mu0, var0 = float(f[z == 0].mean()), float(f[z == 0].var())
    mu1, var1 = float(f[z == 1].mean()), float(f[z == 1].var())
    a0, b0 = fit(mu0, var0)
    a1, b1 = fit(mu1, var1)
    w0 = float((z == 0).mean())
    log_lik_prev = -np.inf
    for _ in range(n_iter):
        log_p0 = beta_dist.logpdf(f, a0, b0) + np.log(max(w0, 1e-9))
        log_p1 = beta_dist.logpdf(f, a1, b1) + np.log(max(1 - w0, 1e-9))
        log_total = np.logaddexp(log_p0, log_p1)
        gamma0 = np.exp(log_p0 - log_total)
        log_lik = log_total.sum()
        if abs(log_lik - log_lik_prev) < tol * abs(log_lik_prev):
            break
        log_lik_prev = log_lik
        n0 = gamma0.sum()
        n1 = (1 - gamma0).sum()
        if n0 < 1 or n1 < 1:
            break
        mu0 = (gamma0 * f).sum() / n0
        mu1 = ((1 - gamma0) * f).sum() / n1
        var0 = (gamma0 * (f - mu0) ** 2).sum() / n0
        var1 = ((1 - gamma0) * (f - mu1) ** 2).sum() / n1
        a0, b0 = fit(mu0, var0)
        a1, b1 = fit(mu1, var1)
        w0 = n0 / (n0 + n1)
    if a0 / (a0 + b0) > a1 / (a1 + b1):
        a0, b0, a1, b1, w0 = a1, b1, a0, b0, 1 - w0
    return a0, b0, a1, b1, w0


def fragmentation_at(coarse_df, fine_df, min_size=8):
    coarse = coarse_df[(coarse_df['cluster_id'] >= 0) & (coarse_df['cluster_size'] >= min_size)][['row_id', 'cluster_id']].rename(columns={'cluster_id': 'coarse_id'})
    fine = fine_df[fine_df['cluster_id'] >= 0][['row_id', 'cluster_id']].rename(columns={'cluster_id': 'fine_id'})
    j = coarse.merge(fine, on='row_id', how='left')
    j['fine_id'] = j['fine_id'].fillna(-1).astype(int)
    g = j.groupby('coarse_id').agg(
        n=('row_id', 'size'),
        n_distinct_fine=('fine_id', lambda s: int((s != -1).sum() and len(set(s) - {-1}))),
    ).reset_index()
    g['fragmentation_rate'] = g['n_distinct_fine'] / g['n'].clip(lower=1)
    return g.rename(columns={'coarse_id': 'cluster_id'})


def labels_fcc14_28(coarse_df, proc_dir, top_n_bulk=100):
    sub = pq.read_table(proc_dir / 'submissions.parquet', columns=['comment_id', 'uploader_uuid']).to_pandas()
    sub = sub[sub['uploader_uuid'].fillna('') != '']
    uploader_freq = sub.groupby('uploader_uuid').size().sort_values(ascending=False)
    top_uploaders = set(uploader_freq.head(top_n_bulk).index)
    members = coarse_df[(coarse_df['cluster_id'] >= 0)][['row_id', 'comment_id', 'cluster_id']]
    members_u = members.merge(sub, on='comment_id', how='left')
    cluster_uploader = (members_u.dropna(subset=['uploader_uuid']).groupby(['cluster_id', 'uploader_uuid']).size().reset_index(name='n'))
    modal = (cluster_uploader.sort_values(['cluster_id', 'n'], ascending=[True, False]).groupby('cluster_id').first().reset_index().rename(columns={'uploader_uuid': 'modal_uploader', 'n': 'modal_uploader_count'}))
    cluster_size_total = members.groupby('cluster_id').size().rename('n_total').reset_index()
    modal = modal.merge(cluster_size_total, on='cluster_id', how='right').fillna(0)
    modal['modal_uploader_frac'] = modal['modal_uploader_count'] / modal['n_total'].clip(lower=1)
    modal['y_bulk'] = modal['modal_uploader'].astype(str).isin(top_uploaders).astype(int)
    modal['y_concentrated'] = (modal['modal_uploader_frac'] >= 0.5).astype(int)

    idx = pq.read_table(proc_dir / 'embedding_index.parquet', columns=['row_id', 'template_size']).to_pandas()
    members_t = members.merge(idx, on='row_id', how='left')
    members_t['template_size'] = members_t['template_size'].fillna(1).astype(int)
    g = members_t.groupby('cluster_id').agg(
        max_template=('template_size', 'max'),
        sum_template=('template_size', 'sum'),
    ).reset_index()
    g['y_template'] = (g['max_template'] >= 5).astype(int)
    g['y_heavy_template'] = (g['max_template'] >= 10).astype(int)
    g['y_sum_heavy'] = (g['sum_template'] >= 50).astype(int)
    return modal[['cluster_id', 'y_bulk', 'y_concentrated']].merge(g[['cluster_id', 'y_template', 'y_heavy_template', 'y_sum_heavy']], on='cluster_id', how='outer').fillna(0)


def labels_cfpb(coarse_df, proc_dir):
    idx = pq.read_table(proc_dir / 'embedding_index.parquet', columns=['row_id', 'template_size']).to_pandas()
    members = coarse_df[(coarse_df['cluster_id'] >= 0)][['row_id', 'cluster_id']]
    members = members.merge(idx, on='row_id', how='left')
    members['template_size'] = members['template_size'].fillna(1).astype(int)
    g = members.groupby('cluster_id').agg(
        max_template=('template_size', 'max'),
        sum_template=('template_size', 'sum'),
    ).reset_index()
    g['y_template'] = (g['max_template'] >= 5).astype(int)
    g['y_heavy_template'] = (g['max_template'] >= 10).astype(int)
    g['y_sum_heavy'] = (g['sum_template'] >= 50).astype(int)
    return g[['cluster_id', 'y_template', 'y_heavy_template', 'y_sum_heavy']]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--proc-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--corpus', choices=['fcc14_28', 'cfpb_2016_0025'], required=True)
    p.add_argument('--alpha', type=float, default=0.10)
    p.add_argument('--min-size', type=int, default=8)
    p.add_argument('--gammas', type=str, default='0.93,0.95,0.96,0.97,0.98')
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f'=== γ_fine sweep on {args.corpus} ===\n')
    coarse = pq.read_table(args.proc_dir / 'clusters_leiden_r0.9.parquet').to_pandas()

    if args.corpus == 'fcc14_28':
        labels = labels_fcc14_28(coarse, args.proc_dir)
        target_cols = ['y_template', 'y_heavy_template', 'y_sum_heavy', 'y_bulk', 'y_concentrated']
    else:
        labels = labels_cfpb(coarse, args.proc_dir)
        target_cols = ['y_template', 'y_heavy_template', 'y_sum_heavy']

    rows = []
    gamma_list = [float(g) for g in args.gammas.split(',')]
    for g_fine in gamma_list:
        path = args.proc_dir / f'clusters_leiden_r{g_fine}.parquet'
        if not path.exists():
            print(f'  γ_fine={g_fine}: NO FILE, skip')
            continue
        fine = pq.read_table(path).to_pandas()
        frag = fragmentation_at(coarse, fine, min_size=args.min_size).merge(labels, on='cluster_id', how='left').fillna(0)
        if len(frag) < 30:
            print(f'  γ_fine={g_fine}: K={len(frag)} too small, skip')
            continue
        f = frag['fragmentation_rate'].to_numpy()
        a0, b0, a1, b1, w0 = fit_beta_2mix_em(f)
        f_clip = np.clip(f, 1e-6, 1 - 1e-6)
        log_e = beta_dist.logpdf(f_clip, a1, b1) - beta_dist.logpdf(f_clip, a0, b0)
        e = np.exp(np.clip(log_e, -700, 700))
        K = len(frag)
        order = np.argsort(-e)
        threshold = K / (args.alpha * np.arange(1, K + 1))
        rej_idx = np.where(e[order] >= threshold)[0]
        k_hat = int(rej_idx.max() + 1) if rej_idx.size else 0
        row = {'gamma_fine': g_fine, 'K': K,
               'g0_mean': a0/(a0+b0), 'g1_mean': a1/(a1+b1), 'g0_weight': w0,
               'mode_separation': a1/(a1+b1) - a0/(a0+b0),
               'k_rejected': k_hat,
               'rejection_rate': k_hat / max(K, 1)}
        for col in target_cols:
            base = float(frag[col].mean())
            try:
                ap = average_precision_score(frag[col].astype(int), log_e) if frag[col].sum() > 0 else float('nan')
            except Exception:
                ap = float('nan')
            row[f'AP_{col}'] = ap
            row[f'base_{col}'] = base
            if k_hat > 0:
                sel = frag.iloc[order[:k_hat]]
                row[f'prec_{col}'] = float(sel[col].mean())
                row[f'rec_{col}'] = float(sel[col].sum() / max(int(frag[col].sum()), 1))
            else:
                row[f'prec_{col}'] = 0.0
                row[f'rec_{col}'] = 0.0
        rows.append(row)
        print(f'  γ_fine={g_fine}: K={K:,}, g0={a0/(a0+b0):.3f}, g1={a1/(a1+b1):.3f}, sep={a1/(a1+b1)-a0/(a0+b0):.3f}, rej={k_hat:,} ({100*k_hat/K:.1f}%)')
        for col in target_cols:
            print(f'    {col}: base={row[f"base_{col}"]:.3f}, AP={row[f"AP_{col}"]:.3f}, prec={row[f"prec_{col}"]:.3f}, rec={row[f"rec_{col}"]:.3f}')

    out = pd.DataFrame(rows)
    out_path = args.output_dir / 'gamma_fine_sweep.csv'
    out.to_csv(out_path, index=False)
    print(f'\nwrote {out_path}')

    if len(rows) > 0:
        print('\n=== Best γ_fine per metric ===')
        for col in target_cols + ['mode_separation', 'rejection_rate']:
            if col in ['mode_separation', 'rejection_rate']:
                vals = pd.Series([r[col] for r in rows])
                gamma_at_max = rows[vals.idxmax()]['gamma_fine']
                print(f'  max {col}: {vals.max():.3f} at γ_fine={gamma_at_max}')
            else:
                ap_col = f'AP_{col}'
                if all(np.isnan(r[ap_col]) for r in rows):
                    continue
                vals = pd.Series([r[ap_col] for r in rows])
                gamma_at_max = rows[vals.idxmax()]['gamma_fine']
                print(f'  max {ap_col}: {vals.max():.3f} at γ_fine={gamma_at_max}')


if __name__ == '__main__':
    main()
