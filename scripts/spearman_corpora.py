"""Spearman correlation: fragmentation log-e vs verbatim-coordination labels.

Expected sign: NEGATIVE (anti-correlated with verbatim coordination by design).
This makes the paper's central thesis (paraphrase-targeting) explicit
across all three corpora.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import spearmanr, beta as beta_dist
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


def fragmentation_at(coarse_df, fine_df, min_size=5):
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--proc-dir', type=Path, required=True)
    p.add_argument('--corpus-name', type=str, required=True)
    p.add_argument('--gamma-fine', type=float, default=0.97)
    p.add_argument('--min-size', type=int, default=5)
    args = p.parse_args()

    print(f'=== Spearman correlations on {args.corpus_name} (γ_fine={args.gamma_fine}, min-size={args.min_size}) ===\n')
    coarse = pq.read_table(args.proc_dir / 'clusters_leiden_r0.9.parquet').to_pandas()
    fine = pq.read_table(args.proc_dir / f'clusters_leiden_r{args.gamma_fine}.parquet').to_pandas()
    frag = fragmentation_at(coarse, fine, min_size=args.min_size)

    members = coarse[(coarse['cluster_id'] >= 0)][['row_id', 'cluster_id']]
    idx = pq.read_table(args.proc_dir / 'embedding_index.parquet', columns=['row_id', 'template_size']).to_pandas()
    member_t = members.merge(idx, on='row_id', how='left')
    member_t['template_size'] = member_t['template_size'].fillna(1).astype(int)
    cluster_t = member_t.groupby('cluster_id').agg(
        max_template=('template_size', 'max'),
        sum_template=('template_size', 'sum'),
        cluster_size=('row_id', 'size'),
    ).reset_index()
    frag = frag.merge(cluster_t, on='cluster_id', how='left').fillna(0)

    f = frag['fragmentation_rate'].to_numpy()
    a0, b0, a1, b1, w0 = fit_beta_2mix_em(f)
    log_e = beta_dist.logpdf(np.clip(f, 1e-6, 1 - 1e-6), a1, b1) - beta_dist.logpdf(np.clip(f, 1e-6, 1 - 1e-6), a0, b0)
    frag['log_e'] = log_e

    print(f'  K = {len(frag):,}')
    print(f'  Beta mixture: g0 mean={a0/(a0+b0):.3f}, g1 mean={a1/(a1+b1):.3f}')
    print()
    print(f'  Spearman ρ between log_e and verbatim-coordination signals (negative = anti-correlated, as expected by design):')

    for col in ['max_template', 'sum_template', 'cluster_size']:
        if col not in frag.columns:
            continue
        rho, pval = spearmanr(frag['log_e'], frag[col])
        print(f'    log_e vs {col:<16}: ρ = {rho:+.4f}  (p = {pval:.2e})')

    if 'submissions.parquet' in [p.name for p in args.proc_dir.iterdir()] and (args.proc_dir / 'submissions.parquet').exists():
        sub = pq.read_table(args.proc_dir / 'submissions.parquet').to_pandas()
        if 'uploader_uuid' in sub.columns:
            sub = sub[sub['uploader_uuid'].fillna('') != '']
            members_u = members.merge(sub[['comment_id', 'uploader_uuid']],
                                       on='comment_id' if 'comment_id' in members.columns else None,
                                       how='left') if 'comment_id' in members.columns else None
            if members_u is not None:
                cu = members_u.dropna(subset=['uploader_uuid']).groupby(['cluster_id', 'uploader_uuid']).size().reset_index(name='n')
                modal = cu.sort_values(['cluster_id', 'n'], ascending=[True, False]).groupby('cluster_id').first().reset_index()
                cluster_size_total = members.groupby('cluster_id').size().rename('n_total').reset_index()
                modal = modal.merge(cluster_size_total, on='cluster_id', how='right').fillna(0)
                modal['modal_uploader_frac'] = modal['n'] / modal['n_total'].clip(lower=1)
                frag2 = frag.merge(modal[['cluster_id', 'modal_uploader_frac']], on='cluster_id', how='left').fillna(0)
                rho, pval = spearmanr(frag2['log_e'], frag2['modal_uploader_frac'])
                print(f'    log_e vs modal_uploader_frac: ρ = {rho:+.4f}  (p = {pval:.2e})')


if __name__ == '__main__':
    main()
