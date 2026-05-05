"""Spot-check the rejected clusters on a second corpus by printing actual
comment texts from the top-N most-rejected clusters.

Output: human-readable report. Look for paraphrase-coordination patterns
(semantically similar but textually distinct) vs random/organic.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import beta as beta_dist
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
        n0 = gamma0.sum(); n1 = (1 - gamma0).sum()
        if n0 < 1 or n1 < 1: break
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
    p.add_argument('--n-clusters', type=int, default=5,
                   help='number of top-rejected clusters to spot-check')
    p.add_argument('--n-comments-per-cluster', type=int, default=4)
    p.add_argument('--min-size', type=int, default=5)
    p.add_argument('--alpha', type=float, default=0.10)
    args = p.parse_args()

    print(f'=== Spot-check rejections on {args.corpus_name} (γ_fine={args.gamma_fine}) ===\n')
    coarse = pq.read_table(args.proc_dir / 'clusters_leiden_r0.9.parquet').to_pandas()
    fine = pq.read_table(args.proc_dir / f'clusters_leiden_r{args.gamma_fine}.parquet').to_pandas()
    comments = pq.read_table(args.proc_dir / 'comments.parquet').to_pandas()
    frag = fragmentation_at(coarse, fine, min_size=args.min_size)
    f_arr = frag['fragmentation_rate'].to_numpy()
    a0, b0, a1, b1, w0 = fit_beta_2mix_em(f_arr)
    log_e = beta_dist.logpdf(np.clip(f_arr, 1e-6, 1 - 1e-6), a1, b1) - beta_dist.logpdf(np.clip(f_arr, 1e-6, 1 - 1e-6), a0, b0)
    e = np.exp(np.clip(log_e, -700, 700))
    K = len(frag)
    order = np.argsort(-e)
    threshold = K / (args.alpha * np.arange(1, K + 1))
    rej_idx = np.where(e[order] >= threshold)[0]
    k_hat = int(rej_idx.max() + 1) if rej_idx.size else 0
    print(f'  K = {K:,}, rejected = {k_hat:,} ({100*k_hat/K:.1f}%)')
    print(f'  g_0 mean = {a0/(a0+b0):.3f}, g_1 mean = {a1/(a1+b1):.3f}')
    print()

    rng = np.random.default_rng(0)
    print(f'=== Top-{args.n_clusters} rejected clusters (highest log_e) ===')
    for rank, oi in enumerate(order[:args.n_clusters]):
        row = frag.iloc[oi]
        cid = int(row['cluster_id'])
        n = int(row['n'])
        log_e_val = float(log_e[oi])
        members = coarse[coarse['cluster_id'] == cid][['row_id']]
        members = members.merge(comments, on='row_id', how='inner')
        if len(members) > args.n_comments_per_cluster:
            members = members.sample(n=args.n_comments_per_cluster, random_state=int(rng.integers(0, 10**9)))
        print(f'\n--- Cluster #{cid}  (size={n}, log_e={log_e_val:.2f}, frag_rate={row["fragmentation_rate"]:.3f}) ---')
        for i, (_, m) in enumerate(members.iterrows()):
            txt = str(m.get('comment_text', ''))[:300].replace('\n', ' ')
            print(f'    [{i+1}] {txt}{"..." if len(str(m.get("comment_text",""))) > 300 else ""}')

    print(f'\n=== Sample of {args.n_clusters} NON-rejected clusters (lowest log_e) ===')
    for rank, oi in enumerate(order[-args.n_clusters:]):
        row = frag.iloc[oi]
        cid = int(row['cluster_id'])
        n = int(row['n'])
        log_e_val = float(log_e[oi])
        members = coarse[coarse['cluster_id'] == cid][['row_id']]
        members = members.merge(comments, on='row_id', how='inner')
        if len(members) > args.n_comments_per_cluster:
            members = members.sample(n=args.n_comments_per_cluster, random_state=int(rng.integers(0, 10**9)))
        print(f'\n--- Cluster #{cid}  (size={n}, log_e={log_e_val:.2f}, frag_rate={row["fragmentation_rate"]:.3f}) ---')
        for i, (_, m) in enumerate(members.iterrows()):
            txt = str(m.get('comment_text', ''))[:300].replace('\n', ' ')
            print(f'    [{i+1}] {txt}{"..." if len(str(m.get("comment_text",""))) > 300 else ""}')


if __name__ == '__main__':
    main()
