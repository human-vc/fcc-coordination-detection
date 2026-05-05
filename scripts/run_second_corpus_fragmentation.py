"""Second-corpus fragmentation analysis (FCC 14-28 or CFPB-2016-0025).

Inputs (in --proc-dir):
    clusters_leiden_r0.9.parquet   (coarse, γ=0.90)
    clusters_leiden_r0.97.parquet  (fine, γ=0.97)
    submissions.parquet            (per-submission metadata)
    embedding_index.parquet        (row_id <-> comment_id <-> template_size)
    comment_uploader.parquet  OR  comment_filer.parquet  (optional, for ground truth)

Output:
    {output_dir}/fragmentation_scores.csv     — per-cluster f_c + labels
    {output_dir}/headline_report.json         — AP, rejection set summary
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


def fcc14_28_labels(coarse_df, proc_dir, top_n_bulk=20):
    """Bulk-uploader ground truth for FCC 14-28: cluster's modal uploader UUID
    is in the top-N most-frequent uploader UUIDs corpus-wide.
    Also: template_size-based and uploader-concentration labels.
    """
    sub = pq.read_table(proc_dir / 'submissions.parquet', columns=['comment_id', 'uploader_uuid']).to_pandas()
    sub = sub[sub['uploader_uuid'].fillna('') != '']

    uploader_freq = sub.groupby('uploader_uuid').size().sort_values(ascending=False)
    top_uploaders = set(uploader_freq.head(top_n_bulk).index)
    print(f'  top {top_n_bulk} uploaders by frequency: {len(top_uploaders)}')
    print(f'  top uploader sizes: {list(uploader_freq.head(5).items())}')

    members = coarse_df[(coarse_df['cluster_id'] >= 0)][['row_id', 'comment_id', 'cluster_id']]
    members = members.merge(sub, on='comment_id', how='left')

    cluster_uploader = (members.dropna(subset=['uploader_uuid'])
                                .groupby(['cluster_id', 'uploader_uuid']).size()
                                .reset_index(name='n'))
    modal = (cluster_uploader.sort_values(['cluster_id', 'n'], ascending=[True, False])
                              .groupby('cluster_id').first().reset_index()
                              .rename(columns={'uploader_uuid': 'modal_uploader',
                                               'n': 'modal_uploader_count'}))
    cluster_size_total = members.groupby('cluster_id').size().rename('n_total').reset_index()
    modal = modal.merge(cluster_size_total, on='cluster_id', how='right')
    modal['modal_uploader_count'] = modal['modal_uploader_count'].fillna(0).astype(int)
    modal['modal_uploader_frac'] = modal['modal_uploader_count'] / modal['n_total'].clip(lower=1)
    modal['y_bulk'] = modal['modal_uploader'].isin(top_uploaders).astype(int)
    modal['y_concentrated'] = (modal['modal_uploader_frac'] >= 0.8).astype(int)
    return modal[['cluster_id', 'modal_uploader', 'modal_uploader_count',
                  'modal_uploader_frac', 'y_bulk', 'y_concentrated']]


def cfpb_labels(coarse_df, proc_dir, template_threshold=10):
    """Coordination labels for CFPB-2016-0025:
    - y_template: cluster's max template_size >= threshold (heavy exact-duplicate signal)
    - y_filer_repeat: cluster's modal filer_name accounts for >50% of comments
    """
    idx = pq.read_table(proc_dir / 'embedding_index.parquet',
                        columns=['row_id', 'template_size']).to_pandas()
    members = coarse_df[(coarse_df['cluster_id'] >= 0)][['row_id', 'cluster_id']]
    members = members.merge(idx, on='row_id', how='left')
    members['template_size'] = members['template_size'].fillna(1).astype(int)
    g = members.groupby('cluster_id').agg(
        n=('row_id', 'size'),
        max_template=('template_size', 'max'),
        sum_template=('template_size', 'sum'),
    ).reset_index()
    g['y_template'] = (g['max_template'] >= template_threshold).astype(int)
    g['y_heavy_template'] = (g['max_template'] >= 50).astype(int)
    return g[['cluster_id', 'max_template', 'sum_template', 'y_template', 'y_heavy_template']]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--proc-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--corpus', choices=['fcc14_28', 'cfpb_2016_0025'], required=True)
    p.add_argument('--alpha', type=float, default=0.10)
    p.add_argument('--min-size', type=int, default=8)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f'=== Second corpus fragmentation: {args.corpus} ===\n')
    coarse = pq.read_table(args.proc_dir / 'clusters_leiden_r0.9.parquet').to_pandas()
    fine = pq.read_table(args.proc_dir / 'clusters_leiden_r0.97.parquet').to_pandas()
    n_candidates = (coarse['cluster_size'] >= args.min_size).sum()
    print(f'  candidates (size >= {args.min_size}): {n_candidates:,}')

    frag = fragmentation_at(coarse, fine, min_size=args.min_size)
    print(f'  fragmentation computed for {len(frag):,} clusters')

    if args.corpus == 'fcc14_28':
        labels = fcc14_28_labels(coarse, args.proc_dir)
        frag = frag.merge(labels, on='cluster_id', how='left')
        target_cols = ['y_bulk', 'y_concentrated']
    else:
        labels = cfpb_labels(coarse, args.proc_dir)
        frag = frag.merge(labels, on='cluster_id', how='left')
        target_cols = ['y_template', 'y_heavy_template']
    frag = frag.fillna(0)

    f = frag['fragmentation_rate'].to_numpy()
    a0, b0, a1, b1, w0 = fit_beta_2mix_em(f)
    print(f'  Beta mixture: g0=Beta({a0:.2f},{b0:.2f}) mean={a0/(a0+b0):.3f} w={w0:.3f}')
    print(f'                g1=Beta({a1:.2f},{b1:.2f}) mean={a1/(a1+b1):.3f} w={1-w0:.3f}')
    f_clip = np.clip(f, 1e-6, 1 - 1e-6)
    log_e = beta_dist.logpdf(f_clip, a1, b1) - beta_dist.logpdf(f_clip, a0, b0)
    e = np.exp(np.clip(log_e, -700, 700))
    frag['log_e'] = log_e

    K = len(frag)
    order = np.argsort(-e)
    threshold = K / (args.alpha * np.arange(1, K + 1))
    rej_idx = np.where(e[order] >= threshold)[0]
    k_hat = int(rej_idx.max() + 1) if rej_idx.size else 0
    print(f'\n  e-BH at α={args.alpha}: rejects {k_hat:,} ({100*k_hat/K:.1f}%)')

    report = {'corpus': args.corpus, 'alpha': args.alpha, 'K': K, 'k_rejected': k_hat,
              'rejection_fraction': k_hat / max(K, 1),
              'g0': {'a': a0, 'b': b0, 'mean': a0/(a0+b0), 'weight': w0},
              'g1': {'a': a1, 'b': b1, 'mean': a1/(a1+b1), 'weight': 1-w0},
              'metrics': {}}
    for col in target_cols:
        if col not in frag.columns or frag[col].sum() == 0:
            continue
        ap = average_precision_score(frag[col].astype(int), log_e)
        sub = frag.iloc[order[:k_hat]]
        prec = float(sub[col].mean()) if k_hat > 0 else 0.0
        rec = float(sub[col].sum() / max(int(frag[col].sum()), 1)) if k_hat > 0 else 0.0
        base = float(frag[col].mean())
        print(f'  {col}: base={base:.3f}, AP={ap:.3f}, prec={prec:.3f}, rec={rec:.3f}')
        report['metrics'][col] = {'base_rate': base, 'AP': ap, 'precision': prec, 'recall': rec}

    frag.to_csv(args.output_dir / 'fragmentation_scores.csv', index=False)
    with (args.output_dir / 'headline_report.json').open('w') as f_out:
        json.dump(report, f_out, indent=2)
    print(f'\nwrote {args.output_dir}/fragmentation_scores.csv')
    print(f'wrote {args.output_dir}/headline_report.json')


if __name__ == '__main__':
    main()
