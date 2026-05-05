from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import beta as beta_dist
from sklearn.mixture import GaussianMixture

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / 'data' / 'processed'
RES = ROOT / 'results'


def fit_beta_2mix_em(f, n_iter=80, tol=1e-5):
    eps = 1e-6
    f = np.clip(f.astype(np.float64), eps, 1 - eps)
    logit = np.log(f / (1 - f)).reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, random_state=0, n_init=5)
    gmm.fit(logit)
    z = gmm.predict(logit)
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
    return (a0, b0, a1, b1, w0)


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


def main():
    print('=== Time-aware analysis: do rejected clusters submit in bursts? ===\n')
    coarse = pq.read_table(PROC / 'clusters_leiden_r0.9.parquet').to_pandas()
    fine = pq.read_table(PROC / 'clusters_leiden_r0.97.parquet').to_pandas()
    base_labels = pd.read_csv(RES / 'fragmentation_scores.csv')[['cluster_id', 'y_astro', 'y_adv']]
    frag = fragmentation_at(coarse, fine).merge(base_labels, on='cluster_id', how='inner')

    f = frag['fragmentation_rate'].to_numpy()
    a0, b0, a1, b1, w0 = fit_beta_2mix_em(f)
    f_clipped = np.clip(f, 1e-6, 1 - 1e-6)
    log_e = beta_dist.logpdf(f_clipped, a1, b1) - beta_dist.logpdf(f_clipped, a0, b0)
    e = np.exp(np.clip(log_e, -700, 700))
    order = np.argsort(-e)
    K = len(frag)
    threshold = K / (0.10 * np.arange(1, K + 1))
    rej_idx = np.where(e[order] >= threshold)[0]
    k_hat = int(rej_idx.max() + 1) if rej_idx.size else 0
    rej_cluster_ids = set(frag.iloc[order[:k_hat]]['cluster_id'].tolist())
    nonrej_cluster_ids = set(frag.iloc[order[k_hat:]]['cluster_id'].tolist())
    print(f'rejected: {len(rej_cluster_ids):,}, non-rejected: {len(nonrej_cluster_ids):,}')

    print('\n[1] Joining with submissions for date_received...')
    subs = pq.read_table(PROC / 'submissions.parquet', columns=['comment_id', 'date_received']).to_pandas()
    subs['date_received'] = pd.to_datetime(subs['date_received'], errors='coerce')

    cluster_members = coarse[(coarse['cluster_id'] >= 0)][['cluster_id', 'comment_id']]
    member_dates = cluster_members.merge(subs, on='comment_id', how='inner')
    print(f'  {len(member_dates):,} cluster-member submission rows with dates')

    def cluster_burst_features(group):
        dts = group['date_received'].dropna().sort_values()
        if len(dts) < 2:
            return pd.Series({'span_days': 0.0, 'iqr_days': 0.0, 'log_mean_gap_seconds': float('nan')})
        span = (dts.iloc[-1] - dts.iloc[0]).total_seconds() / 86400.0
        iqr = (dts.quantile(0.75) - dts.quantile(0.25)).total_seconds() / 86400.0
        gaps = dts.diff().dropna().dt.total_seconds().to_numpy()
        gaps = gaps[gaps > 0]
        log_mg = float(np.log(np.mean(gaps))) if len(gaps) else float('nan')
        return pd.Series({'span_days': span, 'iqr_days': iqr, 'log_mean_gap_seconds': log_mg})

    print('  computing burst features per cluster (sample-based)...')
    rng = np.random.default_rng(0)
    sample_rej = rng.choice(list(rej_cluster_ids), size=min(2000, len(rej_cluster_ids)), replace=False)
    sample_non = rng.choice(list(nonrej_cluster_ids), size=min(2000, len(nonrej_cluster_ids)), replace=False)
    feats = member_dates[member_dates['cluster_id'].isin(np.concatenate([sample_rej, sample_non]))].groupby('cluster_id', as_index=False).apply(cluster_burst_features, include_groups=False).reset_index()
    feats = feats.rename(columns={'level_1': 'tmp'}).drop(columns=[c for c in ['tmp'] if c in feats.columns])
    feats['rejected'] = feats['cluster_id'].isin(rej_cluster_ids)

    print('\n=== Burst statistics by group (medians) ===')
    print(feats.groupby('rejected')[['span_days', 'iqr_days']].agg(['median', 'mean', 'std']).round(2).to_string())

    print('\n=== Distribution of cluster span_days (sample) ===')
    for grp_name, grp in feats.groupby('rejected'):
        sp = grp['span_days'].dropna()
        print(f'  {"REJECTED" if grp_name else "non-rej"}: '
              f'p10={sp.quantile(0.1):.2f}  p25={sp.quantile(0.25):.2f}  '
              f'p50={sp.median():.2f}  p75={sp.quantile(0.75):.2f}  p90={sp.quantile(0.9):.2f} days')

    out = feats.copy()
    out_path = RES / 'time_aware.csv'
    out.to_csv(out_path, index=False)
    print(f'\nwrote {out_path}')


if __name__ == '__main__':
    main()
