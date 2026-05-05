from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import beta as beta_dist
from sklearn.metrics import average_precision_score
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
    return (a0, b0, a1, b1)


def fragmentation_at(coarse, fine, min_size=8):
    coarse = coarse[(coarse['cluster_id'] >= 0) & (coarse['cluster_size'] >= min_size)][['row_id', 'cluster_id']].rename(columns={'cluster_id': 'coarse_id'})
    fine = fine[fine['cluster_id'] >= 0][['row_id', 'cluster_id']].rename(columns={'cluster_id': 'fine_id'})
    j = coarse.merge(fine, on='row_id', how='left')
    j['fine_id'] = j['fine_id'].fillna(-1).astype(int)
    g = j.groupby('coarse_id').agg(
        n=('row_id', 'size'),
        n_distinct_fine=('fine_id', lambda s: int((s != -1).sum() and len(set(s) - {-1}))),
    ).reset_index()
    g['fragmentation_rate'] = g['n_distinct_fine'] / g['n'].clip(lower=1)
    return g.rename(columns={'coarse_id': 'cluster_id'})


def main():
    print('=== Cluster-size strata sensitivity at γ_fine = 0.97 ===\n')
    coarse = pq.read_table(PROC / 'clusters_leiden_r0.9.parquet').to_pandas()
    fine = pq.read_table(PROC / 'clusters_leiden_r0.97.parquet').to_pandas()
    base_labels = pd.read_csv(RES / 'fragmentation_scores.csv')[['cluster_id', 'y_astro', 'y_adv']]
    frag = fragmentation_at(coarse, fine).merge(base_labels, on='cluster_id', how='inner')

    f = frag['fragmentation_rate'].to_numpy()
    a0, b0, a1, b1 = fit_beta_2mix_em(f)
    f_clipped = np.clip(f, 1e-6, 1 - 1e-6)
    log_e = beta_dist.logpdf(f_clipped, a1, b1) - beta_dist.logpdf(f_clipped, a0, b0)
    frag['log_e'] = log_e

    bins = [(8, 16), (16, 32), (32, 64), (64, 128), (128, 100000)]
    rows = []
    for lo, hi in bins:
        sub = frag[(frag['n'] >= lo) & (frag['n'] < hi)]
        if len(sub) < 10:
            continue
        K = len(sub)
        e = np.exp(np.clip(sub['log_e'].to_numpy(), -700, 700))
        order = np.argsort(-e)
        threshold = K / (0.10 * np.arange(1, K + 1))
        rej_idx = np.where(e[order] >= threshold)[0]
        k_hat = int(rej_idx.max() + 1) if rej_idx.size else 0
        if k_hat > 0:
            mask = sub.iloc[order[:k_hat]]
            astro_pct = float(mask['y_astro'].mean())
            astro_recall = mask['y_astro'].sum() / max(int(sub['y_astro'].sum()), 1)
        else:
            astro_pct = astro_recall = 0.0
        try:
            ap = average_precision_score(sub['y_astro'], sub['log_e']) if sub['y_astro'].sum() > 0 else float('nan')
        except Exception:
            ap = float('nan')
        rows.append({
            'size_bucket': f'[{lo}, {hi})' if hi < 100000 else f'>= {lo}',
            'K': K,
            'base_rate': float(sub['y_astro'].mean()),
            'AP': ap,
            'rejected': k_hat,
            'precision_astro': astro_pct,
            'recall_astro': astro_recall,
        })

    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    out_path = RES / 'sensitivity_size.csv'
    out.to_csv(out_path, index=False)
    print(f'\nwrote {out_path}')


if __name__ == '__main__':
    main()
