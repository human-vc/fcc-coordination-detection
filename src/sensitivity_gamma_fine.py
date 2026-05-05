from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import beta as beta_dist
from sklearn.mixture import GaussianMixture
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / 'data' / 'processed'
RES = ROOT / 'results'


def fit_beta_2mix_em(f: np.ndarray, n_iter: int = 80, tol: float = 1e-5):
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
    return ((a0, b0, w0), (a1, b1, 1 - w0))


def fragmentation_at(coarse_clusters: pd.DataFrame, fine_clusters: pd.DataFrame, min_size: int = 8) -> pd.DataFrame:
    coarse = coarse_clusters[(coarse_clusters['cluster_id'] >= 0) & (coarse_clusters['cluster_size'] >= min_size)][['row_id', 'cluster_id']].rename(columns={'cluster_id': 'coarse_id'})
    fine = fine_clusters[fine_clusters['cluster_id'] >= 0][['row_id', 'cluster_id']].rename(columns={'cluster_id': 'fine_id'})
    j = coarse.merge(fine, on='row_id', how='left')
    j['fine_id'] = j['fine_id'].fillna(-1).astype(int)
    g = j.groupby('coarse_id').agg(
        n=('row_id', 'size'),
        n_distinct_fine=('fine_id', lambda s: int((s != -1).sum() and len(set(s) - {-1}))),
        n_unclustered=('fine_id', lambda s: int((s == -1).sum())),
    ).reset_index()
    g['fragmentation_rate'] = g['n_distinct_fine'] / g['n'].clip(lower=1)
    return g.rename(columns={'coarse_id': 'cluster_id'})


def main():
    print('loading coarse (γ=0.90) candidate clusters and labels...')
    coarse = pq.read_table(PROC / 'clusters_leiden_r0.9.parquet').to_pandas()
    labels = pd.read_csv(RES / 'fragmentation_scores.csv')[['cluster_id', 'y_astro', 'y_adv']]

    rows = []
    available = [0.93, 0.94, 0.95, 0.96, 0.97, 0.98]
    for g_fine in available:
        path = PROC / f'clusters_leiden_r{g_fine}.parquet'
        if not path.exists():
            print(f'  γ_fine = {g_fine}: SKIP (no clustering file)')
            continue
        print(f'\nγ_fine = {g_fine}')
        fine = pq.read_table(path).to_pandas()
        frag = fragmentation_at(coarse, fine)
        frag = frag.merge(labels, on='cluster_id', how='inner')
        f = frag['fragmentation_rate'].to_numpy()
        (a0, b0, w0), (a1, b1, w1) = fit_beta_2mix_em(f)
        f_clipped = np.clip(f, 1e-6, 1 - 1e-6)
        log_e = beta_dist.logpdf(f_clipped, a1, b1) - beta_dist.logpdf(f_clipped, a0, b0)
        ap = average_precision_score(frag['y_astro'], log_e)
        K = len(frag)
        e = np.exp(np.clip(log_e, -700, 700))
        order = np.argsort(-e)
        e_sorted = e[order]
        threshold = K / (0.10 * np.arange(1, K + 1))
        rej_idx = np.where(e_sorted >= threshold)[0]
        k_hat = int(rej_idx.max() + 1) if rej_idx.size else 0
        if k_hat > 0:
            mask = frag.iloc[order[:k_hat]]
            astro_pct = float(mask['y_astro'].mean())
            astro_recall = mask['y_astro'].sum() / max(int(frag['y_astro'].sum()), 1)
        else:
            astro_pct = 0.0
            astro_recall = 0.0
        rows.append({
            'gamma_fine': g_fine,
            'g0_mean': a0 / (a0 + b0),
            'g1_mean': a1 / (a1 + b1),
            'g0_weight': w0,
            'AP': ap,
            'k_rejected_eBH': k_hat,
            'precision_astro': astro_pct,
            'recall_astro': astro_recall,
        })
        print(f'  g0 mean={a0/(a0+b0):.3f}, g1 mean={a1/(a1+b1):.3f}, w0={w0:.3f}')
        print(f'  AP={ap:.3f}, e-BH rejects {k_hat:,} at α=0.10, precision={astro_pct:.1%}, recall={astro_recall:.1%}')

    out = pd.DataFrame(rows)
    out.to_csv(RES / 'sensitivity_gamma_fine.csv', index=False)
    print()
    print(out.to_string(index=False))
    print(f'\nwrote {RES}/sensitivity_gamma_fine.csv')


if __name__ == '__main__':
    main()
