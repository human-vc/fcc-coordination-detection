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
    if len(f) < 4 or f.var() < 1e-10:
        return 0.5, 1.0, 0.5, 1.0, 0.5
    logit = np.log(f / (1 - f)).reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, random_state=0, n_init=3)
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
    print('=== Bootstrap CIs on the headline result (γ_coarse=0.90, γ_fine=0.97) ===\n')
    coarse = pq.read_table(PROC / 'clusters_leiden_r0.9.parquet').to_pandas()
    fine = pq.read_table(PROC / 'clusters_leiden_r0.97.parquet').to_pandas()
    base_labels = pd.read_csv(RES / 'fragmentation_scores.csv')[['cluster_id', 'y_astro', 'y_adv']]
    frag = fragmentation_at(coarse, fine).merge(base_labels, on='cluster_id', how='inner')

    # also pull expanded coordination ground truth
    expanded = pd.read_csv(RES / 'expanded_coordination_groundtruth.csv')
    cluster_truth = expanded.groupby('cluster_id').agg(
        expanded_coord=('expanded_coord', 'any'),
    ).reset_index()
    frag = frag.merge(cluster_truth, on='cluster_id', how='left')
    frag['expanded_coord'] = frag['expanded_coord'].fillna(False)

    f = frag['fragmentation_rate'].to_numpy()
    a0, b0, a1, b1, w0 = fit_beta_2mix_em(f)
    f_clipped = np.clip(f, 1e-6, 1 - 1e-6)
    log_e = beta_dist.logpdf(f_clipped, a1, b1) - beta_dist.logpdf(f_clipped, a0, b0)
    frag['log_e'] = log_e

    n_boot = 1000
    rng = np.random.default_rng(0)
    K = len(frag)
    aps_astro = []
    aps_expanded = []
    precs_astro = []
    precs_expanded = []
    recalls = []
    for b in range(n_boot):
        idx = rng.integers(0, K, K)
        sub = frag.iloc[idx]
        try:
            ap_a = average_precision_score(sub['y_astro'], sub['log_e']) if sub['y_astro'].sum() > 0 else float('nan')
            ap_e = average_precision_score(sub['expanded_coord'].astype(int), sub['log_e']) if sub['expanded_coord'].sum() > 0 else float('nan')
        except Exception:
            ap_a = ap_e = float('nan')
        aps_astro.append(ap_a)
        aps_expanded.append(ap_e)
        e = np.exp(np.clip(sub['log_e'].to_numpy(), -700, 700))
        order = np.argsort(-e)
        threshold = K / (0.10 * np.arange(1, K + 1))
        rej_idx = np.where(e[order] >= threshold)[0]
        k_hat = int(rej_idx.max() + 1) if rej_idx.size else 0
        if k_hat > 0:
            mask = sub.iloc[order[:k_hat]]
            precs_astro.append(float(mask['y_astro'].mean()))
            precs_expanded.append(float(mask['expanded_coord'].mean()))
            recalls.append(mask['y_astro'].sum() / max(int(sub['y_astro'].sum()), 1))
        else:
            precs_astro.append(float('nan'))
            precs_expanded.append(float('nan'))
            recalls.append(float('nan'))

    def ci(arr, q1=0.025, q2=0.975):
        a = np.array([x for x in arr if not np.isnan(x)])
        if len(a) == 0:
            return float('nan'), float('nan'), float('nan')
        return float(np.median(a)), float(np.quantile(a, q1)), float(np.quantile(a, q2))

    median_apa, lo_apa, hi_apa = ci(aps_astro)
    median_ape, lo_ape, hi_ape = ci(aps_expanded)
    median_pa, lo_pa, hi_pa = ci(precs_astro)
    median_pe, lo_pe, hi_pe = ci(precs_expanded)
    median_r, lo_r, hi_r = ci(recalls)

    print(f'Bootstrap (n={n_boot}, point ± 95% CI):')
    print(f'  AP astroturf      : {median_apa:.3f}  [{lo_apa:.3f}, {hi_apa:.3f}]')
    print(f'  AP expanded coord : {median_ape:.3f}  [{lo_ape:.3f}, {hi_ape:.3f}]')
    print(f'  Precision astroturf : {median_pa:.3f}  [{lo_pa:.3f}, {hi_pa:.3f}]')
    print(f'  Precision expanded  : {median_pe:.3f}  [{lo_pe:.3f}, {hi_pe:.3f}]')
    print(f'  Recall astroturf  : {median_r:.3f}  [{lo_r:.3f}, {hi_r:.3f}]')

    out = pd.DataFrame({
        'metric': ['ap_astro', 'ap_expanded', 'precision_astro', 'precision_expanded', 'recall_astro'],
        'median': [median_apa, median_ape, median_pa, median_pe, median_r],
        'ci_lo': [lo_apa, lo_ape, lo_pa, lo_pe, lo_r],
        'ci_hi': [hi_apa, hi_ape, hi_pa, hi_pe, hi_r],
    })
    out.to_csv(RES / 'bootstrap_ci.csv', index=False)
    print(f'\nwrote {RES}/bootstrap_ci.csv')


if __name__ == '__main__':
    main()
