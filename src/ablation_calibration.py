from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import beta as beta_dist, gaussian_kde
from sklearn.metrics import average_precision_score
from sklearn.mixture import GaussianMixture

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / 'data' / 'processed'
RES = ROOT / 'results'


def fit_beta_2mix_em(f, n_iter=80, tol=1e-5):
    eps = 1e-6
    f = np.clip(f.astype(np.float64), eps, 1 - eps)
    logit = np.log(f / (1 - f)).reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, random_state=0, n_init=3)
    gmm.fit(logit)
    z = gmm.predict(logit)
    if f[z == 0].mean() > f[z == 1].mean():
        z = 1 - z

    def fit(mu, var):
        if var <= 1e-6:
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


def ebh(e, alpha):
    K = len(e)
    order = np.argsort(-e)
    threshold = K / (alpha * np.arange(1, K + 1))
    rej_idx = np.where(e[order] >= threshold)[0]
    k = int(rej_idx.max() + 1) if rej_idx.size else 0
    return order[:k]


def evaluate(log_e, labels, alpha=0.10, name='cal'):
    e = np.exp(np.clip(log_e, -700, 700))
    K = len(e)
    rej = ebh(e, alpha)
    if len(rej) > 0:
        sel = labels.iloc[rej]
        prec = float(sel['y_astro'].mean())
        rec = sel['y_astro'].sum() / max(int(labels['y_astro'].sum()), 1)
    else:
        prec = rec = 0.0
    ap = average_precision_score(labels['y_astro'], log_e) if labels['y_astro'].sum() > 0 else float('nan')
    return {'calibration': name, 'AP': ap, 'k_rejected': len(rej), 'precision_astro': prec, 'recall_astro': rec}


def cal_beta_em(f):
    a0, b0, a1, b1, w0 = fit_beta_2mix_em(f)
    log_g0 = beta_dist.logpdf(np.clip(f, 1e-6, 1 - 1e-6), a0, b0)
    log_g1 = beta_dist.logpdf(np.clip(f, 1e-6, 1 - 1e-6), a1, b1)
    return log_g1 - log_g0


def cal_gaussian_em(f):
    logit = np.log(np.clip(f, 1e-6, 1 - 1e-6) / (1 - np.clip(f, 1e-6, 1 - 1e-6))).reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, random_state=0, n_init=5)
    gmm.fit(logit)
    means = gmm.means_.flatten()
    if means[0] > means[1]:
        idx0, idx1 = 1, 0
    else:
        idx0, idx1 = 0, 1
    log_p = gmm._estimate_log_prob(logit)
    log_g0 = log_p[:, idx0]
    log_g1 = log_p[:, idx1]
    return log_g1 - log_g0


def cal_kde(f):
    f_clip = np.clip(f, 1e-6, 1 - 1e-6)
    median = np.median(f_clip)
    is_low = f_clip < median
    kde0 = gaussian_kde(f_clip[is_low], bw_method='scott')
    kde1 = gaussian_kde(f_clip[~is_low], bw_method='scott')
    log_g0 = np.log(np.maximum(kde0(f_clip), 1e-12))
    log_g1 = np.log(np.maximum(kde1(f_clip), 1e-12))
    return log_g1 - log_g0


def cal_threshold_split(f, q):
    f_clip = np.clip(f, 1e-6, 1 - 1e-6)
    thr = np.quantile(f_clip, q)
    is_low = f_clip < thr
    a0, b0 = max(f_clip[is_low].mean() * (((1 - f_clip[is_low].mean()) * f_clip[is_low].mean() / max(f_clip[is_low].var(), 1e-6)) - 1), 0.5), max((1 - f_clip[is_low].mean()) * (((1 - f_clip[is_low].mean()) * f_clip[is_low].mean() / max(f_clip[is_low].var(), 1e-6)) - 1), 0.5)
    a1, b1 = max(f_clip[~is_low].mean() * (((1 - f_clip[~is_low].mean()) * f_clip[~is_low].mean() / max(f_clip[~is_low].var(), 1e-6)) - 1), 0.5), max((1 - f_clip[~is_low].mean()) * (((1 - f_clip[~is_low].mean()) * f_clip[~is_low].mean() / max(f_clip[~is_low].var(), 1e-6)) - 1), 0.5)
    log_g0 = beta_dist.logpdf(f_clip, a0, b0)
    log_g1 = beta_dist.logpdf(f_clip, a1, b1)
    return log_g1 - log_g0


def main():
    print('=== C. Calibration robustness: same statistic, different mode-density estimators ===\n')
    coarse = pq.read_table(PROC / 'clusters_leiden_r0.9.parquet').to_pandas()
    fine = pq.read_table(PROC / 'clusters_leiden_r0.97.parquet').to_pandas()
    base_labels = pd.read_csv(RES / 'fragmentation_scores.csv')[['cluster_id', 'y_astro', 'y_adv']]
    frag = fragmentation_at(coarse, fine).merge(base_labels, on='cluster_id', how='inner')
    f = frag['fragmentation_rate'].to_numpy()
    print(f'  K = {len(frag):,}')

    rows = []
    print('\n[1] Beta-mixture EM (headline)...')
    rows.append(evaluate(cal_beta_em(f), frag, name='Beta-mixture EM (ours)'))
    print('[2] Gaussian-mixture EM on logit(f)...')
    rows.append(evaluate(cal_gaussian_em(f), frag, name='Gaussian-mixture EM (logit-space)'))
    print('[3] Kernel density estimation on median-split halves...')
    rows.append(evaluate(cal_kde(f), frag, name='KDE (median-split)'))
    print('[4] Hard quantile-threshold + Beta moment-matching...')
    rows.append(evaluate(cal_threshold_split(f, 0.5), frag, name='Threshold MOM (50/50)'))
    rows.append(evaluate(cal_threshold_split(f, 0.3), frag, name='Threshold MOM (30/70)'))
    rows.append(evaluate(cal_threshold_split(f, 0.7), frag, name='Threshold MOM (70/30)'))

    print('\n[5] Subsample-N analysis: small-sample regime ...')
    rng = np.random.default_rng(0)
    for N in [500, 1000, 2000, 5000]:
        aps = []
        ks = []
        precs = []
        for trial in range(20):
            idx = rng.choice(len(frag), N, replace=False)
            sub = frag.iloc[idx].reset_index(drop=True)
            log_e = cal_beta_em(sub['fragmentation_rate'].to_numpy())
            r = evaluate(log_e, sub, name=f'subN_{N}')
            aps.append(r['AP'] if r['AP'] is not None and not np.isnan(r['AP']) else 0)
            ks.append(r['k_rejected'])
            precs.append(r['precision_astro'])
        rows.append({'calibration': f'Beta-EM, N={N} subsample (20 trials)', 'AP': np.mean(aps), 'k_rejected': int(np.mean(ks)), 'precision_astro': np.mean(precs), 'recall_astro': float('nan')})

    df = pd.DataFrame(rows)
    print()
    print(df.to_string(index=False))
    df.to_csv(RES / 'ablation_calibration.csv', index=False)
    print(f'\nwrote {RES}/ablation_calibration.csv')


if __name__ == '__main__':
    main()
