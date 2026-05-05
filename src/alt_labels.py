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


def evaluate(log_e, labels_col, alpha=0.10):
    e = np.exp(np.clip(log_e, -700, 700))
    K = len(e)
    order = np.argsort(-e)
    threshold = K / (alpha * np.arange(1, K + 1))
    rej_idx = np.where(e[order] >= threshold)[0]
    k_hat = int(rej_idx.max() + 1) if rej_idx.size else 0
    if k_hat > 0:
        sel = labels_col.iloc[order[:k_hat]]
        prec = float(sel.mean())
        rec = float(sel.sum() / max(int(labels_col.sum()), 1))
    else:
        prec = rec = 0.0
    ap = average_precision_score(labels_col, log_e) if labels_col.sum() > 0 else float('nan')
    return ap, k_hat, prec, rec


def main():
    print('=== B. Alternative-label evaluation ===\n')
    coarse = pq.read_table(PROC / 'clusters_leiden_r0.9.parquet').to_pandas()
    fine = pq.read_table(PROC / 'clusters_leiden_r0.97.parquet').to_pandas()
    base_labels = pd.read_csv(RES / 'fragmentation_scores.csv')[['cluster_id', 'y_astro', 'y_adv']]
    frag = fragmentation_at(coarse, fine).merge(base_labels, on='cluster_id', how='inner')

    expanded = pd.read_csv(RES / 'expanded_coordination_groundtruth.csv')
    truth = expanded.groupby('cluster_id').agg(
        stance=('stance', lambda s: s.mode().iloc[0] if len(s.mode()) else 'unclassified'),
        attributed=('attributed', 'any'),
        expanded_coord=('expanded_coord', 'any'),
    ).reset_index()
    frag = frag.merge(truth, on='cluster_id', how='left')
    frag['stance'] = frag['stance'].fillna('unclassified')
    frag['expanded_coord'] = frag['expanded_coord'].fillna(False).astype(int)
    frag['attributed'] = frag['attributed'].fillna(False).astype(int)

    frag['y_keyword_only'] = (frag['stance'] != 'unclassified').astype(int)
    frag['y_pro_or_anti_NN'] = ((frag['stance'] == 'pro-NN') | (frag['stance'] == 'anti-NN')).astype(int)
    frag['y_advocacy_or_astro'] = ((frag['y_astro'] == 1) | (frag['y_adv'] == 1)).astype(int)

    f = frag['fragmentation_rate'].to_numpy()
    a0, b0, a1, b1, w0 = fit_beta_2mix_em(f)
    log_e = beta_dist.logpdf(np.clip(f, 1e-6, 1 - 1e-6), a1, b1) - beta_dist.logpdf(np.clip(f, 1e-6, 1 - 1e-6), a0, b0)

    rows = []
    for col, name in [
        ('y_astro', 'NYAG paid astroturf (primary)'),
        ('attributed', 'Any NYAG-attributed (FOIA paid + advocacy)'),
        ('y_keyword_only', 'Keyword classifier ONLY (no FOIA): pro/anti/mixed'),
        ('y_pro_or_anti_NN', 'Keyword: pro-NN OR anti-NN coordination'),
        ('y_advocacy_or_astro', 'NYAG paid OR NYAG advocacy'),
        ('expanded_coord', 'Expanded coordination (FOIA + keyword)'),
    ]:
        n_pos = int(frag[col].sum())
        ap, k, p, r = evaluate(log_e, frag[col])
        rows.append({'label_source': name, 'n_pos': n_pos, 'base_rate': n_pos / len(frag), 'AP': ap, 'k_rejected': k, 'precision': p, 'recall': r})

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    df.to_csv(RES / 'alt_labels.csv', index=False)
    print(f'\nwrote {RES}/alt_labels.csv')


if __name__ == '__main__':
    main()
