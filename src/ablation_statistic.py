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
    if f.var() < 1e-10:
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
    return (a0, b0, a1, b1, w0)


def fragmentation_at(coarse_df, fine_df, min_size=8):
    coarse = coarse_df[(coarse_df['cluster_id'] >= 0) & (coarse_df['cluster_size'] >= min_size)][['row_id', 'cluster_id']].rename(columns={'cluster_id': 'coarse_id'})
    fine = fine_df[fine_df['cluster_id'] >= 0][['row_id', 'cluster_id']].rename(columns={'cluster_id': 'fine_id'})
    j = coarse.merge(fine, on='row_id', how='left')
    j['fine_id'] = j['fine_id'].fillna(-1).astype(int)
    out = []
    for cid, sub in j.groupby('coarse_id'):
        n = len(sub)
        ids = sub['fine_id'].to_numpy()
        valid = ids[ids != -1]
        if len(valid) == 0:
            out.append({'cluster_id': int(cid), 'n': n, 'frag_rate': 0.0, 'entropy': 0.0, 'gini': 0.0})
            continue
        _, counts = np.unique(valid, return_counts=True)
        p = counts / counts.sum()
        H = float(-(p * np.log(np.maximum(p, 1e-12))).sum())
        if len(p) > 1:
            H_norm = H / np.log(len(p))
        else:
            H_norm = 0.0
        gini = float(1 - (p ** 2).sum())
        out.append({'cluster_id': int(cid), 'n': n, 'frag_rate': len(np.unique(valid)) / n, 'entropy': H_norm, 'gini': gini})
    return pd.DataFrame(out)


def evaluate(stat_values, labels, alpha=0.10, name='stat'):
    f = stat_values
    if f.min() < 0 or f.max() > 1:
        ranks = pd.Series(f).rank(method='average') / (len(f) + 1)
        f = ranks.to_numpy()
    a0, b0, a1, b1, w0 = fit_beta_2mix_em(f)
    f_clip = np.clip(f, 1e-6, 1 - 1e-6)
    log_e = beta_dist.logpdf(f_clip, a1, b1) - beta_dist.logpdf(f_clip, a0, b0)
    e = np.exp(np.clip(log_e, -700, 700))
    K = len(f)
    order = np.argsort(-e)
    threshold = K / (alpha * np.arange(1, K + 1))
    rej_idx = np.where(e[order] >= threshold)[0]
    k_hat = int(rej_idx.max() + 1) if rej_idx.size else 0
    if k_hat > 0:
        sel = labels.iloc[order[:k_hat]]
        prec = float(sel['y_astro'].mean())
        rec = sel['y_astro'].sum() / max(int(labels['y_astro'].sum()), 1)
    else:
        prec = rec = 0.0
    ap = average_precision_score(labels['y_astro'], log_e) if labels['y_astro'].sum() > 0 else float('nan')
    return {
        'statistic': name,
        'AP': ap,
        'k_rejected': k_hat,
        'precision_astro': prec,
        'recall_astro': rec,
        'g0_mean': a0 / (a0 + b0),
        'g1_mean': a1 / (a1 + b1),
    }


def main():
    print('=== D. Statistic ablation: framework with different cluster-level statistics ===\n')
    coarse = pq.read_table(PROC / 'clusters_leiden_r0.9.parquet').to_pandas()
    fine = pq.read_table(PROC / 'clusters_leiden_r0.97.parquet').to_pandas()
    base_labels = pd.read_csv(RES / 'fragmentation_scores.csv')[['cluster_id', 'y_astro', 'y_adv']]
    feats = fragmentation_at(coarse, fine).merge(base_labels, on='cluster_id', how='inner')
    print(f'  K = {len(feats):,} candidate clusters')

    print('\n[1] Loading existing concentration baselines...')
    lrt = pq.read_table(PROC / 'cluster_evalues_lrt_clean.parquet').to_pandas()[['cluster_id', 'kappa_hat', 'log_e']]
    feats = feats.merge(lrt, on='cluster_id', how='left')

    rows = []
    print('\n[2] Each statistic plugged into the same compound-e-value framework (Beta-mixture EM + e-BH):')
    rows.append(evaluate(feats['frag_rate'].to_numpy(), feats, name='fragmentation rate (ours)'))
    rows.append(evaluate(feats['entropy'].to_numpy(), feats, name='entropy of sub-cluster sizes'))
    rows.append(evaluate(feats['gini'].to_numpy(), feats, name='Gini diversity of sub-clusters'))
    if 'kappa_hat' in feats.columns and feats['kappa_hat'].notna().any():
        kf = feats['kappa_hat'].fillna(feats['kappa_hat'].median()).to_numpy()
        rows.append(evaluate(-kf, feats, name='-vMF concentration κ (negated)'))
        rows.append(evaluate(kf, feats, name='+vMF concentration κ (natural)'))
    rows.append(evaluate(-feats['log_e'].fillna(0).to_numpy(), feats, name='-split-LRT vMF (negated)'))

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    df.to_csv(RES / 'ablation_statistic.csv', index=False)
    print(f'\nwrote {RES}/ablation_statistic.csv')


if __name__ == '__main__':
    main()
