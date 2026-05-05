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
    print('=== Confusion analysis on the rejection set at γ_fine = 0.97 ===\n')

    print('[1] Computing fragmentation at γ_fine = 0.97...')
    coarse = pq.read_table(PROC / 'clusters_leiden_r0.9.parquet').to_pandas()
    fine = pq.read_table(PROC / 'clusters_leiden_r0.97.parquet').to_pandas()
    base_labels = pd.read_csv(RES / 'fragmentation_scores.csv')[['cluster_id', 'y_astro', 'y_adv']]
    frag = fragmentation_at(coarse, fine).merge(base_labels, on='cluster_id', how='inner')

    print('[2] Loading expanded coordination ground truth...')
    expanded = pd.read_csv(RES / 'expanded_coordination_groundtruth.csv')
    cluster_truth = expanded.groupby('cluster_id').agg(
        stance=('stance', lambda s: s.mode().iloc[0] if len(s.mode()) else 'unclassified'),
        top_label=('top_label', lambda s: s.dropna().iloc[0] if s.dropna().size else None),
        frac_astroturf=('frac_astroturf', 'first'),
        frac_advocacy=('frac_advocacy', 'first'),
        attributed=('attributed', 'any'),
        expanded_coord=('expanded_coord', 'any'),
    ).reset_index()
    print(f'  expanded GT covers {len(cluster_truth):,} clusters')

    frag = frag.merge(cluster_truth, on='cluster_id', how='left')
    frag['stance'] = frag['stance'].fillna('unclassified')
    frag['expanded_coord'] = frag['expanded_coord'].fillna(False)

    print('[3] Fitting Beta mixture EM, computing rejection set...')
    f = frag['fragmentation_rate'].to_numpy()
    a0, b0, a1, b1 = fit_beta_2mix_em(f)
    f_clipped = np.clip(f, 1e-6, 1 - 1e-6)
    log_e = beta_dist.logpdf(f_clipped, a1, b1) - beta_dist.logpdf(f_clipped, a0, b0)
    e = np.exp(np.clip(log_e, -700, 700))
    order = np.argsort(-e)
    K = len(frag)
    threshold = K / (0.10 * np.arange(1, K + 1))
    rej_idx = np.where(e[order] >= threshold)[0]
    k_hat = int(rej_idx.max() + 1) if rej_idx.size else 0
    rej = frag.iloc[order[:k_hat]].copy()
    print(f'  rejected k = {k_hat:,}')

    print()
    print('=== A. Overall rejection set composition ===')
    n = len(rej)
    n_astro = int(rej['y_astro'].sum())
    n_adv = int(rej['y_adv'].sum())
    n_attributed = int(rej['attributed'].fillna(False).sum())
    n_expanded = int(rej['expanded_coord'].sum())
    n_neither = n - n_attributed - int((~rej['attributed'].fillna(False) & rej['expanded_coord']).sum())
    print(f'  rejected: {n:,}')
    print(f'  NYAG paid astroturf  (y_astro=1):  {n_astro:>5,}  ({100*n_astro/n:.1f}%)')
    print(f'  NYAG advocacy        (y_adv=1):    {n_adv:>5,}  ({100*n_adv/n:.1f}%)')
    print(f'  Any NYAG-attributed (FOIA):       {n_attributed:>5,}  ({100*n_attributed/n:.1f}%)')
    print(f'  Expanded coord (FOIA OR pro/anti): {n_expanded:>5,}  ({100*n_expanded/n:.1f}%)')
    print(f'  NEITHER (true unexplained):        {n - n_expanded:>5,}  ({100*(n-n_expanded)/n:.1f}%)')

    print()
    print('=== B. Stance breakdown of the 13.2% non-astroturf rejected ===')
    nonastro = rej[rej['y_astro'] == 0].copy()
    print(f'  total non-astroturf in rejection: {len(nonastro):,}')
    stance_counts = nonastro['stance'].value_counts()
    for st, cnt in stance_counts.items():
        print(f'    stance={st}: {cnt:>5,}  ({100*cnt/len(nonastro):.1f}%)')

    print()
    print('=== C. Are the "false positives" actually coordinated? ===')
    expanded_in_nonastro = nonastro['expanded_coord'].sum()
    print(f'  non-astroturf rejected with expanded_coord=True: {expanded_in_nonastro:,} of {len(nonastro):,} ({100*expanded_in_nonastro/max(len(nonastro),1):.1f}%)')

    print()
    print('=== D. Headline-style precision under expanded coordination GT ===')
    n_expanded_in_rej = int(rej['expanded_coord'].sum())
    expanded_precision = n_expanded_in_rej / n
    print(f'  Precision (NYAG paid-astroturf): {n_astro/n:.3f} ({100*n_astro/n:.1f}%)')
    print(f'  Precision (any NYAG attribution): {n_attributed/n:.3f} ({100*n_attributed/n:.1f}%)')
    print(f'  Precision (expanded coord, FOIA + keyword classifier): {expanded_precision:.3f} ({100*expanded_precision:.1f}%)')

    print()
    print('=== E. Spot-check examples of "neither" (~unexplained) ===')
    unexplained = rej[~rej['expanded_coord']].head(20)
    if 'top_label' in unexplained.columns:
        sample_labels = unexplained[['cluster_id', 'n', 'fragmentation_rate', 'top_label']].head(15)
        print(sample_labels.to_string(index=False))

    print()
    print('=== F. Recall on expanded GT ===')
    n_expanded_total = int(frag['expanded_coord'].sum())
    expanded_recall = n_expanded_in_rej / max(n_expanded_total, 1)
    print(f'  expanded coord total in candidate set: {n_expanded_total:,}')
    print(f'  recall on expanded coord: {expanded_recall:.3f} ({100*expanded_recall:.1f}%)')

    out = pd.DataFrame([{
        'k_rejected': k_hat,
        'precision_y_astro': n_astro / n,
        'precision_attributed': n_attributed / n,
        'precision_expanded_coord': expanded_precision,
        'recall_y_astro': n_astro / max(int(frag['y_astro'].sum()), 1),
        'recall_expanded_coord': expanded_recall,
        'unexplained_n': n - n_expanded_in_rej,
        'unexplained_pct': (n - n_expanded_in_rej) / n,
    }])
    out_path = RES / 'confusion_analysis_at_097.csv'
    out.to_csv(out_path, index=False)
    print(f'\nwrote {out_path}')


if __name__ == '__main__':
    main()
