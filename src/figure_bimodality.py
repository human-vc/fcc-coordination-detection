from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import beta as beta_dist
from sklearn.mixture import GaussianMixture

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / 'data' / 'processed'
RES = ROOT / 'results'
PAPER_FIG = ROOT / 'paper' / 'figures'
PAPER_FIG.mkdir(parents=True, exist_ok=True)


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
    print('=== Bimodality + γ_fine sweep figure ===\n')
    coarse = pq.read_table(PROC / 'clusters_leiden_r0.9.parquet').to_pandas()
    fine = pq.read_table(PROC / 'clusters_leiden_r0.97.parquet').to_pandas()
    base_labels = pd.read_csv(RES / 'fragmentation_scores.csv')[['cluster_id', 'y_astro', 'y_adv']]
    frag = fragmentation_at(coarse, fine).merge(base_labels, on='cluster_id', how='inner')

    f = frag['fragmentation_rate'].to_numpy()
    a0, b0, a1, b1, w0 = fit_beta_2mix_em(f)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))

    ax = axes[0]
    bins = np.linspace(0, 1, 41)
    is_astro = frag['y_astro'].values.astype(bool)
    ax.hist(f[~is_astro], bins=bins, alpha=0.6, label=f'NYAG non-astroturf ({(~is_astro).sum():,})', color='#4477AA', density=True)
    ax.hist(f[is_astro], bins=bins, alpha=0.6, label=f'NYAG paid astroturf ({is_astro.sum():,})', color='#CC3311', density=True)
    xs = np.linspace(0.005, 0.995, 200)
    g0 = w0 * beta_dist.pdf(xs, a0, b0)
    g1 = (1 - w0) * beta_dist.pdf(xs, a1, b1)
    ax.plot(xs, g0, 'k--', lw=1.5, label=f'$g_0$ (verbatim) Beta({a0:.2f},{b0:.2f})')
    ax.plot(xs, g1, 'k-',  lw=1.5, label=f'$g_1$ (paraphrase) Beta({a1:.2f},{b1:.2f})')
    ax.set_xlabel('cluster fragmentation rate $f_c$')
    ax.set_ylabel('density')
    ax.set_title('Fragmentation distribution\n(γ_coarse=0.90, γ_fine=0.97)')
    ax.legend(loc='upper center', fontsize=8)
    ax.set_xlim(0, 1)

    ax = axes[1]
    sweep = pd.read_csv(RES / 'sensitivity_gamma_fine.csv')
    if 'AP' in sweep.columns:
        ax.plot(sweep['gamma_fine'], sweep['AP'], 'o-', lw=2, markersize=8, color='#CC3311', label='Fragmentation AP')
    ax.axhline(0.386, ls=':', color='gray', lw=1, label='Identity baseline (base rate 0.386)')
    ax.axhline(0.946, ls=':', color='black', lw=1, label='Supervised oracle 0.946')
    ax.axhline(0.354, ls='--', color='#4477AA', lw=1, label='Best concentration baseline (split-LRT 0.354)')
    ax.set_xlabel(r'$\gamma_{\mathrm{fine}}$')
    ax.set_ylabel('AP against NYAG')
    ax.set_title(r'AP vs $\gamma_{\mathrm{fine}}$ (peak at 0.97)')
    ax.legend(fontsize=8, loc='lower right')
    ax.set_ylim(0.1, 1.0)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = PAPER_FIG / 'bimodality_and_gamma_sweep.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    fig.savefig(str(out_path).replace('.png', '.pdf'), bbox_inches='tight')
    print(f'wrote {out_path}')


if __name__ == '__main__':
    main()
