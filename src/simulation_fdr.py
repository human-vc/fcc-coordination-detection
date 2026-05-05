from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import beta as beta_dist
from sklearn.mixture import GaussianMixture

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / 'results'
PAPER_FIG = ROOT / 'paper' / 'figures'
PAPER_FIG.mkdir(parents=True, exist_ok=True)


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
    return (a0, b0, a1, b1, w0)


def ebh(e, alpha):
    K = len(e)
    order = np.argsort(-e)
    e_sorted = e[order]
    threshold = K / (alpha * np.arange(1, K + 1))
    rej_idx = np.where(e_sorted >= threshold)[0]
    k = int(rej_idx.max() + 1) if rej_idx.size else 0
    return order[:k]


def simulate_one(*, K, pi0, alpha, a0, b0, a1, b1, dep='independent', rho=0.0, rng):
    is_null = rng.random(K) < pi0
    n_null = int(is_null.sum())
    n_alt = K - n_null
    f = np.zeros(K)
    if dep == 'independent':
        if n_null > 0:
            f[is_null] = beta_dist.rvs(a0, b0, size=n_null, random_state=rng)
        if n_alt > 0:
            f[~is_null] = beta_dist.rvs(a1, b1, size=n_alt, random_state=rng)
    elif dep == 'positive':
        u_indep = rng.random(K)
        u_shared = rng.random()
        u_eff = (1 - rho) * u_indep + rho * u_shared
        if n_null > 0:
            f[is_null] = beta_dist.ppf(u_eff[is_null], a0, b0)
        if n_alt > 0:
            f[~is_null] = beta_dist.ppf(u_eff[~is_null], a1, b1)
    f = np.clip(f, 1e-6, 1 - 1e-6)
    a0_hat, b0_hat, a1_hat, b1_hat, w0_hat = fit_beta_2mix_em(f)
    log_e = beta_dist.logpdf(f, a1_hat, b1_hat) - beta_dist.logpdf(f, a0_hat, b0_hat)
    e = np.exp(np.clip(log_e, -700, 700))
    rej = ebh(e, alpha)
    if len(rej) == 0:
        return 0.0, 0.0
    fdp = float(is_null[rej].sum() / len(rej))
    power = float((~is_null[rej]).sum() / max(n_alt, 1))
    return fdp, power


def run_grid(*, n_reps=200, K_values=(2000, 5000, 15000), pi0_values=(0.5, 0.7, 0.9, 0.95), alpha_values=(0.05, 0.10, 0.15, 0.20), dep='independent', rho=0.0, base_seed=0):
    a0, b0 = 1.04, 1.37
    a1, b1 = 0.87, 0.10
    rows = []
    for K in K_values:
        for pi0 in pi0_values:
            for alpha in alpha_values:
                fdps = []
                powers = []
                for r in range(n_reps):
                    rng = np.random.default_rng(base_seed * 100000 + hash((K, pi0, alpha, r)) % 1_000_000)
                    fdp, pwr = simulate_one(K=K, pi0=pi0, alpha=alpha, a0=a0, b0=b0, a1=a1, b1=b1, dep=dep, rho=rho, rng=rng)
                    fdps.append(fdp)
                    powers.append(pwr)
                fdps = np.array(fdps)
                powers = np.array(powers)
                rows.append({
                    'K': K, 'pi0': pi0, 'alpha': alpha, 'dep': dep, 'rho': rho,
                    'fdr_mean': float(fdps.mean()),
                    'fdr_se': float(fdps.std(ddof=1) / np.sqrt(n_reps)),
                    'fdr_lo': float(np.quantile(fdps, 0.025)),
                    'fdr_hi': float(np.quantile(fdps, 0.975)),
                    'power_mean': float(powers.mean()),
                    'power_se': float(powers.std(ddof=1) / np.sqrt(n_reps)),
                })
                print(f'  K={K}, pi0={pi0}, alpha={alpha}, dep={dep}: FDR={fdps.mean():.4f} [{np.quantile(fdps, 0.025):.4f}, {np.quantile(fdps, 0.975):.4f}], power={powers.mean():.3f}')
    return pd.DataFrame(rows)


def plot(df_indep, df_pos, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))

    ax = axes[0]
    K_focus = 15000
    sub = df_indep[df_indep['K'] == K_focus]
    pi_colors = {0.5: '#4477AA', 0.7: '#117733', 0.9: '#DDCC77', 0.95: '#CC3311'}
    for pi0, col in pi_colors.items():
        s = sub[sub['pi0'] == pi0].sort_values('alpha')
        if len(s) == 0:
            continue
        ax.fill_between(s['alpha'], s['fdr_lo'], s['fdr_hi'], color=col, alpha=0.15)
        ax.plot(s['alpha'], s['fdr_mean'], 'o-', color=col, lw=1.5, label=f'$\\pi_0={pi0}$')
    ax.plot([0.0, 0.25], [0.0, 0.25], 'k--', lw=1.5, alpha=0.7, label='nominal $\\alpha$')
    ax.set_xlabel(r'nominal level $\alpha$')
    ax.set_ylabel('realized FDR (200 reps)')
    ax.set_title(f'FDR control under independence ($K={K_focus}$)')
    ax.legend(fontsize=8)
    ax.set_xlim(0, 0.22)
    ax.set_ylim(0, 0.22)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    K_focus = 15000
    pi0_focus = 0.7
    rho_palette = ['#4477AA', '#117733', '#DDCC77', '#CC3311']
    for i, rho in enumerate(sorted(df_pos['rho'].unique())):
        s = df_pos[(df_pos['K'] == K_focus) & (df_pos['pi0'] == pi0_focus) & (df_pos['rho'] == rho)].sort_values('alpha')
        if len(s) == 0:
            continue
        col = rho_palette[i % len(rho_palette)]
        ax.fill_between(s['alpha'], s['fdr_lo'], s['fdr_hi'], color=col, alpha=0.15)
        ax.plot(s['alpha'], s['fdr_mean'], 'o-', color=col, lw=1.5, label=f'$\\rho={rho:.1f}$')
    ax.plot([0.0, 0.25], [0.0, 0.25], 'k--', lw=1.5, alpha=0.7, label='nominal $\\alpha$')
    ax.set_xlabel(r'nominal level $\alpha$')
    ax.set_ylabel('realized FDR (200 reps)')
    ax.set_title(f'FDR under positive dependence ($K=15000, \\pi_0=0.7$)')
    ax.legend(fontsize=8)
    ax.set_xlim(0, 0.22)
    ax.set_ylim(0, 0.22)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    fig.savefig(str(out_path).replace('.png', '.pdf'), bbox_inches='tight')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n-reps', type=int, default=200)
    args = p.parse_args()

    print('=== A. FDR verification simulation ===')
    print(f'  n_reps = {args.n_reps}')
    print()
    print('--- Independent nulls ---')
    df_indep = run_grid(n_reps=args.n_reps, K_values=(2000, 5000, 15000),
                         pi0_values=(0.5, 0.7, 0.9, 0.95),
                         alpha_values=(0.05, 0.10, 0.15, 0.20),
                         dep='independent', base_seed=1)
    df_indep.to_csv(RES / 'simulation_fdr_indep.csv', index=False)
    print()
    print('--- Positive dependence (shared latent) ---')
    df_pos_list = []
    for rho in (0.1, 0.3, 0.5, 0.7):
        d = run_grid(n_reps=args.n_reps, K_values=(15000,),
                      pi0_values=(0.7,),
                      alpha_values=(0.05, 0.10, 0.15, 0.20),
                      dep='positive', rho=rho, base_seed=int(2 + 10 * rho))
        df_pos_list.append(d)
    df_pos = pd.concat(df_pos_list, ignore_index=True)
    df_pos.to_csv(RES / 'simulation_fdr_dep.csv', index=False)

    print()
    print('--- Plotting ---')
    plot(df_indep, df_pos, PAPER_FIG / 'simulation_fdr.png')
    print(f'wrote {PAPER_FIG}/simulation_fdr.png')

    print()
    print('=== Summary ===')
    print('Independent: max realized FDR per α (across all π_0, K):')
    summ = df_indep.groupby('alpha')['fdr_mean'].agg(['max', 'mean']).reset_index()
    print(summ.to_string(index=False))


if __name__ == '__main__':
    main()
