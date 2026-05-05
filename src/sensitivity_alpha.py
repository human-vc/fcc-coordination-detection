from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist
from sklearn.mixture import GaussianMixture

ROOT = Path(__file__).resolve().parents[1]
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
    return (a0, b0, a1, b1)


def main():
    frag = pd.read_csv(RES / 'fragmentation_scores.csv')
    f = frag['fragmentation_rate'].to_numpy()
    a0, b0, a1, b1 = fit_beta_2mix_em(f)
    f_clipped = np.clip(f, 1e-6, 1 - 1e-6)
    log_e = beta_dist.logpdf(f_clipped, a1, b1) - beta_dist.logpdf(f_clipped, a0, b0)
    e = np.exp(np.clip(log_e, -700, 700))
    order = np.argsort(-e)
    e_sorted = e[order]
    K = len(frag)
    base_rate = frag['y_astro'].mean()

    rows = []
    for alpha in [0.01, 0.05, 0.10, 0.20]:
        threshold = K / (alpha * np.arange(1, K + 1))
        rej_idx = np.where(e_sorted >= threshold)[0]
        k_hat = int(rej_idx.max() + 1) if rej_idx.size else 0
        if k_hat > 0:
            mask = frag.iloc[order[:k_hat]]
            astro_pct = float(mask['y_astro'].mean())
            adv_pct = float(mask['y_adv'].mean())
            astro_recall = mask['y_astro'].sum() / max(int(frag['y_astro'].sum()), 1)
        else:
            astro_pct = adv_pct = astro_recall = 0.0
        rows.append({
            'alpha': alpha,
            'k_rejected': k_hat,
            'precision_astro': astro_pct,
            'precision_adv': adv_pct,
            'recall_astro': astro_recall,
            'pct_of_K': k_hat / K,
        })

    out = pd.DataFrame(rows)
    out.to_csv(RES / 'sensitivity_alpha.csv', index=False)
    print(out.to_string(index=False))
    print(f'\nbase rate (NYAG y_astro on size≥8): {base_rate:.3f}')
    print(f'wrote {RES}/sensitivity_alpha.csv')


if __name__ == '__main__':
    main()
