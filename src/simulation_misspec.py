"""Misspecification simulation: realized FDR of e-BH on the anchor+cross-fit
construction when (a) the true null density is a 3-component mixture but we
fit a single Beta, (b) anchor clusters are contaminated with a small fraction
of alternatives, (c) the true null and alternative have overlapping support.

The point of this simulation is to address the reviewer concern that the
existing FDR-control plot in the paper samples f_c from the same Beta mixture
used to define E_c — that plot only verifies Wang-Ramdas e-BH validity under
correct specification, not robustness to model misspecification.

Setup
=====
For each scenario, we simulate K = 5000 cluster fragmentation rates f_c with
true null fraction π_0 ∈ {0.5, 0.7, 0.9}, run the anchor + 5-fold cross-fit
pipeline, and report realized FDR averaged over n_reps = 100 replications at
nominal levels α ∈ {0.05, 0.10, 0.15, 0.20}.

Scenarios
=========
S1: correct specification (Beta(2,5) null, Beta(8,2) alternative)
S2: 3-component null (50% Beta(2,5) + 50% Beta(3,8)) fit by single Beta
S3: contaminated null (5% of "anchors" are actually drawn from the alternative)
S4: heavy-tailed null (mixture with a uniform tail)
"""
from __future__ import annotations
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import beta as beta_dist, uniform
from sklearn.model_selection import KFold

EPS = 1e-6


def fit_beta_mom(values):
    v = np.clip(values.astype(np.float64), EPS, 1 - EPS)
    if len(v) < 4 or v.var() < 1e-10:
        return 1.0, 1.0
    mu = float(v.mean()); var = float(v.var())
    c = mu * (1 - mu) / var - 1
    if c <= 0:
        return 1.0, 1.0
    return max(mu * c, 1e-3), max((1 - mu) * c, 1e-3)


def crossfit_log_e(f, anchor, n_folds=5, seed=42):
    n = len(f); log_e = np.zeros(n)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for tr, te in kf.split(np.arange(n)):
        a_tr = anchor[tr]
        if a_tr.sum() < 5 or (~a_tr).sum() < 5:
            continue
        a0, b0 = fit_beta_mom(f[tr][a_tr])
        a1, b1 = fit_beta_mom(f[tr][~a_tr])
        s = np.clip(f[te], EPS, 1 - EPS)
        log_e[te] = beta_dist.logpdf(s, a1, b1) - beta_dist.logpdf(s, a0, b0)
    return log_e


def ebh(log_e, alpha):
    e = np.exp(np.clip(log_e, -700, 700))
    K = len(e); order = np.argsort(-e)
    threshold = K / (alpha * np.arange(1, K + 1))
    rej_idx = np.where(e[order] >= threshold)[0]
    k_hat = int(rej_idx.max() + 1) if rej_idx.size else 0
    return order[:k_hat], k_hat


def sample_S1(rng, K, pi0):
    """Correct specification: null Beta(2,5), alt Beta(8,2)."""
    is_null = rng.random(K) < pi0
    f = np.empty(K)
    f[is_null] = beta_dist.rvs(2, 5, size=is_null.sum(), random_state=rng)
    f[~is_null] = beta_dist.rvs(8, 2, size=(~is_null).sum(), random_state=rng)
    anchor = is_null.copy()
    return f, is_null, anchor


def sample_S2(rng, K, pi0):
    """3-component null: 50% Beta(2,5) + 50% Beta(3,8) — single-Beta fit is misspecified.
    Alt: Beta(8,2). Anchors track the true null."""
    is_null = rng.random(K) < pi0
    f = np.empty(K)
    n_null = is_null.sum()
    sub = rng.random(n_null) < 0.5
    f_null = np.empty(n_null)
    f_null[sub] = beta_dist.rvs(2, 5, size=int(sub.sum()), random_state=rng)
    f_null[~sub] = beta_dist.rvs(3, 8, size=int((~sub).sum()), random_state=rng)
    f[is_null] = f_null
    f[~is_null] = beta_dist.rvs(8, 2, size=(~is_null).sum(), random_state=rng)
    anchor = is_null.copy()
    return f, is_null, anchor


def sample_S3(rng, K, pi0, contam=0.05):
    """Contaminated anchors: 5% of declared anchors are actually drawn from alt.
    Tests robustness to anchor misclassification (e.g., paid astroturf disguised
    as byte-identical form letters)."""
    is_null = rng.random(K) < pi0
    f = np.empty(K)
    f[is_null] = beta_dist.rvs(2, 5, size=is_null.sum(), random_state=rng)
    f[~is_null] = beta_dist.rvs(8, 2, size=(~is_null).sum(), random_state=rng)
    anchor = is_null.copy()
    # Now flip `contam` fraction of anchors to be drawn from alt
    swap = rng.random(K) < contam
    flip = swap & anchor  # only flip anchor=True
    n_flip = int(flip.sum())
    if n_flip > 0:
        f[flip] = beta_dist.rvs(8, 2, size=n_flip, random_state=rng)
    return f, is_null, anchor  # anchor mask unchanged (label noise)


def sample_S4(rng, K, pi0):
    """Heavy-tailed null: 90% Beta(2,5) + 10% Uniform(0,1). Alt: Beta(8,2)."""
    is_null = rng.random(K) < pi0
    f = np.empty(K)
    n_null = is_null.sum()
    sub = rng.random(n_null) < 0.10
    f_null = np.empty(n_null)
    f_null[~sub] = beta_dist.rvs(2, 5, size=int((~sub).sum()), random_state=rng)
    f_null[sub] = uniform.rvs(0, 1, size=int(sub.sum()), random_state=rng)
    f[is_null] = f_null
    f[~is_null] = beta_dist.rvs(8, 2, size=(~is_null).sum(), random_state=rng)
    anchor = is_null.copy()
    return f, is_null, anchor


SCENARIOS = {
    'S1_correct': sample_S1,
    'S2_3comp_null': sample_S2,
    'S3_contam_anchors': sample_S3,
    'S4_heavy_tail_null': sample_S4,
}


def run_scenario(name, sampler, K, pi0, n_reps, alphas, seed=0):
    rows = []
    for r in range(n_reps):
        rng = np.random.default_rng(seed + r * 97)
        f, is_null, anchor = sampler(rng, K, pi0)
        log_e = crossfit_log_e(f, anchor)
        for alpha in alphas:
            rej, k_hat = ebh(log_e, alpha)
            if k_hat == 0:
                fdr = 0.0; tdp = 0.0
            else:
                false_disc = int(is_null[rej].sum())
                fdr = false_disc / k_hat
                tdp = int((~is_null)[rej].sum()) / max(int((~is_null).sum()), 1)
            rows.append({'scenario': name, 'pi0': pi0, 'alpha': alpha, 'rep': r,
                         'k_hat': k_hat, 'fdr': fdr, 'tdp': tdp})
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--K', type=int, default=5000)
    p.add_argument('--n-reps', type=int, default=100)
    p.add_argument('--pi0-list', type=str, default='0.5,0.7,0.9')
    p.add_argument('--alphas', type=str, default='0.05,0.10,0.15,0.20')
    p.add_argument('--output', type=Path, default=Path('results/simulation_misspec.csv'))
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    pi0_list = [float(x) for x in args.pi0_list.split(',')]
    alphas = [float(x) for x in args.alphas.split(',')]

    print(f'=== Misspecification simulation: anchor + 5-fold cross-fit, e-BH ===')
    print(f'  K = {args.K}, n_reps = {args.n_reps}, π_0 ∈ {pi0_list}, α ∈ {alphas}\n')

    all_rows = []
    for name, sampler in SCENARIOS.items():
        for pi0 in pi0_list:
            rows = run_scenario(name, sampler, args.K, pi0, args.n_reps, alphas, seed=args.seed)
            all_rows.extend(rows)
        df = pd.DataFrame([r for r in all_rows if r['scenario'] == name])
        print(f'\nScenario {name}:')
        print(f'  {"π_0":>5} {"α":>6} {"realized FDR (mean)":>22} {"95% CI upper":>14} {"TDP mean":>10} {"PASS":>6}')
        for pi0 in pi0_list:
            for alpha in alphas:
                sub = df[(df['pi0'] == pi0) & (df['alpha'] == alpha)]
                m = sub['fdr'].mean(); ci_u = sub['fdr'].quantile(0.975)
                tdp = sub['tdp'].mean()
                ok = 'PASS' if m <= alpha + 0.02 else 'FAIL'
                print(f'  {pi0:>5} {alpha:>6} {m:>22.4f} {ci_u:>14.4f} {tdp:>10.4f} {ok:>6}')

    pd.DataFrame(all_rows).to_csv(args.output, index=False)
    print(f'\nwrote {args.output}')


if __name__ == '__main__':
    main()
