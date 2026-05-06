"""Triple-fold compound e-value: mode-agnostic, hyperparameter-free.

Construction
============
We compose three layers of compound e-values via Vovk-Wang arithmetic mean:

    1. For each γ_fine ℓ ∈ {γ_1, ..., γ_L}, a paraphrase compound e-value:
           E_para_c(γ_ℓ) = g_1^para_ℓ(f_c(γ_ℓ)) / g_0^para_ℓ(f_c(γ_ℓ))
       where f_c(γ_ℓ) is the fragmentation rate at γ_ℓ and (g_0, g_1) are
       fit by 2-component Beta-mixture EM on the empirical f distribution.

    2. A multi-resolution paraphrase e-value (no γ_fine choice):
           E_para_c = (1/L) Σ_ℓ E_para_c(γ_ℓ)                    (Vovk-Wang AM)

    3. A verbatim compound e-value on τ_c = max_template_size_c / |c|:
           E_verb_c = g_1^verb(τ_c) / g_0^verb(τ_c)
       with (g_0^verb, g_1^verb) fit by 2-component Beta-mixture EM on τ.

    4. The full triple-fold e-value:
           E_full_c = (1/2) (E_para_c + E_verb_c)                (Vovk-Wang AM)

Validity (Vovk & Wang 2021, Theorem 3.2; Wang & Ramdas 2022)
============================================================
Let H_c denote the null for cluster c. Each E_para_c(γ_ℓ) and E_verb_c is
a valid e-value under the IWR §7 mixture-likelihood-ratio identity (Ignatiadis,
Wang & Ramdas 2024, §7), assuming the marginal-mixture and identifiability
assumptions hold for the corresponding test statistic.

By Vovk & Wang (2021, Thm 3.2), if E_1, ..., E_n are valid e-values for the
same null then any convex combination Σ w_i E_i (w_i ≥ 0, Σ w_i = 1) is also
a valid e-value. Applied twice:

    E[E_para_c | H_c]
      = E[(1/L) Σ_ℓ E_para_c(γ_ℓ) | H_c]
      = (1/L) Σ_ℓ E[E_para_c(γ_ℓ) | H_c]                          (linearity)
      ≤ (1/L) Σ_ℓ 1 = 1                                            (each ≤ 1)

    E[E_full_c | H_c]
      = (1/2) (E[E_para_c | H_c] + E[E_verb_c | H_c])
      ≤ (1/2)(1 + 1) = 1.

Hence E_full is a valid compound e-value family, and e-BH at level α on
{E_full_c}_c controls FDR ≤ α under arbitrary dependence (Wang & Ramdas
2022, Thm 1).

Numerical stability
===================
We work in log space throughout:
    log E_para_c = logsumexp_ℓ(log E_para_c(γ_ℓ)) - log L
    log E_full_c = logaddexp(log E_para_c, log E_verb_c) - log 2.

Citations
=========
Vovk, V. & Wang, R. (2021). "E-values: Calibration, combination and
    applications." Annals of Statistics 49(3): 1736-1754.
Wang, R. & Ramdas, A. (2022). "False discovery rate control with e-values."
    JRSS-B 84(3): 822-852.
Ignatiadis, N., Wang, R. & Ramdas, A. (2024). "Asymptotic and compound
    e-values: multiple testing and empirical Bayes." arXiv:2409.19812.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.special import logsumexp
from scipy.stats import beta as beta_dist
from sklearn.metrics import average_precision_score
from sklearn.mixture import GaussianMixture


def fit_beta_2mix_em(f, n_iter: int = 80, tol: float = 1e-5):
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
    return a0, b0, a1, b1, w0


def beta_log_lr(stat: np.ndarray) -> tuple[np.ndarray, dict]:
    a0, b0, a1, b1, w0 = fit_beta_2mix_em(stat)
    s = np.clip(stat, 1e-6, 1 - 1e-6)
    log_e = beta_dist.logpdf(s, a1, b1) - beta_dist.logpdf(s, a0, b0)
    return log_e, {'a0': float(a0), 'b0': float(b0), 'a1': float(a1), 'b1': float(b1),
                   'g0_mean': float(a0 / (a0 + b0)), 'g1_mean': float(a1 / (a1 + b1)),
                   'g0_weight': float(w0)}


def fragmentation_at(coarse, fine, min_size: int):
    coarse_f = coarse[(coarse['cluster_id'] >= 0) & (coarse['cluster_size'] >= min_size)][['row_id', 'cluster_id']].rename(columns={'cluster_id': 'coarse_id'})
    fine_f = fine[fine['cluster_id'] >= 0][['row_id', 'cluster_id']].rename(columns={'cluster_id': 'fine_id'})
    j = coarse_f.merge(fine_f, on='row_id', how='left')
    j['fine_id'] = j['fine_id'].fillna(-1).astype(int)
    g = j.groupby('coarse_id').agg(
        n=('row_id', 'size'),
        n_distinct_fine=('fine_id', lambda s: int((s != -1).sum() and len(set(s) - {-1}))),
    ).reset_index()
    g['fragmentation_rate'] = g['n_distinct_fine'] / g['n'].clip(lower=1)
    return g.rename(columns={'coarse_id': 'cluster_id'})


def cluster_template_signal(coarse, idx, normalized_csv, min_size: int):
    members = coarse[coarse['cluster_id'] >= 0][['row_id', 'cluster_id']]
    if normalized_csv is not None and Path(normalized_csv).exists():
        labels = pd.read_csv(normalized_csv)[['row_id', 'normalized_template_size']]
        labels = labels.rename(columns={'normalized_template_size': 'template_size'})
        members = members.merge(labels, on='row_id', how='left')
    else:
        members = members.merge(idx[['row_id', 'template_size']], on='row_id', how='left')
    members['template_size'] = members['template_size'].fillna(1).astype(int)
    g = members.groupby('cluster_id').agg(
        n=('row_id', 'size'),
        max_template=('template_size', 'max'),
        sum_template=('template_size', 'sum'),
    ).reset_index()
    g = g[g['n'] >= min_size]
    g['verbatim_ratio'] = (g['max_template'] / g['n'].clip(lower=1)).clip(0.0001, 0.9999)
    return g


def ebh(log_e, alpha):
    e = np.exp(np.clip(log_e, -700, 700))
    K = len(e)
    order = np.argsort(-e)
    threshold = K / (alpha * np.arange(1, K + 1))
    rej_idx = np.where(e[order] >= threshold)[0]
    k_hat = int(rej_idx.max() + 1) if rej_idx.size else 0
    return order[:k_hat], k_hat


def evaluate(log_e, label_col, alpha):
    rej, k_hat = ebh(log_e, alpha)
    if k_hat > 0 and label_col.sum() > 0:
        sel = label_col.iloc[rej] if hasattr(label_col, 'iloc') else label_col[rej]
        prec = float(sel.mean()); rec = float(sel.sum() / max(int(label_col.sum()), 1))
    else:
        prec = rec = 0.0
    try:
        ap = average_precision_score(label_col, log_e) if label_col.sum() > 0 else float('nan')
    except Exception:
        ap = float('nan')
    return {'k_rejected': k_hat, 'AP': ap, 'precision': prec, 'recall': rec, 'rejection_rate': k_hat / max(len(log_e), 1)}


def self_test():
    """Validity self-test: empirical FDR of e-BH on the triple-fold should
    track nominal α under a known data-generating process. By Vovk-Wang
    Thm 3.2 + Wang-Ramdas 2022, e-BH on the triple-fold controls FDR ≤ α
    under arbitrary dependence between clusters.

    Test design: 200 reps of (K=2000, π_0=0.7) clusters with
        f_c ~ Beta(2,5) under null, Beta(8,2) under alternative,
        τ_c ~ Beta(2,5) under null, Beta(8,2) under alternative
    Independent across clusters and modes. Run e-BH at α ∈ {0.05, 0.10, 0.20}
    on E_full and report realized FDR."""
    rng = np.random.default_rng(0)
    K = 2000; pi0 = 0.7; n_reps = 200
    a0p, b0p = 2.0, 5.0
    a1p, b1p = 8.0, 2.0
    a0v, b0v = 2.0, 5.0
    a1v, b1v = 8.0, 2.0
    print('  empirical FDR control test:')
    print(f'    K={K}, π_0={pi0}, n_reps={n_reps}, α ∈ [.05, .10, .20]')
    for alpha in [0.05, 0.10, 0.20]:
        fdrs = []
        for r in range(n_reps):
            sub_rng = np.random.default_rng(1_000_000 + r)
            is_null = sub_rng.random(K) < pi0
            f = np.empty(K); tau = np.empty(K)
            f[is_null] = beta_dist.rvs(a0p, b0p, size=is_null.sum(), random_state=sub_rng)
            f[~is_null] = beta_dist.rvs(a1p, b1p, size=(~is_null).sum(), random_state=sub_rng)
            tau[is_null] = beta_dist.rvs(a0v, b0v, size=is_null.sum(), random_state=sub_rng)
            tau[~is_null] = beta_dist.rvs(a1v, b1v, size=(~is_null).sum(), random_state=sub_rng)
            log_e_p, _ = beta_log_lr(f)
            log_e_v, _ = beta_log_lr(tau)
            log_e_full = np.logaddexp(log_e_p, log_e_v) - np.log(2.0)
            rej, k = ebh(log_e_full, alpha)
            if k == 0:
                fdrs.append(0.0)
            else:
                fdrs.append(float(is_null[rej].sum() / k))
        fdrs = np.array(fdrs)
        print(f'    α = {alpha}: realized FDR = {fdrs.mean():.4f} '
              f'(95% CI [{np.quantile(fdrs, 0.025):.4f}, {np.quantile(fdrs, 0.975):.4f}]),  '
              f'{"PASS" if fdrs.mean() <= alpha + 0.02 else "FAIL"}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--proc-dir', type=Path, default=None)
    p.add_argument('--corpus-name', default=None)
    p.add_argument('--gammas', type=str, default='0.95,0.96,0.97,0.98')
    p.add_argument('--min-size', type=int, default=5)
    p.add_argument('--alpha', type=float, default=0.10)
    p.add_argument('--normalized-template-labels-csv', type=Path, default=None)
    p.add_argument('--label-source', type=Path, default=None)
    p.add_argument('--output-dir', type=Path, default=None)
    p.add_argument('--self-test', action='store_true')
    args = p.parse_args()

    if args.self_test:
        print('=== Self-test (Vovk-Wang validity) ===')
        self_test()
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f'=== Triple-fold compound e-value: {args.corpus_name} ===')
    print(f'    γ_list = {args.gammas}, min_size = {args.min_size}, α = {args.alpha}\n')

    coarse = pq.read_table(args.proc_dir / 'clusters_leiden_r0.9.parquet').to_pandas()
    idx = pq.read_table(args.proc_dir / 'embedding_index.parquet').to_pandas()

    gammas = [float(g) for g in args.gammas.split(',')]
    paraphrase_log_es = []
    para_components = []
    used_gammas = []
    base_n_clusters = None
    base_cluster_ids = None
    for gamma in gammas:
        path = args.proc_dir / f'clusters_leiden_r{gamma}.parquet'
        if not path.exists():
            print(f'  γ_fine={gamma}: NO FILE, skip')
            continue
        fine = pq.read_table(path).to_pandas()
        frag = fragmentation_at(coarse, fine, min_size=args.min_size)
        if base_cluster_ids is None:
            base_cluster_ids = frag[['cluster_id', 'n']].copy()
            base_n_clusters = len(frag)
        frag = base_cluster_ids.merge(frag[['cluster_id', 'fragmentation_rate']], on='cluster_id', how='left')
        frag['fragmentation_rate'] = frag['fragmentation_rate'].fillna(0)
        log_e, comp = beta_log_lr(frag['fragmentation_rate'].to_numpy())
        paraphrase_log_es.append(log_e)
        para_components.append({'gamma': gamma, **comp})
        used_gammas.append(gamma)
        print(f'  γ_fine={gamma}: g0={comp["g0_mean"]:.3f}, g1={comp["g1_mean"]:.3f}, w0={comp["g0_weight"]:.3f}')

    L = len(paraphrase_log_es)
    if L == 0:
        raise SystemExit('No fine clustering files found; aborting')
    log_e_para_stack = np.stack(paraphrase_log_es, axis=0)
    log_e_para_multi = logsumexp(log_e_para_stack, axis=0) - np.log(L)

    verbatim = cluster_template_signal(coarse, idx, args.normalized_template_labels_csv, args.min_size)
    df = base_cluster_ids.merge(verbatim[['cluster_id', 'max_template', 'sum_template', 'verbatim_ratio']], on='cluster_id', how='left')
    df['verbatim_ratio'] = df['verbatim_ratio'].fillna(0.0001).clip(0.0001, 0.9999)
    log_e_verb, comp_verb = beta_log_lr(df['verbatim_ratio'].to_numpy())

    log_e_full = np.logaddexp(log_e_para_multi, log_e_verb) - np.log(2.0)
    df['log_e_para_multi'] = log_e_para_multi
    df['log_e_verb'] = log_e_verb
    df['log_e_full'] = log_e_full

    print(f'\n  E_para_multi = (1/{L}) Σ_ℓ E_para(γ_ℓ)  [Vovk-Wang AM across {L} resolutions]')
    print(f'  Verbatim mixture: g0={comp_verb["g0_mean"]:.3f}, g1={comp_verb["g1_mean"]:.3f}, w0={comp_verb["g0_weight"]:.3f}')
    print(f'  E_full = (1/2)(E_para_multi + E_verb)  [Vovk-Wang AM mode-agnostic]\n')

    if args.label_source and args.label_source.exists():
        labels = pd.read_csv(args.label_source)
        cols = [c for c in labels.columns if c.startswith('y_') or c == 'cluster_id']
        df = df.merge(labels[cols].drop_duplicates('cluster_id'), on='cluster_id', how='left').fillna(0)
    if 'max_template' in df.columns:
        df['max_template'] = df['max_template'].fillna(1).astype(int)
        for thr in [5, 50, 500]:
            col = f'y_template_{thr}'
            if col not in df.columns:
                df[col] = (df['max_template'] >= thr).astype(int)

    label_cols = [c for c in df.columns if c.startswith('y_')]
    print(f'  Label columns evaluated: {label_cols}')
    rows = []
    for col in label_cols:
        if col not in df.columns or df[col].sum() == 0:
            continue
        base = float(df[col].astype(int).mean())
        rA = evaluate(log_e_para_multi, df[col].astype(int), args.alpha)
        rV = evaluate(log_e_verb, df[col].astype(int), args.alpha)
        rD = evaluate(log_e_full, df[col].astype(int), args.alpha)
        print(f'\n  {col}: base = {base:.3f}')
        print(f'    E_para_multi: AP={rA["AP"]:.3f}, k={rA["k_rejected"]:,}, prec={rA["precision"]:.3f}, rec={rA["recall"]:.3f}')
        print(f'    E_verb       : AP={rV["AP"]:.3f}, k={rV["k_rejected"]:,}, prec={rV["precision"]:.3f}, rec={rV["recall"]:.3f}')
        print(f'    E_full       : AP={rD["AP"]:.3f}, k={rD["k_rejected"]:,}, prec={rD["precision"]:.3f}, rec={rD["recall"]:.3f}')
        rows.append({'label': col, 'base_rate': base,
                     'AP_para_multi': rA['AP'], 'AP_verb': rV['AP'], 'AP_full': rD['AP'],
                     'k_para_multi': rA['k_rejected'], 'k_verb': rV['k_rejected'], 'k_full': rD['k_rejected'],
                     'prec_para_multi': rA['precision'], 'prec_verb': rV['precision'], 'prec_full': rD['precision'],
                     'rec_para_multi': rA['recall'], 'rec_verb': rV['recall'], 'rec_full': rD['recall']})

    out = pd.DataFrame(rows)
    out.to_csv(args.output_dir / 'triple_fold_evalue.csv', index=False)
    df[['cluster_id', 'n', 'log_e_para_multi', 'log_e_verb', 'log_e_full']].to_csv(
        args.output_dir / 'triple_fold_per_cluster.csv', index=False)
    summary = {'corpus': args.corpus_name, 'L_resolutions': L, 'gammas_used': used_gammas,
               'K': int(len(df)), 'alpha': args.alpha,
               'paraphrase_per_gamma': para_components, 'verbatim': comp_verb,
               'metrics': rows}
    with (args.output_dir / 'triple_fold_summary.json').open('w') as fp:
        json.dump(summary, fp, indent=2)
    print(f'\nwrote {args.output_dir}/triple_fold_evalue.csv')


if __name__ == '__main__':
    main()
