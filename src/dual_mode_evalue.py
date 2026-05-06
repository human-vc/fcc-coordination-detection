"""Dual-mode compound e-value: Vovk-Wang composition of paraphrase and
verbatim sub-detectors.

Construction:
    E_para_c = g_1_para(f_c) / g_0_para(f_c)         (fragmentation, paraphrase target)
    E_verb_c = g_1_verb(tau_c) / g_0_verb(tau_c)     (max-template ratio, verbatim target)
    E_dual_c = (E_para_c + E_verb_c) / 2             (Vovk-Wang arithmetic mean)

Validity:
    By Vovk-Wang (2021) Theorem 3.2, the arithmetic mean of valid e-values
    is itself a valid e-value. Both sub-detectors are valid by the IWR §7
    mixture-likelihood-ratio identity (under Assumptions 1-2 of the paper
    applied per sub-detector). e-BH on (E_dual_c)_c controls FDR at level α
    under arbitrary dependence (Wang-Ramdas 2022).

Test statistics:
    f_c     = fragmentation rate at gamma_fine = (sub-clusters) / |c|
              high → paraphrase-coordinated
    tau_c   = max_template_size / |c|
              high → verbatim-coordinated
              (uses normalized templates if --normalized-template-labels-csv given)
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import beta as beta_dist, spearmanr
from sklearn.metrics import average_precision_score
from sklearn.mixture import GaussianMixture


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
    return a0, b0, a1, b1, w0


def beta_log_lr(stat: np.ndarray) -> tuple[np.ndarray, dict]:
    """Beta-mixture EM on stat; return per-cluster log(g_1/g_0) and components."""
    a0, b0, a1, b1, w0 = fit_beta_2mix_em(stat)
    s = np.clip(stat, 1e-6, 1 - 1e-6)
    log_e = beta_dist.logpdf(s, a1, b1) - beta_dist.logpdf(s, a0, b0)
    return log_e, {'a0': a0, 'b0': b0, 'a1': a1, 'b1': b1, 'g0_mean': a0/(a0+b0),
                   'g1_mean': a1/(a1+b1), 'g0_weight': w0}


def fragmentation_at(coarse_df, fine_df, min_size):
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


def cluster_template_signal(coarse_df, idx_df, normalized_labels_csv=None, min_size=5):
    """Compute per-cluster max-template-size ratio τ_c = max_template / |c|.

    Uses normalized_labels_csv if provided (for FCC 14-28 where prefixes
    inflate template_size variance), else falls back to embedding_index.template_size.
    """
    if normalized_labels_csv is not None and Path(normalized_labels_csv).exists():
        labels = pd.read_csv(normalized_labels_csv)[['row_id', 'normalized_template_size']].rename(columns={'normalized_template_size': 'template_size'})
        members = coarse_df[coarse_df['cluster_id'] >= 0][['row_id', 'cluster_id']].merge(labels, on='row_id', how='left')
    else:
        members = coarse_df[coarse_df['cluster_id'] >= 0][['row_id', 'cluster_id']].merge(idx_df[['row_id', 'template_size']], on='row_id', how='left')
    members['template_size'] = members['template_size'].fillna(1).astype(int)
    g = members.groupby('cluster_id').agg(
        n=('row_id', 'size'),
        max_template=('template_size', 'max'),
        sum_template=('template_size', 'sum'),
    ).reset_index()
    g = g[g['n'] >= min_size]
    g['verbatim_ratio'] = g['max_template'] / g['n'].clip(lower=1)
    g['verbatim_ratio'] = g['verbatim_ratio'].clip(0.0001, 0.9999)
    return g


def evaluate(log_e, label_col, alpha):
    e = np.exp(np.clip(log_e, -700, 700))
    K = len(e)
    order = np.argsort(-e)
    threshold = K / (alpha * np.arange(1, K + 1))
    rej_idx = np.where(e[order] >= threshold)[0]
    k_hat = int(rej_idx.max() + 1) if rej_idx.size else 0
    if k_hat > 0 and label_col.sum() > 0:
        sel = label_col.iloc[order[:k_hat]] if hasattr(label_col, 'iloc') else label_col[order[:k_hat]]
        prec = float(sel.mean())
        rec = float(sel.sum() / max(int(label_col.sum()), 1))
    else:
        prec = rec = 0.0
    try:
        ap = average_precision_score(label_col, log_e) if label_col.sum() > 0 else float('nan')
    except Exception:
        ap = float('nan')
    return {'k_rejected': k_hat, 'AP': ap, 'precision': prec, 'recall': rec, 'rejection_rate': k_hat / max(K, 1)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--proc-dir', type=Path, required=True)
    p.add_argument('--corpus-name', required=True)
    p.add_argument('--gamma-fine', type=float, default=0.97)
    p.add_argument('--min-size', type=int, default=5)
    p.add_argument('--alpha', type=float, default=0.10)
    p.add_argument('--normalized-template-labels-csv', type=Path, default=None)
    p.add_argument('--label-source', type=str, default=None,
                   help='optional CSV with per-cluster {y_astro,y_adv,y_expanded,...}')
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f'=== Dual-mode compound e-value: {args.corpus_name} (γ_fine={args.gamma_fine}, min={args.min_size}) ===\n')

    coarse = pq.read_table(args.proc_dir / 'clusters_leiden_r0.9.parquet').to_pandas()
    fine = pq.read_table(args.proc_dir / f'clusters_leiden_r{args.gamma_fine}.parquet').to_pandas()
    idx = pq.read_table(args.proc_dir / 'embedding_index.parquet').to_pandas()

    frag = fragmentation_at(coarse, fine, min_size=args.min_size)
    verbatim = cluster_template_signal(coarse, idx,
                                       normalized_labels_csv=args.normalized_template_labels_csv,
                                       min_size=args.min_size)
    df = frag.merge(verbatim[['cluster_id', 'max_template', 'sum_template', 'verbatim_ratio']], on='cluster_id', how='inner')
    print(f'  K = {len(df):,} candidate clusters (size ≥ {args.min_size})')

    log_e_para, comp_para = beta_log_lr(df['fragmentation_rate'].to_numpy())
    log_e_verb, comp_verb = beta_log_lr(df['verbatim_ratio'].to_numpy())

    log_e_dual = np.logaddexp(log_e_para, log_e_verb) - np.log(2.0)

    df['log_e_para'] = log_e_para
    df['log_e_verb'] = log_e_verb
    df['log_e_dual'] = log_e_dual

    print(f'\n  Paraphrase mixture: g0_mean={comp_para["g0_mean"]:.3f}, g1_mean={comp_para["g1_mean"]:.3f}, w0={comp_para["g0_weight"]:.3f}')
    print(f'  Verbatim   mixture: g0_mean={comp_verb["g0_mean"]:.3f}, g1_mean={comp_verb["g1_mean"]:.3f}, w0={comp_verb["g0_weight"]:.3f}')

    rho_pp, _ = spearmanr(log_e_para, log_e_verb)
    print(f'\n  Spearman ρ(log E_para, log E_verb) = {rho_pp:+.4f} '
          f'(near zero → modes are largely orthogonal; strongly negative → corpus is one-mode-dominated)')

    if args.label_source and args.label_source != 'auto':
        labels = pd.read_csv(args.label_source)
        df = df.merge(labels, on='cluster_id', how='left').fillna(0)
    elif args.label_source == 'auto':
        for cand in ['fragmentation_scores.csv', 'expanded_coordination_groundtruth.csv']:
            for d in [Path('results'), Path('results_fcc14_28'), Path('results_cfpb_2016_0025'), Path('results_epa_2017_0355')]:
                src = d / cand
                if src.exists():
                    sub = pd.read_csv(src)
                    if 'cluster_id' in sub.columns:
                        cols = [c for c in sub.columns if c.startswith('y_') or c == 'cluster_id']
                        df = df.merge(sub[cols].drop_duplicates('cluster_id'), on='cluster_id', how='left').fillna(0)
                        break

    label_cols = [c for c in df.columns if c.startswith('y_')]
    if not label_cols:
        if 'max_template' in df.columns:
            df['y_template_5'] = (df['max_template'] >= 5).astype(int)
            df['y_template_50'] = (df['max_template'] >= 50).astype(int)
            df['y_template_500'] = (df['max_template'] >= 500).astype(int)
            label_cols = ['y_template_5', 'y_template_50', 'y_template_500']

    print(f'\n  Label columns evaluated: {label_cols}')
    rows = []
    for col in label_cols:
        if col not in df.columns or df[col].sum() == 0:
            continue
        base = float(df[col].astype(int).mean())
        rA = evaluate(log_e_para, df[col].astype(int), args.alpha)
        rV = evaluate(log_e_verb, df[col].astype(int), args.alpha)
        rD = evaluate(log_e_dual, df[col].astype(int), args.alpha)
        print(f'\n  {col}: base rate {base:.3f}')
        print(f'    E_para: AP={rA["AP"]:.3f}, k={rA["k_rejected"]:,}, prec={rA["precision"]:.3f}, rec={rA["recall"]:.3f}')
        print(f'    E_verb: AP={rV["AP"]:.3f}, k={rV["k_rejected"]:,}, prec={rV["precision"]:.3f}, rec={rV["recall"]:.3f}')
        print(f'    E_dual: AP={rD["AP"]:.3f}, k={rD["k_rejected"]:,}, prec={rD["precision"]:.3f}, rec={rD["recall"]:.3f}')
        rows.append({'label': col, 'base_rate': base,
                     'AP_para': rA['AP'], 'AP_verb': rV['AP'], 'AP_dual': rD['AP'],
                     'k_para': rA['k_rejected'], 'k_verb': rV['k_rejected'], 'k_dual': rD['k_rejected'],
                     'prec_para': rA['precision'], 'prec_verb': rV['precision'], 'prec_dual': rD['precision'],
                     'rec_para': rA['recall'], 'rec_verb': rV['recall'], 'rec_dual': rD['recall']})

    out = pd.DataFrame(rows)
    out.to_csv(args.output_dir / 'dual_mode_evalue.csv', index=False)
    df[['cluster_id', 'n', 'fragmentation_rate', 'verbatim_ratio', 'log_e_para', 'log_e_verb', 'log_e_dual']].to_csv(
        args.output_dir / 'dual_mode_per_cluster.csv', index=False)
    summary = {'corpus': args.corpus_name, 'K': int(len(df)), 'alpha': args.alpha,
               'paraphrase_mixture': comp_para, 'verbatim_mixture': comp_verb,
               'spearman_para_verb': float(rho_pp), 'metrics': rows}
    with (args.output_dir / 'dual_mode_summary.json').open('w') as fp:
        json.dump(summary, fp, indent=2)
    print(f'\nwrote {args.output_dir}/dual_mode_evalue.csv')


if __name__ == '__main__':
    main()
