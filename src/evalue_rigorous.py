"""Rigorous compound e-value with anchor-based identification + cross-fitting.

Replaces the EM-orientation + full-empirical-fit construction in triple_fold_evalue.py
with two improvements that address the two main reviewer concerns:

  (1) Anchor-based identification of the null component (Arora et al. 2012, Donoho-Stodden 2003)
      Instead of fitting a 2-component Beta-mixture EM and assuming the lower-mean component
      is the null (a label-verifiable but circular commitment), we identify the null component
      structurally via anchor clusters:
          - For the paraphrase e-value E^para on f_c (fragmentation rate):
              anchor set A_para = {c : tau_c >= 1 - eps}
              These clusters have byte-identical comments, so they are guaranteed verbatim
              (the structural null). Fit g_0^para directly on {f_c : c in A_para}.
              Fit g_1^para on the complement.
          - For the verbatim e-value E^verb on tau_c (max-template ratio):
              anchor set A_verb = {c : f_c(gamma=0.97) >= 1 - eps}
              These clusters have every comment in its own fine sub-cluster, so they are
              guaranteed paraphrase (the structural null for the verbatim test).
              Fit g_0^verb on {tau_c : c in A_verb}, g_1^verb on complement.
      Each anchor signal identifies the null of the OTHER detector — no circularity.
      Validity inherits from IWR §7 with g_0 deterministic (anchor-pinned).

  (2) Five-fold cross-fitting (Chernozhukov et al. 2018, DML)
      To restore the IWR §7 independence requirement (Theorem 6.3 is asymptotic-only when
      g_0, g_1 fit on the same data they are evaluated on), we cross-fit:
          for each fold k:
              fit (g_0^(-k), g_1^(-k)) on clusters NOT in fold k
              evaluate E_c = g_1^(-k)(stat_c) / g_0^(-k)(stat_c) for c in fold k
          concatenate all E_c, run e-BH on the full vector.
      Conditional on the held-out fold, (g_0^(-k), g_1^(-k)) is a deterministic function of
      independent data, so the IWR §5.2 oracle condition holds in finite samples.

Citations
=========
- Ignatiadis, N., Wang, R. & Ramdas, A. (2024). Asymptotic and compound e-values.
  arXiv:2409.19812. — §5.2 oracle, §6 data-driven, §7 simulations.
- Chernozhukov, V. et al. (2018). Double/debiased ML. arXiv:1608.00060. — K-fold cross-fit.
- Arora, S., Ge, R. & Moitra, A. (2012). Learning topic models — going beyond SVD.
  arXiv:1204.1956. — anchor-based identification.
- Kunkel, D. & Peruggia, M. (2020). Anchored Bayesian Gaussian mixture models. EJS.
- Efron, B. (2008). Microarrays, empirical Bayes and the two-groups model. Statist. Sci. 23(1).
- Vovk, V. & Wang, R. (2021). E-values: calibration, combination and applications. AoS 49(3).
- Wang, R. & Ramdas, A. (2022). False discovery rate control with e-values. JRSS-B 84(3).
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.special import logsumexp
from scipy.stats import beta as beta_dist
from sklearn.metrics import average_precision_score
from sklearn.model_selection import KFold

EPS = 1e-6


def fit_beta_mom(values: np.ndarray) -> tuple[float, float]:
    """Fit Beta(a, b) by method-of-moments on bounded values."""
    v = np.clip(values.astype(np.float64), EPS, 1 - EPS)
    if len(v) < 4 or v.var() < 1e-10:
        return 1.0, 1.0
    mu = float(v.mean())
    var = float(v.var())
    c = mu * (1 - mu) / var - 1
    if c <= 0:
        return 1.0, 1.0
    return max(mu * c, 1e-3), max((1 - mu) * c, 1e-3)


def anchor_evalue_one_fold(stat_train: np.ndarray, anchor_train: np.ndarray,
                           stat_eval: np.ndarray) -> tuple[np.ndarray, dict]:
    """Fit g_0 on anchor∩train, g_1 on non-anchor∩train, evaluate log-LR on eval.

    Returns log E_c = log g_1(stat_c) - log g_0(stat_c) for c in eval set.
    """
    if anchor_train.sum() < 20 or (~anchor_train).sum() < 20:
        return np.zeros(len(stat_eval)), {'a0': 1.0, 'b0': 1.0, 'a1': 1.0, 'b1': 1.0,
                                          'g0_mean': 0.5, 'g1_mean': 0.5,
                                          'n_anchor_train': int(anchor_train.sum()),
                                          'n_nonanchor_train': int((~anchor_train).sum())}
    a0, b0 = fit_beta_mom(stat_train[anchor_train])
    a1, b1 = fit_beta_mom(stat_train[~anchor_train])
    s = np.clip(stat_eval, EPS, 1 - EPS)
    log_e = beta_dist.logpdf(s, a1, b1) - beta_dist.logpdf(s, a0, b0)
    return log_e, {'a0': float(a0), 'b0': float(b0), 'a1': float(a1), 'b1': float(b1),
                   'g0_mean': float(a0 / (a0 + b0)), 'g1_mean': float(a1 / (a1 + b1)),
                   'n_anchor_train': int(anchor_train.sum()),
                   'n_nonanchor_train': int((~anchor_train).sum())}


def crossfit_evalue_anchor(stat: np.ndarray, anchor_mask: np.ndarray,
                           n_folds: int = 5, seed: int = 42) -> tuple[np.ndarray, list[dict]]:
    """5-fold cross-fit. For each fold k, fit (g_0, g_1) on non-fold clusters using
    anchor_mask to identify the null, then evaluate E on fold-k clusters.
    """
    n = len(stat)
    log_e = np.zeros(n, dtype=np.float64)
    fold_info = []
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold_idx, (train_idx, eval_idx) in enumerate(kf.split(np.arange(n))):
        log_e_fold, info = anchor_evalue_one_fold(
            stat_train=stat[train_idx],
            anchor_train=anchor_mask[train_idx],
            stat_eval=stat[eval_idx])
        log_e[eval_idx] = log_e_fold
        info['fold'] = fold_idx
        info['n_eval'] = len(eval_idx)
        fold_info.append(info)
    return log_e, fold_info


def ebh(log_e: np.ndarray, alpha: float) -> tuple[np.ndarray, int]:
    e = np.exp(np.clip(log_e, -700, 700))
    K = len(e)
    order = np.argsort(-e)
    threshold = K / (alpha * np.arange(1, K + 1))
    rej_idx = np.where(e[order] >= threshold)[0]
    k_hat = int(rej_idx.max() + 1) if rej_idx.size else 0
    return order[:k_hat], k_hat


def evaluate(log_e: np.ndarray, label_col, alpha: float) -> dict:
    rej, k_hat = ebh(log_e, alpha)
    if k_hat > 0 and label_col.sum() > 0:
        sel = label_col.iloc[rej] if hasattr(label_col, 'iloc') else label_col[rej]
        prec = float(sel.mean())
        rec = float(sel.sum() / max(int(label_col.sum()), 1))
    else:
        prec = rec = 0.0
    try:
        ap = average_precision_score(label_col, log_e) if label_col.sum() > 0 else float('nan')
    except Exception:
        ap = float('nan')
    return {'k_rejected': k_hat, 'AP': ap, 'precision': prec, 'recall': rec,
            'rejection_rate': k_hat / max(len(log_e), 1)}


def fragmentation_at(coarse: pd.DataFrame, fine: pd.DataFrame, min_size: int) -> pd.DataFrame:
    coarse_f = coarse[(coarse['cluster_id'] >= 0) & (coarse['cluster_size'] >= min_size)][
        ['row_id', 'cluster_id']].rename(columns={'cluster_id': 'coarse_id'})
    fine_f = fine[fine['cluster_id'] >= 0][['row_id', 'cluster_id']].rename(
        columns={'cluster_id': 'fine_id'})
    j = coarse_f.merge(fine_f, on='row_id', how='left')
    j['fine_id'] = j['fine_id'].fillna(-1).astype(int)
    g = j.groupby('coarse_id').agg(
        n=('row_id', 'size'),
        n_distinct_fine=('fine_id', lambda s: int((s != -1).sum() and len(set(s) - {-1}))),
    ).reset_index()
    g['fragmentation_rate'] = g['n_distinct_fine'] / g['n'].clip(lower=1)
    return g.rename(columns={'coarse_id': 'cluster_id'})


def template_signal(coarse: pd.DataFrame, idx: pd.DataFrame, min_size: int) -> pd.DataFrame:
    members = coarse[coarse['cluster_id'] >= 0][['row_id', 'cluster_id']]
    members = members.merge(idx[['row_id', 'template_size']], on='row_id', how='left')
    members['template_size'] = members['template_size'].fillna(1).astype(int)
    g = members.groupby('cluster_id').agg(
        n=('row_id', 'size'),
        max_template=('template_size', 'max'),
        sum_template=('template_size', 'sum'),
    ).reset_index()
    g = g[g['n'] >= min_size]
    g['tau'] = (g['max_template'] / g['n'].clip(lower=1)).clip(EPS, 1 - EPS)
    return g


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--proc-dir', type=Path, default=Path('data/processed'))
    p.add_argument('--corpus-name', default='FCC 17-108')
    p.add_argument('--gammas', type=str, default='0.95,0.96,0.97,0.98')
    p.add_argument('--gamma-anchor', type=float, default=0.97,
                   help='Fine resolution used to define f-anchors for the verbatim e-value')
    p.add_argument('--anchor-tau-thr', type=float, default=0.999,
                   help='Anchor threshold on tau (verbatim signal). tau >= 1 means byte-identical.')
    p.add_argument('--anchor-f-thr', type=float, default=0.95,
                   help='Anchor threshold on f at gamma_anchor (paraphrase signal). f near 1 means each comment in own fine sub-cluster.')
    p.add_argument('--min-size', type=int, default=8)
    p.add_argument('--alpha', type=float, default=0.10)
    p.add_argument('--n-folds', type=int, default=5)
    p.add_argument('--label-source', type=Path, default=None)
    p.add_argument('--output-dir', type=Path, default=Path('results'))
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f'=== Rigorous compound e-value (anchor + cross-fit): {args.corpus_name} ===')
    print(f'  γ_list = {args.gammas}, γ_anchor = {args.gamma_anchor}, min_size = {args.min_size}')
    print(f'  anchor thresholds: τ ≥ {args.anchor_tau_thr}, f ≥ {args.anchor_f_thr}')
    print(f'  cross-fit folds = {args.n_folds}, α = {args.alpha}\n')

    coarse = pq.read_table(args.proc_dir / 'clusters_leiden_r0.9.parquet').to_pandas()
    idx = pq.read_table(args.proc_dir / 'embedding_index.parquet').to_pandas()

    # Verbatim signal τ_c (independent of γ_fine)
    tau_df = template_signal(coarse, idx, args.min_size)

    # Fragmentation at each γ_fine
    gammas = [float(g) for g in args.gammas.split(',')]
    frag_by_gamma: dict[float, pd.DataFrame] = {}
    for gamma in gammas:
        path = args.proc_dir / f'clusters_leiden_r{gamma}.parquet'
        if not path.exists():
            print(f'  γ_fine={gamma}: NO FILE, skip')
            continue
        fine = pq.read_table(path).to_pandas()
        frag = fragmentation_at(coarse, fine, min_size=args.min_size)
        frag_by_gamma[gamma] = frag
        print(f'  γ_fine={gamma}: K = {len(frag):,} clusters')

    if not frag_by_gamma:
        raise SystemExit('No fine clustering files; aborting')

    # Build a base table on the intersection of cluster_id sets (use γ_anchor as base)
    if args.gamma_anchor not in frag_by_gamma:
        args.gamma_anchor = sorted(frag_by_gamma.keys())[len(frag_by_gamma) // 2]
        print(f'  γ_anchor not available; falling back to γ_anchor = {args.gamma_anchor}')

    base = frag_by_gamma[args.gamma_anchor][['cluster_id', 'n']].copy()
    base = base.merge(tau_df[['cluster_id', 'max_template', 'tau']], on='cluster_id', how='left')
    base['tau'] = base['tau'].fillna(EPS).clip(EPS, 1 - EPS)
    base['max_template'] = base['max_template'].fillna(1).astype(int)
    base = base.merge(frag_by_gamma[args.gamma_anchor][['cluster_id', 'fragmentation_rate']].rename(
        columns={'fragmentation_rate': 'f_anchor'}), on='cluster_id', how='left')
    base['f_anchor'] = base['f_anchor'].fillna(0).clip(EPS, 1 - EPS)

    K = len(base)
    print(f'\n  K = {K:,} candidate clusters (size ≥ {args.min_size})')

    # Anchor masks
    tau_anchor = (base['tau'] >= args.anchor_tau_thr).to_numpy()
    f_anchor = (base['f_anchor'] >= args.anchor_f_thr).to_numpy()
    print(f'  τ-anchors (verbatim, byte-identical):       {tau_anchor.sum():,} ({100*tau_anchor.mean():.1f}%)')
    print(f'  f-anchors (paraphrase, every comment own fine sub-cluster): {f_anchor.sum():,} ({100*f_anchor.mean():.1f}%)')

    # Paraphrase e-value at each γ_fine: anchor = tau ≥ τ_thr, cross-fit
    print(f'\n--- Paraphrase e-value (anchor = τ-anchors, cross-fit {args.n_folds}-fold) ---')
    para_log_es: dict[float, np.ndarray] = {}
    para_components: list[dict] = []
    for gamma in sorted(frag_by_gamma.keys()):
        f_g = base[['cluster_id']].merge(
            frag_by_gamma[gamma][['cluster_id', 'fragmentation_rate']],
            on='cluster_id', how='left')
        f_vec = f_g['fragmentation_rate'].fillna(0).clip(EPS, 1 - EPS).to_numpy()
        log_e, fold_info = crossfit_evalue_anchor(f_vec, tau_anchor, n_folds=args.n_folds)
        para_log_es[gamma] = log_e
        g0_means = [fi['g0_mean'] for fi in fold_info]
        g1_means = [fi['g1_mean'] for fi in fold_info]
        print(f'  γ={gamma}: g0_mean = {np.mean(g0_means):.3f} (avg across folds), '
              f'g1_mean = {np.mean(g1_means):.3f}')
        para_components.append({'gamma': gamma, 'g0_mean_avg': float(np.mean(g0_means)),
                                'g1_mean_avg': float(np.mean(g1_means)),
                                'fold_info': fold_info})

    # Multi-resolution paraphrase e-value (Vovk-Wang AM across γ_fine)
    log_e_para_stack = np.stack(list(para_log_es.values()), axis=0)
    log_e_para_multi = logsumexp(log_e_para_stack, axis=0) - np.log(len(para_log_es))

    # Verbatim e-value: anchor = f-anchors, cross-fit
    print(f'\n--- Verbatim e-value (anchor = f-anchors at γ={args.gamma_anchor}, cross-fit) ---')
    log_e_verb, verb_fold_info = crossfit_evalue_anchor(
        base['tau'].to_numpy(), f_anchor, n_folds=args.n_folds)
    g0_means_verb = [fi['g0_mean'] for fi in verb_fold_info]
    g1_means_verb = [fi['g1_mean'] for fi in verb_fold_info]
    print(f'  g0_mean = {np.mean(g0_means_verb):.3f} (paraphrase, low τ), '
          f'g1_mean = {np.mean(g1_means_verb):.3f} (verbatim, high τ)')

    # Triple-fold composition: Vovk-Wang AM of (multi-res paraphrase) and (verbatim)
    log_e_full = np.logaddexp(log_e_para_multi, log_e_verb) - np.log(2.0)

    # Save per-cluster
    base['log_e_para_multi'] = log_e_para_multi
    base['log_e_verb'] = log_e_verb
    base['log_e_full'] = log_e_full
    for gamma, le in para_log_es.items():
        base[f'log_e_para_g{gamma}'] = le

    # Evaluate against labels
    if args.label_source and args.label_source.exists():
        labels = pd.read_csv(args.label_source)
        cols = [c for c in labels.columns if c.startswith('y_') or c == 'cluster_id']
        base = base.merge(labels[cols].drop_duplicates('cluster_id'), on='cluster_id', how='left').fillna(0)
    for thr in [5, 50, 500]:
        col = f'y_template_{thr}'
        if col not in base.columns:
            base[col] = (base['max_template'] >= thr).astype(int)

    label_cols = [c for c in base.columns if c.startswith('y_')]
    print(f'\n--- Results at α = {args.alpha} (rigorous: anchor + cross-fit) ---')
    rows = []
    for col in label_cols:
        if base[col].sum() == 0:
            continue
        base_rate = float(base[col].astype(int).mean())
        rA = evaluate(log_e_para_multi, base[col].astype(int), args.alpha)
        rV = evaluate(log_e_verb, base[col].astype(int), args.alpha)
        rD = evaluate(log_e_full, base[col].astype(int), args.alpha)
        for gamma, le in para_log_es.items():
            r = evaluate(le, base[col].astype(int), args.alpha)
            rows.append({'label': col, 'detector': f'E_para_g{gamma}',
                         'base_rate': base_rate, **r})
        rows.append({'label': col, 'detector': 'E_para_multi', 'base_rate': base_rate, **rA})
        rows.append({'label': col, 'detector': 'E_verb', 'base_rate': base_rate, **rV})
        rows.append({'label': col, 'detector': 'E_full', 'base_rate': base_rate, **rD})
        print(f'\n  {col}: base = {base_rate:.3f}')
        print(f'    E_para_multi: AP={rA["AP"]:.3f}, k={rA["k_rejected"]:,}, '
              f'prec={rA["precision"]:.3f}, rec={rA["recall"]:.3f}')
        print(f'    E_verb       : AP={rV["AP"]:.3f}, k={rV["k_rejected"]:,}, '
              f'prec={rV["precision"]:.3f}, rec={rV["recall"]:.3f}')
        print(f'    E_full       : AP={rD["AP"]:.3f}, k={rD["k_rejected"]:,}, '
              f'prec={rD["precision"]:.3f}, rec={rD["recall"]:.3f}')

    pd.DataFrame(rows).to_csv(args.output_dir / 'evalue_rigorous_metrics.csv', index=False)
    base[['cluster_id', 'n', 'tau', 'f_anchor', 'log_e_para_multi', 'log_e_verb', 'log_e_full'] +
         [f'log_e_para_g{g}' for g in para_log_es.keys()]].to_csv(
        args.output_dir / 'evalue_rigorous_per_cluster.csv', index=False)
    summary = {'corpus': args.corpus_name, 'K': int(K), 'alpha': args.alpha,
               'n_folds': args.n_folds,
               'anchor_tau_thr': args.anchor_tau_thr, 'anchor_f_thr': args.anchor_f_thr,
               'n_tau_anchors': int(tau_anchor.sum()), 'n_f_anchors': int(f_anchor.sum()),
               'paraphrase': para_components, 'verbatim_fold_info': verb_fold_info,
               'metrics': rows}
    with (args.output_dir / 'evalue_rigorous_summary.json').open('w') as fp:
        json.dump(summary, fp, indent=2, default=float)
    print(f'\nwrote {args.output_dir}/evalue_rigorous_per_cluster.csv')
    print(f'wrote {args.output_dir}/evalue_rigorous_metrics.csv')


if __name__ == '__main__':
    main()
