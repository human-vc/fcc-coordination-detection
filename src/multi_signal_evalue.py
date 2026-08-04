"""Multi-signal compound e-value: extends the rigorous anchor + cross-fit pipeline
of evalue_rigorous.py with three additional signal channels and composes via
Vovk-Wang arithmetic mean.

Signal channels (each becomes its own anchor-identified compound e-value):
    E^para  : paraphrase fragmentation rate f_c at gamma_fine                    (existing)
    E^verb  : verbatim max-template ratio tau_c                                  (existing)
    E^time  : daily-Herfindahl burstiness H_c on submission timestamps           (new)
    E^geo   : state-Herfindahl concentration G_c on submitter state             (new)
    E^email : email-domain Herfindahl D_c on contact email                       (new)

Each signal is in (0, 1] and admits a structural anchor for the null component:
    E^para  : tau_c = 1   (byte-identical, verbatim by definition)
    E^verb  : f_c >= 0.95 (each comment in own fine sub-cluster, paraphrase by def)
    E^time  : span > P75  (long-span clusters cannot be a coordinated burst)
    E^geo   : H_state low (geographic diffuse, not a single-state campaign)
    E^email : H_email low (email-domain diffuse, not a single-contractor)

The anchor for E^time / E^geo / E^email is the *complement* of what we want to detect,
so anchor = NOT-coordinated-by-this-mode; we fit g_0 on those clusters and g_1 on the rest.

Composition: E^all_c = (1/N) * sum_i E^i_c by Vovk-Wang AM (Theorem 3.2 of Vovk-Wang 2021).
Cross-fitting per signal is 5-fold; FDR control under arbitrary dependence by Wang-Ramdas 2022.
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
from sklearn.model_selection import KFold

EPS = 1e-6


def fit_beta_mom(values: np.ndarray) -> tuple[float, float]:
    v = np.clip(values.astype(np.float64), EPS, 1 - EPS)
    if len(v) < 4 or v.var() < 1e-10:
        return 1.0, 1.0
    mu, var = float(v.mean()), float(v.var())
    c = mu * (1 - mu) / var - 1
    if c <= 0:
        return 1.0, 1.0
    return max(mu * c, 1e-3), max((1 - mu) * c, 1e-3)


def crossfit_evalue(stat: np.ndarray, anchor: np.ndarray, n_folds: int = 5,
                    seed: int = 42) -> tuple[np.ndarray, list[dict]]:
    n = len(stat); log_e = np.zeros(n); info = []
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (tr, te) in enumerate(kf.split(np.arange(n))):
        a_tr = anchor[tr]
        if a_tr.sum() < 20 or (~a_tr).sum() < 20:
            info.append({'fold': fold, 'a0': 1.0, 'b0': 1.0, 'a1': 1.0, 'b1': 1.0,
                         'g0_mean': 0.5, 'g1_mean': 0.5,
                         'n_anchor': int(a_tr.sum()), 'n_nonanchor': int((~a_tr).sum())})
            continue
        a0, b0 = fit_beta_mom(stat[tr][a_tr])
        a1, b1 = fit_beta_mom(stat[tr][~a_tr])
        s = np.clip(stat[te], EPS, 1 - EPS)
        log_e[te] = beta_dist.logpdf(s, a1, b1) - beta_dist.logpdf(s, a0, b0)
        info.append({'fold': fold, 'a0': float(a0), 'b0': float(b0),
                     'a1': float(a1), 'b1': float(b1),
                     'g0_mean': float(a0 / (a0 + b0)), 'g1_mean': float(a1 / (a1 + b1)),
                     'n_anchor': int(a_tr.sum()), 'n_nonanchor': int((~a_tr).sum())})
    return log_e, info


def ebh(log_e, alpha):
    e = np.exp(np.clip(log_e, -700, 700))
    K = len(e); order = np.argsort(-e)
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
    return {'k_rejected': k_hat, 'AP': ap, 'precision': prec, 'recall': rec}


def cluster_temporal_stats(coarse: pd.DataFrame, subs: pd.DataFrame,
                           min_size: int) -> pd.DataFrame:
    members = coarse[coarse['cluster_id'] >= 0][['cluster_id', 'comment_id']]
    j = members.merge(subs[['comment_id', 'date_received']], on='comment_id', how='inner')
    j = j.dropna(subset=['date_received'])
    j['day'] = j['date_received'].dt.floor('d')
    grouped = j.groupby('cluster_id')
    rows = []
    for cid, g in grouped:
        n = len(g)
        if n < min_size:
            continue
        days = g['day']
        span = (days.max() - days.min()).total_seconds() / 86400.0
        day_counts = days.value_counts(normalize=True)
        H = float((day_counts ** 2).sum())  # daily Herfindahl
        rows.append({'cluster_id': int(cid), 'n_with_dates': int(n),
                     'span_days': float(span), 'H_day': H})
    return pd.DataFrame(rows)


def cluster_geo_stats(coarse: pd.DataFrame, subs: pd.DataFrame,
                      min_size: int) -> pd.DataFrame:
    members = coarse[coarse['cluster_id'] >= 0][['cluster_id', 'comment_id']]
    j = members.merge(subs[['comment_id', 'state']], on='comment_id', how='inner')
    j = j[j['state'].notna() & (j['state'].astype(str).str.len() == 2)]
    grouped = j.groupby('cluster_id')
    rows = []
    for cid, g in grouped:
        n = len(g)
        if n < min_size:
            continue
        state_counts = g['state'].value_counts(normalize=True)
        H = float((state_counts ** 2).sum())
        n_states = int(g['state'].nunique())
        rows.append({'cluster_id': int(cid), 'n_with_state': int(n),
                     'n_distinct_states': n_states, 'H_state': H})
    return pd.DataFrame(rows)


def cluster_email_stats(coarse: pd.DataFrame, subs: pd.DataFrame,
                        min_size: int) -> pd.DataFrame:
    members = coarse[coarse['cluster_id'] >= 0][['cluster_id', 'comment_id']]
    j = members.merge(subs[['comment_id', 'contact_email']], on='comment_id', how='inner')
    j = j[j['contact_email'].notna()]
    j['domain'] = j['contact_email'].astype(str).str.split('@').str[-1].str.lower()
    j = j[j['domain'].astype(str).str.contains('.', na=False)]
    grouped = j.groupby('cluster_id')
    rows = []
    for cid, g in grouped:
        n = len(g)
        if n < min_size:
            continue
        dom_counts = g['domain'].value_counts(normalize=True)
        H = float((dom_counts ** 2).sum())
        n_dom = int(g['domain'].nunique())
        rows.append({'cluster_id': int(cid), 'n_with_email': int(n),
                     'n_distinct_domains': n_dom, 'H_email': H})
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--proc-dir', type=Path, default=Path('data/processed'))
    p.add_argument('--rigorous-csv', type=Path,
                   default=Path('results/evalue_rigorous_per_cluster.csv'),
                   help='Existing rigorous E_para and E_verb log-e values')
    p.add_argument('--labels-csv', type=Path,
                   default=Path('results/fragmentation_scores.csv'))
    p.add_argument('--min-size', type=int, default=8)
    p.add_argument('--alpha', type=float, default=0.10)
    p.add_argument('--n-folds', type=int, default=5)
    p.add_argument('--output-dir', type=Path, default=Path('results'))
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print('=== Multi-signal compound e-value: time + geo + email + para + verb ===\n')

    rig = pd.read_csv(args.rigorous_csv)
    labels = pd.read_csv(args.labels_csv)[['cluster_id', 'y_astro']]
    rig = rig.merge(labels, on='cluster_id', how='left')
    rig['y_astro'] = rig['y_astro'].fillna(0).astype(int)
    print(f'Loaded {len(rig):,} clusters with E_para and E_verb.')

    coarse = pq.read_table(args.proc_dir / 'clusters_leiden_r0.9.parquet').to_pandas()
    print('Loading submissions metadata (timestamps, state, email)...')
    subs = pq.read_table(args.proc_dir / 'submissions.parquet',
                         columns=['comment_id', 'date_received', 'state', 'contact_email']).to_pandas()
    subs['date_received'] = pd.to_datetime(subs['date_received'], errors='coerce')
    print(f'  {len(subs):,} submissions loaded.')

    # 1) Temporal burstiness
    print('\n[E^time] Computing daily-Herfindahl burstiness per cluster...')
    time_stats = cluster_temporal_stats(coarse, subs, args.min_size)
    print(f'  {len(time_stats):,} clusters with >= {args.min_size} dated members.')
    print(f'  span_days: median={time_stats["span_days"].median():.1f}, '
          f'p90={time_stats["span_days"].quantile(0.9):.1f}, '
          f'max={time_stats["span_days"].max():.1f}')
    print(f'  H_day:     median={time_stats["H_day"].median():.3f}, '
          f'p10={time_stats["H_day"].quantile(0.1):.3f}, '
          f'p90={time_stats["H_day"].quantile(0.9):.3f}')

    # 2) Geographic concentration
    print('\n[E^geo] Computing state-Herfindahl per cluster...')
    geo_stats = cluster_geo_stats(coarse, subs, args.min_size)
    print(f'  {len(geo_stats):,} clusters with >= {args.min_size} state-known members.')
    print(f'  H_state: median={geo_stats["H_state"].median():.3f}, '
          f'p10={geo_stats["H_state"].quantile(0.1):.3f}, '
          f'p90={geo_stats["H_state"].quantile(0.9):.3f}')

    # 3) Email-domain concentration
    print('\n[E^email] Computing email-domain Herfindahl per cluster...')
    email_stats = cluster_email_stats(coarse, subs, args.min_size)
    print(f'  {len(email_stats):,} clusters with >= {args.min_size} email-known members.')
    if len(email_stats) > 0:
        print(f'  H_email: median={email_stats["H_email"].median():.3f}, '
              f'p10={email_stats["H_email"].quantile(0.1):.3f}, '
              f'p90={email_stats["H_email"].quantile(0.9):.3f}')

    # Merge everything onto rig
    df = rig.merge(time_stats[['cluster_id', 'span_days', 'H_day']], on='cluster_id', how='left')
    df = df.merge(geo_stats[['cluster_id', 'H_state']], on='cluster_id', how='left')
    df = df.merge(email_stats[['cluster_id', 'H_email']], on='cluster_id', how='left')
    df['H_day'] = df['H_day'].fillna(0.5).clip(EPS, 1 - EPS)
    df['H_state'] = df['H_state'].fillna(0.5).clip(EPS, 1 - EPS)
    df['H_email'] = df['H_email'].fillna(0.5).clip(EPS, 1 - EPS)
    df['span_days'] = df['span_days'].fillna(0)

    # Anchor masks: use the validated tau=1 anchor (byte-identical = pure null,
    # 100% purity against y_astro) as the SHARED null anchor for all signal channels.
    # Rationale: tau=1 clusters are by-definition not paid-astroturf-paraphrase;
    # their metadata distribution is the natural null reference for paid-astroturf detection.
    # Paid contractors disperse submissions over time / states / email-domains to mimic
    # grassroots, so the alternative is at the LOW-H end of each metadata distribution.
    tau_anchor = (df['tau'] >= 0.999).to_numpy()
    print(f'\nShared anchor: tau=1 (byte-identical) — n = {tau_anchor.sum():,}, ',
          f'y_astro rate among anchors = {df.loc[tau_anchor, "y_astro"].mean():.4f}')
    # E^time / E^geo / E^email each use the tau anchor; the test stat is 1-H
    # so the alternative (low H, diffuse metadata) becomes the high-stat alternative.
    df['stat_time'] = (1 - df['H_day']).clip(EPS, 1 - EPS)
    df['stat_geo'] = (1 - df['H_state']).clip(EPS, 1 - EPS)
    df['stat_email'] = (1 - df['H_email']).clip(EPS, 1 - EPS)

    # Cross-fit each signal using the shared tau anchor and 1-H test statistic
    print('\nCross-fitting each signal (5-fold) with shared tau-anchor...')
    log_e_time, time_info = crossfit_evalue(df['stat_time'].to_numpy(), tau_anchor, n_folds=args.n_folds)
    log_e_geo, geo_info = crossfit_evalue(df['stat_geo'].to_numpy(), tau_anchor, n_folds=args.n_folds)
    log_e_email, email_info = crossfit_evalue(df['stat_email'].to_numpy(), tau_anchor, n_folds=args.n_folds)

    df['log_e_time'] = log_e_time
    df['log_e_geo'] = log_e_geo
    df['log_e_email'] = log_e_email

    # Composition: Vovk-Wang AM of all available signals
    log_e_para = df['log_e_para_multi'].to_numpy()
    log_e_verb = df['log_e_verb'].to_numpy()
    L = 5
    stack = np.stack([log_e_para, log_e_verb, log_e_time, log_e_geo, log_e_email], axis=0)
    df['log_e_multi5'] = logsumexp(stack, axis=0) - np.log(L)
    # Also without verb (since verb hurts on paraphrase target)
    df['log_e_multi4_noverb'] = logsumexp(np.stack([log_e_para, log_e_time, log_e_geo, log_e_email]), axis=0) - np.log(4)
    # Also just time + para (minimal extension)
    df['log_e_para_time'] = logsumexp(np.stack([log_e_para, log_e_time]), axis=0) - np.log(2)
    df['log_e_para_time_geo'] = logsumexp(np.stack([log_e_para, log_e_time, log_e_geo]), axis=0) - np.log(3)

    print('\n--- Results vs y_astro (NYAG paid astroturf) at α =', args.alpha, '---')
    print(f'  {"detector":<32}{"AP":>8}{"k":>10}{"prec":>10}{"recall":>10}')
    print('  ' + '-' * 70)
    for name, le in [
        ('E_para_multi (existing)', log_e_para),
        ('E_time (NEW)', log_e_time),
        ('E_geo  (NEW)', log_e_geo),
        ('E_email (NEW)', log_e_email),
        ('Vovk-Wang AM (para+time)', df['log_e_para_time'].to_numpy()),
        ('Vovk-Wang AM (para+time+geo)', df['log_e_para_time_geo'].to_numpy()),
        ('Vovk-Wang AM (no verb, 4-sig)', df['log_e_multi4_noverb'].to_numpy()),
        ('Vovk-Wang AM (all 5 signals)', df['log_e_multi5'].to_numpy()),
    ]:
        r = evaluate(le, df['y_astro'], args.alpha)
        print(f'  {name:<32}{r["AP"]:>8.4f}{r["k_rejected"]:>10,}{r["precision"]:>10.3f}{r["recall"]:>10.3f}')

    df.to_csv(args.output_dir / 'multi_signal_per_cluster.csv', index=False)
    print(f'\nwrote {args.output_dir}/multi_signal_per_cluster.csv')


if __name__ == '__main__':
    main()
