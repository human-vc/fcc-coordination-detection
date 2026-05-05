from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / 'data' / 'processed'
RES = ROOT / 'results'

def jaccard_match_clusters(target_cluster_id: int, target_members: set, other_clusters: dict[int, set], threshold: float=0.5) -> int | None:
    best_cid = None
    best_jaccard = 0.0
    for (cid, members) in other_clusters.items():
        inter = len(target_members & members)
        if inter == 0:
            continue
        union = len(target_members | members)
        j = inter / union
        if j > best_jaccard:
            best_jaccard = j
            best_cid = cid
    return best_cid if best_jaccard >= threshold else None

def main() -> None:
    gammas = ['0.85', '0.88', '0.9', '0.93', '0.96']
    print(f'loading e-values + cluster files for γ ∈ {gammas}')
    g_data = {}
    for g in gammas:
        if g == '0.9':
            path = PROC / 'cluster_evalues_lrt_clean.parquet'
        else:
            path = PROC / f'cluster_evalues_lrt_r{g}.parquet'
        ev = pq.read_table(path).to_pandas()
        cl = pq.read_table(PROC / f'clusters_leiden_r{g}.parquet').to_pandas()
        members = cl[cl['cluster_id'].isin(ev['cluster_id'])].groupby('cluster_id')['row_id'].apply(set).to_dict()
        g_data[g] = {'ev': ev, 'members': members}
        print(f'  γ={g}: {len(ev):,} clusters with LRT e-values, {len(members):,} with member sets')
    target_g = '0.9'
    target_ev = g_data[target_g]['ev']
    target_members = g_data[target_g]['members']
    print()
    print('matching γ=0.90 clusters to counterparts at other γ (Jaccard threshold 0.5)...')
    rows = []
    for (_, r) in target_ev.iterrows():
        cid_target = int(r['cluster_id'])
        members = target_members.get(cid_target, set())
        if not members:
            continue
        log_es = {target_g: float(r['log_e'])}
        for g in gammas:
            if g == target_g:
                continue
            other_members = g_data[g]['members']
            match = jaccard_match_clusters(cid_target, members, other_members)
            if match is not None:
                ev_other = g_data[g]['ev']
                row_match = ev_other[ev_other['cluster_id'] == match]
                if len(row_match):
                    log_es[g] = float(row_match['log_e'].iloc[0])
        log_e_values = list(log_es.values())
        from scipy.special import logsumexp
        log_e_avg = float(logsumexp(log_e_values) - np.log(len(log_e_values)))
        rows.append({'cluster_id': cid_target, 'n_resolutions_matched': len(log_es), 'log_e_at_0.9': log_es[target_g], 'log_e_avg': log_e_avg})
    out = pd.DataFrame(rows)
    out.to_parquet(PROC / 'cluster_evalues_lrt_multires.parquet', compression='zstd', index=False)
    print()
    print(f'matched clusters: {len(out):,}')
    print(f'resolution coverage:')
    for (n_res, c) in out['n_resolutions_matched'].value_counts().sort_index().items():
        print(f'  {n_res} resolutions: {c:,}')
    print()
    print('e-value distributions (log space):')
    print(f"  single-γ (γ=0.90 only):  median={out['log_e_at_0.9'].median():.2f},  p10={out['log_e_at_0.9'].quantile(0.1):.2f}")
    print(f"  multi-res averaged:       median={out['log_e_avg'].median():.2f},  p10={out['log_e_avg'].quantile(0.1):.2f}")

    def ebh(scores, alpha=0.1):
        K = len(scores)
        s = np.sort(scores)[::-1]
        e = np.exp(np.clip(s, -700, 700))
        thresholds = K / (alpha * np.arange(1, K + 1))
        rej = e >= thresholds
        if rej.any():
            return int(np.where(rej)[0].max() + 1)
        return 0
    k_single = ebh(out['log_e_at_0.9'].to_numpy())
    k_multi = ebh(out['log_e_avg'].to_numpy())
    print()
    print(f'e-BH at α=0.10:')
    print(f'  single-γ:    {k_single:,} rejected of {len(out):,}')
    print(f'  multi-res:   {k_multi:,} rejected of {len(out):,}')
    att = pd.read_csv(RES / 'attribution_table_r0.9.csv')

    def attribution(out, score_col, alpha=0.1):
        df = out.merge(att[['cluster_id', 'frac_astroturf']], on='cluster_id', how='left')
        df = df.sort_values(score_col, ascending=False).reset_index(drop=True)
        df['e'] = np.exp(np.clip(df[score_col], -700, 700))
        K = len(df)
        df['thr'] = K / (alpha * np.arange(1, K + 1))
        df['rej'] = df['e'] >= df['thr']
        if df['rej'].any():
            k = df.index[df['rej']].max() + 1
            sub = df.iloc[:k]
            n_a = int((sub['frac_astroturf'].fillna(0) >= 0.5).sum())
            return (k, n_a, 100 * n_a / max(k, 1))
        return (0, 0, 0)
    (k1, a1, p1) = attribution(out, 'log_e_at_0.9')
    (k2, a2, p2) = attribution(out, 'log_e_avg')
    print()
    print(f'attribution at α=0.10:')
    print(f"  {'method':<25}{'rejected':>10}{'astro>=0.5':>14}{'astro%':>10}")
    print(f"  {'single-γ (γ=0.90)':<25}{k1:>10,}{a1:>14,}{p1:>9.1f}%")
    print(f"  {'multi-res averaged':<25}{k2:>10,}{a2:>14,}{p2:>9.1f}%")
    print(f'\nwrote {PROC}/cluster_evalues_lrt_multires.parquet')
if __name__ == '__main__':
    main()
