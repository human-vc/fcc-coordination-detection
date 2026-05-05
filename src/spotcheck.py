from __future__ import annotations
import argparse
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / 'data' / 'processed'
RES = ROOT / 'results'
RES.mkdir(exist_ok=True)

def load_method(method: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if method == 'leiden':
        cl = pq.read_table(PROC / 'clusters.parquet').to_pandas()
    else:
        cl = pq.read_table(PROC / f'clusters_{method}.parquet').to_pandas()
    rj = pq.read_table(RES / f'fdr_rejections_{method}.parquet').to_pandas()
    return (cl, rj)

def pick_clusters(rj: pd.DataFrame, cl: pd.DataFrame, *, buckets: list[tuple[int, int]], per_bucket: int, seed: int) -> list[tuple[str, int]]:
    rng = np.random.default_rng(seed)
    rejected = rj[rj['rejected_ebh']]
    picked = []
    for (lo, hi) in buckets:
        in_bucket = rejected[(rejected['n'] >= lo) & (rejected['n'] < hi)]
        if not len(in_bucket):
            continue
        n_pick = min(per_bucket, len(in_bucket))
        chosen = rng.choice(in_bucket['cluster_id'].to_numpy(), size=n_pick, replace=False)
        for cid in chosen:
            picked.append((f'{lo}-{hi}', int(cid)))
    return picked

def get_text_lookup() -> dict[str, str]:
    print('loading comment text lookup (one-time, ~5 sec)...')
    df = pq.read_table(PROC / 'comments.parquet', columns=['comment_id', 'comment_text']).to_pandas()
    return dict(zip(df['comment_id'].astype(str), df['comment_text'].fillna('').astype(str)))

def render_cluster(cid: int, cl: pd.DataFrame, text_lookup: dict[str, str], per_template_strata: int=2) -> str:
    members = cl[cl['cluster_id'] == cid]
    n = len(members)
    out = [f"\n{'=' * 76}", f'cluster_id={cid}  n_members={n}']
    strata = [('singletons', members[members['template_size'] == 1]), ('small (2-10)', members[members['template_size'].between(2, 10)]), ('medium (11-100)', members[members['template_size'].between(11, 100)]), ('large (101+)', members[members['template_size'] >= 101])]
    for (name, sub) in strata:
        if not len(sub):
            continue
        n_strat = min(per_template_strata, len(sub))
        sample = sub.sample(n=n_strat, random_state=0)
        out.append(f'\n  --- {name} ({len(sub)} in stratum, showing {n_strat}) ---')
        for (_, row) in sample.iterrows():
            cid_str = str(row['comment_id'])
            text = text_lookup.get(cid_str, '<text not found>')
            preview = text[:600] + '...' if len(text) > 600 else text
            out.append(f"    [tmpl={row['template_size']:>7,}] {preview!r}")
    return '\n'.join(out)

def main(*, methods: list[str], per_bucket: int, seed: int) -> None:
    text_lookup = get_text_lookup()
    buckets = [(5, 20), (20, 100), (100, 1000), (1000, 1000000)]
    for method in methods:
        try:
            (cl, rj) = load_method(method)
        except FileNotFoundError as e:
            print(f'skip {method}: {e}')
            continue
        picked = pick_clusters(rj, cl, buckets=buckets, per_bucket=per_bucket, seed=seed)
        if not picked:
            continue
        out_path = RES / f'spotcheck_{method}.txt'
        lines = [f'# spot-check for method={method}', f'# {len(picked)} clusters across buckets {[(lo, hi) for (lo, hi) in buckets]}, {per_bucket} per bucket']
        for (bucket, cid) in picked:
            lines.append(f'\n### bucket={bucket}')
            lines.append(render_cluster(cid, cl, text_lookup))
        out_path.write_text('\n'.join(lines))
        print(f'wrote {out_path}  ({len(picked)} clusters)')
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--methods', nargs='+', default=['leiden', 'connected_components', 'hdbscan_emb', 'minhash_lsh'])
    p.add_argument('--per-bucket', type=int, default=3)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    main(methods=args.methods, per_bucket=args.per_bucket, seed=args.seed)
