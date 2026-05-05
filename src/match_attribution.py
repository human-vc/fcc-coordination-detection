from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / 'data' / 'processed'
RES = ROOT / 'results'
RES.mkdir(exist_ok=True)

def text_hash(s: str | float) -> str:
    if not isinstance(s, str):
        return ''
    norm = ' '.join(s.lower().split())
    return hashlib.blake2b(norm.encode('utf-8'), digest_size=12).hexdigest()

def main(*, top_n_uploaders: int=10, cluster_path: Path | None=None, rejections_path: Path | None=None, output_csv: Path | None=None) -> None:
    bf_path = PROC / 'buzzfeed_bulk_uploads.parquet'
    sl_comments = PROC / 'comments.parquet'
    cluster_path = cluster_path or PROC / 'clusters.parquet'
    if rejections_path is not None:
        rejections = rejections_path
    else:
        rejections = RES / 'fdr_rejections_leiden.parquet'
        if not rejections.exists():
            rejections = PROC / 'fdr_rejections.parquet'
    if not bf_path.exists():
        raise SystemExit(f'missing {bf_path}; run src/ingest_buzzfeed.py first')
    print(f'loading BuzzFeed bulk-uploads ({bf_path})...')
    bf = pq.read_table(bf_path, columns=['comments', 'uploader']).to_pandas()
    print(f'  {len(bf):,} rows')
    bf['text_hash'] = bf['comments'].apply(text_hash)
    bf_lookup = bf.dropna(subset=['uploader']).groupby('text_hash')['uploader'].agg(lambda s: s.value_counts().index[0])
    print(f'  {len(bf_lookup):,} unique text hashes -> uploader mapping')
    attr = pq.read_table(PROC / 'buzzfeed_attribution.parquet').to_pandas()
    uploader_to_label = dict(zip(attr['uploader'], attr['ag_attribution']))
    uploader_to_cat = dict(zip(attr['uploader'], attr['category']))
    print(f'loading slnader comments ({sl_comments})...')
    sl = pq.read_table(sl_comments, columns=['comment_id', 'comment_text']).to_pandas()
    print(f'  {len(sl):,} unique slnader comments')
    sl['text_hash'] = sl['comment_text'].apply(text_hash)
    sl['uploader'] = sl['text_hash'].map(bf_lookup)
    matched = sl['uploader'].notna().sum()
    print(f'  matched {matched:,} ({100 * matched / len(sl):.1f}%) to a BuzzFeed uploader')
    sl['category'] = sl['uploader'].map(uploader_to_cat).fillna('')
    sl['label'] = sl['uploader'].map(uploader_to_label).fillna('')
    sl[['comment_id', 'uploader', 'category', 'label']].to_parquet(PROC / 'comment_uploader.parquet', compression='zstd', index=False)
    print('\nMatch breakdown by category:')
    for (cat, n) in sl['category'].value_counts().items():
        label = cat or 'unmatched'
        print(f'  {label:14s} {n:>10,}  ({100 * n / len(sl):.1f}%)')
    print(f'\nloading clusters from {cluster_path}...')
    cl = pq.read_table(cluster_path).to_pandas()
    rj = pq.read_table(rejections).to_pandas()
    rejected_cids = set(rj.loc[rj['rejected_ebh'], 'cluster_id'].astype(int))
    cl_in_rejected = cl[cl['cluster_id'].isin(rejected_cids)].copy()
    cl_in_rejected = cl_in_rejected.merge(sl[['comment_id', 'uploader', 'category', 'label']], on='comment_id', how='left')
    rows = []
    for (cid, group) in cl_in_rejected.groupby('cluster_id'):
        n = len(group)
        n_matched = group['uploader'].notna().sum()
        n_astroturf = (group['category'] == 'astroturf').sum()
        n_advocacy = (group['category'] == 'advocacy').sum()
        labeled = group.loc[group['label'] != '', 'label']
        top_label = labeled.value_counts().index[0] if len(labeled) else ''
        rows.append({'cluster_id': int(cid), 'n_members': n, 'n_matched_buzzfeed': int(n_matched), 'frac_matched_buzzfeed': float(n_matched / n) if n else 0.0, 'n_astroturf': int(n_astroturf), 'frac_astroturf': float(n_astroturf / n) if n else 0.0, 'n_advocacy': int(n_advocacy), 'frac_advocacy': float(n_advocacy / n) if n else 0.0, 'top_label': top_label})
    out = pd.DataFrame(rows).sort_values('frac_astroturf', ascending=False)
    out_path = output_csv or RES / 'attribution_table.csv'
    out.to_csv(out_path, index=False)
    print(f'\nrejected clusters analyzed: {len(out):,}')
    print(f"  avg frac matched to BuzzFeed: {out['frac_matched_buzzfeed'].mean():.3f}")
    print(f"  avg frac astroturf:           {out['frac_astroturf'].mean():.3f}")
    print(f"  avg frac advocacy:            {out['frac_advocacy'].mean():.3f}")
    print(f"  clusters >=50% astroturf:     {(out['frac_astroturf'] >= 0.5).sum():,}")
    print(f"  clusters >=50% advocacy:      {(out['frac_advocacy'] >= 0.5).sum():,}")
    print(f"  clusters >=50% any matched:   {(out['frac_matched_buzzfeed'] >= 0.5).sum():,}")
    print(f'\nwrote {out_path}')
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--top-n-uploaders', type=int, default=10)
    p.add_argument('--cluster-path', type=Path, default=None, help='default: data/processed/clusters.parquet')
    p.add_argument('--rejections-path', type=Path, default=None, help='default: results/fdr_rejections_leiden.parquet or data/processed/fdr_rejections.parquet')
    p.add_argument('--output-csv', type=Path, default=None, help='default: results/attribution_table.csv')
    args = p.parse_args()
    main(top_n_uploaders=args.top_n_uploaders, cluster_path=args.cluster_path, rejections_path=args.rejections_path, output_csv=args.output_csv)
