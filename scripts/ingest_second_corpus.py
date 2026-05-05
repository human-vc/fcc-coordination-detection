from __future__ import annotations
import argparse
import hashlib
import sys
from pathlib import Path
import pandas as pd

def main(*, input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csvs = sorted(input_dir.glob('*.csv'))
    if not csvs:
        sys.exit(f'no CSV files in {input_dir}')
    print(f'found {len(csvs)} CSV files; concatenating...')
    df = pd.concat([pd.read_csv(p, low_memory=False) for p in csvs], ignore_index=True)
    print(f'  total rows: {len(df):,}')
    if 'comment_text' not in df.columns:
        sys.exit(f'missing comment_text column. Columns: {df.columns.tolist()}')
    df['comment_text'] = df['comment_text'].fillna('').astype(str)
    df = df[df['comment_text'].str.len() > 10].reset_index(drop=True)
    print(f'  after filtering empty/short: {len(df):,}')

    def hash_text(t: str) -> str:
        return hashlib.sha256(t.encode('utf-8', errors='replace')).hexdigest()
    if 'comment_id' not in df.columns or df['comment_id'].isna().all():
        df['comment_id'] = df['comment_text'].apply(hash_text)
    else:
        df['comment_id'] = df['comment_id'].fillna(df['comment_text'].apply(hash_text))
    sub_cols = ['submission_id', 'submission_type', 'express_comment', 'date_received', 'city', 'state', 'comment_id']
    submissions = df[[c for c in sub_cols if c in df.columns]].copy()
    submissions['express_comment'] = submissions.get('express_comment', pd.Series(['1'] * len(submissions)))
    submissions.to_parquet(output_dir / 'submissions.parquet', compression='zstd', index=False)
    print(f'  wrote submissions.parquet ({len(submissions):,} rows)')
    uniq = df.groupby('comment_id', as_index=False).first()[['comment_id', 'comment_text']]
    uniq = uniq.reset_index(drop=True)
    uniq['row_id'] = range(len(uniq))
    uniq[['comment_id', 'comment_text', 'row_id']].to_parquet(output_dir / 'comments.parquet', compression='zstd', index=False)
    print(f'  wrote comments.parquet ({len(uniq):,} unique comments)')
    counts = df.groupby('comment_id').size().rename('template_size').reset_index()
    idx = uniq.merge(counts, on='comment_id', how='left')
    idx['template_size'] = idx['template_size'].fillna(1).astype(int)
    idx[['row_id', 'comment_id', 'template_size']].to_parquet(output_dir / 'embedding_index.parquet', compression='zstd', index=False)
    print(f'  wrote embedding_index.parquet')
    print(f"  size 1 (singletons): {(idx['template_size'] == 1).sum():,}")
    print(f"  size 2-5: {((idx['template_size'] >= 2) & (idx['template_size'] <= 5)).sum():,}")
    print(f"  size 6+: {(idx['template_size'] >= 6).sum():,}")
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    main(input_dir=args.input_dir, output_dir=args.output_dir)
