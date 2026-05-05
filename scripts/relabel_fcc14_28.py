"""Re-derive FCC 14-28 template-coordination labels by stripping BuzzFeed's
auto-included FCC submission-ID prefixes and Page-N suffixes from comment text.

Why: the BuzzFeed dataset preserves the FCC-added unique prefix
'<digits>.txt' on each comment, which prevents exact-template-match counting.
After stripping, real template duplicates can be counted.
"""
from __future__ import annotations
import argparse
import hashlib
import re
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PREFIX_RE = re.compile(r'^\d{4,}\.txt\s*', re.IGNORECASE)
SUFFIX_RE = re.compile(r'\s*Page\s+\d+\s*$', re.IGNORECASE)
WHITESPACE_RE = re.compile(r'\s+')


def normalize(text: str) -> str:
    if not isinstance(text, str):
        return ''
    t = PREFIX_RE.sub('', text)
    t = SUFFIX_RE.sub('', t)
    t = WHITESPACE_RE.sub(' ', t).strip().lower()
    return t


def hash_text(t: str) -> str:
    return hashlib.sha256(t.encode('utf-8', errors='replace')).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--proc-dir', type=Path, required=True)
    p.add_argument('--output-csv', type=Path, required=True)
    args = p.parse_args()

    print('reading comments + embedding_index...', flush=True)
    comments = pq.read_table(args.proc_dir / 'comments.parquet').to_pandas()
    idx = pq.read_table(args.proc_dir / 'embedding_index.parquet').to_pandas()
    print(f'  {len(comments):,} unique original-text rows', flush=True)

    print('stripping prefixes/suffixes and re-hashing...', flush=True)
    comments['normalized'] = comments['comment_text'].apply(normalize)
    comments['template_id'] = comments['normalized'].apply(lambda t: hash_text(t) if len(t) >= 10 else f'short_{hash_text(t)}')
    n_unique_normalized = comments['template_id'].nunique()
    print(f'  unique normalized texts: {n_unique_normalized:,}', flush=True)
    print(f'  collapse ratio: {len(comments) / max(n_unique_normalized, 1):.2f}x', flush=True)

    template_size = (comments.groupby('template_id').size().rename('normalized_template_size').reset_index())
    print(f'  largest normalized template: {template_size["normalized_template_size"].max():,}')
    print(f'  templates with size >= 5:  {(template_size["normalized_template_size"] >= 5).sum():,}')
    print(f'  templates with size >= 100: {(template_size["normalized_template_size"] >= 100).sum():,}')
    print(f'  templates with size >= 1000:{(template_size["normalized_template_size"] >= 1000).sum():,}')
    print(f'  top 5 template sizes: {template_size.nlargest(5, "normalized_template_size")["normalized_template_size"].tolist()}')

    out = comments[['comment_id', 'template_id']].merge(template_size, on='template_id')
    out = out.merge(idx[['row_id', 'comment_id']], on='comment_id', how='right')
    out.to_csv(args.output_csv, index=False)
    print(f'\nwrote {args.output_csv}')


if __name__ == '__main__':
    main()
