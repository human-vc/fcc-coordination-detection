from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / 'data' / 'processed'
IDX_PATH = PROC / 'embedding_index.parquet'
SPLIT_PATH = PROC / 'split_assignment.parquet'

def assign(row_id: int, salt: str) -> str:
    h = hashlib.blake2b(f'{salt}-{row_id}'.encode(), digest_size=4).digest()
    return 'A' if h[0] & 1 == 0 else 'B'

def main(*, salt: str='fcc-coord-2026') -> None:
    idx = pq.read_table(IDX_PATH).to_pandas()
    n = len(idx)
    print(f'corpus size: {n:,}')
    splits = np.array([assign(int(r), salt) for r in idx['row_id']], dtype=object)
    out = pd.DataFrame({'row_id': idx['row_id'].astype(np.int32), 'split': splits})
    (a, b) = ((splits == 'A').sum(), (splits == 'B').sum())
    print(f'  discovery (A): {a:,} ({100 * a / n:.1f}%)')
    print(f'  calibration (B): {b:,} ({100 * b / n:.1f}%)')
    out.to_parquet(SPLIT_PATH, compression='zstd', index=False)
    print(f'wrote {SPLIT_PATH}')
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--salt', default='fcc-coord-2026', help='changes the partition; keep fixed for reproducibility')
    args = p.parse_args()
    main(salt=args.salt)
