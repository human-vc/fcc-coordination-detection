from __future__ import annotations
import re
import time
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from datasketch import MinHash, MinHashLSH
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / 'data' / 'processed'
RES = ROOT / 'results'

NUM_PERM = 128
SHINGLE_K = 5
LSH_THRESHOLD = 0.7


def shingles(text: str, k: int = SHINGLE_K):
    text = re.sub(r'\s+', ' ', text.lower().strip())
    if len(text) < k:
        return [text]
    return [text[i:i + k] for i in range(len(text) - k + 1)]


def minhash_for_text(text: str) -> MinHash:
    m = MinHash(num_perm=NUM_PERM)
    for s in set(shingles(text)):
        m.update(s.encode('utf-8'))
    return m


def main():
    print('=== MinHash baseline at γ_coarse = 0.90 candidates ===\n')

    print('[1] Loading candidate-cluster member texts...')
    coarse = pq.read_table(PROC / 'clusters_leiden_r0.9.parquet').to_pandas()
    cands = coarse[(coarse['cluster_id'] >= 0) & (coarse['cluster_size'] >= 8)]
    cand_row_ids = cands['row_id'].tolist()

    con = duckdb.connect()
    con.execute(f"CREATE VIEW c AS SELECT * FROM read_parquet('{PROC / 'comments.parquet'}')")
    con.execute(f"CREATE VIEW i AS SELECT * FROM read_parquet('{PROC / 'embedding_index.parquet'}')")
    con.register('cand_ids', pd.DataFrame({'row_id': cand_row_ids}))
    df = con.execute("""
        SELECT i.row_id, c.comment_text
        FROM i JOIN c USING (comment_id)
        WHERE i.row_id IN (SELECT * FROM cand_ids)
        ORDER BY i.row_id
    """).fetchdf()
    print(f'  fetched {len(df):,} text rows')

    df = df.merge(cands[['row_id', 'cluster_id']], on='row_id', how='inner')

    print(f'\n[2] Computing MinHash signatures (NUM_PERM={NUM_PERM}, shingle_k={SHINGLE_K})...')
    t0 = time.time()
    sigs = []
    for txt in df['comment_text'].fillna(''):
        sigs.append(minhash_for_text(txt))
    print(f'  {len(sigs):,} signatures in {time.time()-t0:.0f}s')

    print(f'\n[3] Computing per-cluster MinHash duplicate density features...')
    rows = []
    for cid, sub in df.groupby('cluster_id'):
        idxs = sub.index.tolist()
        sub_sigs = [sigs[i] for i in idxs]
        n = len(sub_sigs)
        lsh = MinHashLSH(threshold=LSH_THRESHOLD, num_perm=NUM_PERM)
        for k, m in enumerate(sub_sigs):
            lsh.insert(str(k), m)
        n_dupe_pairs = 0
        for k, m in enumerate(sub_sigs):
            res = lsh.query(m)
            n_dupe_pairs += len(res) - 1
        n_dupe_pairs //= 2
        max_pairs = n * (n - 1) // 2
        density = n_dupe_pairs / max(max_pairs, 1)
        n_unique_estimate = max(1, int(round(n * (1 - density))))
        rows.append({
            'cluster_id': int(cid),
            'n': n,
            'minhash_density': density,
            'minhash_unique_est': n_unique_estimate,
            'minhash_unique_rate': n_unique_estimate / n,
        })
    minhash_df = pd.DataFrame(rows)
    print(f'  computed {len(minhash_df):,} cluster-level MinHash features')

    print(f'\n[4] Compare to NYAG astroturf...')
    labels = pd.read_csv(RES / 'fragmentation_scores.csv')[['cluster_id', 'y_astro', 'y_adv']]
    minhash_df = minhash_df.merge(labels, on='cluster_id', how='inner')

    def report(name, score):
        ap = average_precision_score(minhash_df['y_astro'], score)
        print(f'  {name:<40} AP = {ap:.3f}')

    report('MinHash density (higher = duplicate)', minhash_df['minhash_density'])
    report('MinHash density (NEGATIVE = paraphrase)', -minhash_df['minhash_density'])
    report('MinHash unique-rate', minhash_df['minhash_unique_rate'])

    out_path = RES / 'baseline_minhash.csv'
    minhash_df.to_csv(out_path, index=False)
    print(f'\nwrote {out_path}')


if __name__ == '__main__':
    main()
