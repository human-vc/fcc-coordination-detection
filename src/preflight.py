from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / 'data' / 'processed'
REQUIRED_FROM_V1 = [PROC / 'embeddings.npy', PROC / 'embedding_index.parquet', PROC / 'comments.parquet', PROC / 'submissions.parquet']

def check(label: str, ok: bool, detail: str='') -> None:
    mark = '✓' if ok else '✗'
    line = f'  {mark} {label}'
    if detail:
        line += f'  — {detail}'
    print(line)
    if not ok:
        print(f'\nFAILED: {label}\n{detail}')
        sys.exit(1)

def main() -> None:
    print('[1/6] required artifacts...')
    for p in REQUIRED_FROM_V1:
        check(p.name, p.exists(), f'size {p.stat().st_size / 1000000.0:.1f} MB' if p.exists() else 'missing')
    print('\n[2/6] embeddings.npy loadable (mmap)...')
    emb = np.load(PROC / 'embeddings.npy', mmap_mode='r')
    (n_emb, d_emb) = emb.shape
    check('shape', emb.ndim == 2, f'shape={emb.shape}, dtype={emb.dtype}')
    check('dtype', emb.dtype == np.float16, f'dtype={emb.dtype}')
    check('dim 384', d_emb == 384, f'got dim={d_emb}')
    check('nonzero', float(emb[0].astype(np.float32).std()) > 0, 'first vector has variation')
    print('\n[3/6] embedding_index.parquet schema...')
    idx = pq.read_table(PROC / 'embedding_index.parquet').to_pandas()
    check('rows match emb', len(idx) == n_emb, f'idx={len(idx):,} emb={n_emb:,}')
    for col in ('row_id', 'comment_id', 'template_size'):
        check(f'column {col}', col in idx.columns)
    check('row_id contiguous 0..n-1', (idx['row_id'].to_numpy() == np.arange(n_emb)).all(), 'row_id must equal positional index for emb[row_id] to work')
    n_zero = int((idx['template_size'] == 0).sum())
    check('template_size>=0 always', (idx['template_size'] >= 0).all(), f'{n_zero:,} comment_ids have template_size=0 (no submission references them; OK)')
    print('\n[4/6] new v2 code imports cleanly...')
    sys.path.insert(0, str(ROOT / 'src'))
    try:
        from cluster_singletons import main as _cs
        from evalues import fit_beta_mom, size_buckets, vovk_wang
        from resolution_sweep import main as _rs
        from cluster_eval import load_method
        check('cluster_singletons', True)
        check('evalues (Beta + bucketing)', True)
        check('resolution_sweep', True)
        check('cluster_eval', True)
    except Exception as e:
        check('imports', False, repr(e))
    print('\n[5/6] CPM partition class is available in leidenalg...')
    import leidenalg as la
    check('CPMVertexPartition', hasattr(la, 'CPMVertexPartition'))
    print('\n[6/6] Beta tail smoke test...')
    rng = np.random.default_rng(0)
    fake_null = rng.beta(20, 80, size=2000).astype(np.float32)
    (a, b) = fit_beta_mom(fake_null)
    check('Beta MOM positive', a > 0 and b > 0, f'a={a:.2f}, b={b:.2f}')
    from scipy.stats import beta as beta_dist
    sf_at_high = float(beta_dist.sf(0.5, a, b))
    check('Beta tail < 1e-3 at observed high T', sf_at_high < 0.001, f'survival(0.5)={sf_at_high:.2e} (expect tiny since null mean ~0.2)')
    e = float(vovk_wang(np.array([sf_at_high]))[0])
    check('Vovk-Wang e finite > 100', np.isfinite(e) and e > 100, f'e={e:.1f}')
    print('\nALL PREFLIGHT CHECKS PASSED — safe to run ./run_v2.sh')
if __name__ == '__main__':
    main()
