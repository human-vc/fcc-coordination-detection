from __future__ import annotations
import argparse
from pathlib import Path
from time import time
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / 'data' / 'processed'
EMB_PATH = PROC / 'embeddings.npy'

def fit_top_k_components(emb: np.ndarray, *, mean: np.ndarray, k: int, pca_sample: int, rng: np.random.Generator) -> np.ndarray:
    n = emb.shape[0]
    n_sample = min(pca_sample, n)
    idx = rng.choice(n, size=n_sample, replace=False)
    idx.sort()
    sample = emb[idx].astype(np.float32) - mean
    print(f'  PCA on {n_sample:,} sampled rows...')
    (_, s, vt) = np.linalg.svd(sample, full_matrices=False)
    print(f'  top-{k} singular values: {s[:k].round(2).tolist()}  (s_{k + 1}={s[k]:.2f})')
    return vt[:k]

def main(*, k: int=5, pca_sample: int=1000000, input_path: Path | None=None, output_path: Path | None=None, seed: int=0) -> None:
    in_path = input_path or EMB_PATH
    out_path = output_path or PROC / f'embeddings_white_k{k}.npy'
    rng = np.random.default_rng(seed)
    print(f'loading {in_path} (mmap)...')
    emb = np.load(in_path, mmap_mode='r')
    (n, d) = emb.shape
    print(f'  shape {emb.shape}  dtype {emb.dtype}')
    print('computing corpus mean (streaming)...')
    t0 = time()
    chunk = 100000
    mean = np.zeros(d, dtype=np.float64)
    for i in range(0, n, chunk):
        mean += emb[i:i + chunk].astype(np.float64).sum(axis=0)
    mean /= n
    mean_f32 = mean.astype(np.float32)
    print(f'  mean norm: {np.linalg.norm(mean):.4f}   ({time() - t0:.1f}s)')
    print(f'fitting PCA (top-{k} components on {pca_sample:,} subsample)...')
    t0 = time()
    pcs = fit_top_k_components(emb, mean=mean_f32, k=k, pca_sample=pca_sample, rng=rng)
    print(f'  PCA done   ({time() - t0:.1f}s)')
    print(f'projecting off top-{k} PCs and renormalizing (streaming)...')
    out = np.empty((n, d), dtype=np.float16)
    t0 = time()
    n_zero = 0
    for i in range(0, n, chunk):
        x = emb[i:i + chunk].astype(np.float32) - mean_f32
        proj = x @ pcs.T @ pcs
        x -= proj
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        zero_mask = norms[:, 0] < 1e-09
        n_zero += int(zero_mask.sum())
        norms[zero_mask] = 1.0
        x /= norms
        if zero_mask.any():
            x[zero_mask] = 0.0
        out[i:i + len(x)] = x.astype(np.float16)
    print(f'  projection done   ({time() - t0:.1f}s)')
    if n_zero:
        print(f'  warning: {n_zero} rows had zero residual norm; set to zero vector')
    print(f'writing {out_path}  ({out.nbytes / 1000000.0:.1f} MB)')
    np.save(out_path, out)
    sanity_n = min(200000, n)
    print(f'post-whitening sanity check (PCA on {sanity_n:,} of output)...')
    sample_idx = rng.choice(n, size=sanity_n, replace=False)
    sample_idx.sort()
    post = out[sample_idx].astype(np.float32)
    post -= post.mean(axis=0)
    (_, s_post, _) = np.linalg.svd(post, full_matrices=False)
    print(f'  output top-10 singular values: {s_post[:10].round(2).tolist()}')
    s_top = float(s_post[0])
    s_floor = float(s_post[k]) if k < len(s_post) else float(s_post[-1])
    print(f'  ratio s_1 / s_{k + 1} (post-whiten): {s_top / max(s_floor, 1e-09):.2f}  (closer to 1 = more isotropic)')
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--k', type=int, default=5, help='number of top principal components to remove (default 5)')
    p.add_argument('--pca-sample', type=int, default=1000000, help='number of rows for the PCA fit (default 1M)')
    p.add_argument('--input-path', type=Path, default=None, help='default: data/processed/embeddings.npy')
    p.add_argument('--output-path', type=Path, default=None, help='default: data/processed/embeddings_white_k{K}.npy')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    main(k=args.k, pca_sample=args.pca_sample, input_path=args.input_path, output_path=args.output_path, seed=args.seed)
