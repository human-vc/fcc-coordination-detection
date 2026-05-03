"""Cluster-level e-values via cross-half null calibration.

Construction:
  - Clusters S_1, ..., S_m are discovered on the A-half by Leiden.
  - For each S_j, T(S_j) = mean pairwise cosine similarity within S_j (over A
    embeddings only).
  - Null: for each cluster size, draw n_null_draws random size-matched subsets
    from the B-half. Their mean cosine similarity defines the null F_0(t).
  - One-sided p-value:  p_j = (1 + #{null T >= T(S_j)}) / (n_null + 1).
  - e-value via Vovk-Wang (2021) calibrator: e = -log(p) - 1 + 1/p.

Validity: clusters were selected on A independently of B, so the null draws
from B are exchangeable with H_0 elements of A. Hence p_j is super-uniform
under H_0 ("S_j has organic-level cohesion") and the calibrator preserves
validity. e-BH on the resulting e-values controls FDR <= alpha under
arbitrary dependence among the {E_j}.

Caveat: if B contains coordinated content (it does — coordinated templates
straddle the A/B split), the null is biased upward (mixture) → p-values are
*conservative* (over-large) → e-values are conservative (smaller than ideal).
This costs power but preserves type-I control.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
EMB_PATH = PROC / "embeddings.npy"
CLUST_PATH = PROC / "clusters.parquet"
SPLIT_PATH = PROC / "split_assignment.parquet"
OUT_PATH = PROC / "cluster_evalues.parquet"


def cohesion(emb: np.ndarray, idx: np.ndarray, max_pairs: int, rng: np.random.Generator) -> float:
    """Mean pairwise cosine similarity for rows of `emb` indexed by `idx`."""
    n = len(idx)
    if n < 2:
        return 0.0
    if n * (n - 1) // 2 <= max_pairs:
        sub = emb[idx]
        sims = sub @ sub.T
        return float(sims[np.triu_indices(n, k=1)].mean())
    a = rng.integers(0, n, size=max_pairs)
    b = rng.integers(0, n, size=max_pairs)
    diff = a != b
    a, b = a[diff], b[diff]
    return float((emb[idx[a]] * emb[idx[b]]).sum(axis=1).mean())


def vovk_wang(p: np.ndarray) -> np.ndarray:
    """p-to-e calibrator: e = -log(p) - 1 + 1/p (valid for any super-uniform p)."""
    p = np.clip(p, 1e-300, 1.0)
    return -np.log(p) - 1.0 + 1.0 / p


def main(*, n_null_draws: int = 10_000, min_cluster_size: int = 5,
         max_pairs: int = 5000) -> None:
    print(f"loading embeddings from {EMB_PATH}...")
    emb = np.load(EMB_PATH).astype(np.float32)
    print(f"  shape {emb.shape}")

    cl = pq.read_table(CLUST_PATH).to_pandas()
    if not SPLIT_PATH.exists():
        raise SystemExit(f"missing {SPLIT_PATH}; run src/split.py first")
    split = pq.read_table(SPLIT_PATH).to_pandas()

    b_rows = np.where(split["split"].to_numpy() == "B")[0].astype(np.int64)
    print(f"calibration half (B): {len(b_rows):,} rows")
    if len(b_rows) < 1000:
        raise SystemExit("calibration set too small")

    cand = cl[(cl["cluster_id"] >= 0) & (cl["cluster_size"] >= min_cluster_size)]
    cand_groups = cand.groupby("cluster_id")
    sizes = cand_groups.size()
    print(f"candidate clusters with size >= {min_cluster_size}: {len(sizes):,}")
    if not len(sizes):
        return

    rng = np.random.default_rng(0)

    print("computing null T-distribution by size (drawing from B)...")
    unique_sizes = sorted(sizes.unique().tolist())
    null_by_size: dict[int, np.ndarray] = {}
    for s in unique_sizes:
        Ts = np.empty(n_null_draws, dtype=np.float32)
        for d in range(n_null_draws):
            pick = rng.choice(b_rows, size=int(s), replace=False)
            Ts[d] = cohesion(emb, pick, max_pairs=max_pairs, rng=rng)
        null_by_size[int(s)] = np.sort(Ts)

    print("computing observed T and p, e for each candidate cluster...")
    rows = []
    for cid, group in cand_groups:
        member_rows = group.index.to_numpy(dtype=np.int64)
        T_obs = cohesion(emb, member_rows, max_pairs=max_pairs, rng=rng)
        s = int(len(member_rows))
        null = null_by_size[s]
        # super-uniform p (one-sided right tail), with conservative +1 in numerator
        p = (1 + int((null >= T_obs).sum())) / (len(null) + 1)
        e = float(vovk_wang(np.array([p]))[0])
        rows.append({"cluster_id": int(cid), "n": s, "T_obs": float(T_obs),
                     "p": float(p), "e": e})

    out = pd.DataFrame(rows)
    out.to_parquet(OUT_PATH, compression="zstd", index=False)

    print(f"\nwrote {OUT_PATH}  ({len(out):,} candidate clusters)")
    print("p-value distribution:")
    for q in [0.05, 0.5, 0.95]:
        print(f"  q={q:.2f}: p={out['p'].quantile(q):.4g}")
    print("e-value distribution:")
    for q in [0.5, 0.9, 0.99]:
        print(f"  q={q:.2f}: e={out['e'].quantile(q):.3g}")
    print(f"  count e>1:  {(out['e'] > 1).sum():,}")
    print(f"  count e>20: {(out['e'] > 20).sum():,}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-null-draws", type=int, default=10_000)
    p.add_argument("--min-cluster-size", type=int, default=5)
    p.add_argument("--max-pairs", type=int, default=5000)
    args = p.parse_args()
    main(n_null_draws=args.n_null_draws,
         min_cluster_size=args.min_cluster_size, max_pairs=args.max_pairs)
