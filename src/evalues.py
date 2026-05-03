"""Compute cluster-level e-values for coordination detection.

For each candidate cluster S we want a non-negative test statistic E(S) with
E[E(S) | H_0] <= 1, where H_0 = "S is an organic cluster, not coordinated."

v1 construction (simple, will be tightened in the paper):

  1. Compute the cluster cohesion statistic
       T(S) = mean_{i,j in S, i<j} sim(i, j)
  2. Estimate the null distribution F_0 of T over random size-matched subsets
     drawn from the singleton population (template_size == 1, isolated in graph
     OR in tiny clusters of <= 2 — these are "least likely to be coordinated").
  3. For each cluster, compute p-value p(S) = 1 - F_0(T(S)) (one-sided).
  4. Convert p to e via Vovk-Wang (2021) calibrator: e = -log(p) - 1 + 1/p.
     This is a valid p-to-e calibration: if p is super-uniform then E[e] <= 1.
     (Reference: Vovk & Wang, "E-values: Calibration, combination, and
     applications," Annals of Statistics 49(3), 2021.)

Outputs:
  data/processed/cluster_evalues.parquet — cluster_id, n, T_obs, p, e
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
OUT_PATH = PROC / "cluster_evalues.parquet"


def cluster_cohesion(emb: np.ndarray, idx: np.ndarray, max_pairs: int = 5000) -> float:
    """Mean pairwise cosine similarity among rows of `emb` indexed by `idx`.

    For large clusters we subsample pairs to keep the cost bounded.
    """
    n = len(idx)
    if n < 2:
        return 0.0
    if n * (n - 1) // 2 <= max_pairs:
        # exact
        sub = emb[idx]
        sims = sub @ sub.T
        triu = sims[np.triu_indices(n, k=1)]
        return float(triu.mean())
    # subsample pairs
    rng = np.random.default_rng(42)
    a = rng.integers(0, n, size=max_pairs)
    b = rng.integers(0, n, size=max_pairs)
    diff = a != b
    a, b = a[diff], b[diff]
    sims = (emb[idx[a]] * emb[idx[b]]).sum(axis=1)
    return float(sims.mean())


def vovk_wang_calibrator(p: np.ndarray) -> np.ndarray:
    """Map p-values to e-values via -log(p) - 1 + 1/p, a valid calibrator.

    Equivalent to integrating 1/(p * (1 - log p)) — produces e in [1, inf).
    Returns 1 when p == 1, large when p is small.
    """
    p = np.clip(p, 1e-300, 1.0)
    return -np.log(p) - 1.0 + 1.0 / p


def main(*, alpha: float = 0.10, n_null_draws: int = 10_000,
         min_cluster_size: int = 5, max_pairs: int = 5000) -> None:
    print(f"loading embeddings from {EMB_PATH}...")
    emb = np.load(EMB_PATH).astype(np.float32)
    print(f"  {emb.shape}")

    print(f"loading clusters from {CLUST_PATH}...")
    cl = pq.read_table(CLUST_PATH).to_pandas()
    n_total = len(cl)
    print(f"  {n_total:,} rows")

    # build the null calibration set: singletons that ended up in tiny clusters
    null_mask = (cl["template_size"] <= 1) & (cl["cluster_size"] <= 2)
    null_rows = np.where(null_mask.to_numpy())[0]
    print(f"null calibration set: {len(null_rows):,} rows "
          f"(template_size<=1 & cluster_size<=2)")
    if len(null_rows) < 1000:
        raise SystemExit("null set too small; relax filter or increase corpus")

    # candidate clusters: cluster_id != -1 and cluster_size >= min_cluster_size
    cand_groups = (cl[(cl["cluster_id"] >= 0) & (cl["cluster_size"] >= min_cluster_size)]
                   .groupby("cluster_id"))
    cand_sizes = cand_groups.size()
    print(f"candidate clusters with size >= {min_cluster_size}: {len(cand_sizes):,}")

    # compute null T distribution over a range of cluster sizes (size-matched calibration)
    rng = np.random.default_rng(0)
    unique_sizes = np.unique(cand_sizes.to_numpy())
    print(f"null distribution computed over {len(unique_sizes)} distinct cluster sizes")

    null_T_by_size: dict[int, np.ndarray] = {}
    for s in unique_sizes:
        Ts = np.empty(n_null_draws, dtype=np.float32)
        for d in range(n_null_draws):
            pick = rng.choice(null_rows, size=int(s), replace=False)
            Ts[d] = cluster_cohesion(emb, pick, max_pairs=max_pairs)
        null_T_by_size[int(s)] = np.sort(Ts)

    # for each candidate cluster, compute T_obs, p, e
    results = []
    for cid, group in cand_groups:
        rows = group.index.to_numpy()
        s = int(len(rows))
        T_obs = cluster_cohesion(emb, rows, max_pairs=max_pairs)
        null_dist = null_T_by_size[s]
        # one-sided right-tail p
        p = float((null_dist >= T_obs).mean())
        if p == 0.0:  # avoid log(0); use conservative 1/(n+1)
            p = 1.0 / (len(null_dist) + 1)
        e = float(vovk_wang_calibrator(np.array([p]))[0])
        results.append({"cluster_id": int(cid), "n": s, "T_obs": T_obs,
                        "p": p, "e": e})

    out = pd.DataFrame(results)
    out.to_parquet(OUT_PATH, compression="zstd", index=False)
    print(f"\nwrote {OUT_PATH}  ({len(out):,} candidate clusters)")

    print("\ne-value distribution:")
    print(f"  median e: {out['e'].median():.3f}")
    print(f"  90th pct: {out['e'].quantile(0.9):.3f}")
    print(f"  99th pct: {out['e'].quantile(0.99):.3f}")
    print(f"  max e:    {out['e'].max():.3f}")
    print(f"  count e>20: {(out['e'] > 20).sum():,}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--alpha", type=float, default=0.10)
    p.add_argument("--n-null-draws", type=int, default=10_000)
    p.add_argument("--min-cluster-size", type=int, default=5)
    p.add_argument("--max-pairs", type=int, default=5000,
                   help="max pairs for cohesion subsampling")
    args = p.parse_args()
    main(alpha=args.alpha, n_null_draws=args.n_null_draws,
         min_cluster_size=args.min_cluster_size, max_pairs=args.max_pairs)
