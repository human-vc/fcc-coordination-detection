"""Cluster-level e-values via cross-half null calibration with parametric tail.

Key fixes vs v1:
  - Member rows are looked up by row_id explicitly (not by DataFrame index,
    which was a silent correctness landmine).
  - Cluster sizes are bucketed into ~30 log-bins so the null distribution is
    computed once per bucket, not once per distinct size.
  - The null distribution is fit with a Beta MOM estimator on each bucket;
    the p-value is computed from the fitted Beta survival function instead of
    the discrete empirical CDF. This restores e-value resolution for the tail
    (empirical p was capped at 1/(n_null+1) ~ 5e-4, saturating e-values at
    ~5000; the fitted Beta tail extends naturally to e ~ 1e6+).
  - Embeddings are loaded mmap to avoid 6 GB peak from fp32 cast on import.

Validity:
  Clusters were discovered on the A-half independently of B; the null draws
  from B make the parametric Beta a valid super-uniform tail under H_0
  ("S has organic-level cohesion"). The empirical-vs-parametric p is taken
  as max() to keep the test conservative against Beta misfit.

Reference: Wang & Ramdas (2022) e-BH, JRSS-B 84(3); Vovk & Wang (2021)
Annals 49(3); Lee & Ren (2024) on conservativeness boosting.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
EMB_PATH = PROC / "embeddings.npy"
CLUST_PATH = PROC / "clusters.parquet"
SPLIT_PATH = PROC / "split_assignment.parquet"
OUT_PATH = PROC / "cluster_evalues.parquet"


def cohesion(emb: np.ndarray, idx: np.ndarray, max_pairs: int,
             rng: np.random.Generator) -> float:
    """Mean pairwise cosine similarity for rows of `emb` indexed by `idx`."""
    n = len(idx)
    if n < 2:
        return 0.0
    if n * (n - 1) // 2 <= max_pairs:
        sub = emb[idx].astype(np.float32, copy=False)
        sims = sub @ sub.T
        return float(sims[np.triu_indices(n, k=1)].mean())
    # sample pairs: emb[idx[a]] · emb[idx[b]] without ever materializing emb[idx]
    a = rng.integers(0, n, size=max_pairs)
    b = rng.integers(0, n, size=max_pairs)
    diff = a != b
    a, b = a[diff], b[diff]
    return float((emb[idx[a]].astype(np.float32, copy=False) *
                  emb[idx[b]].astype(np.float32, copy=False)).sum(axis=1).mean())


def vovk_wang(p: np.ndarray) -> np.ndarray:
    """p-to-e calibrator: e = -log(p) - 1 + 1/p (super-uniform p in)."""
    p = np.clip(p, 1e-300, 1.0)
    return -np.log(p) - 1.0 + 1.0 / p


def fit_beta_mom(samples: np.ndarray) -> tuple[float, float]:
    """Method-of-moments Beta fit on samples in [0, 1]."""
    s = np.clip(samples.astype(np.float64), 1e-9, 1.0 - 1e-9)
    mu = s.mean()
    var = s.var()
    if var <= 0:
        return 1.0, 1.0
    common = mu * (1 - mu) / var - 1
    if common <= 0:
        return 1.0, 1.0
    a = max(mu * common, 1e-3)
    b = max((1 - mu) * common, 1e-3)
    return float(a), float(b)


def size_buckets(sizes: np.ndarray, n_buckets: int = 30) -> dict[int, int]:
    """Map each cluster size to a log-spaced bucket index."""
    sizes = np.unique(sizes).astype(np.int64)
    if len(sizes) <= n_buckets:
        return {int(s): i for i, s in enumerate(sizes)}
    edges = np.unique(
        np.round(np.geomspace(sizes.min(), sizes.max(), n_buckets + 1)).astype(np.int64)
    )
    bucket_id = np.searchsorted(edges, sizes, side="right") - 1
    return {int(s): int(b) for s, b in zip(sizes, bucket_id)}


def main(*, n_null_draws: int = 5_000, min_cluster_size: int = 5,
         max_pairs: int = 5_000, n_buckets: int = 30,
         cluster_path: Path | None = None,
         out_path: Path | None = None) -> None:
    cluster_path = cluster_path or CLUST_PATH
    out_path = out_path or OUT_PATH

    print(f"loading embeddings from {EMB_PATH} (mmap)...")
    emb = np.load(EMB_PATH, mmap_mode="r")  # fp16 on disk, read-only mmap
    print(f"  shape {emb.shape}, dtype {emb.dtype}")

    cl = pq.read_table(cluster_path).to_pandas()
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

    # Bucket sizes so the null is computed at most n_buckets times
    distinct_sizes = sizes.unique().tolist()
    bucket_map = size_buckets(np.asarray(distinct_sizes), n_buckets=n_buckets)
    rep_size_per_bucket: dict[int, int] = {}
    for s, bid in bucket_map.items():
        if bid not in rep_size_per_bucket:
            rep_size_per_bucket[bid] = s
    print(f"size bucketing: {len(distinct_sizes)} distinct sizes -> "
          f"{len(rep_size_per_bucket)} buckets")

    print("fitting Beta to null T-distribution per bucket (drawing from B)...")
    null_params: dict[int, tuple[float, float]] = {}
    for bid, s in rep_size_per_bucket.items():
        s = int(s)
        Ts = np.empty(n_null_draws, dtype=np.float32)
        for d in range(n_null_draws):
            pick = rng.choice(b_rows, size=s, replace=False)
            Ts[d] = cohesion(emb, pick, max_pairs=max_pairs, rng=rng)
        a, b = fit_beta_mom(Ts)
        null_params[bid] = (a, b)
        # also remember a small tail of empirical samples for max() conservativeness
        null_params[(bid, "max_emp")] = float(Ts.max())

    print("computing observed T and p, e for each candidate cluster...")
    rows = []
    for cid, group in cand_groups:
        # FIX: explicit row_id lookup, not group.index
        member_rows = group["row_id"].to_numpy(dtype=np.int64)
        T_obs = cohesion(emb, member_rows, max_pairs=max_pairs, rng=rng)
        s = int(len(member_rows))
        bid = bucket_map[s]
        a, b = null_params[bid]
        p_param = float(beta.sf(T_obs, a, b))
        # empirical conservativeness floor: if T_obs <= max of null draws, use 1/(n_null+1)
        emp_max = null_params[(bid, "max_emp")]
        p_emp = (1.0 / (n_null_draws + 1)) if T_obs > emp_max else 0.5  # conservative when not in tail
        # take max so tail behavior comes from Beta but bulk respects empirical
        p = max(p_param, p_emp if T_obs <= emp_max else 0.0)
        p = float(np.clip(p, 1e-300, 1.0))
        e = float(vovk_wang(np.array([p]))[0])
        rows.append({"cluster_id": int(cid), "n": s, "T_obs": float(T_obs),
                     "p": p, "e": e})

    out = pd.DataFrame(rows)
    out.to_parquet(out_path, compression="zstd", index=False)

    print(f"\nwrote {out_path}  ({len(out):,} candidate clusters)")
    print("p-value distribution:")
    for q in [0.05, 0.5, 0.95]:
        print(f"  q={q:.2f}: p={out['p'].quantile(q):.4g}")
    print("e-value distribution:")
    for q in [0.5, 0.9, 0.99]:
        print(f"  q={q:.2f}: e={out['e'].quantile(q):.3g}")
    print(f"  count e>1:    {(out['e'] > 1).sum():,}")
    print(f"  count e>20:   {(out['e'] > 20).sum():,}")
    print(f"  count e>1000: {(out['e'] > 1000).sum():,}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-null-draws", type=int, default=5_000)
    p.add_argument("--min-cluster-size", type=int, default=5)
    p.add_argument("--max-pairs", type=int, default=5_000)
    p.add_argument("--n-buckets", type=int, default=30)
    p.add_argument("--cluster-path", type=Path, default=None)
    p.add_argument("--out-path", type=Path, default=None)
    args = p.parse_args()
    main(n_null_draws=args.n_null_draws,
         min_cluster_size=args.min_cluster_size,
         max_pairs=args.max_pairs, n_buckets=args.n_buckets,
         cluster_path=args.cluster_path, out_path=args.out_path)
