"""Cluster-level mixture-LR compound e-values (Ignatiadis–Wang–Ramdas §7).

Uses BuzzFeed-FOIA-attributed clusters as anchors to fit g_1 (the
coordinated-cluster cohesion density), and cross-half random clusters to fit
g_0 (the null cohesion density), both as Beta(α, β) per size bucket.

Per-cluster e-value:
    E_c = g_1(T_c | n_c) / g_0(T_c | n_c)

This is a (compound) e-value:  ∫ g_1(t)/g_0(t) · g_0(t) dt = ∫ g_1(t) dt = 1.
Hence FDR ≤ α via Wang–Ramdas e-BH.

Compared to src/evalues.py (Beta-tail p→e via Vovk-Wang calibrator), this
construction:
  - Uses external label supervision (FOIA-attributed clusters) to inform g_1.
  - Skips the lossy p→e calibration step.
  - Yields a likelihood-ratio interpretation directly.

Output: data/processed/cluster_evalues_mixlr.parquet
        cluster_id, n, T_obs, log_e, e, log_g0, log_g1
"""
from __future__ import annotations

import argparse
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import beta as beta_dist

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
DEFAULT_EMB = PROC / "embeddings.npy"  # use raw embeddings (matches existing T_obs)
DEFAULT_CLUST = PROC / "clusters_leiden_r0.9.parquet"
DEFAULT_TOBS = PROC / "cluster_evalues_leiden_r0.9.parquet"
DEFAULT_ATTR = RES / "attribution_table_r0.9.csv"
DEFAULT_SPLIT = PROC / "split_assignment.parquet"
DEFAULT_OUT = PROC / "cluster_evalues_mixlr.parquet"


def cohesion(emb: np.ndarray, idx: np.ndarray, max_pairs: int,
             rng: np.random.Generator) -> float:
    """Mean pairwise cosine similarity (matches src/evalues.py)."""
    n = len(idx)
    if n < 2:
        return 0.0
    if n * (n - 1) // 2 <= max_pairs:
        sub = emb[idx].astype(np.float32, copy=False)
        sims = sub @ sub.T
        return float(sims[np.triu_indices(n, k=1)].mean())
    a = rng.integers(0, n, size=max_pairs)
    b = rng.integers(0, n, size=max_pairs)
    diff = a != b
    a, b = a[diff], b[diff]
    return float((emb[idx[a]].astype(np.float32, copy=False) *
                  emb[idx[b]].astype(np.float32, copy=False)).sum(axis=1).mean())


def fit_beta_mom(samples: np.ndarray, *, eps: float = 1e-6) -> tuple[float, float]:
    """Method-of-moments Beta fit on samples in (0, 1)."""
    s = np.clip(samples.astype(np.float64), eps, 1.0 - eps)
    mu, var = float(s.mean()), float(s.var())
    if var <= eps:
        return 1.0, 1.0
    common = mu * (1 - mu) / var - 1
    if common <= 0:
        return 1.0, 1.0
    a = max(mu * common, 1e-3)
    b = max((1 - mu) * common, 1e-3)
    return float(a), float(b)


def size_buckets(sizes: np.ndarray, n_buckets: int = 30) -> dict[int, int]:
    sizes = np.unique(sizes).astype(np.int64)
    if len(sizes) <= n_buckets:
        return {int(s): i for i, s in enumerate(sizes)}
    edges = np.unique(
        np.round(np.geomspace(sizes.min(), sizes.max(), n_buckets + 1)).astype(np.int64)
    )
    bucket_id = np.searchsorted(edges, sizes, side="right") - 1
    return {int(s): int(b) for s, b in zip(sizes, bucket_id)}


def main(*, embedding_path: Path | None = None,
         cluster_path: Path | None = None,
         tobs_path: Path | None = None,
         attr_path: Path | None = None,
         split_path: Path | None = None,
         output_path: Path | None = None,
         n_null_draws: int = 2_000,
         max_pairs: int = 5_000,
         n_buckets: int = 30,
         seed: int = 0) -> None:
    emb_path = embedding_path or DEFAULT_EMB
    cl_path = cluster_path or DEFAULT_CLUST
    tobs_path = tobs_path or DEFAULT_TOBS
    attr_path = attr_path or DEFAULT_ATTR
    split_path = split_path or DEFAULT_SPLIT
    out_path = output_path or DEFAULT_OUT

    print(f"loading embeddings (mmap): {emb_path}")
    emb = np.load(emb_path, mmap_mode="r")
    print(f"  shape {emb.shape}  dtype {emb.dtype}")

    print(f"loading clusters: {cl_path}")
    cl = pq.read_table(cl_path).to_pandas()

    print(f"loading T_obs from existing pipeline: {tobs_path}")
    tobs = pq.read_table(tobs_path).to_pandas()
    print(f"  {len(tobs):,} clusters with T_obs")

    print(f"loading attribution labels: {attr_path}")
    att = pd.read_csv(attr_path)
    print(f"  {len(att):,} attribution rows")

    print(f"loading B-half split: {split_path}")
    split = pq.read_table(split_path).to_pandas()
    b_rows = np.where(split["split"].to_numpy() == "B")[0].astype(np.int64)
    print(f"  B-half: {len(b_rows):,} rows")

    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # 1. Fit g_0 per size bucket via random null draws from B-half
    # ------------------------------------------------------------------
    sizes = tobs["n"].to_numpy()
    distinct_sizes = np.unique(sizes)
    bucket_map = size_buckets(distinct_sizes, n_buckets=n_buckets)
    rep_size_per_bucket: dict[int, int] = {}
    for s, bid in bucket_map.items():
        if bid not in rep_size_per_bucket:
            rep_size_per_bucket[bid] = int(s)
    n_b = len(rep_size_per_bucket)
    print(f"\nfitting g_0 (Beta MOM null) on {n_b} size buckets, "
          f"{n_null_draws} draws each...")
    g0_params: dict[int, tuple[float, float]] = {}
    null_max: dict[int, float] = {}
    t0 = time()
    for bid, s in rep_size_per_bucket.items():
        Ts = np.empty(n_null_draws, dtype=np.float32)
        for d in range(n_null_draws):
            pick = rng.choice(b_rows, size=s, replace=False)
            Ts[d] = cohesion(emb, pick, max_pairs=max_pairs, rng=rng)
        a0, b0 = fit_beta_mom(Ts)
        g0_params[bid] = (a0, b0)
        null_max[bid] = float(Ts.max())
    print(f"  g_0 fit done   ({time() - t0:.1f}s)")

    # ------------------------------------------------------------------
    # 2. Identify labeled astroturf clusters; fit g_1 per size bucket
    # ------------------------------------------------------------------
    labeled = tobs.merge(att[["cluster_id", "frac_astroturf", "frac_advocacy",
                              "top_label"]],
                          on="cluster_id", how="left")
    astroturf_mask = (labeled["frac_astroturf"].fillna(0.0) >= 0.5)
    advocacy_mask = (labeled["frac_advocacy"].fillna(0.0) >= 0.5)
    coord_mask = astroturf_mask | advocacy_mask
    print(f"\nlabeled clusters: astroturf>=0.5: {astroturf_mask.sum():,}, "
          f"advocacy>=0.5: {advocacy_mask.sum():,}, "
          f"either coord: {coord_mask.sum():,}")

    coord_df = labeled[coord_mask]

    print(f"fitting g_1 (Beta MOM on coordinated clusters) per bucket...")
    g1_params: dict[int, tuple[float, float]] = {}
    n_per_bucket: dict[int, int] = {}
    for bid, s in rep_size_per_bucket.items():
        # use all coord clusters whose size maps to this bucket
        bucket_sizes_set = {int(ss) for ss, bb in bucket_map.items() if bb == bid}
        sub = coord_df[coord_df["n"].isin(bucket_sizes_set)]
        n_per_bucket[bid] = len(sub)
        if len(sub) >= 5:
            a1, b1 = fit_beta_mom(sub["T_obs"].to_numpy())
        else:
            # fallback: pool all coord clusters across buckets
            a1, b1 = fit_beta_mom(coord_df["T_obs"].to_numpy())
        g1_params[bid] = (a1, b1)

    print(f"  bucket-level coord counts: median {np.median(list(n_per_bucket.values())):.0f}, "
          f"min {min(n_per_bucket.values())}, max {max(n_per_bucket.values())}")
    pooled_a1, pooled_b1 = fit_beta_mom(coord_df["T_obs"].to_numpy())
    print(f"  pooled g_1 (Beta): a={pooled_a1:.2f}, b={pooled_b1:.2f}, "
          f"mean={pooled_a1/(pooled_a1+pooled_b1):.4f}")
    pooled_a0, pooled_b0 = fit_beta_mom(np.concatenate([
        np.array([cohesion(emb,
                          rng.choice(b_rows, size=8, replace=False),
                          max_pairs=max_pairs, rng=rng)
                 for _ in range(2000)])
    ]))
    print(f"  pooled g_0 (size~8): a={pooled_a0:.2f}, b={pooled_b0:.2f}, "
          f"mean={pooled_a0/(pooled_a0+pooled_b0):.4f}")

    # ------------------------------------------------------------------
    # 3. Compute mixture-LR e-value per cluster
    # ------------------------------------------------------------------
    print(f"\ncomputing per-cluster e-values...")
    rows = []
    for _, r in tobs.iterrows():
        s = int(r["n"])
        bid = bucket_map[s]
        a0, b0 = g0_params[bid]
        a1, b1 = g1_params[bid]
        T = float(r["T_obs"])
        # clip to (eps, 1-eps) for numerical stability of Beta logpdf
        T_c = float(np.clip(T, 1e-6, 1.0 - 1e-6))
        log_g0 = float(beta_dist.logpdf(T_c, a0, b0))
        log_g1 = float(beta_dist.logpdf(T_c, a1, b1))
        log_e = log_g1 - log_g0
        # safety: cap at +/-700 to avoid overflow on exp
        log_e_capped = float(np.clip(log_e, -700.0, 700.0))
        e = float(np.exp(log_e_capped))
        rows.append({
            "cluster_id": int(r["cluster_id"]),
            "n": s, "T_obs": T,
            "log_g0": log_g0, "log_g1": log_g1,
            "log_e": log_e, "e": e,
            "bucket": bid,
        })

    out = pd.DataFrame(rows)
    out.to_parquet(out_path, compression="zstd", index=False)
    print(f"\nwrote {out_path}  ({len(out):,} clusters)")

    # ------------------------------------------------------------------
    # 4. Diagnostic: e-value distribution by attribution group
    # ------------------------------------------------------------------
    print()
    out_lab = out.merge(att[["cluster_id", "frac_astroturf",
                              "frac_advocacy", "top_label"]],
                         on="cluster_id", how="left")
    grp_mask = {
        "all":           out_lab["log_e"] == out_lab["log_e"],
        "astroturf>=0.5": out_lab["frac_astroturf"].fillna(0.0) >= 0.5,
        "advocacy>=0.5": out_lab["frac_advocacy"].fillna(0.0) >= 0.5,
        "unlabeled":     out_lab["top_label"].isna(),
    }
    print("e-value distribution (log) by attribution group:")
    for name, mask in grp_mask.items():
        sub = out_lab[mask]
        if len(sub) == 0:
            continue
        print(f"  {name:18s}  n={len(sub):,}  "
              f"log_e p10={sub['log_e'].quantile(0.1):6.2f}  "
              f"p50={sub['log_e'].quantile(0.5):6.2f}  "
              f"p90={sub['log_e'].quantile(0.9):6.2f}  "
              f"p99={sub['log_e'].quantile(0.99):6.2f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--embedding-path", type=Path, default=None)
    p.add_argument("--cluster-path", type=Path, default=None)
    p.add_argument("--tobs-path", type=Path, default=None)
    p.add_argument("--attr-path", type=Path, default=None)
    p.add_argument("--split-path", type=Path, default=None)
    p.add_argument("--output-path", type=Path, default=None)
    p.add_argument("--n-null-draws", type=int, default=2_000)
    p.add_argument("--max-pairs", type=int, default=5_000)
    p.add_argument("--n-buckets", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    main(embedding_path=args.embedding_path,
         cluster_path=args.cluster_path,
         tobs_path=args.tobs_path,
         attr_path=args.attr_path,
         split_path=args.split_path,
         output_path=args.output_path,
         n_null_draws=args.n_null_draws,
         max_pairs=args.max_pairs,
         n_buckets=args.n_buckets,
         seed=args.seed)
