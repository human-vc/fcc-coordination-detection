"""Baseline coordination-detection methods to compare against the
Leiden + e-BH pipeline.

Each baseline produces a clusters_<method>.parquet in the same schema as
clusters.parquet so the evalues.py / ebh.py / eval.py downstream stages can
run unchanged on top of it.

Baselines (each a recognized prior-art method for near-duplicate or coordination):
  - minhash_lsh: MinHash + LSH on word 5-grams. Classical near-duplicate
    detection (Broder 1997). Used in 2017 FCC astroturf analyses.
  - connected_components: threshold the kNN graph at sim>=threshold and take
    connected components. Simplest possible "graph + cluster" baseline; no
    Leiden modularity optimization.
  - hdbscan_emb: HDBSCAN density clustering directly on the embeddings, no
    graph at all. Tests whether the graph step adds anything.

All baselines run on the discovery (A) half only, matching the sample-split
protocol.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from time import time

import duckdb
import igraph as ig
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
IDX_PATH = PROC / "embedding_index.parquet"
SPLIT_PATH = PROC / "split_assignment.parquet"
EMB_PATH = PROC / "embeddings.npy"
EDGES_PATH = PROC / "knn_edges.parquet"
COMMENTS_PATH = PROC / "comments.parquet"


def load_split() -> tuple[pd.DataFrame, np.ndarray]:
    idx = pq.read_table(IDX_PATH).to_pandas()
    split = pq.read_table(SPLIT_PATH).to_pandas()
    a_mask = (split["split"].to_numpy() == "A")
    return idx, a_mask


def write_clusters(idx: pd.DataFrame, cluster_for_row: np.ndarray, name: str,
                   min_cluster_size: int) -> None:
    out = idx.copy()
    out["cluster_id"] = cluster_for_row
    sizes = pd.Series(cluster_for_row).value_counts()
    out["cluster_size"] = out["cluster_id"].map(sizes).fillna(1).astype(np.int64)
    out.loc[out["cluster_id"] < 0, "cluster_size"] = 1

    big = sizes[(sizes.index >= 0) & (sizes >= min_cluster_size)]
    print(f"  {name}: {len(big):,} clusters with size >= {min_cluster_size}; "
          f"largest = {int(big.iloc[0]) if len(big) else 0}")

    out_path = PROC / f"clusters_{name}.parquet"
    out.to_parquet(out_path, compression="zstd", index=False)
    print(f"  wrote {out_path}")


# --- baseline 1: MinHash + LSH ----------------------------------------------

_TOK = re.compile(r"\w+")


def shingle(text: str, n: int = 5) -> set[str]:
    toks = _TOK.findall((text or "").lower())
    if len(toks) < n:
        return {" ".join(toks)} if toks else set()
    return {" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def run_minhash_lsh(idx: pd.DataFrame, a_mask: np.ndarray, *,
                   threshold: float = 0.85, num_perm: int = 128,
                   shingle_n: int = 5, min_cluster_size: int = 5,
                   subsample: int | None = None) -> None:
    from datasketch import MinHash, MinHashLSH

    a_rows = np.where(a_mask)[0]
    if subsample and len(a_rows) > subsample:
        rng = np.random.default_rng(0)
        a_rows = rng.choice(a_rows, size=subsample, replace=False)
        print(f"[minhash_lsh] subsampled to {len(a_rows):,} A-rows for tractability")

    print(f"[minhash_lsh] loading comment_text via pandas merge...")
    comments = pq.read_table(COMMENTS_PATH, columns=["comment_id", "comment_text"]).to_pandas()
    a_idx = idx.iloc[a_rows][["row_id", "comment_id"]].copy()
    merged = a_idx.merge(comments, on="comment_id", how="left")
    merged["comment_text"] = merged["comment_text"].fillna("")
    print(f"  merged {len(merged):,} rows; missing text: {(merged['comment_text']=='').sum():,}")
    row_to_text = dict(zip(merged["row_id"].astype(int), merged["comment_text"]))

    print(f"[minhash_lsh] hashing {len(a_rows):,} comments (n={shingle_n} shingles, perm={num_perm})...")
    t0 = time()
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    minhashes: dict[int, "MinHash"] = {}
    for r in a_rows:
        text = row_to_text.get(int(r), "")
        sh = shingle(text, n=shingle_n)
        m = MinHash(num_perm=num_perm)
        for s in sh:
            m.update(s.encode("utf-8"))
        lsh.insert(str(int(r)), m)
        minhashes[int(r)] = m
    print(f"  hashed in {time()-t0:.1f}s")

    print("[minhash_lsh] querying LSH for near-duplicate pairs and building graph...")
    edges_a, edges_b = [], []
    for r, m in minhashes.items():
        cand = lsh.query(m)
        for c in cand:
            ci = int(c)
            if ci != r:
                a, b = (r, ci) if r < ci else (ci, r)
                edges_a.append(a); edges_b.append(b)

    if not edges_a:
        cluster_for_row = np.full(len(idx), -1, dtype=np.int64)
        cluster_for_row[~a_mask] = -2
        write_clusters(idx, cluster_for_row, "minhash_lsh", min_cluster_size)
        return

    df_edges = pd.DataFrame({"src": edges_a, "dst": edges_b}).drop_duplicates()
    nodes = pd.unique(pd.concat([df_edges["src"], df_edges["dst"]])).astype(np.int64)
    row_to_g = {int(r): i for i, r in enumerate(nodes)}
    src_g = df_edges["src"].map(row_to_g).to_numpy(dtype=np.int64)
    dst_g = df_edges["dst"].map(row_to_g).to_numpy(dtype=np.int64)
    g = ig.Graph(n=len(nodes), edges=list(zip(src_g, dst_g)), directed=False)
    g.simplify()
    cc = g.connected_components()
    membership = np.asarray(cc.membership, dtype=np.int64)

    cluster_for_row = np.full(len(idx), -1, dtype=np.int64)
    cluster_for_row[nodes] = membership
    cluster_for_row[~a_mask] = -2
    write_clusters(idx, cluster_for_row, "minhash_lsh", min_cluster_size)


# --- baseline 2: connected components on threshold graph --------------------

def run_connected_components(idx: pd.DataFrame, a_mask: np.ndarray, *,
                             min_cluster_size: int = 5) -> None:
    print("[connected_components] loading existing kNN edges and taking components...")
    edges = pq.read_table(EDGES_PATH).to_pandas()
    nodes = pd.unique(pd.concat([edges["src_row"], edges["dst_row"]])).astype(np.int64)
    row_to_g = {int(r): i for i, r in enumerate(nodes)}
    src_g = edges["src_row"].map(row_to_g).to_numpy(dtype=np.int64)
    dst_g = edges["dst_row"].map(row_to_g).to_numpy(dtype=np.int64)
    g = ig.Graph(n=len(nodes), edges=list(zip(src_g, dst_g)), directed=False)
    cc = g.connected_components()
    membership = np.asarray(cc.membership, dtype=np.int64)

    cluster_for_row = np.full(len(idx), -1, dtype=np.int64)
    cluster_for_row[nodes] = membership
    cluster_for_row[~a_mask] = -2
    write_clusters(idx, cluster_for_row, "connected_components", min_cluster_size)


# --- baseline 3: HDBSCAN on raw embeddings ----------------------------------

def run_hdbscan(idx: pd.DataFrame, a_mask: np.ndarray, *,
                min_samples: int = 5, min_cluster_size: int = 5,
                subsample: int = 200_000) -> None:
    import hdbscan

    print("[hdbscan_emb] loading embeddings...")
    emb = np.load(EMB_PATH).astype(np.float32)
    a_rows = np.where(a_mask)[0]
    if len(a_rows) > subsample:
        rng = np.random.default_rng(0)
        a_rows_sub = rng.choice(a_rows, size=subsample, replace=False)
        print(f"[hdbscan_emb] HDBSCAN does not scale to 1.9M; "
              f"subsampling to {subsample:,} A-rows (random seed=0)")
    else:
        a_rows_sub = a_rows
    sub = emb[a_rows_sub]
    print(f"[hdbscan_emb] running HDBSCAN on {len(sub):,} vectors...")
    t0 = time()
    clusterer = hdbscan.HDBSCAN(
        metric="euclidean", min_cluster_size=min_cluster_size,
        min_samples=min_samples, core_dist_n_jobs=-1,
        algorithm="boruvka_kdtree",
    )
    labels = clusterer.fit_predict(sub)
    print(f"  HDBSCAN done in {time()-t0:.1f}s")

    cluster_for_row = np.full(len(idx), -1, dtype=np.int64)
    cluster_for_row[a_rows_sub] = labels  # HDBSCAN already uses -1 for noise
    # rows in A but not in subsample → leave as -1 (treated as not-clustered)
    cluster_for_row[~a_mask] = -2
    write_clusters(idx, cluster_for_row, "hdbscan_emb", min_cluster_size)


def main(*, methods: list[str], threshold: float, min_cluster_size: int,
         minhash_subsample: int | None = 500_000,
         minhash_num_perm: int = 64,
         hdbscan_subsample: int = 200_000) -> None:
    idx, a_mask = load_split()

    if "minhash_lsh" in methods:
        run_minhash_lsh(idx, a_mask, threshold=threshold,
                        num_perm=minhash_num_perm,
                        min_cluster_size=min_cluster_size,
                        subsample=minhash_subsample)
    if "connected_components" in methods:
        run_connected_components(idx, a_mask, min_cluster_size=min_cluster_size)
    if "hdbscan_emb" in methods:
        run_hdbscan(idx, a_mask, min_cluster_size=min_cluster_size,
                    subsample=hdbscan_subsample)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--methods", nargs="+",
                   default=["minhash_lsh", "connected_components", "hdbscan_emb"],
                   choices=["minhash_lsh", "connected_components", "hdbscan_emb"])
    p.add_argument("--threshold", type=float, default=0.85,
                   help="MinHash Jaccard threshold (also used for kNN)")
    p.add_argument("--min-cluster-size", type=int, default=5)
    p.add_argument("--minhash-subsample", type=int, default=500_000,
                   help="cap MinHash to this many A-rows (OOM at 1.9M)")
    p.add_argument("--minhash-num-perm", type=int, default=64,
                   help="MinHash permutations (lower = less memory)")
    p.add_argument("--hdbscan-subsample", type=int, default=200_000,
                   help="HDBSCAN doesn't scale to >~300K; subsample first")
    args = p.parse_args()
    main(methods=args.methods, threshold=args.threshold,
         min_cluster_size=args.min_cluster_size,
         minhash_subsample=args.minhash_subsample,
         minhash_num_perm=args.minhash_num_perm,
         hdbscan_subsample=args.hdbscan_subsample)
