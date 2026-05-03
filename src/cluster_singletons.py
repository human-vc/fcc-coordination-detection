"""Run Leiden community detection on the kNN similarity graph.

Operates on whichever half graph_singletons.py wrote (default: A). Rows in the
held-out half receive cluster_id = -2; non-clustered active rows get -1.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import igraph as ig
import leidenalg as la
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
EDGES_PATH = PROC / "knn_edges.parquet"
IDX_PATH = PROC / "embedding_index.parquet"
SPLIT_PATH = PROC / "split_assignment.parquet"
OUT_PATH = PROC / "clusters.parquet"


def main(*, resolution: float = 1.0, min_cluster_size: int = 5) -> None:
    print(f"loading edges from {EDGES_PATH}...")
    edges = pq.read_table(EDGES_PATH).to_pandas()
    print(f"loaded {len(edges):,} edges")

    idx = pq.read_table(IDX_PATH).to_pandas()
    n = len(idx)
    print(f"corpus size: {n:,}")

    if SPLIT_PATH.exists():
        split = pq.read_table(SPLIT_PATH).to_pandas()
        held_out = (split["split"].to_numpy() == "B")
    else:
        held_out = np.zeros(n, dtype=bool)

    active_rows = pd.unique(pd.concat([edges["src_row"], edges["dst_row"]])).astype(np.int64)
    print(f"active (with edges): {len(active_rows):,}")

    row_to_g = {int(r): i for i, r in enumerate(active_rows)}
    src_g = edges["src_row"].map(row_to_g).to_numpy(dtype=np.int64)
    dst_g = edges["dst_row"].map(row_to_g).to_numpy(dtype=np.int64)
    weights = edges["similarity"].to_numpy(dtype=np.float32)

    g = ig.Graph(n=len(active_rows), edges=list(zip(src_g, dst_g)), directed=False)
    g.es["weight"] = weights.tolist()
    g.simplify(combine_edges={"weight": "max"})
    print(f"graph: {g.vcount():,} nodes, {g.ecount():,} edges")

    print(f"running Leiden (resolution={resolution})...")
    parts = la.find_partition(
        g, la.RBConfigurationVertexPartition,
        weights="weight", resolution_parameter=resolution,
        n_iterations=-1, seed=42,
    )
    membership = np.asarray(parts.membership, dtype=np.int64)
    n_clusters = int(membership.max() + 1) if len(membership) else 0
    print(f"clusters found: {n_clusters:,}")

    cluster_for_row = np.full(n, -1, dtype=np.int64)
    cluster_for_row[active_rows] = membership
    cluster_for_row[held_out] = -2

    out = idx.copy()
    out["cluster_id"] = cluster_for_row
    sizes = pd.Series(cluster_for_row).value_counts()
    out["cluster_size"] = out["cluster_id"].map(sizes).fillna(1).astype(np.int64)
    out.loc[out["cluster_id"] < 0, "cluster_size"] = 1

    big = sizes[(sizes.index >= 0) & (sizes >= min_cluster_size)]
    print(f"clusters with size >= {min_cluster_size}: {len(big):,}")
    if len(big):
        print(f"  total rows in those clusters: {int(big.sum()):,}")
        print(f"  largest cluster: {int(big.iloc[0]):,} rows")

    out.to_parquet(OUT_PATH, compression="zstd", index=False)
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--resolution", type=float, default=1.0)
    p.add_argument("--min-cluster-size", type=int, default=5)
    args = p.parse_args()
    main(resolution=args.resolution, min_cluster_size=args.min_cluster_size)
