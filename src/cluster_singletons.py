"""Run Leiden community detection on the kNN similarity graph.

Output: data/processed/clusters.parquet with columns
    row_id, comment_id, template_size, cluster_id, cluster_size
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
OUT_PATH = PROC / "clusters.parquet"


def main(*, resolution: float = 1.0, min_cluster_size: int = 5) -> None:
    print(f"loading edges from {EDGES_PATH}...")
    edges = pq.read_table(EDGES_PATH).to_pandas()
    print(f"loaded {len(edges):,} edges")

    print(f"loading index from {IDX_PATH}...")
    idx = pq.read_table(IDX_PATH).to_pandas()
    n = len(idx)
    print(f"index has {n:,} rows")

    # Only nodes that appear in edges are in the active subgraph; everyone else is a singleton.
    active_rows = pd.unique(pd.concat([edges["src_row"], edges["dst_row"]]))
    print(f"active (non-isolated) rows: {len(active_rows):,}")

    # build mapping: original row_id -> compact graph idx
    row_to_gidx = pd.Series(np.arange(len(active_rows), dtype=np.int64), index=active_rows)
    src_g = row_to_gidx.loc[edges["src_row"].values].to_numpy()
    dst_g = row_to_gidx.loc[edges["dst_row"].values].to_numpy()
    weights = edges["similarity"].to_numpy()

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
    membership = np.array(parts.membership, dtype=np.int64)
    print(f"clusters found: {membership.max() + 1:,}")

    # map back to full index space; isolated nodes get cluster_id = -1
    cluster_for_row = np.full(n, -1, dtype=np.int64)
    cluster_for_row[active_rows] = membership

    out = idx.copy()
    out["cluster_id"] = cluster_for_row
    sizes = pd.Series(cluster_for_row).value_counts()
    out["cluster_size"] = out["cluster_id"].map(sizes).fillna(0).astype(np.int64)
    # don't count -1 isolates as a cluster of huge size
    out.loc[out["cluster_id"] == -1, "cluster_size"] = 1

    big = sizes[(sizes.index >= 0) & (sizes >= min_cluster_size)]
    print(f"clusters with size >= {min_cluster_size}: {len(big):,}")
    print(f"  total rows in those clusters: {int(big.sum()):,}")
    if len(big):
        print(f"  largest cluster: {int(big.iloc[0]):,} rows")

    out.to_parquet(OUT_PATH, compression="zstd", index=False)
    print(f"\nwrote {OUT_PATH}  ({OUT_PATH.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--resolution", type=float, default=1.0)
    p.add_argument("--min-cluster-size", type=int, default=5)
    args = p.parse_args()
    main(resolution=args.resolution, min_cluster_size=args.min_cluster_size)
