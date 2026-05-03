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


def main(*, resolution: float = 0.90, min_cluster_size: int = 5,
         partition: str = "cpm", out_path: Path | None = None) -> None:
    """Run Leiden community detection on the kNN similarity graph.

    partition='cpm' uses Constant Potts Model (resolution-limit-free; resolution
    parameter has the interpretation of a similarity threshold for the cosine-
    weighted graph). partition='rb' uses RBConfigurationVertexPartition, which
    has a known resolution limit on graphs with heterogeneous community sizes
    and produced the 213K mega-cluster failure mode in v1.

    See Traag, Van Dooren, Nesterov 2011 (arXiv:1104.3083) on RB resolution limits
    and the leidenalg docs for the CPM density interpretation.
    """
    out_path = out_path or OUT_PATH
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
    edge_array = np.column_stack((src_g, dst_g))  # avoid Python list of tuples (~6 GB at 50M edges)
    del edges

    g = ig.Graph(n=len(active_rows), edges=edge_array, directed=False)
    g.es["weight"] = weights.tolist()
    g.simplify(combine_edges={"weight": "max"})
    print(f"graph: {g.vcount():,} nodes, {g.ecount():,} edges")

    if partition == "cpm":
        partition_cls = la.CPMVertexPartition
        print(f"running Leiden / CPM (gamma={resolution}, similarity threshold)...")
    elif partition == "rb":
        partition_cls = la.RBConfigurationVertexPartition
        print(f"running Leiden / RBConfig (resolution={resolution}, has resolution limit)...")
    else:
        raise ValueError(f"unknown partition: {partition}")
    parts = la.find_partition(
        g, partition_cls,
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

    out.to_parquet(out_path, compression="zstd", index=False)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--resolution", type=float, default=0.90,
                   help="for CPM, similarity threshold in [0,1]; for RB, modularity scale")
    p.add_argument("--partition", choices=["cpm", "rb"], default="cpm",
                   help="cpm = Constant Potts (resolution-limit-free); rb = RBConfiguration")
    p.add_argument("--min-cluster-size", type=int, default=5)
    p.add_argument("--out-path", type=Path, default=None)
    args = p.parse_args()
    main(resolution=args.resolution, partition=args.partition,
         min_cluster_size=args.min_cluster_size, out_path=args.out_path)
