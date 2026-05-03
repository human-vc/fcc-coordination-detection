"""Build a coordination graph from the precomputed near/exact duplicate edges.

The slnader/fcc-comments dataset already contains near_duplicates and
exact_duplicates tables — these are the strongest possible coordination signal
and are essentially free.  We build an undirected graph at the document level,
then map back to submissions and run Leiden community detection to identify
candidate coordination clusters.

Output: data/processed/coordination_clusters.parquet with columns
    submission_id, document_id, cluster_id, cluster_size
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import igraph as ig
import leidenalg as la
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"


def main(*, min_cluster_size: int = 5, edge_kind: str = "near") -> None:
    """Build duplicate-edge graph and detect clusters.

    edge_kind ∈ {'near', 'exact', 'both'} — which precomputed edge tables to use.
    """
    con = duckdb.connect()
    for p in PROC.glob("*.parquet"):
        con.execute(f"CREATE OR REPLACE VIEW {p.stem} AS SELECT * FROM read_parquet('{p}')")

    if edge_kind == "near":
        edges_q = "SELECT target_document_id, duplicate_document_id FROM near_duplicates"
    elif edge_kind == "exact":
        edges_q = "SELECT target_document_id, duplicate_document_id FROM exact_duplicates"
    else:
        edges_q = ("SELECT target_document_id, duplicate_document_id FROM near_duplicates "
                   "UNION SELECT target_document_id, duplicate_document_id FROM exact_duplicates")

    edges = con.execute(edges_q).fetchdf()
    print(f"loaded {len(edges):,} duplicate edges ({edge_kind})")

    nodes = pd.unique(pd.concat([edges["target_document_id"], edges["duplicate_document_id"]]))
    print(f"unique documents in graph: {len(nodes):,}")

    node_to_idx = {n: i for i, n in enumerate(nodes)}
    src = edges["target_document_id"].map(node_to_idx).to_numpy()
    dst = edges["duplicate_document_id"].map(node_to_idx).to_numpy()

    g = ig.Graph(n=len(nodes), edges=list(zip(src, dst)), directed=False)
    g.simplify()
    print(f"graph: {g.vcount():,} nodes, {g.ecount():,} edges")

    print("running Leiden community detection...")
    parts = la.find_partition(g, la.ModularityVertexPartition, n_iterations=-1, seed=42)
    print(f"clusters found: {len(parts):,}")

    membership = parts.membership
    cluster_sizes = pd.Series(membership).value_counts()
    big = cluster_sizes[cluster_sizes >= min_cluster_size]
    print(f"clusters with size >= {min_cluster_size}: {len(big):,}")
    print(f"  total docs in those clusters: {big.sum():,}")
    print(f"  largest cluster: {cluster_sizes.iloc[0]:,} docs")

    out = pd.DataFrame({
        "document_id": nodes,
        "cluster_id": membership,
    })
    out["cluster_size"] = out["cluster_id"].map(cluster_sizes)

    docs_to_subs = con.execute(
        "SELECT submission_id, document_id FROM documents"
    ).fetchdf()
    out = out.merge(docs_to_subs, on="document_id", how="left")

    out_path = PROC / f"coordination_clusters_{edge_kind}.parquet"
    out.to_parquet(out_path, compression="zstd")
    print(f"\nwrote {out_path}  ({len(out):,} rows)")


if __name__ == "__main__":
    main()
