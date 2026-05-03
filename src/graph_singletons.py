"""Build a kNN similarity graph over the embedded comments.

For each comment, find the top-k nearest neighbors by cosine similarity
(embeddings are L2-normalized by sentence-transformers, so dot product = cosine).
Keep edges with similarity >= threshold; output sparse edge list.

Usage:
  python src/graph_singletons.py --k 50 --threshold 0.85
"""
from __future__ import annotations

import argparse
from pathlib import Path
from time import time

import faiss
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
EMB_PATH = PROC / "embeddings.npy"
IDX_PATH = PROC / "embedding_index.parquet"
EDGES_PATH = PROC / "knn_edges.parquet"


def main(*, k: int = 50, threshold: float = 0.85,
         restrict_singletons: bool = False) -> None:
    print(f"loading embeddings from {EMB_PATH}...")
    emb = np.load(EMB_PATH).astype(np.float32)  # FAISS needs fp32
    n, d = emb.shape
    print(f"loaded {n:,} embeddings, dim {d}")

    idx_df = pq.read_table(IDX_PATH).to_pandas()
    print(f"index has {len(idx_df):,} rows")

    if restrict_singletons:
        mask = idx_df["template_size"].to_numpy() <= 1
        sel = np.where(mask)[0]
        print(f"restricting to {len(sel):,} singleton rows (template_size<=1)")
        emb_q = emb[sel]
    else:
        sel = np.arange(n, dtype=np.int64)
        emb_q = emb

    print(f"building FAISS index ({len(emb_q):,} vectors)...")
    index = faiss.IndexFlatIP(d)  # inner product = cosine on normalized vecs
    index.add(emb_q)

    print(f"querying top-{k} neighbors...")
    t0 = time()
    sims, neighbors = index.search(emb_q, k + 1)  # +1 because self is included
    print(f"  search took {time()-t0:.1f}s")

    # drop self-edges and apply threshold
    src_rows, dst_rows, sim_vals = [], [], []
    for i in range(len(emb_q)):
        for j_pos in range(k + 1):
            j = neighbors[i, j_pos]
            s = sims[i, j_pos]
            if j == i or j < 0:
                continue
            if s < threshold:
                break  # neighbors are sorted by similarity desc
            # store undirected (smaller, larger) to dedupe later
            a, b = (sel[i], sel[j]) if sel[i] < sel[j] else (sel[j], sel[i])
            src_rows.append(int(a))
            dst_rows.append(int(b))
            sim_vals.append(float(s))

    print(f"\nraw edges (with duplicates): {len(src_rows):,}")

    # dedupe (a, b) pairs, keep max similarity
    edge_dict: dict[tuple[int, int], float] = {}
    for a, b, s in zip(src_rows, dst_rows, sim_vals):
        key = (a, b)
        if key not in edge_dict or s > edge_dict[key]:
            edge_dict[key] = s
    print(f"unique edges: {len(edge_dict):,}")

    if not edge_dict:
        print("no edges above threshold; lower --threshold")
        return

    src = np.fromiter((a for a, _ in edge_dict.keys()), dtype=np.int64, count=len(edge_dict))
    dst = np.fromiter((b for _, b in edge_dict.keys()), dtype=np.int64, count=len(edge_dict))
    sim = np.fromiter(edge_dict.values(), dtype=np.float32, count=len(edge_dict))

    table = pa.table({"src_row": src, "dst_row": dst, "similarity": sim})
    pq.write_table(table, EDGES_PATH, compression="zstd")
    print(f"\nwrote {EDGES_PATH}  ({EDGES_PATH.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=50)
    p.add_argument("--threshold", type=float, default=0.85)
    p.add_argument("--singletons", action="store_true",
                   help="restrict to template_size<=1 (the hard regime)")
    args = p.parse_args()
    main(k=args.k, threshold=args.threshold, restrict_singletons=args.singletons)
