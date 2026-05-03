"""Build a kNN similarity graph over the embedded corpus.

Sample-splitting is enabled by default: edges are constructed only among rows
in the discovery half (A), so that downstream cluster discovery is independent
of the calibration half (B). The B half is reserved for null-distribution
estimation in evalues.py.

Output: edges between A-rows above similarity threshold; row indices are the
original embedding row_ids (not compacted).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from time import time

import faiss
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
EMB_PATH = PROC / "embeddings.npy"
SPLIT_PATH = PROC / "split_assignment.parquet"
EDGES_PATH = PROC / "knn_edges.parquet"


def main(*, k: int = 50, threshold: float = 0.85, half: str = "A") -> None:
    print(f"loading embeddings from {EMB_PATH}...")
    emb = np.load(EMB_PATH).astype(np.float32)
    n, d = emb.shape
    print(f"loaded {n:,} embeddings, dim {d}")

    if half == "full":
        sel = np.arange(n, dtype=np.int64)
    else:
        if not SPLIT_PATH.exists():
            raise SystemExit(f"missing {SPLIT_PATH}; run src/split.py first")
        split = pq.read_table(SPLIT_PATH).to_pandas()
        sel = np.where(split["split"].to_numpy() == half)[0].astype(np.int64)
        print(f"using half {half!r}: {len(sel):,} rows")

    emb_q = np.ascontiguousarray(emb[sel])

    print(f"building FAISS index ({len(emb_q):,} vectors)...")
    index = faiss.IndexFlatIP(d)
    index.add(emb_q)

    print(f"querying top-{k} neighbors...")
    t0 = time()
    sims, neighbors = index.search(emb_q, k + 1)
    print(f"  search took {time()-t0:.1f}s")

    # vectorized edge extraction with similarity threshold
    edges_a, edges_b, edges_s = [], [], []
    for col in range(k + 1):
        col_sims = sims[:, col]
        col_nbrs = neighbors[:, col]
        mask = (col_sims >= threshold) & (col_nbrs != np.arange(len(emb_q))) & (col_nbrs >= 0)
        i_local = np.where(mask)[0]
        if len(i_local) == 0:
            continue
        a = sel[i_local]
        b = sel[col_nbrs[i_local]]
        # canonical undirected order
        lo = np.minimum(a, b)
        hi = np.maximum(a, b)
        edges_a.append(lo)
        edges_b.append(hi)
        edges_s.append(col_sims[i_local])

    if not edges_a:
        print("no edges above threshold; lower --threshold")
        return

    src = np.concatenate(edges_a)
    dst = np.concatenate(edges_b)
    sim = np.concatenate(edges_s).astype(np.float32)
    print(f"raw edges (with duplicates): {len(src):,}")

    # dedupe (src, dst) pairs keeping max similarity, via pandas groupby
    df = pd.DataFrame({"src": src, "dst": dst, "sim": sim})
    df = df.groupby(["src", "dst"], as_index=False, sort=False)["sim"].max()
    print(f"unique edges: {len(df):,}")

    table = pa.table({
        "src_row": df["src"].to_numpy(dtype=np.int64),
        "dst_row": df["dst"].to_numpy(dtype=np.int64),
        "similarity": df["sim"].to_numpy(dtype=np.float32),
    })
    pq.write_table(table, EDGES_PATH, compression="zstd")
    print(f"\nwrote {EDGES_PATH}  ({EDGES_PATH.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=50)
    p.add_argument("--threshold", type=float, default=0.85)
    p.add_argument("--half", choices=["A", "B", "full"], default="A",
                   help="which half of the sample-split to build the graph on")
    args = p.parse_args()
    main(k=args.k, threshold=args.threshold, half=args.half)
