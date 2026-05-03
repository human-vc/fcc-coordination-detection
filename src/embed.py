"""Embed every unique comment_text in the comments table with MiniLM-L6-v2.

Output:
  data/processed/embeddings.npy      — (N, 384) float16 array
  data/processed/embedding_index.parquet — row_id, comment_id, template_size

template_size is the count of submissions sharing that comment_id (computed
from the submissions table). This is the partial-label proxy: comment_ids with
high template_size are gold-positive coordination; singletons are the
hard-regime population we want to detect soft coordination within.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from time import time

import duckdb
import numpy as np
import pyarrow.parquet as pq
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
EMB_PATH = PROC / "embeddings.npy"
IDX_PATH = PROC / "embedding_index.parquet"


def device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def default_batch_size(dev: str) -> int:
    return {"cuda": 1024, "mps": 256, "cpu": 64}[dev]


def main(*, batch_size: int | None = None, model_name: str = "all-MiniLM-L6-v2",
         limit: int | None = None) -> None:
    dev = device()
    if batch_size is None:
        batch_size = default_batch_size(dev)
    print(f"device: {dev}  batch_size: {batch_size}")

    print("loading template-size counts...")
    con = duckdb.connect()
    con.execute(f"CREATE VIEW c AS SELECT * FROM read_parquet('{PROC / 'comments.parquet'}')")
    con.execute(f"CREATE VIEW s AS SELECT * FROM read_parquet('{PROC / 'submissions.parquet'}')")

    # join: each unique comment + how many submissions reference it
    sql = """
        SELECT c.comment_id,
               c.comment_text,
               coalesce(t.n_submissions, 0) AS template_size
        FROM c
        LEFT JOIN (
            SELECT comment_id, count(*) AS n_submissions
            FROM s WHERE comment_id IS NOT NULL
            GROUP BY comment_id
        ) t USING (comment_id)
        ORDER BY c.row_id
    """
    if limit:
        sql += f" LIMIT {limit}"
    df = con.execute(sql).fetchdf()
    n = len(df)
    print(f"comments to embed: {n:,}")

    # save the index immediately so we can restart embedding if it crashes
    idx = df[["comment_id", "template_size"]].copy()
    idx["row_id"] = np.arange(n, dtype=np.int32)
    idx[["row_id", "comment_id", "template_size"]].to_parquet(
        IDX_PATH, compression="zstd", index=False
    )
    print(f"wrote index to {IDX_PATH}")

    print(f"loading model: {model_name}")
    model = SentenceTransformer(model_name, device=dev)
    dim = model.get_sentence_embedding_dimension()
    print(f"embedding dim: {dim}")

    # preallocate fp16 array
    out = np.zeros((n, dim), dtype=np.float16)
    texts = df["comment_text"].fillna("").tolist()

    t0 = time()
    written = 0
    for i in tqdm(range(0, n, batch_size), desc="embed"):
        batch = texts[i:i + batch_size]
        with torch.inference_mode():
            emb = model.encode(batch, batch_size=batch_size, convert_to_numpy=True,
                               show_progress_bar=False, normalize_embeddings=True)
        out[i:i + len(batch)] = emb.astype(np.float16)
        written += len(batch)
    elapsed = time() - t0
    print(f"\nembedded {written:,} in {elapsed:.1f}s ({written/elapsed:.0f}/s)")

    np.save(EMB_PATH, out)
    print(f"wrote embeddings to {EMB_PATH}  ({out.nbytes/1e6:.1f} MB)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=None,
                   help="default: 1024 on CUDA, 256 on MPS, 64 on CPU")
    p.add_argument("--model", default="all-MiniLM-L6-v2")
    p.add_argument("--limit", type=int, default=None,
                   help="for testing: embed only first N rows")
    args = p.parse_args()
    main(batch_size=args.batch_size, model_name=args.model, limit=args.limit)
