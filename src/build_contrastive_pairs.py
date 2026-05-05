"""Build contrastive training pairs from BuzzFeed bulk-uploads.

Positive pairs: two comments uploaded by the SAME Box.com account (same
coordination source — synonymized paraphrases of one campaign template).

Hard negative pairs: two comments uploaded by DIFFERENT accounts but with
high baseline cosine similarity (topically related but stylistically
different sources).

This is weak supervision: an uploader account is a noisy proxy for "same
source," and FOIA gives us this metadata for free at the FCC 17-108 docket.

Output:
  data/processed/contrastive_pairs.parquet — anchor_text, positive_text,
                                              hard_negative_text (optional)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OUT_PATH = PROC / "contrastive_pairs.parquet"


def main(*, n_positive_per_uploader: int = 200,
         min_uploader_count: int = 100,
         max_uploaders: int = 200,
         seed: int = 0) -> None:
    bf_path = PROC / "buzzfeed_bulk_uploads.parquet"
    print(f"loading {bf_path}...")
    bf = pq.read_table(bf_path, columns=["comments", "uploader"]).to_pandas()
    bf = bf.dropna(subset=["comments", "uploader"])
    bf["comments"] = bf["comments"].astype(str)
    bf["uploader"] = bf["uploader"].astype(str)
    print(f"  {len(bf):,} rows")

    # filter to uploaders with enough comments to sample pairs from
    counts = bf["uploader"].value_counts()
    eligible = counts[counts >= min_uploader_count].index.tolist()
    eligible = eligible[:max_uploaders]
    print(f"  {len(eligible):,} uploaders with >= {min_uploader_count} comments")

    bf = bf[bf["uploader"].isin(eligible)]
    print(f"  filtered to {len(bf):,} rows from eligible uploaders")

    rng = np.random.default_rng(seed)
    rows = []
    for u in eligible:
        sub = bf[bf["uploader"] == u]
        n = len(sub)
        n_pairs = min(n_positive_per_uploader, n // 2)
        if n_pairs < 2:
            continue
        # sample without-replacement pairs
        idxs = rng.choice(n, size=2 * n_pairs, replace=False)
        a_idxs = idxs[:n_pairs]
        b_idxs = idxs[n_pairs:]
        anchors = sub["comments"].iloc[a_idxs].tolist()
        positives = sub["comments"].iloc[b_idxs].tolist()
        for a, p in zip(anchors, positives):
            if a != p and len(a.strip()) > 10 and len(p.strip()) > 10:
                rows.append({"anchor": a, "positive": p, "uploader": u})

    df = pd.DataFrame(rows)
    print(f"\nbuilt {len(df):,} positive pairs across {df['uploader'].nunique()} uploaders")
    print(f"  per-uploader breakdown (top 10):")
    print(df["uploader"].value_counts().head(10).to_string())

    df.to_parquet(OUT_PATH, compression="zstd", index=False)
    print(f"\nwrote {OUT_PATH}  ({OUT_PATH.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-positive-per-uploader", type=int, default=200,
                   help="how many positive pairs to draw per uploader")
    p.add_argument("--min-uploader-count", type=int, default=100,
                   help="ignore uploaders with fewer than this many comments")
    p.add_argument("--max-uploaders", type=int, default=200,
                   help="cap to top-N uploaders by volume")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    main(n_positive_per_uploader=args.n_positive_per_uploader,
         min_uploader_count=args.min_uploader_count,
         max_uploaders=args.max_uploaders, seed=args.seed)
