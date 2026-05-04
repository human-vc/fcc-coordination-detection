"""Match slnader 17-108 rejected clusters to BuzzFeed bulk-upload uploaders.

For each comment in slnader/fcc-comments, search the BuzzFeed FOIA bulk-uploads
file for an exact text match. If found, attach the uploader Box.com account
UUID. Then for each rejected cluster compute: what fraction of its members
were uploaded by the top-N uploaders (the ones NY AG named as fake-comment
contractors)?

This gives a real-world precision number for the methodology against named
coordination campaigns.

Outputs:
  data/processed/comment_uploader.parquet  — comment_id -> uploader_uuid
  results/attribution_table.csv            — per-cluster attribution stats
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
RES.mkdir(exist_ok=True)


def text_hash(s: str | float) -> str:
    """Stable hash of normalized comment text for cross-corpus matching."""
    if not isinstance(s, str):
        return ""
    norm = " ".join(s.lower().split())
    return hashlib.blake2b(norm.encode("utf-8"), digest_size=12).hexdigest()


def main(*, top_n_uploaders: int = 10,
         cluster_path: Path | None = None) -> None:
    bf_path = PROC / "buzzfeed_bulk_uploads.parquet"
    sl_comments = PROC / "comments.parquet"
    cluster_path = cluster_path or (PROC / "clusters.parquet")
    rejections = RES / "fdr_rejections_leiden.parquet"
    if not rejections.exists():
        rejections = PROC / "fdr_rejections.parquet"

    if not bf_path.exists():
        raise SystemExit(f"missing {bf_path}; run src/ingest_buzzfeed.py first")

    print(f"loading BuzzFeed bulk-uploads ({bf_path})...")
    bf = pq.read_table(bf_path, columns=["comments", "uploader"]).to_pandas()
    print(f"  {len(bf):,} rows")
    bf["text_hash"] = bf["comments"].apply(text_hash)
    bf_lookup = (bf.dropna(subset=["uploader"])
                   .groupby("text_hash")["uploader"]
                   .agg(lambda s: s.value_counts().index[0]))
    print(f"  {len(bf_lookup):,} unique text hashes -> uploader mapping")

    print(f"loading slnader comments ({sl_comments})...")
    sl = pq.read_table(sl_comments, columns=["comment_id", "comment_text"]).to_pandas()
    print(f"  {len(sl):,} unique slnader comments")
    sl["text_hash"] = sl["comment_text"].apply(text_hash)
    sl["uploader"] = sl["text_hash"].map(bf_lookup)
    matched = sl["uploader"].notna().sum()
    print(f"  matched {matched:,} ({100*matched/len(sl):.1f}%) to a BuzzFeed uploader")

    sl[["comment_id", "uploader"]].to_parquet(
        PROC / "comment_uploader.parquet", compression="zstd", index=False
    )

    # uploader volume + ranking
    vol = (sl.dropna(subset=["uploader"])
             .groupby("uploader").size().sort_values(ascending=False))
    top_set = set(vol.head(top_n_uploaders).index)
    print(f"\nTop {top_n_uploaders} uploaders by matched-volume:")
    for r, (u, n) in enumerate(vol.head(top_n_uploaders).items(), 1):
        print(f"  rank {r:2d}: {u}  matches={n:>10,}")

    # per-cluster attribution
    print(f"\nloading clusters from {cluster_path}...")
    cl = pq.read_table(cluster_path).to_pandas()
    rj = pq.read_table(rejections).to_pandas()
    rejected_cids = set(rj.loc[rj["rejected_ebh"], "cluster_id"].astype(int))
    cl_in_rejected = cl[cl["cluster_id"].isin(rejected_cids)].copy()
    cl_in_rejected = cl_in_rejected.merge(
        sl[["comment_id", "uploader"]], on="comment_id", how="left"
    )

    rows = []
    for cid, group in cl_in_rejected.groupby("cluster_id"):
        n = len(group)
        n_matched = group["uploader"].notna().sum()
        n_top = group["uploader"].isin(top_set).sum()
        rows.append({
            "cluster_id": int(cid),
            "n_members": n,
            "n_matched_buzzfeed": int(n_matched),
            "frac_matched_buzzfeed": float(n_matched / n) if n else 0.0,
            "n_top_uploader": int(n_top),
            "frac_top_uploader": float(n_top / n) if n else 0.0,
        })
    out = pd.DataFrame(rows).sort_values("frac_top_uploader", ascending=False)
    out.to_csv(RES / "attribution_table.csv", index=False)

    print(f"\nrejected clusters analyzed: {len(out):,}")
    print(f"  avg frac matched to BuzzFeed: {out['frac_matched_buzzfeed'].mean():.3f}")
    print(f"  avg frac top-{top_n_uploaders} uploader: {out['frac_top_uploader'].mean():.3f}")
    print(f"  clusters with frac_top_uploader >= 0.5: "
          f"{(out['frac_top_uploader'] >= 0.5).sum():,}")
    print(f"\nwrote {RES/'attribution_table.csv'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--top-n-uploaders", type=int, default=10)
    p.add_argument("--cluster-path", type=Path, default=None,
                   help="default: data/processed/clusters.parquet")
    args = p.parse_args()
    main(top_n_uploaders=args.top_n_uploaders, cluster_path=args.cluster_path)
