"""Ingest BuzzFeed FOIA-derived bulk-uploads file for FCC docket 17-108.

Source: https://archive.org/details/fcc-comments-and-bulk-uploads
File:   bulk-uploads-17-108-with-uuids.csv.zip

This file has the unique-per-corpus property of carrying *uploader Box.com
account IDs* (as UUIDs) — the metadata the NY AG used to attribute fake
comments to specific contractors (Fluent, React2Media, Opt-Intelligence,
Media Bridge LLC, etc.). The NY AG settlements (May 2021 / 2023) named the
organizations and approximate volumes:

    Fluent, Inc.            ~7.7M comments
    Opt-Intelligence Inc.   ~250K
    React2Media             ~329K
    Media Bridge LLC        (Broadband for America vendor)
    LCX Digital             smaller
    Center for Individual Freedom (CFIF)

Because BuzzFeed replaced raw uploader emails with UUIDs, we cannot identify
which UUID corresponds to which org by name. We use:

  1. Volume rank — the largest uploader UUID by submitted-comment count is
     Fluent; second is one of {Opt-Intelligence, React2Media, ...}.
  2. The breach-sample files — 'breaches-17-108-mb-sample.csv' is a
     10,000-address sample from Media Bridge specifically; the 1,000-address
     sample for 8 uploaders covers the other major contractors. Their
     email-UUIDs are linked to uploader-UUIDs through the bulk-uploads file.

Outputs:
  data/processed/buzzfeed_bulk_uploads.parquet    — all bulk-uploaded comments
  data/processed/buzzfeed_uploader_volume.parquet — uploader_uuid -> count
  data/processed/buzzfeed_attribution.parquet     — uploader_uuid -> {label, rank}
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "buzzfeed"
PROC = ROOT / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)


def main() -> None:
    csv_path = RAW / "bulk-uploads-17-108-with-uuids.csv"
    if not csv_path.exists():
        zip_path = RAW / "bulk-uploads-17-108-with-uuids.csv.zip"
        if not zip_path.exists():
            raise SystemExit(f"missing {zip_path}; download first")
        import zipfile
        print(f"unzipping {zip_path}...")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(RAW)

    print(f"reading {csv_path}...")
    # full schema per BuzzFeed README:
    # date, comments, name_and_location, email_address,
    # email_address_nonstandard, email_domain, file, uploader
    df = pd.read_csv(csv_path, low_memory=False, dtype=str)
    print(f"loaded {len(df):,} rows; columns: {list(df.columns)}")

    # write parquet
    out_main = PROC / "buzzfeed_bulk_uploads.parquet"
    df.to_parquet(out_main, compression="zstd", index=False)
    print(f"wrote {out_main}  ({out_main.stat().st_size/1e6:.1f} MB)")

    # uploader volume rank
    vol = df.groupby("uploader", dropna=False).size().reset_index(name="n_comments")
    vol = vol.sort_values("n_comments", ascending=False).reset_index(drop=True)
    vol["rank"] = vol.index + 1
    out_vol = PROC / "buzzfeed_uploader_volume.parquet"
    vol.to_parquet(out_vol, compression="zstd", index=False)
    print(f"wrote {out_vol}")

    # rough attribution by NY AG-published volumes
    # Fluent ~7.7M, React2Media ~329K, Opt-Intelligence ~250K
    # Anything in top-10 by volume is presumptively a named contractor;
    # we tag with NY-AG-likely-attribution and let the eval bind precise IDs.
    AG_LABELS = {
        # rank → likely organization (NY AG report 2021)
        1: "Fluent (likely)",
        2: "React2Media or Opt-Intelligence (likely)",
        3: "React2Media or Opt-Intelligence (likely)",
        4: "Media Bridge LLC (likely)",
    }
    vol["ag_attribution"] = vol["rank"].map(AG_LABELS).fillna("")
    vol[["uploader", "rank", "n_comments", "ag_attribution"]].to_parquet(
        PROC / "buzzfeed_attribution.parquet", compression="zstd", index=False
    )

    print("\nTop 20 uploaders by volume:")
    print(vol.head(20).to_string(index=False))
    print(f"\nTotal uploaders: {len(vol):,}")
    print(f"Top 10 uploaders share: {vol.head(10)['n_comments'].sum() / vol['n_comments'].sum():.1%}")


if __name__ == "__main__":
    main()
