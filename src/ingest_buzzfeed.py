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
import pyarrow.csv as pa_csv
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "buzzfeed"
PROC = ROOT / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)


def stream_csv_to_parquet(csv_path: Path, out_path: Path) -> int:
    """Stream a large CSV to Parquet without loading it all in RAM."""
    read_opts = pa_csv.ReadOptions(block_size=64 * 1024 * 1024)  # 64 MB blocks
    parse_opts = pa_csv.ParseOptions(quote_char='"', escape_char=None,
                                      newlines_in_values=True)
    convert_opts = pa_csv.ConvertOptions(
        # force everything as string; we'll cast downstream as needed
        strings_can_be_null=True, null_values=[""],
    )
    writer = None
    n_rows = 0
    with pa_csv.open_csv(csv_path, read_options=read_opts,
                         parse_options=parse_opts,
                         convert_options=convert_opts) as reader:
        for batch in reader:
            # cast all to string for stable schema
            cols = {f.name: batch.column(f.name).cast(pa.string())
                    for f in batch.schema}
            tbl = pa.table(cols)
            if writer is None:
                writer = pq.ParquetWriter(out_path, tbl.schema, compression="zstd")
            writer.write_table(tbl)
            n_rows += tbl.num_rows
            if n_rows % 1_000_000 < tbl.num_rows:
                print(f"  ...{n_rows:,} rows written")
    if writer is not None:
        writer.close()
    return n_rows


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

    out_main = PROC / "buzzfeed_bulk_uploads.parquet"
    print(f"streaming {csv_path} -> {out_main}...")
    n = stream_csv_to_parquet(csv_path, out_main)
    print(f"wrote {out_main}  ({n:,} rows; {out_main.stat().st_size/1e6:.1f} MB)")

    # for downstream stats, read just the columns we need
    print("\nreading uploader+domain columns for volume stats...")
    df = pq.read_table(out_main, columns=["uploader"]).to_pandas()
    print(f"  {len(df):,} rows in memory ({df.memory_usage(deep=True).sum()/1e6:.0f} MB)")

    # uploader volume rank
    vol = df.groupby("uploader", dropna=False).size().reset_index(name="n_comments")
    vol = vol.sort_values("n_comments", ascending=False).reset_index(drop=True)
    vol["rank"] = vol.index + 1
    out_vol = PROC / "buzzfeed_uploader_volume.parquet"
    vol.to_parquet(out_vol, compression="zstd", index=False)
    print(f"wrote {out_vol}")

    # The `uploader` column is the plaintext email of the Box.com account,
    # so we can attribute directly by domain match instead of guessing by rank.
    # Categorization of the top accounts:
    #   ASTROTURF (NY AG-named or strongly-suggested fake):
    #     - shane@mediabridgellc.com  -> Media Bridge LLC (NY AG 2021 explicit)
    #     - esmisc@mac.com            -> likely Fluent (4.3M volume matches NY AG ~7.7M Fluent)
    #     - fccfreedom@hmamail.com    -> anonymous (HideMyAss) email; astroturf signal
    #   LEGITIMATE ADVOCACY (public organizations mobilizing supporters):
    #     - mike@fightforthefuture.org, karen@momsrising.org, dutch@freepress.net,
    #       kurt@demandprogress.org, advocacy@mozilla.com, action@aclu.org,
    #       meaghan@mandatemedia.com, david@openmedia.org, info@mpowerchange.org,
    #       eve@revolutionmessaging.com
    UPLOADER_LABELS: dict[str, str] = {
        "shane@mediabridgellc.com":     "astroturf:Media Bridge LLC (NY AG)",
        "esmisc@mac.com":                "astroturf:likely Fluent",
        "fccfreedom@hmamail.com":        "astroturf:anonymous (HMA)",
        "mike@fightforthefuture.org":    "advocacy:Fight for the Future",
        "karen@momsrising.org":          "advocacy:MomsRising",
        "dutch@freepress.net":           "advocacy:Free Press",
        "kurt@demandprogress.org":       "advocacy:Demand Progress",
        "advocacy@mozilla.com":          "advocacy:Mozilla",
        "action@aclu.org":                "advocacy:ACLU",
        "meaghan@mandatemedia.com":      "advocacy:Mandate Media (agency)",
        "david@openmedia.org":            "advocacy:OpenMedia",
        "info@mpowerchange.org":          "advocacy:MPower Change",
        "eve@revolutionmessaging.com":    "advocacy:Revolution Messaging",
        "wyden@mandatemedia.com":         "advocacy:Sen. Wyden / Mandate Media",
        "tom@cashmusic.org":              "advocacy:CASH Music",
        "tom+netneutrality@cashmusic.org": "advocacy:CASH Music",
        "ncatalano@ofa.us":               "advocacy:Organizing for Action",
        "info@betheimpakt.com":           "advocacy:Be The Impakt",
    }
    vol["ag_attribution"] = vol["uploader"].map(UPLOADER_LABELS).fillna("")
    vol["category"] = vol["ag_attribution"].apply(
        lambda s: s.split(":", 1)[0] if s else "")
    vol[["uploader", "rank", "n_comments", "ag_attribution", "category"]].to_parquet(
        PROC / "buzzfeed_attribution.parquet", compression="zstd", index=False
    )

    print("\nTop 20 uploaders by volume:")
    print(vol.head(20).to_string(index=False))
    print(f"\nTotal uploaders: {len(vol):,}")
    print(f"Top 10 uploaders share: {vol.head(10)['n_comments'].sum() / vol['n_comments'].sum():.1%}")

    print("\nCategory breakdown (by comments):")
    by_cat = vol.groupby("category")["n_comments"].sum().sort_values(ascending=False)
    total = vol["n_comments"].sum()
    for cat, n in by_cat.items():
        cat_label = cat or "unlabeled"
        print(f"  {cat_label:14s} {n:>10,}  ({100*n/total:.1f}%)")


if __name__ == "__main__":
    main()
