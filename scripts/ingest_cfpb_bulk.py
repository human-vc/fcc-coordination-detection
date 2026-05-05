"""Ingest the regulations.gov bulk-download CSV for CFPB-2016-0025
into our parquet schema.

Source: regulations.gov/bulkdownload UI, email-delivered CSV.

CSV columns of interest:
    Document ID, Docket ID, Tracking Number, Posted Date, Received Date,
    First Name, Last Name, City, State/Province, Zip/Postal Code, Country,
    Organization Name, Comment, Content Files, Attachment Files, ...

Output schema:
    submission_id, comment_id, comment_text, date_received,
    submission_type, express_comment, city, state, filer_name
"""
from __future__ import annotations
import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd


def hash_text(t: str) -> str:
    return hashlib.sha256(t.encode("utf-8", errors="replace")).hexdigest()


SEE_ATTACHED_PATTERNS = ("see attach", "see the attached", "please see")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-csv", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--min-len", type=int, default=50)
    p.add_argument("--drop-see-attached", action="store_true", default=True)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"reading {args.input_csv} ...", flush=True)
    df = pd.read_csv(args.input_csv, low_memory=False, dtype=str, on_bad_lines="skip")
    print(f"  total rows: {len(df):,}", flush=True)
    print(f"  columns: {df.columns.tolist()[:10]} ...", flush=True)

    df["comment_text"] = df["Comment"].fillna("").astype(str).str.strip()
    print(f"  with non-empty comment: {df['comment_text'].str.len().gt(0).sum():,}", flush=True)

    df = df[df["comment_text"].str.len() >= args.min_len].reset_index(drop=True)
    print(f"  with comment len >= {args.min_len}: {len(df):,}", flush=True)

    if args.drop_see_attached:
        text_low = df["comment_text"].str.lower()
        mask = pd.Series([False] * len(df))
        for pat in SEE_ATTACHED_PATTERNS:
            mask = mask | text_low.str.contains(pat, na=False)
        keep = (~mask) | (df["comment_text"].str.len() >= 200)
        df = df[keep].reset_index(drop=True)
        print(f"  after removing 'see attached' stubs (kept long ones): {len(df):,}", flush=True)

    df["comment_id"] = df["comment_text"].apply(hash_text)
    df["submission_id"] = df["Document ID"].fillna("").astype(str)
    if "Received Date" in df.columns:
        df["date_received"] = df["Received Date"].fillna("")
    elif "Posted Date" in df.columns:
        df["date_received"] = df["Posted Date"].fillna("")
    else:
        df["date_received"] = ""
    df["submission_type"] = "comment"
    df["express_comment"] = "1"
    df["city"] = df.get("City", "").fillna("")
    df["state"] = df.get("State/Province", "").fillna("")
    df["filer_name"] = (df.get("First Name", "").fillna("") + " " + df.get("Last Name", "").fillna("")).str.strip()
    df["organization"] = df.get("Organization Name", "").fillna("")

    sub_cols = ["submission_id", "submission_type", "express_comment",
                "date_received", "city", "state", "comment_id",
                "filer_name", "organization"]
    submissions = df[[c for c in sub_cols if c in df.columns]].copy()
    submissions.to_parquet(args.output_dir / "submissions.parquet",
                           compression="zstd", index=False)
    print(f"  wrote submissions.parquet ({len(submissions):,} rows)", flush=True)

    uniq = (df.groupby("comment_id", as_index=False).first()
              [["comment_id", "comment_text"]])
    uniq = uniq.reset_index(drop=True)
    uniq["row_id"] = range(len(uniq))
    uniq[["comment_id", "comment_text", "row_id"]].to_parquet(
        args.output_dir / "comments.parquet", compression="zstd", index=False)
    print(f"  wrote comments.parquet ({len(uniq):,} unique comments)", flush=True)

    counts = df.groupby("comment_id").size().rename("template_size").reset_index()
    idx = uniq.merge(counts, on="comment_id", how="left")
    idx["template_size"] = idx["template_size"].fillna(1).astype(int)
    idx[["row_id", "comment_id", "template_size"]].to_parquet(
        args.output_dir / "embedding_index.parquet", compression="zstd", index=False)
    print(f"  wrote embedding_index.parquet", flush=True)
    print(f"  size 1 (singletons): {(idx['template_size'] == 1).sum():,}")
    print(f"  size 2-5: {((idx['template_size'] >= 2) & (idx['template_size'] <= 5)).sum():,}")
    print(f"  size 6+: {(idx['template_size'] >= 6).sum():,}")
    print(f"  largest template: {idx['template_size'].max():,}")

    if "filer_name" in df.columns:
        filer_counts = df.groupby(["comment_id", "filer_name"]).size().reset_index(name="n")
        top_filer = (filer_counts.sort_values(["comment_id", "n"], ascending=[True, False])
                                 .groupby("comment_id").first().reset_index())
        top_filer[["comment_id", "filer_name", "n"]].to_parquet(
            args.output_dir / "comment_filer.parquet", compression="zstd", index=False)
        print(f"  wrote comment_filer.parquet ({len(top_filer):,} rows)")


if __name__ == "__main__":
    main()
