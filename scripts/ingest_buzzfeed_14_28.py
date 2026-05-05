"""Ingest BuzzFeed FCC 14-28 packaged corpus into our parquet schema.

Source: https://archive.org/download/fcc-comments-and-bulk-uploads/
        comments-14-28-with-uuids.csv.zip

Expected columns (from BuzzFeedNews/2019-10-fcc-comments README):
    id_submission, text_data, date_received, contact_email_domain_uuid,
    name_and_location_uuid, ...

We normalize to:
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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--min-len", type=int, default=10)
    p.add_argument("--text-col-candidates", type=str,
                   default="text_data,comment_text,comment,text",
                   help="comma-separated candidate column names for comment text")
    p.add_argument("--id-col-candidates", type=str,
                   default="id_submission,submission_id,id,comment_id",
                   help="comma-separated candidate column names for submission id")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csvs = sorted(args.input_dir.glob("*.csv"))
    if not csvs:
        sys.exit(f"no CSVs in {args.input_dir}")
    print(f"found {len(csvs)} CSV(s); concatenating...", flush=True)

    dfs = []
    for path in csvs:
        print(f"  reading {path.name}", flush=True)
        df_chunk = pd.read_csv(path, low_memory=False, dtype=str, on_bad_lines="skip")
        dfs.append(df_chunk)
    df = pd.concat(dfs, ignore_index=True)
    print(f"  total rows: {len(df):,}", flush=True)
    print(f"  columns: {df.columns.tolist()}", flush=True)

    text_candidates = args.text_col_candidates.split(",")
    text_col = next((c for c in text_candidates if c in df.columns), None)
    if text_col is None:
        sys.exit(f"no text column found among {text_candidates}; columns: {df.columns.tolist()}")
    print(f"  using text column: {text_col}", flush=True)

    id_candidates = args.id_col_candidates.split(",")
    id_col = next((c for c in id_candidates if c in df.columns), None)
    if id_col is None:
        print(f"  WARN: no submission id column among {id_candidates}; will hash text", flush=True)

    df["comment_text"] = df[text_col].fillna("").astype(str).str.strip()
    df = df[df["comment_text"].str.len() >= args.min_len].reset_index(drop=True)
    print(f"  after filtering empty/short: {len(df):,}", flush=True)

    df["comment_id"] = df["comment_text"].apply(hash_text)
    df["submission_id"] = (df[id_col].astype(str) if id_col else df["comment_id"])
    df["date_received"] = df.get("date_received", pd.Series([""] * len(df)))
    df["submission_type"] = df.get("submission_type", "comment")
    df["express_comment"] = "1"
    df["city"] = df.get("city", "")
    df["state"] = df.get("state", "")
    if "contact_email_domain_uuid" in df.columns:
        df["uploader_uuid"] = df["contact_email_domain_uuid"].fillna("")
    elif "name_and_location_uuid" in df.columns:
        df["uploader_uuid"] = df["name_and_location_uuid"].fillna("")
    else:
        df["uploader_uuid"] = ""
    df["filer_name"] = df["uploader_uuid"]

    sub_cols = ["submission_id", "submission_type", "express_comment",
                "date_received", "city", "state", "comment_id", "uploader_uuid"]
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

    if "uploader_uuid" in df.columns:
        uploader_counts = (df[df["uploader_uuid"] != ""]
                              .groupby(["comment_id", "uploader_uuid"]).size()
                              .reset_index(name="n"))
        top_uploader_per_comment = (uploader_counts
                                       .sort_values(["comment_id", "n"], ascending=[True, False])
                                       .groupby("comment_id").first().reset_index())
        top_uploader_per_comment[["comment_id", "uploader_uuid", "n"]].to_parquet(
            args.output_dir / "comment_uploader.parquet",
            compression="zstd", index=False)
        print(f"  wrote comment_uploader.parquet ({len(top_uploader_per_comment):,} rows)", flush=True)


if __name__ == "__main__":
    main()
