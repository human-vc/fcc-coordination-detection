"""Fetch FCC comments from ECFS (Electronic Comment Filing System) bulk API.

For docket 14-28 (the 2014 Open Internet proceeding) or any other proceeding,
pulls all express comments and saves as CSV.

Requires an FCC API key (free at https://api.regulations.gov/ — uses the
data.gov DATA_GOV_API_KEY which works for FCC ECFS too).
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import requests


def fetch_proceeding(*, proceeding: str, output_dir: Path,
                     api_key: str, page_size: int = 100,
                     max_pages: int | None = None) -> None:
    """Page through ECFS API until exhausted."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"comments_{proceeding}.csv"
    meta_path = output_dir / f"meta_{proceeding}.json"

    base_url = "https://publicapi.fcc.gov/ecfs/filings"
    params = {
        "api_key": api_key,
        "proceedings.name": proceeding,
        "limit": page_size,
        "sort": "date_received,ASC",
    }

    n_total = 0
    n_pages = 0
    page_token = None

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "submission_id", "comment_id", "comment_text",
            "date_received", "submission_type", "city", "state",
        ])
        writer.writeheader()
        while True:
            if page_token:
                params["page_token"] = page_token
            try:
                r = requests.get(base_url, params=params, timeout=60)
            except Exception as e:
                print(f"  request error (sleeping 30s): {e}")
                time.sleep(30)
                continue
            if r.status_code != 200:
                print(f"  status {r.status_code}: {r.text[:300]}")
                if r.status_code in (429, 500, 502, 503):
                    time.sleep(30)
                    continue
                break
            data = r.json()
            filings = data.get("filings", []) or data.get("Filings", []) or data.get("data", [])
            if not filings:
                break
            for filing in filings:
                row = {
                    "submission_id": filing.get("id_submission") or filing.get("id"),
                    "comment_id": filing.get("id_comment") or filing.get("comment_id"),
                    "comment_text": (filing.get("text_data") or
                                     filing.get("express_comment") or
                                     filing.get("comment_text") or ""),
                    "date_received": filing.get("date_received") or filing.get("date_disseminated"),
                    "submission_type": (filing.get("submissiontype", {}) or {}).get("description")
                                       if isinstance(filing.get("submissiontype"), dict)
                                       else filing.get("submission_type"),
                    "city": (filing.get("addressentity", {}) or {}).get("city"),
                    "state": (filing.get("addressentity", {}) or {}).get("state"),
                }
                writer.writerow(row)
                n_total += 1
            n_pages += 1
            page_token = (data.get("metadata", {}) or {}).get("page_token") or \
                         data.get("next_page_token")
            if not page_token:
                break
            if max_pages and n_pages >= max_pages:
                break
            print(f"  page {n_pages}, total {n_total:,}")
            time.sleep(0.2)  # be polite

    with meta_path.open("w") as f:
        json.dump({"proceeding": proceeding, "total_comments": n_total,
                   "n_pages": n_pages}, f, indent=2)
    print(f"\nwrote {csv_path}  ({n_total:,} comments, {n_pages} pages)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--proceeding", required=True,
                   help="FCC docket, e.g. '14-28' or '17-108'")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--api-key", required=True,
                   help="FCC ECFS / data.gov API key")
    p.add_argument("--page-size", type=int, default=100)
    p.add_argument("--max-pages", type=int, default=None)
    args = p.parse_args()
    fetch_proceeding(proceeding=args.proceeding,
                     output_dir=args.output_dir,
                     api_key=args.api_key,
                     page_size=args.page_size,
                     max_pages=args.max_pages)
