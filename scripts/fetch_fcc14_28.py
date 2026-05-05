"""Paginated FCC 14-28 download via ECFS public API.

Strategy:
  - Filter to express_comment=1 (~1.59M of 2.19M total filings)
  - Sort by id_submission ASC for deterministic pagination
  - Paginate via offset+limit
  - Save incrementally to CSV (resume-safe)
  - 0.25s delay between calls to be polite (~50min minimum)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import requests

API_BASE = "https://publicapi.fcc.gov/ecfs/filings"


def fetch_page(*, api_key: str, proceeding: str, offset: int,
               limit: int, sort: str = "id_submission,ASC",
               max_retries: int = 3) -> list[dict]:
    params = {
        "proceedings.name": proceeding,
        "api_key": api_key,
        "limit": limit,
        "offset": offset,
        "sort": sort,
        "express_comment": "1",
    }
    for attempt in range(max_retries):
        try:
            r = requests.get(API_BASE, params=params, timeout=60)
            if r.status_code == 200:
                return r.json().get("filing", [])
            elif r.status_code == 429:
                wait = 60 * (attempt + 1)
                print(f"  [offset {offset}] 429 rate-limited; sleeping {wait}s",
                      file=sys.stderr, flush=True)
                time.sleep(wait)
            elif r.status_code in (500, 502, 503, 504):
                wait = 30 * (attempt + 1)
                print(f"  [offset {offset}] {r.status_code}; sleeping {wait}s",
                      file=sys.stderr, flush=True)
                time.sleep(wait)
            else:
                print(f"  [offset {offset}] status {r.status_code}: {r.text[:200]}",
                      file=sys.stderr, flush=True)
                return []
        except Exception as e:
            print(f"  [offset {offset}] exception {e}; sleeping 30s",
                  file=sys.stderr, flush=True)
            time.sleep(30)
    return []


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--proceeding", default="14-28")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--api-key", required=True)
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--start-offset", type=int, default=0)
    p.add_argument("--max-records", type=int, default=2_000_000)
    p.add_argument("--delay", type=float, default=0.25)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / f"comments_{args.proceeding}.csv"
    meta_path = args.output_dir / f"meta_{args.proceeding}.json"
    state_path = args.output_dir / f"state_{args.proceeding}.json"

    # Resume from prior state if present
    start_offset = args.start_offset
    n_total_existing = 0
    mode = "w"
    if state_path.exists():
        st = json.loads(state_path.read_text())
        start_offset = st.get("next_offset", start_offset)
        n_total_existing = st.get("n_total", 0)
        mode = "a"
        print(f"resuming from offset={start_offset}, "
              f"already have {n_total_existing} records",
              flush=True)

    fields = ["submission_id", "comment_id", "comment_text", "date_received",
              "submission_type", "express_comment", "filer_name", "city", "state"]
    f = csv_path.open(mode, newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=fields)
    if mode == "w":
        writer.writeheader()

    n_total = n_total_existing
    n_empty_pages = 0
    n_pages = 0
    t0 = time.time()
    offset = start_offset
    try:
        while n_total < args.max_records:
            filings = fetch_page(api_key=args.api_key,
                                  proceeding=args.proceeding,
                                  offset=offset, limit=args.limit)
            if not filings:
                n_empty_pages += 1
                if n_empty_pages >= 3:
                    print(f"three empty pages in a row at offset {offset}; stopping",
                          flush=True)
                    break
                offset += args.limit
                time.sleep(args.delay * 2)
                continue
            n_empty_pages = 0
            for filing in filings:
                txt = (filing.get("text_data") or "").strip()
                if not txt:
                    continue  # skip empty (we want comment text)
                row = {
                    "submission_id": filing.get("id_submission"),
                    "comment_id": filing.get("id_submission"),
                    "comment_text": txt,
                    "date_received": filing.get("date_received") or
                                     filing.get("date_disseminated"),
                    "submission_type": (filing.get("submissiontype") or {}).get("description"),
                    "express_comment": filing.get("express_comment"),
                    "filer_name": (filing.get("filers") or [{}])[0].get("name", ""),
                    "city": "", "state": "",
                }
                writer.writerow(row)
                n_total += 1
            f.flush()

            n_pages += 1
            offset += args.limit
            # save state
            state_path.write_text(json.dumps({
                "next_offset": offset, "n_total": n_total,
                "n_pages": n_pages, "elapsed": time.time() - t0,
            }))
            if n_pages % 20 == 0:
                rate = n_pages / max(time.time() - t0, 1)
                print(f"  pages={n_pages}, offset={offset}, total={n_total:,}, "
                      f"rate={rate:.1f} pg/s", flush=True)
            time.sleep(args.delay)

    finally:
        f.close()
        meta_path.write_text(json.dumps({
            "proceeding": args.proceeding,
            "total_comments": n_total,
            "n_pages": n_pages,
            "final_offset": offset,
            "elapsed_seconds": time.time() - t0,
        }, indent=2))
        print(f"\nDONE: {n_total:,} comments in {n_pages} pages, "
              f"{(time.time() - t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
