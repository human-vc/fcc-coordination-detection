"""Fetch CFPB-2016-0025 (payday lending rule) comments via regulations.gov v4 API.

Strategy:
  - 5000-row deep-pagination cap on regulations.gov v4 → use date-windowing
  - Recursive date-bisect: if a window returns >=4900 rows, split it in half
  - Resume-safe: appends to CSV + saves processed-window state JSON
  - Saves each comment as one CSV row with {comment_id, comment_text,
    date_received, submission_type, ...} matching the project schema

Regulations.gov v4 API:
  - https://api.regulations.gov/v4/comments?filter[docketId]=CFPB-2016-0025
  - filter[lastModifiedDate][ge|le]=YYYY-MM-DD
  - sort=lastModifiedDate
  - page[size] up to 250, page[number] until 5000-row cap
  - Comment text in attributes.comment (inline)
  - 1000 req/hour default rate limit
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests

API = "https://api.regulations.gov/v4/comments"
DOCKET = "CFPB-2016-0025"
PAGE_SIZE = 250
WINDOW_CAP = 4900  # bisect if a window returns this many or more


def fetch_window_paginated(*, api_key: str, date_from: str, date_to: str,
                           docket_id: str, max_retries: int = 3) -> list[dict]:
    """Pull all comments in [date_from, date_to] up to ~5000 rows.

    Returns list of comment dicts (attributes section, with id added).
    """
    out: list[dict] = []
    page = 1
    while True:
        params = {
            "filter[docketId]": docket_id,
            "filter[lastModifiedDate][ge]": date_from,
            "filter[lastModifiedDate][le]": date_to,
            "page[size]": PAGE_SIZE,
            "page[number]": page,
            "sort": "lastModifiedDate",
            "api_key": api_key,
        }
        for attempt in range(max_retries):
            try:
                r = requests.get(API, params=params, timeout=60)
            except Exception as e:
                print(f"    [page {page}] exception {e}; sleeping 30s",
                      file=sys.stderr, flush=True)
                time.sleep(30)
                continue
            if r.status_code == 200:
                break
            if r.status_code == 429:
                wait = 60 * (attempt + 1)
                print(f"    [page {page}] 429; sleeping {wait}s",
                      file=sys.stderr, flush=True)
                time.sleep(wait)
                continue
            if r.status_code in (500, 502, 503, 504):
                wait = 30 * (attempt + 1)
                print(f"    [page {page}] {r.status_code}; sleeping {wait}s",
                      file=sys.stderr, flush=True)
                time.sleep(wait)
                continue
            print(f"    [page {page}] status {r.status_code}: {r.text[:200]}",
                  file=sys.stderr, flush=True)
            return out
        else:
            print(f"    [page {page}] gave up after {max_retries}",
                  file=sys.stderr, flush=True)
            return out

        body = r.json()
        items = body.get("data", [])
        if not items:
            return out
        for item in items:
            attrs = item.get("attributes", {}) or {}
            attrs["_id"] = item.get("id")
            out.append(attrs)
        if len(items) < PAGE_SIZE:
            return out
        page += 1
        time.sleep(0.4)


def date_midpoint(d_from: str, d_to: str) -> str:
    f = datetime.fromisoformat(d_from)
    t = datetime.fromisoformat(d_to)
    mid = f + (t - f) / 2
    return mid.strftime("%Y-%m-%d")


def fetch_recursive(*, api_key: str, date_from: str, date_to: str,
                    docket_id: str, depth: int = 0) -> list[dict]:
    """Date-bisect: if a window hits the cap, split and recurse."""
    indent = "  " * depth
    print(f"{indent}window [{date_from} .. {date_to}]",
          flush=True)
    items = fetch_window_paginated(
        api_key=api_key, date_from=date_from, date_to=date_to,
        docket_id=docket_id,
    )
    print(f"{indent}  -> {len(items):,} comments", flush=True)
    if len(items) >= WINDOW_CAP and date_from != date_to:
        mid = date_midpoint(date_from, date_to)
        if mid == date_from or mid == date_to:
            print(f"{indent}  cannot bisect further (single day); "
                  f"may be losing data past cap", flush=True)
            return items
        f = datetime.fromisoformat(date_from)
        m = datetime.fromisoformat(mid)
        left_end = (m - timedelta(days=1)).strftime("%Y-%m-%d")
        if datetime.fromisoformat(left_end) < f:
            left_end = date_from
        left = fetch_recursive(
            api_key=api_key, date_from=date_from, date_to=left_end,
            docket_id=docket_id, depth=depth + 1,
        )
        right = fetch_recursive(
            api_key=api_key, date_from=mid, date_to=date_to,
            docket_id=docket_id, depth=depth + 1,
        )
        return left + right
    return items


def daterange(start: str, end: str, days: int):
    """Yield (window_from, window_to) tuples covering [start, end]."""
    s = datetime.fromisoformat(start)
    e = datetime.fromisoformat(end)
    cur = s
    while cur <= e:
        nxt = min(cur + timedelta(days=days - 1), e)
        yield cur.strftime("%Y-%m-%d"), nxt.strftime("%Y-%m-%d")
        cur = nxt + timedelta(days=1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--api-key", required=True)
    p.add_argument("--docket-id", default=DOCKET)
    p.add_argument("--date-from", default="2016-06-01")
    p.add_argument("--date-to", default="2018-12-31")
    p.add_argument("--initial-window-days", type=int, default=14)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / f"comments_{args.docket_id}.csv"
    state_path = args.output_dir / f"state_{args.docket_id}.json"
    meta_path = args.output_dir / f"meta_{args.docket_id}.json"

    done_windows: set[str] = set()
    n_total_existing = 0
    mode = "w"
    if state_path.exists():
        st = json.loads(state_path.read_text())
        done_windows = set(st.get("done_windows", []))
        n_total_existing = st.get("n_total", 0)
        mode = "a"
        print(f"resuming: {len(done_windows)} windows done, "
              f"{n_total_existing:,} comments saved", flush=True)

    fields = ["submission_id", "comment_id", "comment_text",
              "date_received", "submission_type", "express_comment",
              "city", "state", "filer_name"]
    f = csv_path.open(mode, newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=fields)
    if mode == "w":
        writer.writeheader()

    n_total = n_total_existing
    n_windows_processed = 0
    t0 = time.time()
    try:
        for win_from, win_to in daterange(args.date_from, args.date_to,
                                           args.initial_window_days):
            win_key = f"{win_from}_{win_to}"
            if win_key in done_windows:
                continue
            comments = fetch_recursive(
                api_key=args.api_key,
                date_from=win_from, date_to=win_to,
                docket_id=args.docket_id,
            )
            for c in comments:
                txt = (c.get("comment") or "").strip()
                if len(txt) < 10:
                    continue
                row = {
                    "submission_id": c.get("_id"),
                    "comment_id": c.get("_id"),
                    "comment_text": txt,
                    "date_received": c.get("postedDate") or c.get("receiveDate"),
                    "submission_type": "comment",
                    "express_comment": "1",
                    "city": c.get("city") or "",
                    "state": c.get("stateProvinceRegion") or "",
                    "filer_name": (c.get("submitterName") or
                                    f"{c.get('firstName','')} {c.get('lastName','')}".strip()),
                }
                writer.writerow(row)
                n_total += 1
            f.flush()
            done_windows.add(win_key)
            n_windows_processed += 1
            elapsed = time.time() - t0
            print(f"  [done {len(done_windows)} windows, "
                  f"{n_total:,} comments, {elapsed/60:.1f} min, "
                  f"{n_total / max(elapsed, 1):.0f} c/s]",
                  flush=True)
            state_path.write_text(json.dumps({
                "done_windows": sorted(done_windows),
                "n_total": n_total,
                "n_windows": n_windows_processed,
                "elapsed_seconds": elapsed,
            }))
    finally:
        f.close()
        meta_path.write_text(json.dumps({
            "docket_id": args.docket_id,
            "total_comments": n_total,
            "n_windows": len(done_windows),
            "elapsed_seconds": time.time() - t0,
        }, indent=2))
        print(f"\nDONE: {n_total:,} comments in {len(done_windows)} windows, "
              f"{(time.time() - t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
