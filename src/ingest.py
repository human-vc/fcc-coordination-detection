"""Stream-parse fcc.pgsql into Parquet files, no Postgres required.

The dump is plain SQL with tab-separated COPY blocks. We extract the rows for
the tables that matter for coordination analysis and write them as Parquet.

Tables extracted:
  - submissions: metadata (timing, location, type) for ~24M submissions
  - comments: express-comment text (~3.8M rows)
  - near_duplicates: precomputed near-duplicate document pairs
  - exact_duplicates: precomputed exact-duplicate document pairs
  - interest_groups: commenter-type labels (interest-group / business / individual)
  - filers: filer names
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "fcc.pgsql"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

# table -> (column names, pyarrow types)
SCHEMAS = {
    "submissions": (
        ["submission_id", "submission_type", "express_comment", "date_received",
         "contact_email", "city", "address_line_1", "address_line_2",
         "state", "zip_code", "comment_id"],
        [pa.string()] * 2 + [pa.string(), pa.string()] + [pa.string()] * 7,
    ),
    "comments": (
        ["comment_id", "comment_text", "row_id"],
        [pa.string(), pa.string(), pa.int64()],
    ),
    "near_duplicates": (
        ["target_document_id", "duplicate_document_id"],
        [pa.string(), pa.string()],
    ),
    "exact_duplicates": (
        ["target_document_id", "duplicate_document_id"],
        [pa.string(), pa.string()],
    ),
    "interest_groups": (
        ["submission_id", "business"],
        [pa.string(), pa.string()],
    ),
    "filers": (
        ["submission_id", "filer_name"],
        [pa.string(), pa.string()],
    ),
    "documents": (
        ["submission_id", "document_name", "download_status",
         "document_id", "file_extension"],
        [pa.string()] * 5,
    ),
    "docs_cited": (
        ["cite_id", "submission_id", "document_id"],
        [pa.string()] * 3,
    ),
}

COPY_RE = re.compile(r"^COPY public\.(\w+) \([^)]+\) FROM stdin;\s*$")
BATCH = 100_000


def unesc(field: str) -> str | None:
    """Postgres COPY uses \\N for NULL and standard backslash escapes."""
    if field == r"\N":
        return None
    return (field.replace(r"\t", "\t").replace(r"\n", "\n")
                 .replace(r"\r", "\r").replace(r"\\", "\\"))


def write_batch(writer: pq.ParquetWriter, cols: list[str], types: list[pa.DataType],
                buf: list[list[str | None]]) -> None:
    arrays = []
    for i, t in enumerate(types):
        col = [row[i] for row in buf]
        if pa.types.is_integer(t):
            col = [int(x) if x is not None else None for x in col]
        arrays.append(pa.array(col, type=t))
    writer.write_table(pa.Table.from_arrays(arrays, names=cols))


def main() -> None:
    if not RAW.exists():
        sys.exit(f"missing {RAW}; extract fcc.tar.gz first")

    total = RAW.stat().st_size
    bar = tqdm(total=total, unit="B", unit_scale=True, desc="parse")

    with RAW.open("r", encoding="utf-8", errors="replace") as f:
        line = f.readline()
        bar.update(len(line.encode("utf-8", errors="replace")))
        active = None  # (table, writer, cols, types, buf)

        while line:
            if active is None:
                m = COPY_RE.match(line)
                if m and m.group(1) in SCHEMAS:
                    table = m.group(1)
                    cols, types = SCHEMAS[table]
                    schema = pa.schema(list(zip(cols, types)))
                    out_path = OUT / f"{table}.parquet"
                    writer = pq.ParquetWriter(out_path, schema, compression="zstd")
                    active = (table, writer, cols, types, [], 0)
                    bar.set_postfix(table=table, rows=0)
            else:
                if line.startswith(r"\.") and (len(line) == 2 or line[2] in "\r\n"):
                    table, writer, cols, types, buf, written = active
                    if buf:
                        write_batch(writer, cols, types, buf)
                        written += len(buf)
                    writer.close()
                    bar.set_postfix(table=table, rows=f"{written:,} done")
                    active = None
                else:
                    fields = line.rstrip("\n").split("\t")
                    if len(fields) == len(active[2]):
                        active[4].append([unesc(x) for x in fields])
                        if len(active[4]) >= BATCH:
                            write_batch(active[1], active[2], active[3], active[4])
                            active = (active[0], active[1], active[2], active[3], [], active[5] + len(active[4]))
                            bar.set_postfix(table=active[0], rows=f"{active[5]:,}+")

            line = f.readline()
            bar.update(len(line.encode("utf-8", errors="replace")))

        if active is not None:
            if active[4]:
                write_batch(active[1], active[2], active[3], active[4])
            active[1].close()

    bar.close()
    print("\nParquet files written to", OUT)
    for p in sorted(OUT.glob("*.parquet")):
        meta = pq.read_metadata(p)
        print(f"  {p.name:24s} {meta.num_rows:>12,} rows  {p.stat().st_size/1e6:>7.1f} MB")


if __name__ == "__main__":
    main()
