from __future__ import annotations
from pathlib import Path
import duckdb
ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / 'data' / 'processed'

def main() -> None:
    con = duckdb.connect()
    for p in sorted(PROC.glob('*.parquet')):
        con.execute(f"CREATE OR REPLACE VIEW {p.stem} AS SELECT * FROM read_parquet('{p}')")
    print('Tables:')
    for p in sorted(PROC.glob('*.parquet')):
        n = con.execute(f'SELECT count(*) FROM {p.stem}').fetchone()[0]
        print(f'  {p.stem:24s} {n:>12,} rows  ({p.stat().st_size / 1000000.0:>6.1f} MB)')
    print('\nSubmissions by year:')
    rows = con.execute('\n        SELECT extract(year from cast(date_received as date)) as y, count(*) as n\n        FROM submissions WHERE date_received IS NOT NULL\n        GROUP BY y ORDER BY y\n    ').fetchall()
    for (y, n) in rows:
        print(f'  {int(y)}: {n:>10,}')
    print('\nSubmission types (top 10):')
    rows = con.execute('\n        SELECT submission_type, count(*) as n FROM submissions\n        GROUP BY submission_type ORDER BY n DESC LIMIT 10\n    ').fetchall()
    for (t, n) in rows:
        print(f'  {str(t):30s} {n:>10,}')
    print('\nExpress vs standard:')
    rows = con.execute("\n        SELECT CASE WHEN express_comment = '1' THEN 'express' ELSE 'standard' END as kind,\n               count(*) as n\n        FROM submissions GROUP BY kind\n    ").fetchall()
    for (k, n) in rows:
        print(f'  {k:15s} {n:>10,}')
    print('\nNear-duplicate edges:', con.execute('SELECT count(*) FROM near_duplicates').fetchone()[0])
    print('Exact-duplicate edges:', con.execute('SELECT count(*) FROM exact_duplicates').fetchone()[0])
    print('\nInterest-group labels (top business categories):')
    rows = con.execute('\n        SELECT business, count(*) as n FROM interest_groups\n        GROUP BY business ORDER BY n DESC LIMIT 10\n    ').fetchall()
    for (b, n) in rows:
        print(f'  {str(b)[:40]:40s} {n:>8,}')
    print('\nDegree distribution of near-duplicate graph (top documents):')
    rows = con.execute('\n        WITH deg AS (\n          SELECT target_document_id AS doc, count(*) AS d FROM near_duplicates GROUP BY doc\n          UNION ALL\n          SELECT duplicate_document_id AS doc, count(*) AS d FROM near_duplicates GROUP BY doc\n        )\n        SELECT doc, sum(d) as total_d FROM deg GROUP BY doc ORDER BY total_d DESC LIMIT 10\n    ').fetchall()
    for (d, td) in rows:
        print(f'  {d:30s} {td:>8,} edges')
if __name__ == '__main__':
    main()
