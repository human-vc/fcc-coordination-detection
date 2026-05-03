"""End-to-end pipeline: run e-values + e-BH + eval for the Leiden method
and each baseline on the same sample-split.

Reads:
  data/processed/clusters.parquet                  (Leiden)
  data/processed/clusters_minhash_lsh.parquet
  data/processed/clusters_connected_components.parquet
  data/processed/clusters_hdbscan_emb.parquet

Writes:
  results/method_comparison.csv
  results/predictions_<method>.parquet
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

CLUST_FILE = PROC / "clusters.parquet"
EVAL_FILE = PROC / "cluster_evalues.parquet"
FDR_FILE = PROC / "fdr_rejections.parquet"


def run_for_method(method: str, alpha: float, *, n_null_draws: int,
                   min_cluster_size: int) -> dict:
    """Run evalues -> ebh -> read rejections; return summary dict."""
    src_clust = PROC / f"clusters_{method}.parquet" if method != "leiden" else CLUST_FILE
    if not src_clust.exists():
        print(f"  [{method}] skip — no cluster file at {src_clust}")
        return {"method": method, "status": "missing"}

    # swap clusters file in place so evalues.py / ebh.py find it
    if method != "leiden":
        backup = CLUST_FILE.with_suffix(".parquet.bak")
        if CLUST_FILE.exists():
            shutil.move(CLUST_FILE, backup)
        shutil.copy(src_clust, CLUST_FILE)
    try:
        py = sys.executable
        subprocess.run([py, str(ROOT / "src" / "evalues.py"),
                        "--n-null-draws", str(n_null_draws),
                        "--min-cluster-size", str(min_cluster_size)], check=True)
        subprocess.run([py, str(ROOT / "src" / "ebh.py"),
                        "--alpha", str(alpha)], check=True)
        rj = pq.read_table(FDR_FILE).to_pandas()
        ev = pq.read_table(EVAL_FILE).to_pandas()
    finally:
        if method != "leiden":
            backup = CLUST_FILE.with_suffix(".parquet.bak")
            if backup.exists():
                shutil.move(backup, CLUST_FILE)

    # carry per-method copies for later inspection
    shutil.copy(EVAL_FILE, RES / f"cluster_evalues_{method}.parquet")
    shutil.copy(FDR_FILE, RES / f"fdr_rejections_{method}.parquet")

    summary = {
        "method": method,
        "candidate_clusters": int(len(rj)),
        "median_e": float(ev["e"].median()) if len(ev) else 0.0,
        "rej_bh": int(rj["rejected_bh"].sum()),
        "rej_by": int(rj["rejected_by"].sum()),
        "rej_ebh": int(rj["rejected_ebh"].sum()),
        "rows_in_ebh_rej": int(rj.loc[rj["rejected_ebh"], "n"].sum()),
    }
    print(f"  [{method}] candidates={summary['candidate_clusters']:,}, "
          f"BH={summary['rej_bh']}, BY={summary['rej_by']}, "
          f"e-BH={summary['rej_ebh']}, rows={summary['rows_in_ebh_rej']:,}")
    return summary


def main(*, alpha: float = 0.10, n_null_draws: int = 5000,
         min_cluster_size: int = 5) -> None:
    methods = ["leiden", "connected_components", "hdbscan_emb", "minhash_lsh"]
    rows = []
    for m in methods:
        print(f"\n=== {m} ===")
        rows.append(run_for_method(m, alpha,
                                   n_null_draws=n_null_draws,
                                   min_cluster_size=min_cluster_size))

    out = pd.DataFrame(rows)
    out_path = RES / "method_comparison.csv"
    out.to_csv(out_path, index=False)

    print("\n=== final comparison (alpha=" + str(alpha) + ") ===")
    print(out.to_string(index=False))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--alpha", type=float, default=0.10)
    p.add_argument("--n-null-draws", type=int, default=5000)
    p.add_argument("--min-cluster-size", type=int, default=5)
    args = p.parse_args()
    main(alpha=args.alpha, n_null_draws=args.n_null_draws,
         min_cluster_size=args.min_cluster_size)
