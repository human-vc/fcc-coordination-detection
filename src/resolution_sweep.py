"""Re-run Leiden at multiple resolutions and pipeline at each.

Higher resolution → more, smaller clusters. We want to find the sweet spot
where rejected clusters are interpretably coordinated (cluster_eval shows
high cluster-precision) without becoming so fine-grained that real
campaigns are split apart.

For each resolution r in the sweep:
  1. cluster_singletons.py with --resolution r → clusters.parquet
  2. evalues.py → cluster_evalues.parquet
  3. ebh.py --alpha alpha → fdr_rejections.parquet
  4. cluster_eval.py → cluster_eval_table_r{r}.csv

Existing clusters.parquet is restored after the sweep.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
RES.mkdir(exist_ok=True)
PY = sys.executable


def run(cmd: list[str]) -> None:
    print(f"  $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main(*, resolutions: list[float], alpha: float, n_null_draws: int,
         min_cluster_size: int) -> None:
    clust_path = PROC / "clusters.parquet"
    backup = PROC / "clusters.parquet.bak"
    if clust_path.exists():
        shutil.copy(clust_path, backup)

    summary = []
    for r in resolutions:
        print(f"\n=== resolution {r} ===")
        run([PY, str(ROOT / "src" / "cluster_singletons.py"),
             "--resolution", str(r),
             "--min-cluster-size", str(min_cluster_size)])
        run([PY, str(ROOT / "src" / "evalues.py"),
             "--n-null-draws", str(n_null_draws),
             "--min-cluster-size", str(min_cluster_size)])
        run([PY, str(ROOT / "src" / "ebh.py"), "--alpha", str(alpha)])

        # snapshot cluster + rejection for this resolution
        shutil.copy(clust_path, PROC / f"clusters_leiden_r{r}.parquet")
        shutil.copy(PROC / "fdr_rejections.parquet",
                    RES / f"fdr_rejections_leiden_r{r}.parquet")

        rj = pq.read_table(PROC / "fdr_rejections.parquet").to_pandas()
        cl = pq.read_table(clust_path).to_pandas()
        n_clusters = int((cl["cluster_id"] >= 0).sum())  # actually rows in clusters
        n_unique_clusters = int(cl[cl["cluster_id"] >= 0]["cluster_id"].nunique())
        n_rej = int(rj["rejected_ebh"].sum())
        rej_rows = int(rj.loc[rj["rejected_ebh"], "n"].sum())
        summary.append({
            "resolution": r,
            "candidate_clusters": int(len(rj)),
            "unique_clusters_total": n_unique_clusters,
            "rej_ebh": n_rej,
            "rows_in_rej": rej_rows,
        })

    # restore original clusters.parquet
    if backup.exists():
        shutil.move(backup, clust_path)

    out = pd.DataFrame(summary)
    out_path = RES / "resolution_sweep.csv"
    out.to_csv(out_path, index=False)
    print("\n=== resolution sweep summary ===")
    print(out.to_string(index=False))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--resolutions", nargs="+", type=float,
                   default=[0.5, 1.0, 2.0, 5.0, 10.0])
    p.add_argument("--alpha", type=float, default=0.10)
    p.add_argument("--n-null-draws", type=int, default=5000)
    p.add_argument("--min-cluster-size", type=int, default=5)
    args = p.parse_args()
    main(resolutions=args.resolutions, alpha=args.alpha,
         n_null_draws=args.n_null_draws,
         min_cluster_size=args.min_cluster_size)
