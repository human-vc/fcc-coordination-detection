"""FDR control on cluster-level p/e-values: BH, BY, and e-BH.

We compute three rejection sets at the same alpha so we can compare power:

  - BH (Benjamini-Hochberg 1995): valid under independence / PRDS.
  - BY (Benjamini-Yekutieli 2001): valid under arbitrary dependence
    but pays a c(m) = sum_{k=1..m} 1/k correction (loses ~log m power).
  - e-BH (Wang & Ramdas 2022, JRSS-B): valid under arbitrary dependence
    among the e-values, no extra correction. Comparable to BY in worst-case
    guarantee but typically more powerful with sharp e-values.

References:
  Benjamini & Hochberg (1995). JRSS-B 57(1).
  Benjamini & Yekutieli (2001). Annals of Statistics 29(4).
  Wang & Ramdas (2022). JRSS-B 84(3).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
EVAL_PATH = PROC / "cluster_evalues.parquet"
OUT_PATH = PROC / "fdr_rejections.parquet"


def bh(p: np.ndarray, alpha: float) -> np.ndarray:
    """Benjamini-Hochberg rejection mask."""
    m = len(p)
    order = np.argsort(p)
    p_sorted = p[order]
    thresholds = np.arange(1, m + 1) * alpha / m
    valid = p_sorted <= thresholds
    if not valid.any():
        return np.zeros(m, dtype=bool)
    k = int(np.where(valid)[0].max() + 1)
    rejected = np.zeros(m, dtype=bool)
    rejected[order[:k]] = True
    return rejected


def by(p: np.ndarray, alpha: float) -> np.ndarray:
    """Benjamini-Yekutieli rejection mask (arbitrary dependence)."""
    m = len(p)
    cm = np.sum(1.0 / np.arange(1, m + 1))
    return bh(p, alpha / cm)


def ebh(e: np.ndarray, alpha: float) -> tuple[np.ndarray, int, float]:
    """e-BH rejection: largest k with E_(k) >= m / (k * alpha)."""
    m = len(e)
    order = np.argsort(-e)
    e_sorted = e[order]
    thresholds = m / (np.arange(1, m + 1) * alpha)
    valid = e_sorted >= thresholds
    if not valid.any():
        return np.zeros(m, dtype=bool), 0, float("inf")
    k = int(np.where(valid)[0].max() + 1)
    threshold = float(e_sorted[k - 1])
    rejected = np.zeros(m, dtype=bool)
    rejected[order[:k]] = True
    return rejected, k, threshold


def main(*, alpha: float = 0.10) -> None:
    df = pq.read_table(EVAL_PATH).to_pandas()
    p = df["p"].to_numpy(dtype=np.float64)
    e = df["e"].to_numpy(dtype=np.float64)
    m = len(df)
    print(f"candidates: {m:,}  alpha: {alpha}")

    bh_rej = bh(p, alpha)
    by_rej = by(p, alpha)
    ebh_rej, ebh_k, ebh_thr = ebh(e, alpha)

    df["rejected_bh"] = bh_rej
    df["rejected_by"] = by_rej
    df["rejected_ebh"] = ebh_rej
    df["rejected"] = ebh_rej  # default for downstream eval.py

    print(f"\n{'method':<6} {'rejected':>10} {'rows in clusters':>20}")
    for name, mask in [("BH", bh_rej), ("BY", by_rej), ("e-BH", ebh_rej)]:
        n_rej = int(mask.sum())
        n_rows = int(df.loc[mask, "n"].sum()) if n_rej else 0
        print(f"{name:<6} {n_rej:>10,} {n_rows:>20,}")

    df.to_parquet(OUT_PATH, compression="zstd", index=False)
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--alpha", type=float, default=0.10)
    args = p.parse_args()
    main(alpha=args.alpha)
