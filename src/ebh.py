"""Apply the e-BH procedure (Wang & Ramdas 2022, JRSS-B).

Given e-values E_1, ..., E_m, the e-BH procedure rejects the largest k for
which sorting in *descending* order yields E_(k) >= m / (k * alpha).  The
resulting rejection set has FDR <= alpha under arbitrary dependence among
the E_j (no PRDS or independence assumption needed).

Reference:
  Wang, R., & Ramdas, A. (2022). False discovery rate control with e-values.
  Journal of the Royal Statistical Society Series B, 84(3), 822-852.
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
OUT_PATH = PROC / "ebh_rejections.parquet"


def ebh(e: np.ndarray, alpha: float) -> tuple[np.ndarray, int, float]:
    """e-BH procedure. Returns (rejected_mask, k_star, threshold).

    Rejects the indices with the k_star largest e-values.
    """
    m = len(e)
    order = np.argsort(-e)  # descending
    e_sorted = e[order]
    # find largest k such that e_(k) >= m / (k * alpha)
    ranks = np.arange(1, m + 1)
    threshold_per_k = m / (ranks * alpha)
    valid = e_sorted >= threshold_per_k
    if not valid.any():
        return np.zeros(m, dtype=bool), 0, float("inf")
    k_star = int(np.where(valid)[0].max() + 1)
    threshold = float(e_sorted[k_star - 1])
    rejected = np.zeros(m, dtype=bool)
    rejected[order[:k_star]] = True
    return rejected, k_star, threshold


def main(*, alpha: float = 0.10) -> None:
    print(f"loading e-values from {EVAL_PATH}")
    df = pq.read_table(EVAL_PATH).to_pandas()
    e = df["e"].to_numpy()
    m = len(e)
    print(f"  {m:,} candidate clusters")

    rejected, k_star, threshold = ebh(e, alpha)
    print(f"\ne-BH at alpha={alpha}")
    print(f"  rejected k* = {k_star:,} clusters out of {m:,}")
    print(f"  e threshold = {threshold:.3f}")
    if k_star > 0:
        n_in_rejected = int(df.loc[rejected, "n"].sum())
        print(f"  rows in rejected clusters: {n_in_rejected:,}")
        print(f"  cluster size range: {int(df.loc[rejected, 'n'].min())}"
              f" – {int(df.loc[rejected, 'n'].max())}")

    df["rejected"] = rejected
    df.to_parquet(OUT_PATH, compression="zstd", index=False)
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--alpha", type=float, default=0.10)
    args = p.parse_args()
    main(alpha=args.alpha)
