"""Evaluate detection quality against template-size gold labels.

Gold definition:
    coordinated_gold = template_size >= GOLD_THRESHOLD
where template_size is the count of submissions sharing the same comment_id.

The interesting evaluation regime is the *hard* one: among singletons
(template_size == 1), how many does the e-BH procedure flag as part of a
rejected cluster, and what does spot-checking reveal?

Outputs both a confusion matrix on the all-clusters regime and a singleton-
only analysis.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
RES.mkdir(exist_ok=True)


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    tp = int(((y_true) & (y_pred)).sum())
    fp = int((~y_true & (y_pred)).sum())
    fn = int(((y_true) & ~y_pred).sum())
    tn = int((~y_true & ~y_pred).sum())
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn}


def metrics(c: dict[str, int]) -> dict[str, float]:
    p = c["TP"] / max(1, c["TP"] + c["FP"])
    r = c["TP"] / max(1, c["TP"] + c["FN"])
    return {"precision": p, "recall": r, "f1": 2 * p * r / max(1e-12, p + r)}


def main(*, gold_threshold: int = 10) -> None:
    cl = pq.read_table(PROC / "clusters.parquet").to_pandas()
    rj = pq.read_table(PROC / "ebh_rejections.parquet").to_pandas()

    rejected_cids = set(rj.loc[rj["rejected"], "cluster_id"].astype(int).tolist())
    cl["predicted_coordinated"] = cl["cluster_id"].isin(rejected_cids)
    cl["gold_coordinated"] = cl["template_size"] >= gold_threshold

    print(f"corpus size: {len(cl):,} unique comments")
    print(f"gold threshold: template_size >= {gold_threshold}")
    print(f"  gold positive: {int(cl['gold_coordinated'].sum()):,} comments")
    print(f"  predicted positive: {int(cl['predicted_coordinated'].sum()):,} comments")

    c_all = confusion(cl["gold_coordinated"].to_numpy(),
                      cl["predicted_coordinated"].to_numpy())
    m_all = metrics(c_all)
    print("\nFull corpus (per-comment):")
    print(f"  TP={c_all['TP']:,}  FP={c_all['FP']:,}  "
          f"FN={c_all['FN']:,}  TN={c_all['TN']:,}")
    print(f"  precision={m_all['precision']:.3f}  recall={m_all['recall']:.3f}  "
          f"f1={m_all['f1']:.3f}")

    # singleton-only regime: comments with template_size == 1
    sg = cl[cl["template_size"] == 1].copy()
    print(f"\nSingleton regime (template_size == 1): {len(sg):,} comments")
    sg_pred = int(sg["predicted_coordinated"].sum())
    print(f"  predicted coordinated among singletons: {sg_pred:,}")
    print(f"  → these are the soft-coordination detections that template-counting misses")

    # cluster-level summary
    print("\nRejected cluster size distribution:")
    rej_sizes = rj.loc[rj["rejected"], "n"]
    if len(rej_sizes):
        for q in [0.1, 0.5, 0.9]:
            print(f"  size at quantile {q:.1f}: {int(rej_sizes.quantile(q))}")
        print(f"  largest rejected cluster: {int(rej_sizes.max())}")

    # save row-level predictions
    out_path = RES / "predictions.parquet"
    cl[["row_id", "comment_id", "template_size", "cluster_id",
        "predicted_coordinated", "gold_coordinated"]].to_parquet(
        out_path, compression="zstd", index=False)
    print(f"\nwrote per-comment predictions to {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--gold-threshold", type=int, default=10)
    args = p.parse_args()
    main(gold_threshold=args.gold_threshold)
