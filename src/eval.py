"""Evaluate detection quality across methods and gold definitions.

Two evaluation regimes:

1. **Mega-template sanity check.** Comments with template_size >= 100 are
   essentially-certain coordination (they were submitted thousands of times
   identically). A working detector must rediscover them with high precision.
   This is *not* the headline result — it just confirms the pipeline is sane.

2. **Singleton soft-coordination regime.** Among comments with template_size
   == 1 (genuinely unique submissions), how many does each method flag as
   part of an FDR-rejected cluster? These are the *interesting* detections —
   coordinated content paraphrased to evade exact-match counting. There is no
   clean ground truth; the v1 paper reports counts and qualitative
   spot-checks.
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
    return {
        "TP": int((y_true & y_pred).sum()),
        "FP": int((~y_true & y_pred).sum()),
        "FN": int((y_true & ~y_pred).sum()),
        "TN": int((~y_true & ~y_pred).sum()),
    }


def metrics(c: dict[str, int]) -> dict[str, float]:
    p = c["TP"] / max(1, c["TP"] + c["FP"])
    r = c["TP"] / max(1, c["TP"] + c["FN"])
    return {"precision": p, "recall": r,
            "f1": 2 * p * r / max(1e-12, p + r)}


def load_method(method: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if method == "leiden":
        cl = pq.read_table(PROC / "clusters.parquet").to_pandas()
    else:
        cl = pq.read_table(PROC / f"clusters_{method}.parquet").to_pandas()
    rj = pq.read_table(RES / f"fdr_rejections_{method}.parquet").to_pandas()
    return cl, rj


def evaluate_method(method: str, *, gold_thresholds: list[int],
                    spot_check_n: int, rng: np.random.Generator) -> list[dict]:
    cl, rj = load_method(method)
    rejected_cids = set(rj.loc[rj["rejected_ebh"], "cluster_id"].astype(int).tolist())
    cl["pred"] = cl["cluster_id"].isin(rejected_cids)

    rows = []
    for gt in gold_thresholds:
        cl["gold"] = cl["template_size"] >= gt
        c = confusion(cl["gold"].to_numpy(), cl["pred"].to_numpy())
        m = metrics(c)
        rows.append({"method": method, "gold_threshold": gt,
                     **c, **m})

    # singleton soft-coordination
    sing = cl[cl["template_size"] == 1]
    n_sing = len(sing)
    n_sing_pred = int(sing["pred"].sum())
    rows.append({"method": method, "gold_threshold": -1,
                 "TP": -1, "FP": -1, "FN": -1, "TN": -1,
                 "precision": float("nan"), "recall": float("nan"),
                 "f1": float("nan"),
                 "singletons_total": n_sing,
                 "singletons_predicted_coord": n_sing_pred})

    # spot-check sample of singleton-only clusters
    sing_clusters = (sing.loc[sing["pred"]]
                       .groupby("cluster_id").size()
                       .sort_values(ascending=False))
    if len(sing_clusters):
        pick = sing_clusters.iloc[:min(len(sing_clusters), spot_check_n)]
        spot_path = RES / f"spotcheck_{method}.csv"
        spot_rows = []
        for cid, sz in pick.items():
            mems = sing[sing["cluster_id"] == cid]["row_id"].head(5).tolist()
            spot_rows.append({"cluster_id": int(cid), "size": int(sz),
                              "row_ids_sample": mems})
        pd.DataFrame(spot_rows).to_csv(spot_path, index=False)

    return rows


def main(*, methods: list[str], gold_thresholds: list[int],
         spot_check_n: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    all_rows = []
    for method in methods:
        try:
            all_rows.extend(evaluate_method(method, gold_thresholds=gold_thresholds,
                                            spot_check_n=spot_check_n, rng=rng))
        except FileNotFoundError as e:
            print(f"  skip {method}: {e}")
    out = pd.DataFrame(all_rows)
    out.to_csv(RES / "eval_table.csv", index=False)
    print(out.to_string(index=False))
    print(f"\nwrote {RES/'eval_table.csv'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--methods", nargs="+",
                   default=["leiden", "connected_components",
                            "hdbscan_emb", "minhash_lsh"])
    p.add_argument("--gold-thresholds", nargs="+", type=int,
                   default=[10, 100, 1000, 10000])
    p.add_argument("--spot-check-n", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    main(methods=args.methods, gold_thresholds=args.gold_thresholds,
         spot_check_n=args.spot_check_n, seed=args.seed)
