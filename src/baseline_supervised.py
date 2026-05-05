"""Supervised classifier baseline on NYAG-attributed astroturf labels.

For reviewers asking "why not just train on the labels you have?" — this
trains logistic-regression and gradient-boosted classifiers on per-cluster
features from the existing pipelines (cohesion, kappa, log e-values),
using FOIA-attributed astroturf as the positive class. Reports cross-validated
precision/recall/F1/AUC.

Comparison framing: the supervised classifier *requires* labels and
generalizes only to the labelled-population's distribution, while the
unsupervised LRT/cohesion procedures provide FDR control without labels.
The right comparison is at matched recall: how does supervised precision
on attributed astroturf compare to LRT's precision at the same recall?
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, precision_score,
                              recall_score, roc_auc_score, f1_score)
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"


def assemble_features(att_path: Path = RES / "attribution_table_r0.9.csv",
                       lrt_path: Path = PROC / "cluster_evalues_lrt_cluster_aware.parquet",
                       mixlr_path: Path = PROC / "cluster_evalues_mixlr_r0.9.parquet",
                       cohesion_path: Path = PROC / "cluster_evalues_leiden_r0.9.parquet"
                       ) -> pd.DataFrame:
    att = pd.read_csv(att_path)
    lrt = pq.read_table(lrt_path).to_pandas()[["cluster_id", "n", "kappa_hat", "log_e"]]
    lrt = lrt.rename(columns={"log_e": "log_e_lrt"})
    mixlr = pq.read_table(mixlr_path).to_pandas()[["cluster_id", "log_e", "T_obs"]]
    mixlr = mixlr.rename(columns={"log_e": "log_e_mixlr"})
    coh = pq.read_table(cohesion_path).to_pandas()[["cluster_id", "T_obs", "p", "e"]]
    coh = coh.rename(columns={"T_obs": "T_obs_coh", "p": "p_coh", "e": "e_coh"})

    df = lrt.merge(mixlr, on="cluster_id", how="inner") \
            .merge(coh, on="cluster_id", how="inner") \
            .merge(att[["cluster_id", "frac_astroturf", "frac_advocacy"]],
                   on="cluster_id", how="left")
    df["y"] = (df["frac_astroturf"].fillna(0) >= 0.5).astype(int)
    df["log_e_coh"] = np.log(df["e_coh"].astype(float).clip(1e-300, 1e300))
    df = df.drop(columns=["e_coh"])
    return df


def main() -> None:
    df = assemble_features()
    print(f"feature table: {len(df):,} clusters with all features")
    print(f"  astroturf positives: {int(df['y'].sum()):,} ({100*df['y'].mean():.1f}%)")

    feature_cols = ["n", "kappa_hat", "log_e_lrt", "log_e_mixlr",
                     "T_obs", "T_obs_coh", "p_coh", "log_e_coh"]
    X = df[feature_cols].fillna(df[feature_cols].median()).to_numpy()
    y = df["y"].to_numpy()
    print(f"features: {feature_cols}")

    # Stratified 5-fold CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print()
    print(f"{'model':<25}{'AUC':>8}{'AP':>8}{'P@best F1':>11}{'R@best F1':>11}{'F1':>8}")
    print('-' * 75)
    for name, Model in [("logistic", LogisticRegression),
                          ("gradient_boosted", GradientBoostingClassifier)]:
        aucs, aps, ps, rs, f1s = [], [], [], [], []
        for tr, te in skf.split(X, y):
            if Model == LogisticRegression:
                clf = Model(max_iter=2000, class_weight="balanced")
            else:
                clf = Model(n_estimators=200, max_depth=3, random_state=0)
            clf.fit(X[tr], y[tr])
            scores = clf.predict_proba(X[te])[:, 1]
            aucs.append(roc_auc_score(y[te], scores))
            aps.append(average_precision_score(y[te], scores))
            # find threshold optimizing F1
            best_f1, best_p, best_r = 0, 0, 0
            for t in np.linspace(0.01, 0.99, 99):
                pred = (scores >= t).astype(int)
                if pred.sum() == 0:
                    continue
                p = precision_score(y[te], pred, zero_division=0)
                r = recall_score(y[te], pred, zero_division=0)
                f1 = 2*p*r/(p+r) if (p+r) > 0 else 0
                if f1 > best_f1:
                    best_f1, best_p, best_r = f1, p, r
            ps.append(best_p); rs.append(best_r); f1s.append(best_f1)
        print(f"{name:<25}"
              f"{np.mean(aucs):>8.3f}"
              f"{np.mean(aps):>8.3f}"
              f"{np.mean(ps):>11.3f}"
              f"{np.mean(rs):>11.3f}"
              f"{np.mean(f1s):>8.3f}")

    # Comparison at matched recall: at recall = 100% on astroturf, what's precision?
    # Train on all, evaluate on all (to give the supervised classifier its best shot)
    print()
    print("=== full-data-trained (best-case supervised) ===")
    for name, Model in [("logistic", LogisticRegression),
                          ("gradient_boosted", GradientBoostingClassifier)]:
        if Model == LogisticRegression:
            clf = Model(max_iter=2000, class_weight="balanced")
        else:
            clf = Model(n_estimators=200, max_depth=3, random_state=0)
        clf.fit(X, y)
        scores = clf.predict_proba(X)[:, 1]
        # At threshold giving recall = 1.0 on astroturf
        for target_recall in [1.0, 0.99, 0.95, 0.90]:
            sorted_scores = np.sort(scores[y == 1])
            n_target = max(1, int(np.ceil(len(sorted_scores) * (1 - target_recall))))
            t = sorted_scores[n_target - 1] if n_target <= len(sorted_scores) else 0
            pred = (scores >= t).astype(int)
            p = precision_score(y, pred, zero_division=0)
            r = recall_score(y, pred, zero_division=0)
            print(f"  {name} @ recall={target_recall:.2f}: precision={100*p:.1f}%, "
                  f"actual_recall={100*r:.1f}%, n_flagged={int(pred.sum()):,}")

    # Save the best classifier's outputs for table-building
    clf = GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=0)
    clf.fit(X, y)
    df["score_supervised"] = clf.predict_proba(X)[:, 1]
    df.to_csv(RES / "supervised_baseline_scores.csv", index=False)
    print()
    print(f"wrote {RES}/supervised_baseline_scores.csv")


if __name__ == "__main__":
    main()
