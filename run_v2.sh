#!/usr/bin/env bash
# v2 analysis: cluster-level eval, resolution sweep, qualitative spot-check.
# Assumes v1 already ran (clusters.parquet, cluster_evalues.parquet,
# fdr_rejections.parquet for at least the leiden method exist).

set -euo pipefail
cd "$(dirname "$0")"

step() {
  echo
  echo "=== [$1] $(date +%H:%M:%S) ==="
  shift
  time "$@"
}

step "cluster_eval_v1" python src/cluster_eval.py \
  --methods leiden connected_components hdbscan_emb \
  --gold-thresholds 10 100 1000

step "spotcheck_v1" python src/spotcheck.py \
  --methods leiden connected_components hdbscan_emb \
  --per-bucket 3

step "resolution_sweep" python src/resolution_sweep.py \
  --resolutions 0.5 1.0 2.0 5.0 10.0 \
  --alpha 0.10 --n-null-draws 2000

step "cluster_eval_after_sweep" python src/cluster_eval.py \
  --methods leiden \
  --gold-thresholds 10 100 1000

echo
echo "=== v2 done $(date +%H:%M:%S) ==="
echo "key outputs in results/:"
ls -1 results/*.csv results/spotcheck_*.txt 2>/dev/null | sed 's/^/  /'
