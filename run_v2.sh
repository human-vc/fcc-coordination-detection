#!/usr/bin/env bash
# v2 analysis with all P0 fixes from the pre-flight audit applied.
#
# Critical changes vs v1:
#   - Leiden uses CPM (resolution-limit-free) instead of RBConfiguration; the
#     resolution parameter is now a similarity threshold in [0.85, 0.99].
#     Eliminates the 213K mega-cluster that caused e-value saturation.
#   - evalues.py uses a Beta MOM tail fit on size-bucketed null draws; e-values
#     now span 1 to ~1e6 instead of saturating at ~5e3.
#   - resolution_sweep snapshots per-resolution outputs; no shared-state race.
#
# Assumes: data/processed/embeddings.npy + embedding_index.parquet exist.

set -euo pipefail
cd "$(dirname "$0")"

step() {
  echo
  echo "=== [$1] $(date +%H:%M:%S) ==="
  shift
  time "$@"
}

# Phase 1: regenerate clusters with CPM (must redo because v1 RBConfig
# produced the resolution-limit failure mode).
step "split"            python src/split.py
step "graph"            python src/graph_singletons.py --k 50 --threshold 0.85
step "cluster_cpm"      python src/cluster_singletons.py --partition cpm --resolution 0.90
step "evalues"          python src/evalues.py --n-null-draws 5000
step "ebh"              python src/ebh.py --alpha 0.10

# also baselines on the same A-half (skip minhash by default after the v1 OOM)
step "baselines"        python src/baselines.py --methods connected_components hdbscan_emb
step "evalues+ebh-base" python src/run_pipeline.py --alpha 0.10 --n-null-draws 5000

# Phase 2: cluster-level eval at the new partition.
step "cluster_eval"     python src/cluster_eval.py \
                          --methods leiden connected_components hdbscan_emb \
                          --gold-thresholds 10 100 1000

step "spotcheck"        python src/spotcheck.py \
                          --methods leiden connected_components hdbscan_emb \
                          --per-bucket 3

# Phase 3: CPM resolution sweep (similarity-threshold semantics).
step "resolution_sweep" python src/resolution_sweep.py \
                          --partition cpm \
                          --resolutions 0.85 0.88 0.90 0.93 0.96 \
                          --alpha 0.10 --n-null-draws 5000

# Phase 4: per-resolution cluster_eval (uses snapshot files, no state race).
for r in 0.85 0.88 0.90 0.93 0.96; do
  step "cluster_eval_r$r" python src/cluster_eval.py \
    --methods leiden \
    --cluster-path data/processed/clusters_leiden_r${r}.parquet \
    --rejections-path results/fdr_rejections_leiden_r${r}.parquet \
    --output results/cluster_eval_table_r${r}.csv \
    --gold-thresholds 10 100 1000
done

echo
echo "=== v2 done $(date +%H:%M:%S) ==="
echo "key outputs in results/:"
ls -1 results/*.csv results/spotcheck_*.txt 2>/dev/null | sed 's/^/  /'
