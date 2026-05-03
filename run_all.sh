#!/usr/bin/env bash
# Run the post-embedding pipeline end-to-end.
# Assumes: data/processed/embeddings.npy and embedding_index.parquet already exist.

set -euo pipefail
cd "$(dirname "$0")"

step() {
  echo
  echo "=== [$1] $(date +%H:%M:%S) ==="
  shift
  time "$@"
}

step "split"            python src/split.py
step "graph"            python src/graph_singletons.py --k 50 --threshold 0.85
step "cluster"          python src/cluster_singletons.py
step "baselines"        python src/baselines.py
step "fdr+evalues"      python src/run_pipeline.py --alpha 0.10 --n-null-draws 5000
step "eval"             python src/eval.py

echo
echo "=== done $(date +%H:%M:%S) ==="
echo "results: $(ls -1 results/ | wc -l) files in results/"
