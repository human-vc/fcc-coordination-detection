#!/usr/bin/env bash
# Overnight second-corpus pipeline.
#
# Run as:
#   cd /Users/jacobcrainic/fcc-coordination-detection
#   nohup ./scripts/run_second_corpus_overnight.sh > overnight.log 2>&1 &
#   disown
#
# Each step is idempotent: re-running skips completed steps.
# Edit STEP_1_SOURCE below to pick where the second-corpus comments come from.

set -uo pipefail
cd "$(dirname "$0")/.."

# ============== CONFIG ==============
CORPUS=${CORPUS:-fcc14_28}
RAW_DIR=data/raw_${CORPUS}
PROC_DIR=data/processed_${CORPUS}
RES_DIR=results_${CORPUS}
mkdir -p "$RAW_DIR" "$PROC_DIR" "$RES_DIR"

# Pick ONE source. Default: FCC ECFS API (needs FCC_API_KEY env var).
# Other options below.
STEP_1_SOURCE=${STEP_1_SOURCE:-fcc_ecfs}   # fcc_ecfs | regulations_gov | manual

# Whitening / q̂ / pipeline knobs
WHITEN_K=${WHITEN_K:-5}
Q_K=${Q_K:-50}
ALPHA=${ALPHA:-0.10}
MIN_CLUSTER_SIZE=${MIN_CLUSTER_SIZE:-8}

PY=.venv/bin/python
log() { echo "[$(date '+%H:%M:%S')] $*"; }
done_marker() { test -f "${PROC_DIR}/.${1}_done"; }
mark_done() { touch "${PROC_DIR}/.${1}_done"; }

# ============== STEP 1: download corpus ==============
if ! done_marker step1_raw; then
  log "STEP 1: download corpus ($STEP_1_SOURCE)"
  case "$STEP_1_SOURCE" in
    fcc_ecfs)
      if [ -z "${FCC_API_KEY:-}" ]; then
        log "ERROR: FCC_API_KEY env var not set."
        log "  Get one at: https://api.regulations.gov/  (or the FCC ECFS site)"
        log "  Then run:  export FCC_API_KEY=...  before launching this script."
        exit 1
      fi
      $PY scripts/fetch_fcc14_28.py --proceeding 14-28 \
          --output-dir "$RAW_DIR" --api-key "$FCC_API_KEY" \
          --max-records 1600000 --limit 100 --delay 0.15 \
          || { log "fetch failed"; exit 2; }
      ;;
    regulations_gov)
      if [ -z "${REGULATIONS_API_KEY:-}" ]; then
        log "ERROR: REGULATIONS_API_KEY not set."
        exit 1
      fi
      $PY scripts/fetch_regulations_gov.py --docket "EPA-HQ-OAR-2025-0194" \
          --output-dir "$RAW_DIR" --api-key "$REGULATIONS_API_KEY" \
          || { log "fetch failed"; exit 2; }
      ;;
    manual)
      log "expected pre-existing files in $RAW_DIR (e.g., comments.csv or .parquet)"
      [ "$(ls -A "$RAW_DIR" 2>/dev/null)" ] || { log "ERROR: $RAW_DIR is empty"; exit 1; }
      ;;
    *)
      log "ERROR: unknown STEP_1_SOURCE: $STEP_1_SOURCE"
      exit 1
      ;;
  esac
  mark_done step1_raw
else
  log "STEP 1 already done — skipping"
fi

# ============== STEP 2: ingest to parquet ==============
if ! done_marker step2_parquet; then
  log "STEP 2: ingest raw data to parquet"
  $PY scripts/ingest_second_corpus.py --input-dir "$RAW_DIR" \
      --output-dir "$PROC_DIR" \
      || { log "ingest failed"; exit 2; }
  mark_done step2_parquet
else
  log "STEP 2 already done — skipping"
fi

# ============== STEP 3: embed via SBERT ==============
if ! done_marker step3_embed; then
  log "STEP 3: embed with SBERT (MiniLM-L6-v2)"
  log "  uses MPS on M3 Pro by default; on Brev set CUDA_VISIBLE_DEVICES first."
  $PY src/embed.py \
    --output-path "$PROC_DIR/embeddings.npy" \
    --batch-size ${EMBED_BATCH_SIZE:-256} \
    || { log "embed failed"; exit 2; }
  mark_done step3_embed
else
  log "STEP 3 already done — skipping"
fi

# ============== STEP 4: whiten ==============
if ! done_marker step4_whiten; then
  log "STEP 4: all-but-the-top whitening (k=$WHITEN_K)"
  $PY src/whiten.py --k $WHITEN_K \
    --input-path "$PROC_DIR/embeddings.npy" \
    --output-path "$PROC_DIR/embeddings_white_k${WHITEN_K}.npy" \
    || { log "whiten failed"; exit 2; }
  mark_done step4_whiten
else
  log "STEP 4 already done — skipping"
fi

# ============== STEP 5: cluster (Leiden CPM) ==============
if ! done_marker step5_cluster; then
  log "STEP 5: Leiden CPM clustering at γ=0.90"
  $PY src/cluster_singletons.py --resolution 0.90 \
      --min-cluster-size 5 \
      --embedding-path "$PROC_DIR/embeddings_white_k${WHITEN_K}.npy" \
      --output-path "$PROC_DIR/clusters_leiden_r0.9.parquet" \
      || { log "cluster failed"; exit 2; }
  mark_done step5_cluster
else
  log "STEP 5 already done — skipping"
fi

# ============== STEP 6: fit q̂ on cluster-disjoint singletons ==============
if ! done_marker step6_q; then
  log "STEP 6: fit movMF q̂ on singletons"
  $PY src/q_movmf.py \
    --input-path "$PROC_DIR/embeddings_white_k${WHITEN_K}.npy" \
    --output-path "$PROC_DIR/q_movmf_singletons.pkl" \
    --Ks $Q_K \
    || { log "q̂ failed"; exit 2; }
  mark_done step6_q
else
  log "STEP 6 already done — skipping"
fi

# ============== STEP 7: split-LRT compound e-values ==============
if ! done_marker step7_lrt; then
  log "STEP 7: split-LRT compound e-values"
  $PY src/evalues_lrt.py \
    --embedding-path "$PROC_DIR/embeddings_white_k${WHITEN_K}.npy" \
    --cluster-path "$PROC_DIR/clusters_leiden_r0.9.parquet" \
    --q-path "$PROC_DIR/q_movmf_singletons.pkl" \
    --output-path "$PROC_DIR/cluster_evalues_lrt.parquet" \
    --min-cluster-size $MIN_CLUSTER_SIZE \
    || { log "evalues_lrt failed"; exit 2; }
  mark_done step7_lrt
else
  log "STEP 7 already done — skipping"
fi

# ============== STEP 8: e-BH + report ==============
if ! done_marker step8_report; then
  log "STEP 8: e-BH at α=$ALPHA + headline report"
  $PY scripts/report_second_corpus.py \
      --evalues-path "$PROC_DIR/cluster_evalues_lrt.parquet" \
      --alpha $ALPHA \
      --output-path "$RES_DIR/headline_report.json" \
      || { log "report failed"; exit 2; }
  mark_done step8_report
else
  log "STEP 8 already done — skipping"
fi

log ""
log "=========================================================="
log "OVERNIGHT PIPELINE FINISHED"
log "Headline report: $RES_DIR/headline_report.json"
log "=========================================================="
cat "$RES_DIR/headline_report.json" 2>/dev/null
