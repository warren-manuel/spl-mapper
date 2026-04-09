#!/usr/bin/env bash

set -euo pipefail

RESULT_DIR="results/20260317_medgemma"
CSV_PATH="results/contra_gold_100_1.csv"
MODEL_ID="google/medgemma-27b-text-it"
GPU_IDS="0"
START_STEP="${START_STEP:-1}"
LOG_FILE="${PIPELINE_LOG_FILE:-$RESULT_DIR/pipeline.log}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if ! [[ "$START_STEP" =~ ^[1-7]$ ]]; then
  echo "START_STEP must be an integer from 1 to 7. Got: $START_STEP" >&2
  exit 1
fi

mkdir -p "$RESULT_DIR"

if [[ "${PIPELINE_LOG_INITIALIZED:-0}" != "1" ]]; then
  export PIPELINE_LOG_INITIALIZED=1
  exec > >(tee -a "$LOG_FILE") 2>&1
fi

if (( START_STEP <= 1 )); then
  echo "[1/7] CONTRA extraction"
  python multi_gpu_contra_extract.py \
    --csv_path "$CSV_PATH" \
    --text_col CONTRA_TEXT \
    --out_dir "$RESULT_DIR" \
    --model_id "$MODEL_ID" \
    --max_new_tokens 512 \
    --gpus "$GPU_IDS" \
    --index_col SPL_SET_ID
fi

shopt -s nullglob
pred_jsonls=( "$RESULT_DIR"/out_gpu*.jsonl )
shopt -u nullglob

if [[ ${#pred_jsonls[@]} -eq 0 ]]; then
  echo "No extraction shard files found under $RESULT_DIR/out_gpu*.jsonl" >&2
  exit 1
fi

if (( START_STEP <= 2 )); then
  echo "[2/7] Mapper"
  python hyb_mapper.py \
    --pred-jsonl "${pred_jsonls[@]}" \
    --out-jsonl "$RESULT_DIR/mapped_hits.jsonl" \
    --run-es \
    --item-term-keys ci_text condition_text substance_text severity_span course_span
fi

if (( START_STEP <= 3 )); then
  echo "[3/7] Map verifier"
  python3 map_verify.py \
    --mapped-jsonl "$RESULT_DIR/mapped_hits.jsonl" \
    --out-jsonl "$RESULT_DIR/verified_hits.jsonl" \
    --gpu-ids "$GPU_IDS"
fi

if (( START_STEP <= 4 )); then
  echo "[4/7] Prefilter"
  python3 prefilter.py \
    --mapped-jsonl "$RESULT_DIR/mapped_hits.jsonl" \
    --out-json "$RESULT_DIR/prefilter_cache.json" \
    --snomed-source-dir snomed_us_source \
    --max-workers 4
fi

if (( START_STEP <= 5 )); then
  echo "[5/7] Postcoord"
  python3 postcord.py \
    --mapped-jsonl "$RESULT_DIR/mapped_hits.jsonl" \
    --verified-jsonl "$RESULT_DIR/verified_hits.jsonl" \
    --out-jsonl "$RESULT_DIR/postcoord_hits.jsonl" \
    --gpu-ids "$GPU_IDS" \
    --filter-by-range \
    --prefilter-cache "$RESULT_DIR/prefilter_cache.json"
fi

if (( START_STEP <= 6 )); then
  echo "[6/7] Aggregate results"
  python3 result_agg.py \
    --verified-jsonl "$RESULT_DIR/verified_hits.jsonl" \
    --postcoord-jsonl "$RESULT_DIR/postcoord_hits.jsonl" \
    --out-jsonl "$RESULT_DIR/aggregated_hits.jsonl" \
    --out-csv "$RESULT_DIR/aggregated_hits.csv"
fi

if (( START_STEP <= 7 )); then
  echo "[7/7] Evaluate"
  python3 evaluate_agg_results_1.py \
    --pred-csv "$RESULT_DIR/aggregated_hits.csv" \
    --gold-csv "$CSV_PATH" \
    --out-json "$RESULT_DIR/eval_metrics.json" \
    --out-details-csv "$RESULT_DIR/eval_details.csv" \
    --discard-na-gold-expression \
    --require-semantic
fi

echo "Pipeline completed successfully."
