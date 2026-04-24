#!/usr/bin/env bash
# Factorial isolation test runner: 2^3 = 8 combinations of
#   USE_BM25_TUNING  (1=tuned b=0.5, 0=original b=0.75)
#   USE_ANCESTOR_PATHS (1=on, 0=off)
#   USE_STRICT_PROMPTS (1=strict 20260407 prompt, 0=original prompt)
#
# Prerequisites:
#   - snomed_ct_es_index (tuned, b=0.5) must already exist.
#   - snomed_ct_es_index_original (b=0.75) must be pre-built once:
#       python build_es_index.py --index-name snomed_ct_es_index_original --bm25-b 0.75 --rebuild
#
# Usage:  bash run_isolation_tests.sh
set -euo pipefail

SPL_LIST="results/setid_100.txt"
GOLD_CSV="results/contra_gold_100_3.csv"
BASE_DIR="results/$(date +%Y%m%d)_isolation"

# Shared extraction cache — built on the first run, reused by all subsequent runs.
# This skips the extract_items LLM call (7 of 8 runs save ~N_SPL * extraction_time).
EXTRACTION_CACHE="${BASE_DIR}/extraction_cache.jsonl"

run_exp() {
    local tag=$1 bm25=$2 anc=$3 strict=$4
    local out="$BASE_DIR/$tag"
    mkdir -p "$out"
    echo ""
    echo "=== Running: $tag  (BM25=$bm25, ANC=$anc, STRICT=$strict) ==="
    USE_BM25_TUNING=$bm25 \
    USE_ANCESTOR_PATHS=$anc \
    USE_STRICT_PROMPTS=$strict \
        python langgraph_agent_runner.py \
            --spl-list               "$SPL_LIST" \
            --out-jsonl              "$out/agent_results.jsonl" \
            --aggregated-jsonl       "$out/aggregated_results.jsonl" \
            --aggregated-csv         "$out/aggregated_hits.csv" \
            --gold-csv               "$GOLD_CSV" \
            --eval-json              "$out/eval_metrics.json" \
            --eval-details-csv       "$out/evaluation_details.csv" \
            --audit-jsonl            "$out/runtime_audit.jsonl" \
            --extracted-items-cache  "$EXTRACTION_CACHE"
}

# ── 8 combinations (binary factorial) ──────────────────────────────────────
#        tag                   bm25  anc  strict
run_exp  "000_baseline"          0    0    0
run_exp  "100_bm25_only"         1    0    0
run_exp  "010_anc_only"          0    1    0
run_exp  "001_strict_only"       0    0    1
run_exp  "110_bm25_anc"          1    1    0
run_exp  "101_bm25_strict"       1    0    1
run_exp  "011_anc_strict"        0    1    1
run_exp  "111_all_on"            1    1    1

echo ""
echo "=== All 8 runs complete. Results in $BASE_DIR ==="
echo ""
echo "── Concept-level F1 summary ──"
for d in "$BASE_DIR"/*/; do
    tag=$(basename "$d")
    f1=$(python3 -c "
import json, sys
try:
    d = json.load(open('${d}eval_metrics.json'))
    print(d.get('concept', {}).get('f1', '?'))
except Exception as e:
    print('err')
" 2>/dev/null)
    printf "  %-25s concept_f1=%s\n" "$tag" "$f1"
done | sort -k1
