# Plan: Restructure agent_results.jsonl Output Schema

## Context

The current output schema has several issues that make manual review difficult:
- `SPL_SET_ID` is duplicated inside every item (already on the SPL-level envelope)
- `post_decision` is always "N/A" since `route_or_fill_enabled=False` — stale field
- `fills` (dict keyed by slot name) differs from `refinements` (list from `postcoord_pattern`) — the trace already has the richer, SCTID-keyed list form
- `decision` and `confidence` are buried inside `trace.postcoord_pattern` rather than surfaced at the item level
- Top-level key `results` is generic; rename to `contraindications`
- Evaluator CSV columns don't include `status`, `decision`, or `confidence` and use stale naming

Goal: clean item schema for manual review + a CSV that surfaces all relevant fields flat.

---

## New item_result schemas

### DIRECT
```json
{
  "item_index": 0,
  "query_text": "...",
  "status": "DIRECT",
  "primary_concept_id": "...",
  "fsn": "...",
  "trace": {"direct_verify": {...}},
  "extracted_item": {...}
}
```

### POSTCOORD / MINIMAL
```json
{
  "item_index": 0,
  "query_text": "...",
  "status": "POSTCOORD",
  "primary_concept_id": "419076005",
  "fsn": "Allergic reaction (disorder)",
  "decision": "POSTCOORD_OK",
  "confidence": 0.85,
  "refinements": [
    {"attribute_sctid": "246075003", "attribute_fsn": "Causative agent (attribute)", "value_sctid": "...", "value_term": "..."}
  ],
  "expression": "419076005:{246075003=...}",
  "trace": {"direct_verify": {}, "focus_selector": [...], "mrcm_mapper": [...], "postcoord_pattern": {...}},
  "extracted_item": {...}
}
```

### REVIEW
Same as POSTCOORD/MINIMAL plus `"review_flag": true`.

---

## Changes — Part 1: `scripts/run_pipeline.py`

### A. `assemble_direct_node` (~line 1662)
- Remove `"SPL_SET_ID": state["spl_set_id"]`
- Rename `"selected_id"` → `"primary_concept_id"` (source: `direct.get("selected_id", "N/A")`)
- Rename `"selected_term"` → `"fsn"` (source: `direct.get("selected_term", "N/A")`)

### B. `assemble_postcoord_node` (~line 1680)
- Remove `"SPL_SET_ID": state["spl_set_id"]`
- Remove `"post_decision": post_decision` key (keep local variable — still needed for `status` derivation)
- Rename `"selected_problem_id"` → `"primary_concept_id"` (source: `state["selected_problem_id"]`)
- Rename `"selected_focus_term"` → `"fsn"` (source: `state["selected_focus_term"]`)
- Remove `"fills": state["fills_detail"]`
- Add `"decision": state.get("postcoord_pattern", {}).get("decision", "N/A")`
- Add `"confidence": state.get("postcoord_pattern", {}).get("confidence")`
- Add `"refinements": state.get("postcoord_pattern", {}).get("refinements", [])`

### C. `assemble_review_node` (~line 1723)
Same changes as B. Keep `"review_flag": True`.

### D. `finalize_node` (~line 2039)
- Rename `"results": item_results` → `"contraindications": item_results`
- Also rename `"results": []` in the error branch (~line 2031) → `"contraindications": []`

---

## Changes — Part 2: `src/evaluation/evaluator.py`

### E. Add SCTID → slot-key mapping (module level, near line 79)
```python
_ATTR_SCTID_TO_KEY = {
    "246075003": "causative_agent",
    "246112005": "severity",
    "263502005": "clinical_course",
}
```

### F. `aggregate_agent_results` (~line 231)
- Read `row.get("contraindications")` instead of `row.get("results")`
- Pass `spl_set_id` and `product_name` into `_aggregate_result_item`:
  ```python
  product_name = str(row.get("product_name", ""))
  for result in row.get("contraindications") or []:
      grouped[spl_set_id].append(
          _aggregate_result_item(result, spl_set_id=spl_set_id, product_name=product_name)
      )
  ```

### G. `_aggregate_result_item` (~line 157) — full replacement
New signature: `_aggregate_result_item(result, spl_set_id="", product_name="")`

Logic:
- `base` dict: `SPL_SET_ID` from param, `product_name` from param, `item_index`, `query_text`, `status`
- DIRECT branch: `primary_concept_id` ← `result["primary_concept_id"]`, `fsn` ← `result["fsn"]`, `decision="N/A"`, `confidence=""`, `expression` = primary_concept_id, `attributes={}`
- POSTCOORD/MINIMAL/REVIEW branch: same field names; `attributes` built by iterating `result["refinements"]` and matching `attribute_sctid` to `_ATTR_SCTID_TO_KEY`
- Fallback branch: all N/A

### H. `AGG_CSV_COLUMNS` (~line 79) — replacement
```python
AGG_CSV_COLUMNS = [
    "SPL_SET_ID",
    "product_name",
    "item_index",
    "query_text",
    "status",
    "primary_concept_id",
    "fsn",
    "decision",
    "confidence",
    "expression",
    "causative_agent_id",
    "causative_agent_term",
    "severity_id",
    "severity_term",
    "clinical_course_id",
    "clinical_course_term",
]
```

### I. `_aggregated_item_to_csv_row` (~line 208) — replacement
Read from new field names; extract attribute columns by matching from `attributes` dict (keyed by `causative_agent`, `severity`, `clinical_course`).

---

## Critical files

| File | Changes |
|---|---|
| `scripts/run_pipeline.py` | A–D: assemble_direct, assemble_postcoord, assemble_review, finalize_node |
| `src/evaluation/evaluator.py` | E–I: SCTID map, aggregate_agent_results, _aggregate_result_item, AGG_CSV_COLUMNS, _aggregated_item_to_csv_row |

---

## Order of implementation

1. `scripts/run_pipeline.py` changes (A–D) — produces new JSONL
2. `src/evaluation/evaluator.py` changes (E–I) — reads new JSONL, produces clean CSV
3. Evaluator metric functions (`evaluate_aggregated_predictions` etc.) deferred — evaluation shifting to manual

---

## Verification

Point evaluator at existing `results/20260526_check/agent_results.jsonl` before re-running pipeline:

```python
import json
from src.evaluation.evaluator import aggregate_agent_results, write_csv_rows, AGG_CSV_COLUMNS

with open("results/20260526_check/agent_results.jsonl") as f:
    rows = [json.loads(l) for l in f]

_, csv_rows = aggregate_agent_results(rows)
write_csv_rows("/tmp/review.csv", csv_rows, AGG_CSV_COLUMNS)
```

Check `/tmp/review.csv`:
- `status` column populated (DIRECT / POSTCOORD / MINIMAL / REVIEW)
- `decision` and `confidence` populated for non-DIRECT rows
- `causative_agent_id/term`, `severity_id/term` populated where MRCM mapper found values
- No stale columns: `mapping_source`, `final_concept_id`, `postcoord_expression`
