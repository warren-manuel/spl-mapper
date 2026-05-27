# Plan: Restructure agent_results.jsonl Output Schema

## Context

The current JSONL output has stale/redundant fields that make manual review difficult:
- `fills` is always all-N/A — the real data lives in `trace.postcoord_pattern.refinements`
- `expression` is the bare focus SCTID (e.g. `419076005`), not a real SNOMED post-coordinated expression
- `selected_problem_id` / `selected_focus_term` use opaque names; `selected_focus_term` can be dirty reasoning text
- `post_decision` is always "N/A" (`route_or_fill_enabled=False`)
- `SPL_SET_ID` is duplicated inside every item (already on the SPL envelope)
- `results` key is generic; `contraindications` is more precise
- CSV columns don't surface `status`, `decision`, `confidence`, or `ci_text`

Only `src/evaluation/evaluator.py` reads `agent_results.jsonl`. No other downstream consumers.

---

## New item schema

### DIRECT
```json
{
  "item_index": 0,
  "ci_text": "...",
  "query_text": "...",
  "status": "DIRECT",
  "primary_concept_id": "419076005",
  "fsn": "Allergic reaction (disorder)",
  "trace": {"direct_verify": {...}}
}
```

### POSTCOORD / MINIMAL / REVIEW
```json
{
  "item_index": 0,
  "ci_text": "...",
  "query_text": "...",
  "status": "POSTCOORD",
  "primary_concept_id": "419076005",
  "fsn": "Allergic reaction (disorder)",
  "decision": "POSTCOORD_OK",
  "confidence": 0.9,
  "refinements": [
    {"attribute_sctid": "246075003", "attribute_fsn": "Causative agent (attribute)", "value_sctid": "871830002", "value_term": "Vaccine product containing Corynebacterium diphtheriae toxoid antigen"},
    {"attribute_sctid": "246112005", "attribute_fsn": "Severity (attribute)", "value_sctid": "24484000", "value_term": "Severe"}
  ],
  "expression": "419076005:246075003=871830002,246112005=24484000",
  "trace": {"direct_verify": {...}, "focus_selector": [...], "mrcm_mapper": [...], "postcoord_pattern": {...}}
}
```
REVIEW also has `"review_flag": true`.

---

## Expression format

Ungrouped SNOMED compositional grammar (no role-group info from agent):
```
{focus_sctid}:{attr1_sctid}={val1_sctid},{attr2_sctid}={val2_sctid}
```
- Empty refinements → bare focus SCTID
- DIRECT → bare focus SCTID (no refinements)

Helper function to add in `scripts/run_pipeline.py` (module level, before assemble nodes):

```python
def _build_snomed_expression(focus_sctid: str, refinements: list) -> str:
    if not refinements:
        return str(focus_sctid)
    attrs = ",".join(
        f"{r['attribute_sctid']}={r['value_sctid']}"
        for r in refinements
        if r.get("attribute_sctid") and r.get("value_sctid")
    )
    return f"{focus_sctid}:{attrs}" if attrs else str(focus_sctid)
```

---

## Changes — Part 1: `scripts/run_pipeline.py`

### A. Add `_build_snomed_expression` helper (module level, before assemble nodes)
See expression format above.

### B. `assemble_direct_node` (~line 1662)
- Remove `"SPL_SET_ID"`, `"post_decision"`, `"fills"`
- Rename `"selected_id"` → `"primary_concept_id"`, `"selected_term"` → `"fsn"`
- Add `"ci_text": state.get("extracted_item", {}).get("ci_text", "")`
- `"expression"` = primary_concept_id (bare SCTID, DIRECT has no refinements)

### C. `assemble_postcoord_node` (~line 1680)
- Remove `"SPL_SET_ID"`, `"post_decision"`, `"fills"`
- Rename `"selected_problem_id"` → `"primary_concept_id"`, `"selected_focus_term"` → `"fsn"`
  - `fsn` sanitization: if `state["selected_focus_term"]` is >60 chars or "N/A", call `graph_client.lookup_concept(state["selected_problem_id"])` to get clean preferred term (same logic already in `mrcm_attribute_mapper_node`)
- Add `"ci_text": state.get("extracted_item", {}).get("ci_text", "")`
- Add `"decision": state.get("postcoord_pattern", {}).get("decision", "N/A")`
- Add `"confidence": state.get("postcoord_pattern", {}).get("confidence")`
- Add `"refinements": state.get("postcoord_pattern", {}).get("refinements", [])`
- Build `"expression"` = `_build_snomed_expression(state["selected_problem_id"], refinements)`

### D. `assemble_review_node` (~line 1723)
Same as C. Keep `"review_flag": True`.

### E. `finalize_node` (~line 2039)
- Rename `"results": item_results` → `"contraindications": item_results`
- Rename `"results": []` in the error branch (~line 2031) → `"contraindications": []`

---

## Changes — Part 2: `src/evaluation/evaluator.py`

### F. Add SCTID → slot-key mapping (module level, near line 79)
```python
_ATTR_SCTID_TO_KEY = {
    "246075003": "causative_agent",
    "246112005": "severity",
    "263502005": "clinical_course",
}
```

### G. `AGG_CSV_COLUMNS` (~line 79) — replacement
```python
AGG_CSV_COLUMNS = [
    "SPL_SET_ID",
    "product_name",
    "item_index",
    "ci_text",
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

### H. `aggregate_agent_results` (~line 231)
- Read `row.get("contraindications")` instead of `row.get("results")`
- Pass `spl_set_id` and `product_name` as params to `_aggregate_result_item`:
  ```python
  product_name = str(row.get("product_name", ""))
  for result in row.get("contraindications") or []:
      grouped[spl_set_id].append(
          _aggregate_result_item(result, spl_set_id=spl_set_id, product_name=product_name)
      )
  ```

### I. `_aggregate_result_item` (~line 157) — full replacement
New signature: `_aggregate_result_item(result, spl_set_id="", product_name="")`

- `base` dict: `SPL_SET_ID` from param, `product_name` from param, `item_index`, `ci_text` (from `result.get("ci_text", "")`), `query_text`, `status`
- DIRECT branch: `primary_concept_id`, `fsn`, `decision="N/A"`, `confidence=""`, `expression=primary_concept_id`, `attributes={}`
- POSTCOORD/MINIMAL/REVIEW branch: same new field names; `attributes` built by iterating `result["refinements"]` and matching `attribute_sctid` to `_ATTR_SCTID_TO_KEY`; `expression` from `result["expression"]`
- Fallback: all N/A

### J. `_aggregated_item_to_csv_row` (~line 208) — replacement
Read from new field names; extract `causative_agent_id/term`, `severity_id/term`, `clinical_course_id/term` from the `attributes` dict.

---

## Critical files

| File | Changes |
|---|---|
| `scripts/run_pipeline.py` | Add `_build_snomed_expression`; update `assemble_direct_node`, `assemble_postcoord_node`, `assemble_review_node`, `finalize_node` |
| `src/evaluation/evaluator.py` | Add `_ATTR_SCTID_TO_KEY`; replace `AGG_CSV_COLUMNS`, `aggregate_agent_results`, `_aggregate_result_item`, `_aggregated_item_to_csv_row` |

---

## Order of implementation

1. `scripts/run_pipeline.py` — helper + assemble nodes + finalize (A–E)
2. `src/evaluation/evaluator.py` — CSV reader (F–J)
3. Metric functions (`evaluate_aggregated_predictions` etc.) — deferred; evaluation shifting to manual

---

## Verification

After pipeline changes produce a new JSONL, or against an existing file to test evaluator alone:

```python
import json
from src.evaluation.evaluator import aggregate_agent_results, write_csv_rows, AGG_CSV_COLUMNS

with open("results/20260526/agent_results.jsonl") as f:
    rows = [json.loads(l) for l in f if l.strip()]

_, csv_rows = aggregate_agent_results(rows)
write_csv_rows("/tmp/review.csv", csv_rows, AGG_CSV_COLUMNS)
```

Check `/tmp/review.csv`:
- `ci_text` column populated
- `status` populated (DIRECT / POSTCOORD / MINIMAL / REVIEW)
- `decision` and `confidence` populated for non-DIRECT rows
- `expression` is a real SNOMED expression string (`419076005:246075003=871830002,...`), not a bare SCTID
- `causative_agent_id/term`, `severity_id/term` populated where MRCM mapper found values
- No stale columns (`mapping_source`, `final_concept_id`, `postcoord_expression`, `fills`, `post_decision`)
