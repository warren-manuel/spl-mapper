# Plan: verify_concept_specificity — direct match over/under-specificity correction

## Context

The direct match agent incorrectly accepts pre-coordinated concepts that are more specific
than the query. Example: query "Hypersensitivity to ASCORBIC ACID" → agent accepts
"Allergy to ascorbic acid" reasoning "clinically equivalent." In SNOMED, Allergy IS-A
Hypersensitivity — the match is too specific. The agent already calls `get_logical_definition`
and sees `Has realization → Allergic process`, but fails to recognise the specificity gap.

The fix is a new tool `verify_concept_specificity` that the agent calls when it detects
a potential mismatch. The tool uses the SNOMED graph to determine — purely dynamically,
with zero hardcoding — whether any role value in the concept's logical definition is a
subtype of what the query implies.

---

## Algorithm — no hardcoded SCTIDs, FSN fragments, or concept names

For each `RoleTriple` in the logical definition of the candidate concept:

1. Compute word overlap between the **role value's preferred term** and the **query text**
   (case-insensitive, words longer than 3 characters only).
2. If overlap → role value is consistent with the query → continue to next role.
3. If no overlap → call `get_ancestors(destination_sctid, max_depth=2)`.
4. Check each ancestor's preferred term for word overlap with the query.
   - **Ancestor overlaps query** → the role value IS-A something that matches the query →
     the concept is **more specific** than the query implies → record as `too_specific`.
5. Return `{"verdict": "too_specific" | "compatible", "details": [...]}`.

### Why this handles all role types without hardcoding

For query `"Hypersensitivity to ASCORBIC ACID"`:
- Role `Causative agent → Ascorbic acid`: words `{"ascorbic", "acid"}` overlap with query
  → consistent → skipped automatically.
- Role `Has realization → Allergic process`: words `{"allergic", "process"}` do NOT overlap
  with query → check ancestors → `Hypersensitivity process` → `"hypersensitivity"` is in
  query → **too_specific** recorded.

---

## Critical files

### 1. `src/snomed/graph_client.py` — new method

Insert after `get_siblings` (currently method #7), before `check_concept_exists`.
Uses only existing `get_logical_definition` and `get_ancestors` (both already cached —
zero extra latency if the agent called them in the same ReAct loop).

```python
def verify_concept_specificity(self, sctid: str, query_text: str) -> Dict[str, Any]:
    import re
    roles = self.get_logical_definition(sctid)
    if not roles:
        return {"verdict": "compatible", "details": [],
                "note": "primitive concept — no logical definition to check"}

    query_words = {w for w in re.findall(r'\b\w+\b', query_text.lower()) if len(w) > 3}
    details: List[Dict[str, Any]] = []

    for role in roles:
        value_term = (role.destination_preferred_term or "").lower()
        value_words = {w for w in re.findall(r'\b\w+\b', value_term) if len(w) > 3}

        if value_words & query_words:
            continue  # role value is consistent with query

        ancestors = self.get_ancestors(role.destination_sctid, max_depth=2)
        for ancestor in ancestors:
            anc_words = {w for w in re.findall(r'\b\w+\b',
                         (ancestor.preferred_term or "").lower()) if len(w) > 3}
            if anc_words & query_words:
                details.append({
                    "attribute": role.type_fsn,
                    "concept_value": role.destination_preferred_term,
                    "ancestor_matching_query": ancestor.preferred_term,
                    "interpretation": (
                        f"'{role.destination_preferred_term}' IS-A "
                        f"'{ancestor.preferred_term}' — the query implies the broader "
                        f"concept; this match is more specific than the query"
                    ),
                })
                break

    verdict = "too_specific" if details else "compatible"
    return {"verdict": verdict, "details": details}
```

### 2. `scripts/run_pipeline.py` — one new branch in `_run_ontology_tool`

Add after the `"get_ancestors"` branch (~line 1014):

```python
if tool == "verify_concept_specificity":
    return graph_client.verify_concept_specificity(
        args.get("sctid", ""), args.get("query_text", "")
    )
```

Also add `"verify_concept_specificity"` to the direct_match tool list (budget check / trace).

### 3. `agents/direct_match_agent.md` — two additions

**Add to `### Available Tools`:**
```
**`verify_concept_specificity(sctid, query_text)`**
Checks whether any role value in the concept's logical definition is more specific
than what the query implies, by traversing that value's IS-A ancestors in SNOMED.
Call after `get_logical_definition` when the concept's label and the query use related
but different clinical terms.
Returns:
  {"verdict": "too_specific" | "compatible",
   "details": [{"attribute": "...", "concept_value": "...",
                "ancestor_matching_query": "...", "interpretation": "..."}]}
If verdict is "too_specific" you may attempt one additional search using the
`ancestor_matching_query` term before deciding direct_match: false.
```

**Update STEP 3 — VERIFY PRE-COORDINATION** to add after the existing text:
```
If the logical definition is non-empty and the concept's label uses different clinical
terminology from the query, call `verify_concept_specificity(sctid, query_text)` to
check for a specificity mismatch.
- verdict "compatible" → proceed to accept
- verdict "too_specific" → the concept is a subtype of what the query describes;
  optionally retry search with the `ancestor_matching_query` term, then return false.
```

**Bump budget:** `Maximum tool calls: 8` → `10`

---

## What does NOT change

- No changes to `_run_ontology_tool` static method signature
- No changes to `ContraState`, `ItemState`, or graph wiring
- No changes to `focus_selector` or `mrcm_attribute_mapper` agents
- The tool is passively available — agent calls it only when needed

---

## Verification

```bash
# Unit test — confirm the tool returns "too_specific"
conda run -n splmap python3 -c "
import sys; sys.path.insert(0, '.')
from src.snomed.graph_client import SnomedGraphClient
gc = SnomedGraphClient()
result = gc.verify_concept_specificity('294940003', 'Hypersensitivity to ASCORBIC ACID')
print(result)
# Expected: verdict='too_specific', details shows Allergic process / Hypersensitivity process
"

# Confirm compatible verdict for a correctly-specific match
conda run -n splmap python3 -c "
import sys; sys.path.insert(0, '.')
from src.snomed.graph_client import SnomedGraphClient
gc = SnomedGraphClient()
result = gc.verify_concept_specificity('428321000124101',
    'anaphylaxis after a dose of hepatitis B-containing vaccine')
print(result)
# Expected: verdict='compatible'
"
```

End-to-end: re-run pipeline — "Hypersensitivity to ASCORBIC ACID" should route to
post-coordination rather than accepting "Allergy to ascorbic acid".
