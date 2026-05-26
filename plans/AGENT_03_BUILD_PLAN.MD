# Agent 3 — Post-Coordination Expression Agent: Build Plan

**Status:** v1 ⛔ deprecated (`pattern_finder_enabled=False`) | v2 ✅ implemented + enhanced

> **Note (2026-05-07):** v2 `mrcm_attribute_mapper_node` was implemented in Phases D-E of
> `AGENT_IMPLEMENTER_PLAN.md`, then enhanced to a **full ReAct agent with self-contained
> retrieval** in Pipeline Restructure Phase V. `route_or_fill_node` and `pattern_finder_node`
> are both disabled by default (`route_or_fill_enabled=False`, `pattern_finder_enabled=False`).
> References to `AGENT_025_BUILD_PLAN.MD` (focus_selector) are superseded by Phase IV.
> Current design: `plans/AGENT_IMPLEMENTER_PLAN.md` § "Phase V".

Aligns with Agent 3 (`postcord_agent`) in `agents/README.md`.

---

## Problem

`route_or_fill` selects the focus concept AND fills 3 hardcoded attributes
(`causative_agent`, `severity`, `clinical_course`) from a flat ranked candidate pool with
no structural SNOMED context. The 3-attribute constraint is pipeline-imposed, not
ontology-derived. Components outside that envelope (body structure, finding site,
associated morphology) are silently dropped.

The v1 Agent 3 (`pattern_finder` + `expression_validator`) was a partial fix — it refines
existing `fills_norm` using Neo4j role context with confidence-gated overrides. But the
slot schema remains hardcoded and `route_or_fill` is still the primary selection mechanism.

---

## v1 — Implemented (refines route_or_fill)

### Item-Graph (current)

```
categorize_slots → route_or_fill → normalize → pattern_finder → expression_validator → [assemble_postcoord | assemble_review]
```

### What pattern_finder does
- Queries Neo4j `get_logical_definition(focus_sctid)` and `get_ancestors()` for structural context
- Makes one LLM call (system: `agents/postcord_agent.md`) to propose refined fills
- Accepts refinements when `confidence >= 0.7`
- `expression_validator` runs MRCM range checks and `validate_postcoord_with_mrcm`; sets
  `review_flag=True` on failure or low confidence
- `assemble_review_node` routes review items to `status="REVIEW"` (preserved for triage)

### Config (v1)
| Var | Default | Purpose |
|---|---|---|
| `AGENT_MAX_TOKENS_PATTERN_FINDER` | `256` | Token budget for pattern_finder call |
| `PATTERN_FINDER_ENABLED` | `1` | Set to `0` — falls back to pre-v1 behavior |

---

## v2 — Planned (MRCM-agnostic redesign)

### Goal
Replace `route_or_fill` as the primary slot-filling mechanism with two new agents:
- **Agent 2.5** (`focus_selector`) — agentic ReAct focus concept selection
- **Agent 3 redesign** (`mrcm_attribute_mapper`) — MRCM-driven attribute assignment from
  per-component candidates

### Item-Graph (target)

```
categorize_slots
  → focus_selector (Agent 2.5)
  → retrieve_component_candidates
  → mrcm_attribute_mapper (Agent 3 redesign)
  → expression_validator
  → [assemble_postcoord | assemble_review]
```

`route_or_fill` node code is left untouched but is removed from the graph wiring.

---

## Component → Attribute Mapping (v2)

Replaces `_HIERARCHY_TO_SLOT` dict and hardcoded `attribute_table`. Attribute validity is
determined at runtime per focus concept via `get_attribute_domain_rules(focus_sctid)`.

| Agent 2 hierarchy | Role in expression |
|---|---|
| Clinical Finding / Procedure / Disorder / Finding | → focus concept (not a fill) |
| Substance / Pharma/Bio Product / Organism | → causative_agent-type attribute (MRCM lookup) |
| Qualifier Value (severity cue) | → severity-type attribute (MRCM lookup) |
| Qualifier Value (course cue) | → clinical_course-type attribute (MRCM lookup) |
| Body Structure | → finding_site-type attribute (MRCM lookup) |

Output format changes from `fills_norm: Dict[str, str]` (3 fixed keys) to:
```python
refinements: List[Dict]  # [{attribute_sctid, attribute_fsn, value_sctid, value_term}, ...]
```

---

## Agent 2.5 — Focus Selector

See `AGENT_025_BUILD_PLAN.MD` for the detailed build plan.

Summary: A ReAct-loop node with tool access (`lookup_concept`, `get_logical_definition`,
`get_ancestors`, `get_siblings`) that selects the abstract SNOMED focus concept by
structural analogy — finding sibling pre-coordinated concepts and inferring the shared
abstract parent.

---

## `mrcm_attribute_mapper` (Agent 3 v2 node)

### Input
- `selected_problem_id` from `focus_selector`
- `slot_hierarchies` (Agent 2 component output)
- Per-component candidates from `retrieve_component_candidates`
- `graph_client` for MRCM domain rule lookup

### Processing
1. `get_attribute_domain_rules(focus_sctid)` → allowed attribute SCTIDs for this concept
2. For each Agent 2 component: map hierarchy → best matching allowed attribute
3. One LLM call (system: `agents/postcord_agent.md` v2) to select value from component candidates
4. Output: `refinements[]` — variable length

### Config (v2 additions)
| Var | Default | Purpose |
|---|---|---|
| `MRCM_MAPPER_ENABLED` | `1` | Set to `0` to skip (falls back to v1 pattern_finder) |
| `AGENT_MAX_TOKENS_MRCM_MAPPER` | `512` | Token budget for mrcm_attribute_mapper LLM call |

---

## `retrieve_component_candidates` (new node)

Runs after `focus_selector`, before `mrcm_attribute_mapper`. For each non-focus component
in `slot_hierarchies` with a resolved preferred_term (from Agent 2 Phase 2), calls
`retrieve_candidates_for_item` with a single-slot item dict. Produces per-component
candidate lists keyed by component index.

This is Priority 3 (Slot-Specific Targeted Retrieval) from CLAUDE.md.

---

## Files to Change (v2)

| File | Change |
|---|---|
| `scripts/run_pipeline.py` | Add `retrieve_component_candidates_node`, `mrcm_attribute_mapper_node`; updated wiring |
| `src/llm/prompts.py` | New `mrcm_attribute_mapper` build/parse/run functions |
| `agents/postcord_agent.md` | Rewrite for MRCM-driven attribute assignment (v2 prompt) |
| `src/snomed/graph_client.py` | Reuse existing `get_attribute_domain_rules` — no new methods needed |

---

## Verification (v2)

1. `MRCM_MAPPER_ENABLED=0`: pipeline falls back to v1 `pattern_finder` behavior
2. Full run with `--enable-graph-client`: confirm `refinements` array in `item_result.trace`
   contains non-empty entries with `attribute_sctid` fields from MRCM lookup
3. For a "hypersensitivity to X" item where Agent 2.5 correctly selects abstract focus:
   confirm `attribute_sctid=246075003` (causative_agent) appears in refinements with the
   correct substance value, rather than being encoded in the focus concept itself
4. Items with Body Structure components: confirm `finding_site` attribute (363698007) appears
   in refinements when MRCM domain rules allow it for the focus concept
5. Items failing MRCM validation: `status="REVIEW"` in output (unchanged from v1)
