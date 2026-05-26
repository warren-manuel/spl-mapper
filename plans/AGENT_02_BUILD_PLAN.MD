# Agent 2 — Compositional Extractor: Build Plan

**Status:** Phase 1 ✅ | Phase 2 ✅ | Phase 3 ✅ superseded

> **Note (2026-05-07):** Phase 3 (per-component retrieval via `retrieve_component_candidates_node`)
> was implemented as Phase B/C of `AGENT_IMPLEMENTER_PLAN.md`, then **superseded by Pipeline
> Restructure Phase IV** — `focus_selector` now calls `search_snomed` itself and no pre-built
> candidate pool is needed. The node was removed from the graph.
> Current design: `plans/AGENT_IMPLEMENTER_PLAN.md` § "Phase IV".

Aligns with Agent 2 (`compositional_extractor`) in `agents/README.md`.

---

## Problem

After `decompose_coordinations`, all item slots (`contraindication_state_text`,
`substance_text`, `severity_span`, `course_span`) are null. The `route_or_fill` LLM
receives no structured signal — it must infer everything from `ci_text` alone.

## Solution

A new `categorize_slots` node reads `agents/snomed_conventions.md` as its system prompt
and segments each decomposed span into labelled SNOMED components. The component hierarchy
labels are used to populate item slots before `route_or_fill` runs.

---

## Item-Graph Change

```
# Before
direct_match → prefilter → route_or_fill → normalize → validate → assemble

# After Phase 1 (prefilter bypassed)
direct_match → categorize_slots → route_or_fill → normalize → pattern_finder → expression_validator → assemble

# After Phase 3 (planned — per-component retrieval)
direct_match → categorize_slots → focus_selector → retrieve_component_candidates → mrcm_attribute_mapper → expression_validator → assemble
```

`prefilter_node` code is left untouched but is no longer wired into this path.

---

## Component → Slot Mapping

| LLM-assigned Hierarchy | Target item field |
|---|---|
| Clinical Finding | `contraindication_state_text` |
| Procedure | `contraindication_state_text` |
| Substance | `substance_text` |
| Pharmaceutical/Biological Product | `substance_text` |
| Organism | `substance_text` |
| Qualifier Value | `severity_span` |
| Body Structure | `slot_hierarchies` only |

Mapping is direct from hierarchy label — no keyword analysis. First match wins per slot.

---

## Phase 1 — Single-shot (implemented)

### `src/llm/prompts.py`
- `_load_snomed_conventions()` — lazy loader for `agents/snomed_conventions.md`
- `CATEGORIZE_SLOTS_USER_PROMPT` — user prompt template `{ci_text}`
- `parse_categorize_slots_output(raw)` — extracts `components[]` from LLM output
- `categorize_item_slots(chat_fn, item, *, max_tokens, stop, retries)` — LLM call + parse

### `scripts/run_pipeline.py`
- `AgentRunConfig`: adds `categorize_slots_max_tokens` (default 256), `categorize_slots_enabled` (default True)
- `ItemState`: adds `slot_hierarchies: Dict[str, Any]`
- `categorize_slots_node`: calls LLM, maps components to item fields, stores hierarchy labels
- Graph: `direct_match` no-match path now goes `→ categorize_slots → route_or_fill`

### Config env vars
| Var | Default | Purpose |
|---|---|---|
| `AGENT_MAX_TOKENS_CATEGORIZE` | `256` | Token budget for categorize_slots call |
| `CATEGORIZE_SLOTS_ENABLED` | `1` | Set to `0` to skip node (backward compat) |

---

## Phase 2 — Tool-augmented lookup (implemented)

When LLM returns `lookup_needed: true` for a component, `SnomedGraphClient.lookup_concept()`
resolves the raw span text to the SNOMED preferred term. The resolved term replaces the raw
text in the item slot, giving downstream nodes a better normalized hint.

### Changes implemented in Phase 2
- `ContraLangGraphAgent.__init__` accepts optional `graph_client: SnomedGraphClient`
- `categorize_slots_node` calls `graph_client.lookup_concept()` for flagged components;
  stores `resolved_preferred_term` and `resolved_sctid` in `slot_hierarchies`
- CLI: `--enable-graph-client` flag wires the Neo4j client at runtime

### Phase 2 is not a second LLM call
The graph client lookup is a fast DB query (~1-5ms). The resolved preferred_term is stored
alongside the component and used when writing to item slots.

---

## Phase 3 — Per-component retrieval (planned)

`slot_hierarchies` output feeds a second retrieval step (`retrieve_component_candidates`).
Each component with a resolved preferred_term is used as a per-slot query against the same
BM25+dense retrieval stack, producing component-specific candidate lists. This implements
Priority 3 (Slot-Specific Targeted Retrieval) from CLAUDE.md and provides the input for
Agent 2.5 (`focus_selector`) and Agent 3 (`mrcm_attribute_mapper`).

Phase 3 is tracked in `AGENT_03_BUILD_PLAN.MD`.

---

## Verification
1. `--extraction-only` run: no change expected (categorize_slots is item-graph only)
2. Full run: check `runtime_audit.jsonl` for `call_name="categorize_slots"` events
3. Check `extracted_item` in results: `contraindication_state_text` / `substance_text`
   should now be non-null for items that previously had all-null slots
4. Phase 2: run with `--enable-graph-client`; check `slot_hierarchies` for `resolved_sctid`
