# Agent Implementer Plan

**Covers:** All required additions to realize the agent architecture described in `agents/README.md`.
**Derived from:** Gap analysis against the current `scripts/run_pipeline.py` state.
**Execute in phase order** — each phase has explicit dependencies on the previous.

---

## Current Item Graph (as of 2026-05-07 — all phases complete)

```
direct_match (ReAct, self-contained — calls search_snomed + get_logical_definition)
  → [DIRECT] assemble_direct → END
  → [NO MATCH] categorize_slots
      → focus_selector (ReAct — calls search_snomed + get_logical_definition + get_ancestors)
          → normalize (builds fills_norm from refinements[])
              → mrcm_attribute_mapper (ReAct — calls get_domain_attributes + get_attribute_range + search_snomed)
                  → [assemble_postcoord | assemble_review] → END
```

**Nodes removed from graph:** `retrieve_candidates`, `prefilter`, `retrieve_component_candidates`,
`expression_validator` | **Disabled (in code, not wired):** `pattern_finder`, `route_or_fill`

## Historical Baseline (pre-implementation — for reference)

```
retrieve_candidates → direct_match
  → [DIRECT] assemble_direct
  → [NO MATCH] categorize_slots
      → focus_selector          ← wrong input (full-text focus_candidates, not component pool)
      → route_or_fill           ← focus pick partially overridden by focus_selector
      → normalize               ← hardcoded 3 slots
      → pattern_finder          ← uses postcord_agent.md v2 prompt, v1 parser (MISMATCH)
      → expression_validator    ← reads v1 keys (proposed_focus_id, proposed_fills)
      → [assemble_postcoord | assemble_review]
```

---

## Phase A — Fix Active Bug: postcord_agent.md v2 / parser v1 mismatch

**Dependency:** None — do first.

**Problem:** `postcord_agent.md` was rewritten to output `{"refinements": [...], "decision":
..., "confidence": ...}`. `parse_pattern_finder_output()` accepts it (because `"decision"`
is present) but `expression_validator_node` reads `postcoord_pattern.get("proposed_focus_id")`
and `postcoord_pattern.get("proposed_fills")` — both missing in v2 output. Result: the
pattern_finder LLM call succeeds but all refinements are silently discarded.

**Fix — update `expression_validator_node` to handle both formats:**

In `scripts/run_pipeline.py::expression_validator_node`, after the existing
`postcoord_pattern.get("confidence", 0) >= 0.7` refinement block, add a second branch:

```python
# v2 format: refinements[] array from mrcm_attribute_mapper / updated pattern_finder
elif postcoord_pattern.get("confidence", 0) >= 0.7 and "refinements" in postcoord_pattern:
    for ref in postcoord_pattern.get("refinements", []):
        attr_fsn = ref.get("attribute_fsn", "").lower()
        val_sctid = str(ref.get("value_sctid", "N/A"))
        if val_sctid == "N/A":
            continue
        if "causative" in attr_fsn:
            fills_norm["causative_agent"] = val_sctid
        elif "severity" in attr_fsn:
            fills_norm["severity"] = val_sctid
        elif "course" in attr_fsn or "clinical course" in attr_fsn:
            fills_norm["clinical_course"] = val_sctid
```

This is a bridge patch — it maps v2 `refinements[]` back to the existing 3-slot `fills_norm`
so the rest of the downstream pipeline (normalize, assemble) keeps working unchanged until
Phase E replaces them properly.

**Files:** `scripts/run_pipeline.py`

---

## Phase B — `retrieve_component_candidates_node`

**Dependency:** Phase A complete.

**Purpose:** Run a second, per-component BM25+dense retrieval pass using each component's
resolved text as the query. Produces a mixed candidate pool used by Agent 2.5 and Agent 3.

### B.1 — `ItemState` addition

```python
component_candidates: List[Dict[str, Any]]   # flat list, all component candidates merged
```

The list format (not dict) is intentional — Agent 2.5 and Agent 3 receive one unseparated
pool and must use Agent 2 hierarchy labels to identify focus vs fill candidates.

### B.2 — New node in `scripts/run_pipeline.py`

```python
def retrieve_component_candidates_node(state: ItemState) -> ItemState:
    slot_hierarchies = state.get("slot_hierarchies", {})
    all_candidates: List[Dict[str, Any]] = []

    for text, meta in slot_hierarchies.items():
        query_text = meta.get("resolved_preferred_term") or text
        if not query_text:
            continue
        # Build a minimal item dict with ci_text as the component query
        component_item = {
            "ci_text": query_text,
            "contraindication_state_text": None,
            "substance_text": None,
            "severity_span": None,
            "course_span": None,
        }
        try:
            cands = retrieve_candidates_for_item(component_item)
            # Tag each candidate with which component it came from
            focus_cands = cands.get("focus_candidates", [])
            for c in focus_cands:
                c["source_component"] = text
                c["source_hierarchy"] = meta.get("hierarchy", "")
            all_candidates.extend(focus_cands)
        except Exception:
            pass  # per-component retrieval failure is non-fatal

    return {**state, "component_candidates": all_candidates}
```

### B.3 — Graph wiring change

```python
# Remove:
graph.add_edge("categorize_slots", "focus_selector")

# Add:
graph.add_node("retrieve_component_candidates",
    instrument_item_node("retrieve_component_candidates", retrieve_component_candidates_node))
graph.add_edge("categorize_slots", "retrieve_component_candidates")
graph.add_edge("retrieve_component_candidates", "focus_selector")
```

**Files:** `scripts/run_pipeline.py`

---

## Phase C — Fix `focus_selector_node` Input

**Dependency:** Phase B complete.

**Problem:** `focus_selector_node` currently reads `state["candidates"].get("focus_candidates", [])`
(full-text candidates from the first retrieval). It should read from `state["component_candidates"]`
(the mixed pool from Phase B) so the LLM can apply the selection guides described in
`agents/focus_selector.md`.

### C.1 — Update `focus_selector_node` in `scripts/run_pipeline.py`

Change the fallback and main path to use `component_candidates`:

```python
def focus_selector_node(state: ItemState) -> ItemState:
    # Use component candidate pool (mixed); fall back to full-text focus_candidates
    component_pool = state.get("component_candidates") or \
                     state["candidates"].get("focus_candidates", [])

    if not self.cfg.focus_selector_enabled or self.graph_client is None:
        # Fallback: pick top Clinical Finding candidate from pool
        top = _pick_top_focus_candidate(component_pool, state.get("slot_hierarchies", {}))
        return {
            **state,
            "selected_problem_id": str(top.get("id", "N/A")),
            "selected_focus_term": str(top.get("label") or top.get("term", "N/A")),
            "focus_selector_trace": [],
        }
    ...
    # Pass component_pool to prompt builder instead of focus_candidates
    user = build_focus_selector_user_prompt(
        state["item"], focus_text, component_pool, state.get("slot_hierarchies", {})
    )
    ...
```

New helper `_find_focus_component_text` already exists. Add:

```python
def _pick_top_focus_candidate(
    pool: List[Dict[str, Any]], slot_hierarchies: Dict[str, Any]
) -> Dict[str, Any]:
    """From a mixed pool, return the highest-ranked Clinical Finding candidate."""
    focus_tags = {"clinical finding", "procedure", "disorder", "finding"}
    for c in pool:
        src_hier = str(c.get("source_hierarchy", "")).lower()
        if src_hier in focus_tags:
            return c
    return pool[0] if pool else {}
```

### C.2 — Update `build_focus_selector_user_prompt` in `src/llm/prompts.py`

Add `slot_hierarchies` parameter to provide Agent 2 component labels alongside candidates:

```python
def build_focus_selector_user_prompt(
    item: Dict[str, Any],
    focus_component_text: str,
    candidate_pool: List[Dict[str, Any]],
    slot_hierarchies: Optional[Dict[str, Any]] = None,
) -> str:
    lines = []
    lines.append(f"FULL CONTRAINDICATION: {item.get('ci_text', '')}")
    lines.append(f"FOCUS COMPONENT TEXT (from Agent 2): {focus_component_text or 'not identified'}")

    if slot_hierarchies:
        lines.append("\nAGENT 2 COMPONENTS:")
        for text, meta in slot_hierarchies.items():
            hier = meta.get("hierarchy", "?")
            resolved = meta.get("resolved_preferred_term") or text
            lines.append(f"  [{hier}] {resolved}")

    lines.append(f"\nCANDIDATE POOL (all components, mixed — {len(candidate_pool)} total):")
    for i, c in enumerate(candidate_pool[:15], 1):
        cid = c.get("id", "?")
        label = c.get("label") or c.get("term", "?")
        src = c.get("source_hierarchy", "?")
        lines.append(f"  {i}) {cid} | {label} | source_hierarchy={src}")

    lines.append(
        "\nIdentify the correct ABSTRACT focus concept. "
        "Use tools to verify. Return final answer with <<END_JSON>>."
    )
    return "\n".join(lines)
```

**Files:** `scripts/run_pipeline.py`, `src/llm/prompts.py`

---

## Phase D — `mrcm_attribute_mapper_node` (Agent 3 v2)

**Dependency:** Phase C complete. Phase A bridge patch in place.

**Purpose:** Replaces `pattern_finder_node`. Uses MRCM domain rules to determine valid
attributes for the focus concept, then assigns each non-focus component to an attribute
and selects a value from its candidates.

### D.1 — New functions in `src/llm/prompts.py`

**`build_mrcm_mapper_user_prompt(item, slot_hierarchies, component_candidates, mrcm_rules, focus_sctid, focus_term) -> str`**

Sections:
- `FOCUS CONCEPT: {focus_sctid} | {focus_term}`
- `MRCM ALLOWED ATTRIBUTES:` — list of `{attribute_sctid} | {attribute_fsn}` from domain rules
- `NON-FOCUS COMPONENTS (from Agent 2):` — list of `[{hierarchy}] {text}` excluding focus
- `CANDIDATE POOL:` — all component candidates (same pool as Agent 2.5 received)
- `CONTRAINDICATION: {ci_text}`
- Output instruction: return `{"decision": ..., "refinements": [...], "confidence": ...}<<END_JSON>>`

**`parse_mrcm_mapper_output(raw) -> Dict`**

```python
def parse_mrcm_mapper_output(raw: str) -> Dict[str, Any]:
    cleaned = trim_after_end_json_token(raw, token=END_JSON_TOKEN, include_token=False)
    parsed = extract_json(cleaned)
    if not isinstance(parsed, dict):
        return {}
    if "refinements" not in parsed and "decision" not in parsed:
        return {}
    return parsed
```

**`run_mrcm_mapper(chat_fn, item, slot_hierarchies, component_candidates, mrcm_rules, focus_sctid, focus_term, *, max_tokens, stop, retries) -> Tuple[Dict, str]`**

Same retry pattern as `run_pattern_finder`. Loads system prompt via `_load_postcord_agent()`.

### D.2 — New `AgentRunConfig` fields

```python
mrcm_mapper_max_tokens: int = 512
mrcm_mapper_enabled:    bool = True
```

Env vars: `AGENT_MAX_TOKENS_MRCM_MAPPER`, `MRCM_MAPPER_ENABLED`

### D.3 — New `ItemState` field

```python
refinements: List[Dict[str, Any]]   # [{attribute_sctid, attribute_fsn, value_sctid, value_term}, ...]
```

### D.4 — New `mrcm_attribute_mapper_node` in `scripts/run_pipeline.py`

```python
def mrcm_attribute_mapper_node(state: ItemState) -> ItemState:
    if not self.cfg.mrcm_mapper_enabled:
        return {**state, "refinements": []}

    focus_sctid = state.get("selected_problem_id", "N/A")
    focus_term  = state.get("selected_focus_term", "N/A")
    slot_hierarchies   = state.get("slot_hierarchies", {})
    component_pool     = state.get("component_candidates", [])

    # Get MRCM domain rules for this focus concept
    mrcm_rules = []
    if self.graph_client is not None and focus_sctid.isdigit():
        try:
            mrcm_rules = self.graph_client.get_attribute_domain_rules(focus_sctid)
        except Exception:
            pass

    mapper_started = time.perf_counter()
    parsed, raw = run_mrcm_mapper(
        self.llm.chat,
        state["item"],
        slot_hierarchies,
        component_pool,
        mrcm_rules,
        focus_sctid,
        focus_term,
        max_tokens=self.cfg.mrcm_mapper_max_tokens,
        stop=None,
        retries=self.cfg.retries,
    )
    self._log_llm_call(
        call_name="mrcm_attribute_mapper",
        system=_load_postcord_agent()[:200] + "…",
        user=f"FOCUS:{focus_sctid} CI:{state['item'].get('ci_text','')}",
        max_tokens=self.cfg.mrcm_mapper_max_tokens,
        effective_max_tokens=self.cfg.mrcm_mapper_max_tokens,
        raw=raw,
        parsed=parsed or None,
        duration_s=time.perf_counter() - mapper_started,
    )
    return {**state, "refinements": parsed.get("refinements", []), "postcoord_pattern": parsed}
```

Note: `postcoord_pattern` is also set so the Phase A bridge patch in `expression_validator_node`
still receives the full v2 output structure.

### D.5 — Graph wiring: replace `pattern_finder` with `mrcm_attribute_mapper`

```python
# Remove:
graph.add_node("pattern_finder", ...)
graph.add_edge("normalize", "pattern_finder")
graph.add_edge("pattern_finder", "expression_validator")

# Add:
graph.add_node("mrcm_attribute_mapper",
    instrument_item_node("mrcm_attribute_mapper", mrcm_attribute_mapper_node))
graph.add_edge("normalize", "mrcm_attribute_mapper")
graph.add_edge("mrcm_attribute_mapper", "expression_validator")
```

`pattern_finder_node` code is left defined but removed from graph wiring.

**Files:** `src/llm/prompts.py`, `scripts/run_pipeline.py`

---

## Phase E — Downstream Node Updates

**Dependency:** Phase D complete.

### E.1 — `normalize_node` — generalize to variable refinements

Current: hardcodes 3 keys from `route_or_fill.fills`. With `mrcm_attribute_mapper` in
place, `route_or_fill` no longer provides fills — they come from `refinements[]`.

```python
def normalize_node(state: ItemState) -> ItemState:
    selected_problem_id = state.get("selected_problem_id", "N/A")
    selected_focus_term = state.get("selected_focus_term", "N/A")

    # Build fills_norm from refinements[] (v2) or route_or_fill.fills (v1 fallback)
    refinements = state.get("refinements", [])
    fills_norm: Dict[str, str] = {}
    fills_detail: Dict[str, Dict[str, str]] = {}

    if refinements:
        for ref in refinements:
            attr_sctid = str(ref.get("attribute_sctid", ""))
            val_sctid  = str(ref.get("value_sctid", "N/A"))
            val_term   = str(ref.get("value_term", "N/A"))
            if attr_sctid and val_sctid != "N/A":
                fills_norm[attr_sctid]   = val_sctid
                fills_detail[attr_sctid] = {"id": val_sctid, "term": val_term,
                                             "attribute_fsn": ref.get("attribute_fsn", "")}
    else:
        # v1 fallback path (route_or_fill still provides fills when mrcm_mapper disabled)
        route_fill = state.get("route_or_fill", {})
        fills = route_fill.get("fills", {}) or {}
        for key in ("causative_agent", "severity", "clinical_course"):
            value = fills.get(key, "N/A")
            if isinstance(value, dict):
                value = value.get("id", "N/A")
            fills_norm[key] = str(value)
            fills_detail[key] = {
                "id": fills_norm[key],
                "term": candidate_label_by_id(
                    state["candidates"].get(f"{key}_candidates", []), fills_norm[key]
                ),
            }

    return {
        **state,
        "selected_problem_id": selected_problem_id,
        "selected_focus_term": selected_focus_term,
        "fills_norm": fills_norm,
        "fills_detail": fills_detail,
    }
```

### E.2 — `assemble_postcoord_node` — generalize `ax_pairs`

Current: iterates `self.cfg.attribute_table` (3 hardcoded entries).
Required: iterate `fills_norm` keys directly (variable-length, SCTID-keyed when v2).

```python
def assemble_postcoord_node(state: ItemState) -> ItemState:
    ax_pairs: List[str] = []
    fills_norm = state.get("fills_norm", {})
    fills_detail = state.get("fills_detail", {})

    for attr_key, val_id in fills_norm.items():
        if val_id == "N/A":
            continue
        # attr_key is either a SCTID string (v2) or slot name like "causative_agent" (v1)
        attr_sctid = attr_key if attr_key.isdigit() else str(
            self.cfg.attribute_table.get(attr_key, "")
        )
        if attr_sctid:
            ax_pairs.append(f"{attr_sctid}={val_id}")

    expression = (
        f"{state['selected_problem_id']}:{{{','.join(ax_pairs)}}}"
        if ax_pairs else state["selected_problem_id"]
    )
    ...
    # fills in item_result now uses fills_detail directly (variable keys, same structure)
```

### E.3 — `expression_validator_node` — remove Phase A bridge, use refinements natively

After Phase D, replace the Phase A bridge patch with a proper v2 handler. The validator
should iterate `state.get("refinements", [])` to check MRCM range constraints, rather
than mapping back to 3 hardcoded slot names. `validate_postcoord_with_mrcm` remains the
final gate but receives the SCTID-keyed `fills_norm` from `normalize_node`.

The `review_flag` logic stays unchanged:
```python
review_flag = (not ok) or (postcoord_pattern.get("confidence", 1.0) < 0.5)
```

**Files:** `scripts/run_pipeline.py`

---

## Summary of All File Changes (Phases A–E) ✅

| File | Phase | Change | Status |
|---|---|---|---|
| `scripts/run_pipeline.py` | A | `expression_validator_node` v1/v2 bridge | ✅ Done |
| `scripts/run_pipeline.py` | B | `retrieve_component_candidates_node` + `ItemState.component_candidates` + wiring | ✅ Done |
| `scripts/run_pipeline.py` | C | `focus_selector_node` reads `component_candidates`; `_pick_top_focus_candidate` helper | ✅ Done |
| `src/llm/prompts.py` | C | `build_focus_selector_user_prompt` adds `slot_hierarchies` param + mixed pool formatting | ✅ Done |
| `src/llm/prompts.py` | D | `build_mrcm_mapper_user_prompt`, `parse_mrcm_mapper_output`, `run_mrcm_mapper` | ✅ Done |
| `scripts/run_pipeline.py` | D | `mrcm_attribute_mapper_node`, `AgentRunConfig` fields, `ItemState.refinements`, wiring | ✅ Done |
| `scripts/run_pipeline.py` | E | `normalize_node` generalized; `assemble_postcoord_node` generalized; `expression_validator_node` v2 native | ✅ Done |

## Env Vars Added (Phases A–E)

| Var | Default | Phase | Purpose |
|---|---|---|---|
| `MRCM_MAPPER_ENABLED` | `1` | D | Set to `0` to skip mrcm_attribute_mapper (falls back to v1 pattern_finder) |
| `AGENT_MAX_TOKENS_MRCM_MAPPER` | `512` | D | Token budget for mrcm_attribute_mapper LLM call |

## Verification (Phases A–E)

1. **Phase A:** Run a full pipeline on 2 SPLs; inspect `runtime_audit.jsonl` — `pattern_finder`
   events should have non-empty `parsed_output.refinements` instead of `{}`.
2. **Phase B:** Confirm `component_candidates` key is present in `ItemState` trace output;
   length should be > `focus_candidates` alone for items with multiple components.
3. **Phase C:** For "hypersensitivity to X" items, confirm `focus_selector_trace` shows
   `get_logical_definition` tool calls; `selected_problem_id` resolves to abstract parent
   (e.g. Hypersensitivity disorder) not pre-coordinated child.
4. **Phase D:** Confirm `mrcm_attribute_mapper` events in audit; `item_result.trace.postcoord_pattern.refinements`
   is non-empty; `attribute_sctid` values come from MRCM domain rules, not `attribute_table`.
5. **Phase E:** Expression string for a hypersensitivity item uses SCTID-keyed fills:
   `267038008:{246075003=387517004}` not `267038008:{causative_agent=387517004}`.
6. **Regression (`MRCM_MAPPER_ENABLED=0`):** Output identical to pre-Phase-D pipeline —
   `pattern_finder` runs, fills_norm has 3 hardcoded slots, expression assembles as before.

---

---

# Agent Architecture Decisions

**Scope:** Determines which of the 8 LLM call sites should be upgraded to ReAct agents
and which should remain single-shot chains. Outputs feed a downstream drug attribute
representation model — decision quality is the priority; latency is not a constraint.

---

## LLM Call Site Audit

| # | Node | Lines | Current Pattern | Decision Type |
|---|------|-------|-----------------|---------------|
| 1 | `extract_items_node` | ~1525 | Simple chain | List extraction from SPL text |
| 2 | `decompose_coordinations_node` | ~1566 | Simple chain (per-item loop) | Linguistic decomposition |
| 3 | `direct_match_node` | ~721 | Simple chain | Binary match + SCTID selection |
| 4 | `categorize_slots_node` | ~839 | Simple chain | SNOMED hierarchy classification |
| 5 | `focus_selector_node` | ~967 | **ReAct Python loop** ✅ | SCTID selection w/ ontology verification |
| 6 | `route_or_fill_node` | ~1037 | Simple chain (v1 legacy) | Focus + 3-slot fill selection |
| 7 | `mrcm_attribute_mapper_node` | ~1140 | Simple chain | MRCM attribute assignment |
| 8 | `pattern_finder_node` | ~1094 | Simple chain (v1 legacy) | Post-coordination pattern scoring |

**Rule of thumb:** Use ReAct when the correct answer depends on what is found in an
ontology lookup. Use a simple chain when input text fully determines the output.

---

## Decision by Node

### Keep as Simple Chain

**`extract_items_node`** — Text extraction only. No SNOMED lookup needed. Keep.

**`decompose_coordinations_node`** — Linguistic splitting. Purely syntactic. Keep.

**`categorize_slots_node`** — Fixed-taxonomy classification (Clinical Finding / Substance /
Qualifier Value). Post-call rule-based resolution via `graph_client.lookup_concept` already
handles low-confidence cases. Keep.

### Upgrade to ReAct Agent

**`direct_match_node`** — **Priority: High.** Root cause of Failure Mode 1 (42–49% of
concept failures): LLM picks a pre-coordinated concept as a semantic approximation when
the query text does not closely match. A 2–3 call ReAct loop should:
1. Identify best candidate from pool
2. Call `get_logical_definition(sctid)` on it
3. If non-empty (pre-coordinated) AND query text does NOT closely match concept label
   → `{"direct_match": false}` → route to postcoord path
4. If non-empty AND query text closely matches → valid direct match
5. If empty (abstract) → standard text-match evaluation

**Rationale:** A pre-coordinated concept is correct when the query literally matches it
(e.g. "Allergy to ibuprofen" → concept "Allergy to ibuprofen"). The failure is when the
LLM uses it as an approximation for text that doesn't match — the encoded slots are then
no longer faithful to the query, and the postcoord path must decompose it properly.

**`mrcm_attribute_mapper_node`** — **Priority: High.** Currently uses 4 hardcoded attribute
SCTIDs pre-fetched before the LLM call. True MRCM-agnostic operation requires the LLM to
discover applicable attributes dynamically. ReAct loop should:
1. Call `get_domain_attributes(focus_sctid)` to discover applicable attrs for this focus
2. For each discovered attribute, call `get_attribute_range(attr_sctid)` to get ECL range
3. Match component candidates to discovered attrs → output `refinements[]`

Removes the hardcoded `_std_attrs` dict; mapper works for any focus concept hierarchy.

### Already ReAct — Maintain

**`focus_selector_node`** ✅ — Python-loop ReAct, max 5 tool calls. Correct design.

**Backend note:** LangGraph native `ToolNode` + `bind_tools()` requires native tool-calling
API support. Two valid implementation paths:

| Backend | ReAct Pattern |
|---------|--------------|
| HuggingFace `transformers` (current) | Python loop with JSON tool call parsing (current `focus_selector_node` pattern) |
| **vLLM OpenAI-compatible endpoint** | **LangGraph `ToolNode` + `bind_tools()`** — model returns structured `ToolCall` objects |

If the pipeline moves to vLLM (`ChatOpenAI(base_url="http://localhost:<port>/v1")`),
`gemma-4-31B-it` supports function calling and `ToolNode` becomes the preferred pattern
for better LangSmith trace observability (each tool call visible as a named graph node).

### Deprecate (v1 legacy)

**`route_or_fill_node`** — Focus output overridden by `focus_selector_node`; fill output
overridden by `mrcm_attribute_mapper_node` in v2 path. Convert to explicit v1 fallback
gated by `route_or_fill_enabled=False`, remove when v2 path is stable.

**`pattern_finder_node`** — Superseded by `mrcm_attribute_mapper_node`. Its `postcoord_pattern`
output is only used for the `review_flag` confidence check. Gate with
`pattern_finder_enabled=False` as new default once v2 is validated.

---

## Phase F — Upgrade `direct_match_node` to ReAct

**Files:** `scripts/run_pipeline.py`, `src/llm/prompts.py`, `agents/direct_match_agent.md`

**Also fixes:** Line 1023 bug — `candidate_label_by_id(focus_candidates, selected_id)`
where `focus_candidates` is out of scope; fix to `component_pool`.

1. Write `agents/direct_match_agent.md`: ReAct system prompt with 3-step procedure
   (identify top candidate → `get_logical_definition` → decide match/no-match)
2. Add `build_direct_match_agent_user_prompt` + `parse_direct_match_agent_output` to
   `src/llm/prompts.py`
3. Convert `_verify_direct_match` in `scripts/run_pipeline.py` to a bounded loop (max 3
   tool calls) using the same Python loop pattern as `focus_selector_node`
4. Add to `AgentRunConfig`:
   - `direct_match_max_tool_calls: int = 3`
   - `direct_match_agent_enabled: bool = True`
5. Fix line 1023 bug in `focus_selector_node`

**Verification:** Run 2 SPLs; `direct_match` events in `runtime_audit.jsonl` where a
pre-coordinated concept was top-ranked should show `direct_match=false` and route to
postcoord path.

---

## Phase G — Upgrade `mrcm_attribute_mapper_node` to ReAct

**Files:** `scripts/run_pipeline.py`, `src/llm/prompts.py`, `src/snomed/graph_client.py`,
`agents/postcord_agent.md`

1. Add `get_domain_attributes(focus_sctid) -> List[AttributeMatch]` to
   `src/snomed/graph_client.py` — queries MRCM domain constraints for applicable attrs
2. Rewrite `agents/postcord_agent.md` to ReAct format:
   - Step 1: `get_domain_attributes(focus_sctid)` → discover applicable attrs
   - Step 2: For each attr → `get_attribute_range(attr_sctid)` → get ECL range
   - Step 3: Match non-focus components → output `refinements[]`
3. Convert `run_mrcm_mapper` in `src/llm/prompts.py` to a bounded loop (max 6 tool calls)
4. Add `mrcm_mapper_max_tool_calls: int = 6` to `AgentRunConfig`
5. Remove hardcoded `_std_attrs` dict from `mrcm_attribute_mapper_node`

**Verification:** `mrcm_attribute_mapper` audit events should show tool call traces with
dynamically discovered attributes; `attribute_sctid` values should not be limited to the
4 hardcoded standard attrs.

---

## Phase H — Deprecate v1 Legacy Nodes

**File:** `scripts/run_pipeline.py`

1. Set `pattern_finder_enabled: bool = False` as new default in `AgentRunConfig`
2. Add `route_or_fill_enabled: bool = False` to `AgentRunConfig`
3. Update graph wiring: skip both nodes when their flags are `False`; `normalize_node`
   reads `refinements[]` only (v1 fallback path stays in code but unreachable by default)

**Verification:** `route_or_fill` and `pattern_finder` events should not appear in
`runtime_audit.jsonl` by default.

---

## Summary of All File Changes (Phases F–H)

| File | Phase | Change |
|---|---|---|
| `agents/direct_match_agent.md` | F | New ReAct system prompt for direct match agent |
| `src/llm/prompts.py` | F | `build_direct_match_agent_user_prompt`, `parse_direct_match_agent_output` |
| `scripts/run_pipeline.py` | F | `_verify_direct_match` → bounded loop; config fields; line 1023 bug fix |
| `src/snomed/graph_client.py` | G | `get_domain_attributes(focus_sctid)` |
| `agents/postcord_agent.md` | G | Rewrite to ReAct format |
| `src/llm/prompts.py` | G | `run_mrcm_mapper` → bounded loop |
| `scripts/run_pipeline.py` | G | Remove `_std_attrs`; add `mrcm_mapper_max_tool_calls` config |
| `scripts/run_pipeline.py` | H | `pattern_finder_enabled=False`; `route_or_fill_enabled=False`; graph wiring update |

---

## Pending Bug Fixes — focus_selector_node

### Bug I — Component pool ordering in `retrieve_component_candidates_node`

**File:** `scripts/run_pipeline.py`

**Problem:** Candidates are appended in `slot_hierarchies` dict iteration order (i.e. the
order the LLM returned components from `categorize_slots_node`). If a Substance component
is iterated first and returns 20 candidates, `component_pool[:15]` in
`build_focus_selector_user_prompt` contains only substance candidates — the Clinical
Finding candidates are appended after position 15 and are never shown to the agent.

**Fix:** After building `all_candidates`, sort by `source_hierarchy` so focus-hierarchy
candidates (Clinical Finding / Disorder / Procedure) appear before fill-hierarchy candidates
(Substance, Qualifier Value), using the same `focus_tags` set already defined in
`_pick_top_focus_candidate`:

```python
_focus_tags = {"clinical finding", "procedure", "disorder", "finding", "regime/therapy"}
all_candidates.sort(
    key=lambda c: 0 if c.get("source_hierarchy", "").lower() in _focus_tags else 1
)
```

This is a stable sort — relative rank within each tier is preserved.

**Location:** `retrieve_component_candidates_node`, just before `return {**state, "component_candidates": all_candidates}`.

---

### Bug II — Term resolution for parent concepts returned by tool calls

**File:** `scripts/run_pipeline.py`

**Problem:** The agent may return a `focus_sctid` that came from a `get_ancestors` tool
call (the abstract parent of a pre-coordinated candidate). That SCTID is not in
`component_pool`, so `candidate_label_by_id(component_pool, selected_id)` returns `"N/A"`.
The current fallback uses `parsed.get("reasoning", "N/A")[:80]` as the term — this is
prose, not a concept label.

**Fix:** When `candidate_label_by_id` returns `"N/A"`, scan the `trace` for the SCTID
in any tool result before falling back to reasoning text. The `get_ancestors` and
`lookup_concept` results contain `preferred_term` in their `ConceptMatch.__dict__`:

```python
selected_id = str(parsed.get("focus_sctid", "N/A"))
selected_term = candidate_label_by_id(component_pool, selected_id)
if not selected_term or selected_term == "N/A":
    # Search tool results in trace for the SCTID (e.g. from get_ancestors)
    for step in trace:
        tool_result = step.get("parsed", {})
        for ancestor in tool_result.get("ancestors", []):
            if ancestor.get("sctid") == selected_id:
                selected_term = ancestor.get("preferred_term", "N/A")
                break
        for hit in tool_result.get("results", []):  # lookup_concept
            if hit.get("sctid") == selected_id:
                selected_term = hit.get("preferred_term", "N/A")
                break
        if selected_term and selected_term != "N/A":
            break
if not selected_term or selected_term == "N/A":
    selected_term = str(parsed.get("reasoning", "N/A"))[:80]
```

**Location:** `focus_selector_node`, lines 1164–1167 in `scripts/run_pipeline.py`.

---

---

# Pipeline Restructure — Phases I–V

## Context

Five architectural changes to make all three ReAct agents fully autonomous:
agents own their own retrieval, the shared BM25+FAISS stack replaces Neo4j text search,
now-redundant nodes (`retrieve_candidates`, `retrieve_component_candidates`,
`expression_validator`) are removed, and graph_client responses are enriched so the LLM
can reason over them without a secondary lookup.

---

## Target Graph (after all phases)

```
direct_match (ReAct, self-contained retrieval)
  → [DIRECT] assemble_direct
  → [NO MATCH] categorize_slots
      → focus_selector (ReAct, calls search_snomed itself)
          → mrcm_attribute_mapper (ReAct, calls search_snomed, validates inline)
              → normalize
                  → [assemble_postcoord | assemble_review]
```

**Nodes removed:** `retrieve_candidates_node`, `prefilter_node`,
`retrieve_component_candidates_node`, `expression_validator_node`

---

## Phase I — Shared `search_snomed` tool

**Context:** All three ReAct agents (direct_match, focus_selector, mrcm_attribute_mapper)
need BM25+FAISS retrieval as a tool call. One implementation prevents three divergent
wrappers.

**Files:** `src/retrieval/hybrid_mapper.py`, `scripts/run_pipeline.py`

### I.1 — Add `search_snomed_concepts` to `hybrid_mapper.py`

Uses `search_query()` from `src/retrieval/search_utils.py` directly (the same primitive
`retrieve_candidates_for_item` calls), accessed via the cached `get_cached_mapper_resources()`
resource bundle. Avoids the fake item-dict pattern; gives agents a clean
`(query, hierarchy_filter?, k?)` interface.

```python
def search_snomed_concepts(
    query: str,
    hierarchy_filter: Optional[str] = None,
    k: int = 10,
) -> List[Dict[str, Any]]:
    resources = get_cached_mapper_resources(
        snomed_source_dir=os.environ.get("SNOMED_SOURCE_DIR", "snomed_us_source"),
        es_index=os.environ.get("MAPPER_ES_INDEX", "snomed_ct_bm25"),
        dense_index_path=os.environ.get("MAPPER_DENSE_INDEX_PATH", ""),
        model_name=os.environ.get("MAPPER_MODEL_NAME", ""),
        device=os.environ.get("MAPPER_DEVICE", "cuda"),
    )
    results = search_query(
        query,
        model=resources.st_model,
        faiss_index=resources.faiss_index,
        concept_meta_df=resources.concept_meta_df,
        es=resources.es,
        bm25_index=resources.es_index_name,
        k_dense=resources.k_dense,
        k_bm25=resources.k_bm25,
        k_final=k,
        is_a_graph=getattr(resources, "is_a_graph", None),
    )
    if hierarchy_filter:
        results = [
            r for r in results
            if r.get("top_level_hierarchy", "").lower() == hierarchy_filter.lower()
        ]
    return results[:k]
```

`get_cached_mapper_resources()` holds resources in a module-level cache — no reload
penalty on subsequent calls.

### I.2 — Add `search_snomed` case to `_run_ontology_tool` in `run_pipeline.py`

```python
if tool == "search_snomed":
    results = search_snomed_concepts(
        args.get("query", ""),
        hierarchy_filter=args.get("hierarchy_filter"),
        k=int(args.get("k", 10)),
    )
    return {"candidates": results}
```

Available to all three agents through the existing tool dispatch.

### I.3 — Import `search_snomed_concepts` in `run_pipeline.py`

Add to the `from src.retrieval.hybrid_mapper import` block.

---

## Phase II — `graph_client` response enrichment

**Context:** `get_domain_attributes` returns only bare SCTIDs — no names. LLMs cannot
reason over bare SCTIDs. Methods that return concept-level data must include
`preferred_term` and `fsn`.

**File:** `src/snomed/graph_client.py`

### II.1 — `get_domain_attributes` — join with Concept node

Add `MATCH (attr:Concept {sctid: d.attribute_sctid})` and return `attr.preferred_term`,
`attr.fsn`. New return format:

```python
{"attribute_sctid": "246075003", "preferred_term": "Causative agent", "fsn": "Causative agent (attribute)"}
```

Full new Cypher:
```cypher
MATCH (focus:Concept {sctid: $sctid})-[:IS_A*0..30]->(ancestor:Concept)
MATCH (d:MRCMAttributeDomain) WHERE d.domain_id = ancestor.sctid
MATCH (attr:Concept {sctid: d.attribute_sctid})
RETURN DISTINCT d.attribute_sctid AS attribute_sctid,
       attr.preferred_term AS preferred_term,
       attr.fsn AS fsn
```

### II.2 — `get_attribute_range` — add attribute name to `MRCMRange`

```python
@dataclass
class MRCMRange:
    attribute_sctid: str
    attribute_preferred_term: str   # NEW
    attribute_fsn: str              # NEW
    range_constraint: str
    attribute_rule: str
    rule_strength_id: str
    content_type_id: str
```

Query addition: `MATCH (attr:Concept {sctid: $attr})` and return `attr.preferred_term`,
`attr.fsn`.

---

## Phase III — `direct_match_node` — fully self-contained ReAct

**Context:** Remove dependency on pre-fetched `state["candidates"]`. Agent formulates
its own normalized query, calls `search_snomed`, optionally calls ontology tools, decides.

**Files:** `scripts/run_pipeline.py`, `agents/direct_match_agent.md`

### III.1 — Rewrite `agents/direct_match_agent.md`

Reasoning procedure:
- **Step 1**: Normalize `ci_text` to canonical clinical phrase (drop qualifiers, resolve
  "e.g." examples)
- **Step 2**: `search_snomed(normalized_query)`, optionally `hierarchy_filter="Clinical Finding"`
- **Step 3**: `get_logical_definition(sctid)` on best hit
- **Step 4**: Apply pre-coordination rule; return final answer

Available tools: `search_snomed`, `get_logical_definition`, `get_ancestors`
Max tool calls: 8

### III.2 — Update `_verify_direct_match_react`

- Remove `focus_candidates` parameter — no pool received
- Signature: `_verify_direct_match_react(self, ci_text: str) -> Dict`
- User prompt: only `ci_text`, no candidate list

### III.3 — `AgentRunConfig`

`direct_match_max_tool_calls: int = 8` (was 3); env var default `"8"`.

### III.4 — Remove `retrieve_candidates_node` from graph

Remove node, edges, and entry point. New entry: `"direct_match"`.
`ItemState.candidates` kept for backward compat, never populated.

---

## Phase IV — `focus_selector_node` — self-contained retrieval

**Context:** Remove `retrieve_component_candidates_node`. Agent receives `slot_hierarchies`
(Agent 2 output) and calls `search_snomed` itself.

**Files:** `scripts/run_pipeline.py`, `agents/focus_selector.md`, `src/llm/prompts.py`

### IV.1 — Rewrite `agents/focus_selector.md`

Reasoning procedure:
- **Step 1**: From `slot_hierarchies`, identify Clinical Finding / Disorder component
- **Step 2**: Normalize that text; `search_snomed(query, hierarchy_filter="Clinical Finding")`
- **Step 3**: Retry with different query if results are poor
- **Step 4**: `get_logical_definition` on top hits; pre-coordinated → `get_ancestors(max_depth=1)`
- **Step 5**: Return `{"focus_sctid": "...", "reasoning": "..."}<<END_JSON>>`

Available tools: `search_snomed`, `get_logical_definition`, `get_ancestors`, `get_siblings`
(`lookup_concept` removed — replaced by `search_snomed`)
Max tool calls: 10

### IV.2 — Update `build_focus_selector_user_prompt` in `src/llm/prompts.py`

Remove `candidate_pool` parameter. New structure:
```
FULL CONTRAINDICATION: {ci_text}
AGENT 2 COMPONENTS:
  [Clinical Finding] hypersensitivity
  [Substance] ibuprofen
Use search_snomed to retrieve candidates for the focus component.
Return final answer with <<END_JSON>>.
```

### IV.3 — Update `focus_selector_node` in `run_pipeline.py`

Remove `component_pool` construction. Term resolution: scan trace `candidates` from
`search_snomed` results for `preferred_term` (same pattern as existing ancestor scan).

### IV.4 — Remove `retrieve_component_candidates_node` from graph

Remove node and edges. `categorize_slots` → `focus_selector` directly.
`ItemState.component_candidates` kept, never populated.

### IV.5 — `AgentRunConfig`

`focus_selector_max_tool_calls: int = 10` (was 5).

---

## Phase V — `mrcm_attribute_mapper_node` — self-contained retrieval + inline validation

**Context:** Agent discovers attributes via tools, searches for values, validates inline.
Replaces `expression_validator_node`.

**Files:** `scripts/run_pipeline.py`, `agents/postcord_agent.md`, `src/llm/prompts.py`

### V.1 — Rewrite `agents/postcord_agent.md`

Reasoning procedure:
- **Step 1**: `get_domain_attributes(focus_sctid)` → `[{attribute_sctid, preferred_term, fsn}]`
- **Step 2**: `get_attribute_range(attr_sctid)` for each relevant attribute → ECL + name
- **Step 3**: Match each non-focus component from `slot_hierarchies` to an attribute
- **Step 4**: `search_snomed(component_text, hierarchy_filter=<range_hierarchy>)` per component
- **Step 5**: Output `refinements[]`, `decision`, `confidence`; set `review_flag: true`
  if `confidence < 0.5`

Available tools: `search_snomed`, `get_domain_attributes`, `get_attribute_range`,
`get_logical_definition`
Max tool calls: 12

### V.2 — Update `build_mrcm_mapper_react_user_prompt`

Remove `mrcm_rules` parameter. Inputs: `slot_hierarchies`, `focus_sctid`, `focus_term`,
`ci_text`. Agent discovers MRCM rules itself via tools.

### V.3 — `review_flag` from mrcm_attribute_mapper

```python
review_flag = parsed.get("confidence", 1.0) < 0.5 if parsed else True
return {**state, "refinements": refinements, "postcoord_pattern": parsed or {},
        "review_flag": review_flag}
```

### V.4 — Rename `route_after_validator` → `route_after_mapper`

Reads `state.get("review_flag", False)` — unchanged logic, new source.

### V.5 — Remove `expression_validator_node` from graph

Remove node and edges. `mrcm_attribute_mapper` → `normalize` directly.

### V.6 — `AgentRunConfig`

`mrcm_mapper_max_tool_calls: int = 12` (was 6).

---

## `ItemState` field changes (after all phases)

| Field | Status |
|---|---|
| `candidates` | Unused — keep for compat |
| `component_candidates` | Unused — keep for compat |
| `slot_hierarchies` | Active |
| `refinements` | Active |
| `review_flag` | Active — now set by mrcm_attribute_mapper |
| `validation` | Remove — was set by expression_validator_node |

---

## Summary of all file changes (Phases I–V)

| File | Phase | Change | Status |
|---|---|---|---|
| `src/retrieval/hybrid_mapper.py` | I | `search_snomed_concepts(query, hierarchy_filter, k)` via `search_query()` | ✅ Done |
| `scripts/run_pipeline.py` | I | `search_snomed` case in `_run_ontology_tool`; import | ✅ Done |
| `src/snomed/graph_client.py` | II | Enrich `get_domain_attributes` + `get_attribute_range` with preferred_term/fsn | ✅ Done |
| `agents/direct_match_agent.md` | III | Rewrite: query formulation + `search_snomed` tool | ✅ Done |
| `scripts/run_pipeline.py` | III | Remove pool param from `_verify_direct_match_react`; remove `retrieve_candidates_node`; budget → 8 | ✅ Done |
| `agents/focus_selector.md` | IV | Rewrite: agent calls `search_snomed`; remove `lookup_concept` | ✅ Done |
| `src/llm/prompts.py` | IV | `build_focus_selector_user_prompt` removes `candidate_pool` param | ✅ Done |
| `scripts/run_pipeline.py` | IV | Remove `retrieve_component_candidates_node`; budget → 10 | ✅ Done |
| `agents/postcord_agent.md` | V | Rewrite: full retrieval + inline validation + `review_flag` | ✅ Done |
| `src/llm/prompts.py` | V | `build_mrcm_mapper_react_user_prompt` removes `mrcm_rules` param | ✅ Done |
| `scripts/run_pipeline.py` | V | Remove `expression_validator_node`; mrcm_mapper sets `review_flag`; budget → 12 | ✅ Done |

---

## Verification

**Phase I:** `search_snomed` tool calls appear in direct_match, focus_selector, and
mrcm_attribute_mapper traces in `runtime_audit.jsonl`. Results include `id`, `label`,
`ancestor_path`.

**Phase II:** `get_domain_attributes` tool results include `preferred_term` and `fsn`.
`get_attribute_range` results include `attribute_preferred_term`.

**Phase III:** No pre-fetched candidate list in `direct_match` audit events. Trace shows
`search_snomed` with normalized query distinct from raw `ci_text` for compound items.

**Phase IV:** `focus_selector_trace` shows `search_snomed` before `get_logical_definition`.
No `retrieve_component_candidates` events in audit.

**Phase V:** `mrcm_attribute_mapper` trace shows `get_domain_attributes` →
`get_attribute_range` → `search_snomed` per component. No `expression_validator` events.
`review_flag` in item state set by mrcm_attribute_mapper.

---

---

# Phase VI — Generation-Time Stop Token (`PostThinkingStoppingCriteria`)

## Context

Model outputs noise tokens after `<<END_JSON>>` (e.g. `du du du...` visible in
`runtime_audit.jsonl`) because `stop` in `LocalLLM.generate()` is applied as
post-generation string truncation — the model fills `max_new_tokens` freely, then
the decoded string is split at the stop marker. Two compounding problems:

1. **Wasted compute** — model fills `max_new_tokens` with garbage after the end token.
2. **Premature truncation** — when `enable_thinking=True` (Gemma 4 `<|think|>` mode),
   the model may emit `<<END_JSON>>` as part of its internal reasoning chain inside
   `<think>...</think>`. The current `response.split(marker)[0]` fires on the first
   occurrence, discarding the actual output that follows `</think>`. This is why call
   sites that use thinking mode currently pass `stop=None` to avoid the truncation at all
   — letting the model run to `max_new_tokens` and relying on the parser to find the
   real JSON.

## Root Cause in Code

**File:** [src/llm/backends.py](src/llm/backends.py) lines 156–160

```python
if stop:
    for marker in stop:
        if marker and marker in response:
            response = response.split(marker)[0]   # fires on FIRST occurrence
            break
```

Post-generation truncation only. No awareness of `<think>` state.

## Fix

### 1 — Add `PostThinkingStoppingCriteria` class to `src/llm/backends.py`

```python
from transformers import StoppingCriteria, StoppingCriteriaList

class PostThinkingStoppingCriteria(StoppingCriteria):
    """
    Fires on stop_str only AFTER </think> has been seen in generated output.
    Prevents premature stop when the model mentions the stop token inside its
    reasoning chain. Accumulates decoded text token-by-token (O(1) per step).
    """
    def __init__(self, tokenizer: Any, stop_str: str, think_end: str = "</think>"):
        self.tokenizer = tokenizer
        self.stop_str = stop_str
        self.think_end = think_end
        self._buf = ""
        self._thinking_done = False

    def reset(self) -> None:
        self._buf = ""
        self._thinking_done = False

    def __call__(self, input_ids: Any, scores: Any, **kwargs) -> bool:
        tok = self.tokenizer.decode(
            [input_ids[0, -1].item()], skip_special_tokens=False
        )
        self._buf += tok
        if not self._thinking_done and self.think_end in self._buf:
            self._thinking_done = True
        if self._thinking_done and self.stop_str in self._buf:
            return True
        return False
```

Place the class definition immediately before the `LocalLLM` dataclass (around line 60).

### 2 — Use `StoppingCriteriaList` in `LocalLLM.generate()`

In `generate()`, before calling `self.model.generate(...)`, build stopping criteria
when `enable_thinking=True` and `stop` is provided:

```python
stopping_criteria = None
if stop and self.enable_thinking:
    criteria = [
        PostThinkingStoppingCriteria(self.tokenizer, marker)
        for marker in stop if marker
    ]
    stopping_criteria = StoppingCriteriaList(criteria)

generate_kwargs = { "max_new_tokens": max_new_tokens, ... }
if stopping_criteria:
    generate_kwargs["stopping_criteria"] = stopping_criteria
```

Keep the existing post-processing truncation block (lines 156–160) as a safety fallback
for non-thinking models and edge cases where the criteria fires one token late.

### 3 — Remove `stop=None` overrides at all call sites

Once `PostThinkingStoppingCriteria` is in place, call sites that bypass stop to avoid
premature thinking-block truncation can be restored to use the configured stop list.

**`scripts/run_pipeline.py`** — 7 occurrences of `stop=None` at lines 822, 1038, 1189,
1332, 1371, 1724, 1765. All are in ReAct loop `.chat()` calls and simple chain calls
that currently pass `stop=None` defensively. Change: remove the `stop=None` override
and let the call inherit `cfg.stop` (i.e. `["<<END_JSON>>"]`).

**`src/llm/prompts.py`** — All `stop` usages in prompts.py already pass the `stop`
argument through from the caller without overriding. No changes needed here provided
call sites in `run_pipeline.py` stop passing `stop=None`.

---

## Status — ⛔ SUPERSEDED by Phase VII

Phase VI is cancelled. vLLM (Phase VII) handles stop tokens and thinking-mode natively.
The `PostThinkingStoppingCriteria` class is not needed. The `stop=None` overrides in
`run_pipeline.py` can be left in place — vLLM stops correctly regardless.

---

---

# Phases VII–IX: vLLM + MCP + ToolNode

## Context

Migrate from HuggingFace `transformers` local serving to vLLM OpenAI-compatible endpoint.
This solves Phase VI for free, enables native structured tool calling across all modern
models, and creates an MCP server seam for external orchestrators.

Roadmap: **Phase VII** (drop-in vLLM swap) → **Phase VIII** (MCP tool server) → **Phase IX** (ToolNode per agent).

---

## Phase VII — vLLM Backend

**Goal:** Same pipeline, different serving layer. Python-loop ReAct unchanged. Stop noise gone.

### Deployment

```bash
vllm serve google/gemma-4-31B-it \
  --port 8000 --tensor-parallel-size 3 \
  --gpu-memory-utilization 0.90 \
  --enable-auto-tool-choice --tool-call-parser pythonic \
  --max-model-len 8192
```

Tool-call-parser by model: Gemma 4 → `pythonic` | Qwen 3.x → `hermes` | Llama 3.1/3.3 → `llama3_json`

### New code in `scripts/run_pipeline.py`

**`VLLMLLMConfig`** dataclass — reads `VLLM_BASE_URL`, `VLLM_MODEL_ID`, `VLLM_API_KEY`.

**`VLLMChatLLM`** class — wraps `openai.OpenAI(base_url=..., api_key=...)` with same `.chat()` signature as `HuggingFaceChatLLM`. Passes `stop` list directly to completions API.

**`build_llm()` extension:**
```python
if backend == "vllm":
    return VLLMChatLLM(VLLMLLMConfig.from_env())
```

Add `"vllm"` to argparse choices.

### New env vars

| Var | Default | Purpose |
|---|---|---|
| `VLLM_BASE_URL` | `http://localhost:8000/v1` | Server endpoint |
| `VLLM_MODEL_ID` | `google/gemma-4-31B-it` | Model name in API calls |
| `VLLM_API_KEY` | `token` | Auth token (any non-empty string) |
| `VLLM_TIMEOUT_SECONDS` | `120` | Per-request timeout |
| `LANGGRAPH_LLM_BACKEND` | `huggingface` | Add `vllm` as valid choice |

### Files changed

| File | Change | Status |
|---|---|---|
| `scripts/run_pipeline.py` | `VLLMLLMConfig`, `VLLMChatLLM`, extend `build_llm()` + argparse | ⬜ TODO |
| `.env` | Add `VLLM_BASE_URL`, `VLLM_MODEL_ID`, set `LANGGRAPH_LLM_BACKEND=vllm` | ⬜ TODO |

### Verification
Run 5-SPL test with `LANGGRAPH_LLM_BACKEND=vllm`. Confirm: no trailing noise in `raw_output`,
`duration_s` lower than transformers baseline, all items complete without crash.

---

## Phase VIII — MCP Tool Server

**Goal:** Expose all 7 SNOMED tools as MCP servers for external orchestrator access.

### New file: `src/tools/snomed_mcp_server.py`

FastMCP server wrapping `search_snomed_concepts()` and all `SnomedGraphClient` methods.
7 tools: `search_snomed`, `get_logical_definition`, `get_ancestors`, `get_siblings`,
`get_domain_attributes`, `get_attribute_range`, `lookup_concept`.

### Transport

**HTTP** (pipeline integration):
```bash
fastmcp run src/tools/snomed_mcp_server.py --transport http --port 8001
```

**Stdio** (Claude Code / external agents): register in `.mcp.json`.

### Integration in `run_pipeline.py`

`_run_ontology_tool` gets optional `mcp_client` param. When `SNOMED_TOOLS_BACKEND=mcp`,
routes dispatch through MCP client instead of direct Python calls. Default: `"direct"`.

### New env vars

| Var | Default | Purpose |
|---|---|---|
| `SNOMED_TOOLS_BACKEND` | `direct` | `direct` = Python; `mcp` = MCP client |
| `SNOMED_MCP_URL` | `http://localhost:8001` | MCP server URL |

### Files changed

| File | Change | Status |
|---|---|---|
| `src/tools/snomed_mcp_server.py` | New FastMCP server | ⬜ TODO |
| `scripts/run_pipeline.py` | Optional MCP dispatch in `_run_ontology_tool` | ⬜ TODO |
| `.mcp.json` | Register server for Claude Code | ⬜ TODO |
| `environment.yml` | Add `fastmcp` dependency | ⬜ TODO |

### Verification
Start MCP server; run pipeline with `SNOMED_TOOLS_BACKEND=mcp`. Output identical to direct.
Test with Claude Code: manually call `search_snomed("hypersensitivity disorder")`.

---

## Phase IX — LangGraph ToolNode Conversion

**Goal:** Replace Python-loop ReAct with LangGraph `create_react_agent` + `ToolNode`.
Requires Phase VII. Convert one agent at a time behind feature flags.

### Tool schemas

New file `src/tools/schemas.py`: LangChain `@tool`-decorated wrappers for all 7 tools.
These are passed to both `bind_tools()` (tells LLM what's available) and `ToolNode`
(handles execution).

### `VLLMChatLLM` extension

Add `get_langchain_llm() -> ChatOpenAI` method returning `ChatOpenAI(base_url=..., model=...)`.
Used to call `llm.get_langchain_llm().bind_tools(AGENT_TOOLS)`.

### Per-agent conversion pattern

Each node gets a feature flag. When enabled, `create_react_agent` replaces the Python loop:
```python
agent = create_react_agent(
    model=self.llm.get_langchain_llm().bind_tools(tools),
    tools=tools,
    state_modifier=SystemMessage(content=_load_agent_md()),
)
```

Conversion order: `direct_match_node` → `focus_selector_node` → `mrcm_attribute_mapper_node`.

### Feature flags

| Var | Default | Effect |
|---|---|---|
| `DIRECT_MATCH_USE_TOOLNODE` | `false` | Use ToolNode for direct_match |
| `FOCUS_SELECTOR_USE_TOOLNODE` | `false` | Use ToolNode for focus_selector |
| `MRCM_MAPPER_USE_TOOLNODE` | `false` | Use ToolNode for mrcm_attribute_mapper |

Enable one at a time. Validate concept-level F1 ≥ Python-loop baseline before proceeding.

### Files changed

| File | Change | Status |
|---|---|---|
| `src/tools/schemas.py` | New — LangChain `@tool` definitions | ⬜ TODO |
| `scripts/run_pipeline.py` | `VLLMChatLLM.get_langchain_llm()`; per-agent ToolNode branches | ⬜ TODO |
| `environment.yml` | Add `langchain-openai` dependency | ⬜ TODO |

### Verification
Per agent after enabling ToolNode: audit shows structured `ToolCall` objects in traces;
concept-level F1 ≥ Python-loop baseline; agent `.md` system prompts still loaded unchanged.

---

## Phase dependencies

```
Phase VII ──→ Phase IX (required — ToolNode needs native tool calling)
Phase VII ──→ Phase VIII (independent, can run in parallel)
Phase VIII ──→ Phase IX (optional — ToolNode can use direct Python OR MCP transport)
```
