# Agents

This directory contains agentic modules that extend the core LangGraph pipeline. Each
agent operates within the architectural invariant:

> **LLM proposes; ontology/rules validate.**

Agents call into `src/` modules — they do not bypass ontology validation.

---

## Agents

### 1. SNOMED Concept Navigator (`snomed_concept_reader`)

**Purpose:** Load and structure the top-level SNOMED CT hierarchy as navigational context
for LLM prompts. Produces `snomed_conventions.md` as a stable offline artifact.

**Interface:**
- Reads from `src/snomed/snomed_utils.py` (`load_snomed_dataframes`, IS-A graph)
- Produces `agents/snomed_conventions.md` documenting hierarchy naming conventions
- **NOT a runtime agent** — runs once per SNOMED release as offline pre-computation
- Output feeds into `src/llm/prompts.py` via `_load_snomed_conventions()`

**Status:** Implemented (snomed_conventions.md generated; Neo4j graph built)

---

### 2. Compositional Extractor (`compositional_extractor`)

**Purpose:** Given a decomposed contraindication span, segment it into SNOMED-labelled
components (Clinical Finding, Substance, Qualifier Value, Body Structure, etc.) and
populate item slots before candidate retrieval.

**Interface:**
- System prompt: `agents/snomed_conventions.md`
- Input: `ci_text` from decomposed item
- Output: `slot_hierarchies` dict + populated item slots (`contraindication_state_text`,
  `substance_text`, `severity_span`) in `ItemState`
- Phase 2: `slot_hierarchies` components with `lookup_needed: true` are resolved to SNOMED
  preferred terms via `SnomedGraphClient.lookup_concept()`
- Phase 3 (planned): `slot_hierarchies` feeds `retrieve_component_candidates` — each
  component with a resolved preferred_term drives a per-slot BM25+dense retrieval query,
  producing component-specific candidate lists (Priority 3, CLAUDE.md)
- Implemented in: `scripts/run_pipeline.py::categorize_slots_node`
- Prompt loader: `src/llm/prompts.py::_load_snomed_conventions()`

**Status:** Implemented (Phase 1 + Phase 2) | Phase 3 planned

---

### 2.5. Focus Selector (`focus_selector`)

**Purpose:** Given the focus component identified by Agent 2 and the full-text focus
candidates from retrieval, select the most accurate SNOMED focus concept for
post-coordination by reasoning over ontology structure.

**Interface:**
- System prompt: `agents/focus_selector.md`
- Input: mixed per-component candidate pool from `retrieve_component_candidates`; Agent 2
  component labels from `slot_hierarchies`
- Tools available (candidates-first, tools for verification): `get_logical_definition`,
  `get_ancestors`, `get_siblings`, `lookup_concept` (fallback only) — via `SnomedGraphClient`
- Output: `selected_problem_id` (abstract focus concept SCTID), `focus_selector_trace`
  (full tool call history stored in `ItemState` for audit)
- Implemented in: `scripts/run_pipeline.py::focus_selector_node`
- Prompt loader: `src/llm/prompts.py::_load_focus_selector()`

**Key challenge:** The candidate pool contains ALL component candidates mixed (focus
concepts, substances, qualifiers) — they are not separated by component type. Agent 2.5
must first identify focus-level candidates using Agent 2's hierarchy labels, then verify
each is abstract (not pre-coordinated) before selecting.

**Key capability — Abstraction detection:**

For "hypersensitivity to ibuprofen" (candidates pool includes `405543001` and `267038008`):
1. `get_logical_definition("405543001")` → non-empty → pre-coordinated → not the focus
2. `get_ancestors("405543001", max_depth=1)` → `267038008` (Hypersensitivity disorder)
3. `267038008` is in pool; `get_logical_definition("267038008")` → empty → abstract → correct
4. Selected focus: `267038008` — ibuprofen becomes a causative_agent fill for Agent 3

**Fallback:** If `graph_client` is None or `FOCUS_SELECTOR_ENABLED=0`, picks the top
focus-tagged candidate without tool reasoning.

**Status:** In progress

---

### 3. Post-Coordination Expression Agent (`postcord_agent`)

**Purpose:** Given a focus concept and non-focus components from Agent 2 with their
per-component candidates, determine the valid MRCM attributes for the focus concept
and assign each component to the appropriate attribute fill.

**Interface:**
- System prompt: `agents/postcord_agent.md`
- Input: `selected_problem_id` from Agent 2.5; non-focus components from `slot_hierarchies`;
  per-component candidate lists from `retrieve_component_candidates`
- MRCM lookup: `SnomedGraphClient.get_attribute_domain_rules(focus_sctid)` determines which
  attributes are valid for this concept (replaces hardcoded `attribute_table`)
- Output: `refinements: [{attribute_sctid, attribute_fsn, value_sctid, value_term}, ...]` —
  variable length, MRCM-driven (not constrained to 3 hardcoded slots)
- Expression Validator subagent: verifies expression against MRCM range constraints;
  routes low-confidence items to `status="REVIEW"` for human triage
- Implemented in: `scripts/run_pipeline.py` — `pattern_finder_node` (v1) →
  `mrcm_attribute_mapper_node` (v2, planned)

**v1 (current):** Refines existing `fills_norm` from `route_or_fill` using Neo4j role context
and confidence-gated overrides. Slot schema still hardcoded to 3 attributes.

**v2 (planned):** Replaces `route_or_fill` slot selection entirely. MRCM domain rules
determine the valid attribute set at runtime per focus concept.

**Status:** Implemented (v1 — pattern_finder) | v2 redesign planned (MRCM-agnostic)

---

## Integration Contract

All agents must:
1. Call into `src/` modules for ontology access — never duplicate SNOMED logic
2. Return structured output compatible with `ContraState` or `ItemState` in `scripts/run_pipeline.py`
3. Pass results through the validation layer in `src/snomed/snomed_utils.py` before committing
4. Be parallelizable via LangGraph `Send` API when processing multiple items
