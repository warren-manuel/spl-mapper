# Pipeline Context: SPL Contraindication Mapping

## What This Project Does

LangGraph-based pipeline that extracts contraindications from SPL (Structured Product Label) records and maps them to SNOMED CT concepts. Entry point: `scripts/run_pipeline.py`.

---

## Current Architecture (as of 2026-05-21)

### Two Nested LangGraph Graphs

**SPL-level graph** (one per SPL record):
```
resolve_contra_section → extract_items → [loop] prepare_item → process_item → advance_item → finalize
```

**Item-level graph** (one per extracted contraindication):
```
direct_match (ReAct)
  → [DIRECT] assemble_direct → END
  → [NO MATCH] categorize_slots
      → focus_selector (ReAct)
          → normalize
              → mrcm_attribute_mapper (ReAct)
                  → [assemble_postcoord | assemble_review] → END
```

All three ReAct agents are **self-contained**: they call `search_snomed` and ontology tools themselves. No pre-built candidate pools are passed in.

### Three ReAct Agents (Python-loop, bounded tool calls)

| Agent | Node | Tools | Budget |
|---|---|---|---|
| Direct Match | `direct_match_node` | `search_snomed`, `get_logical_definition`, `get_ancestors` | 8 calls |
| Focus Selector | `focus_selector_node` | `search_snomed`, `get_logical_definition`, `get_ancestors`, `get_siblings` | 10 calls |
| MRCM Attribute Mapper | `mrcm_attribute_mapper_node` | `get_domain_attributes`, `get_attribute_range`, `search_snomed`, `get_logical_definition` | 12 calls |

### Two Simple-Chain LLM Calls Per SPL

| Node | Call | Pattern |
|---|---|---|
| `extract_items_node` | Extract discrete contraindication items from SPL section text | Single-shot |
| `decompose_coordinations_node` | Linguistically split coordinated items (e.g. "A and B" → [A, B]) | Single-shot per item |

### Retrieval Stack
```
BM25 (multi-tier: exact → phrase → shingle → token) ─┐
                                                       ├──▶ RRF ──▶ cross-encoder re-rank ──▶ top-K
FAISS (dense, embeddinggemma-300m)                    ─┘
```
- `search_snomed_concepts()` in `src/retrieval/hybrid_mapper.py` is the shared entry point for all agent tool calls
- Backed by `search_query()` from `src/retrieval/search_utils.py`
- FAISS index cached in `get_cached_mapper_resources()` — no reload across calls
- Cross-encoder reranker: off by default, enabled via `RERANKER_MODEL_ID` env var
- Each candidate enriched with IS-A ancestor path via `get_longest_ancestor_path`

### Ontology Backend

**Neo4j graph** (`src/snomed/graph_client.py`): IS-A traversal, HAS_ROLE logical definitions, MRCM domain/range lookups. All methods return concept names alongside SCTIDs.

**SNOMED RF2 flatfiles** (`src/snomed/snomed_utils.py`): ECL evaluation, MRCM constraint lookup, prefilter membership.

### State Models
- `ContraState` — SPL-level: raw record, set ID, product name, contra text, extracted items, item index, results, errors
- `ItemState` — Item-level: extracted item, slot_hierarchies (Agent 2 output), selected_problem_id, selected_focus_term, refinements (list of `{attribute_sctid, attribute_fsn, value_sctid, value_term}`), fills_norm, fills_detail, review_flag, expression, status, item_result

---

## What Has Been Done

### Phase A–E: Agent Architecture Foundation
Implemented `categorize_slots` (Agent 2), `focus_selector` (Agent 2.5), and `mrcm_attribute_mapper` (Agent 3 v2). Wired `expression_validator` bridge, `normalize_node` generalization, `assemble_postcoord_node` generalization. Deprecated `pattern_finder` and `route_or_fill` (disabled by default).
→ Details: `plans/AGENT_02_BUILD_PLAN.MD`, `plans/AGENT_03_BUILD_PLAN.MD`, `plans/AGENT_IMPLEMENTER_PLAN.md` §Phases A–E

### Phase F–H: ReAct Upgrades + Legacy Deprecation
Upgraded `direct_match_node` and `mrcm_attribute_mapper_node` to full ReAct agents. Wrote `agents/direct_match_agent.md`. Set `pattern_finder_enabled=False` and `route_or_fill_enabled=False` as defaults.
→ Details: `plans/AGENT_IMPLEMENTER_PLAN.md` §Phases F–H

### Phase I–V: Self-Contained Retrieval Restructure
Removed pre-built candidate pools from all three agents. Added `search_snomed_concepts()` shared tool. Enriched `graph_client` responses with preferred_term/fsn. Removed `retrieve_candidates_node`, `prefilter_node`, `retrieve_component_candidates_node`, `expression_validator_node` from the graph. Rewrote all three agent `.md` system prompts.
→ Details: `plans/AGENT_IMPLEMENTER_PLAN.md` §Phases I–V

### Phase VII: vLLM Backend (2026-05-13)
Replaced HuggingFace `transformers` with vLLM OpenAI-compatible endpoint. `VLLMLLMConfig` + `VLLMChatLLM` added to `scripts/run_pipeline.py`. Select via `LANGGRAPH_LLM_BACKEND=vllm`. Phase VI (`PostThinkingStoppingCriteria`) cancelled — vLLM handles stop tokens natively. Added `<|think|>` prefix to all three ReAct agent `.md` system prompts to enable Gemma 4 thinking mode.
→ Details: `plans/plan_VLLM.md` §Phase VII

### Bug Fixes + Trace Output Fixes (2026-05-13/14)
- Fixed `normalize_node` KeyError on `state["candidates"]` (retrieve_candidates_node removed)
- Fixed `assemble_postcoord_node` / `assemble_review_node` KeyError on `state["route_or_fill"]` (disabled)
- Fixed graph wiring: `route_or_fill` now conditional on `cfg.route_or_fill_enabled`
- **Fix 2:** Added `mrcm_mapper_trace` to `ItemState`; `mrcm_attribute_mapper_node` now returns it; both assemble nodes include `focus_selector` + `mrcm_mapper` traces in `item_result.trace` (replaced dead `route_or_fill`/`validation` keys)
- **Fix 3:** Added `graph_client.lookup_concept` fallback in `focus_selector_node` when trace-scan term resolution returns "N/A"; removed dead `focus_candidates` lookup from `normalize_node`
- **Fix 1 (server):** vLLM restarted with `--max-model-len 32768` (was 8192, caused context overflow at focus_selector)
→ Details: `plans/plan_VLLM.md` §Active Fixes

### Phase VIII: MCP Tool Server (2026-05-14, files present — verification pending)
Created `src/tools/snomed_mcp_server.py` (FastMCP server with 7 tools) and `src/tools/schemas.py` (LangChain `@tool` wrappers). Registered in `.mcp.json` for Claude Code access. Pipeline default stays `SNOMED_TOOLS_BACKEND=direct`.
→ Details: `plans/plan_VLLM.md` §Phase VIII

### Post-vLLM Pipeline Fixes (2026-05-19/21)

**Parsing robustness — `categorize_slots`:**
- `parse_categorize_slots_output` now accepts both `"components"` and `"segments"` top-level keys, and both `"text"` and `"span"` component text fields (LLM was outputting `"segments"/"span"` due to prompt wording)
- User prompt reworded from "Segment this span…" to "Identify each clinical component…" with explicit JSON schema example
- `categorize_slots_node` builder uses `comp.get("text") or comp.get("span", "")` for consistency
- **Impact:** `NON-FOCUS COMPONENTS: (none identified)` in mrcm_attribute_mapper resolved; Agent 2 slots now reach Agent 3

**Focus term sanitization:**
- `mrcm_attribute_mapper_node` now sanitizes `focus_term` before prompt builder: if term is >60 chars (reasoning text fallback) or "N/A", calls `graph_client.lookup_concept(focus_sctid)` to get the clean preferred term
- **Impact:** `FOCUS CONCEPT: 419076005 | Allergic reaction (disorder)` instead of `FOCUS CONCEPT: 419076005 | The focus component is 'allergic reaction'...`

**MRCM agent output parity:**
- `agents/postcord_agent.md`: added `reasoning` field to output format — aligns with `direct_match_agent.md` and `focus_selector.md`

**Graph client fixes (`src/snomed/graph_client.py`):**
- `get_domain_attributes`: changed inner `MATCH (attr:Concept)` to `OPTIONAL MATCH`; fallbacks to SCTID string when Concept node missing (attribute concepts were not loaded)
- `get_attribute_range`: same `OPTIONAL MATCH` fix; fallback to SCTID
- New `get_hierarchy_map()`: single bulk Cypher query returning `{sctid_int: top_level_hierarchy}` for all concepts; used to pre-populate `MapperResources`

**Retrieval enrichment (search hits now include `top_level_hierarchy` + `semantic_tag`):**
- `MapperResources` (hybrid_mapper.py): new `hierarchy_map: Optional[Dict[int, str]]` field
- `get_cached_mapper_resources`: new optional `graph_client` param — calls `get_hierarchy_map()` once and caches result; `USE_ANCESTOR_PATHS` env var check moved here (was only in standalone wrappers)
- `search_query` (search_utils.py): enriches all hits with `top_level_hierarchy` (from `hierarchy_map`) and `semantic_tag` (from `concept_meta_df`)
- `hierarchy_filter` in `search_snomed_concepts` now works — was always returning empty because hits never had `top_level_hierarchy` before this fix
- `search_snomed_for_agent` gains `graph_client=None` kwarg; call sites within `ContraLangGraphAgent` pass `self.graph_client`

**SNOMED graph rebuild with Attribute hierarchy (completed 2026-05-26):**
- `build_snomed_graph.py`: `246061005` (Attribute) added to `TOP_LEVEL_HIERARCHIES`, `CONTRAINDICATION_RELEVANT`, `HIERARCHY_PRIORITY`
- Graph rebuilt — `get_domain_attributes` now returns proper attribute names (e.g. "Causative agent") instead of bare SCTIDs

**Code quality fixes:**
- `_call_llm_json`: temperature hardcode changed `1.0` → `0.0`; `.env` duplicate `HF_TEMPERATURE=0.1` block commented out
- All `_log_llm_call` call sites: removed `system[:200] + "…"` truncation — full system prompt now logged
- ToolNode `create_react_agent` calls: `state_modifier=` → `prompt=` (LangGraph 1.0 API rename)

**Phase IX ToolNode — issue identified, flags reset:**
- ToolNode path produces `{tool, args, trace}` in `direct_verify` instead of `{direct_match, selected_id, reasoning, trace}` because agent `.md` prompts are written for Python-loop JSON-text output, not native LangGraph tool calls
- All three ToolNode flags set to `false` in `.env` pending proper Phase IX prompt rewrite + validation
→ Details: `plans/plan_VLLM.md` §Phase IX

### SNOMED Neo4j Knowledge Graph
Built Neo4j graph from SNOMED RF2 source. Loaded Concept, IS_A, HAS_ROLE, MRCMAttributeDomain, MRCMAttributeRange nodes/edges.
→ Details: `plans/SNOMED_BUILD_PLAN.md`, `plans/SNOMED_BUILD_LOG.md`

### Codebase Restructure
Migrated from monolithic script to `src/` package layout.
→ Details: `plans/RESTRUCTURE_PLAN.md`, `plans/RESTRUCTURE_LOG.md`, `plans/RESTRUCTURE_AUDIT.md`

---

## Current Metrics (baseline — `results/20260401-02/`)

| Model | Level | Precision | Recall | F1 |
|---|---|---|---|---|
| Qwen 3.5 | Extraction | 0.871 | 0.901 | 0.886 |
| Qwen 3.5 | Contraindication | 0.409 | 0.424 | 0.416 |
| Qwen 3.5 | Concept | 0.580 | 0.524 | 0.551 |
| MedGemma | Extraction | 0.844 | 0.860 | 0.852 |
| MedGemma | Contraindication | 0.356 | 0.363 | 0.360 |
| MedGemma | Concept | 0.552 | 0.439 | 0.489 |

**Optimization target:** concept-level F1. Acceptance criterion: contraindication-level F1.

### Diagnosed Failure Modes
- **Failure Mode 1 (42–49% of concept failures):** Direct-match fires on pre-coordinated concepts used as semantic approximations. Fixed by ReAct direct_match agent with `get_logical_definition` check.
- **Failure Mode 2 (40–45% of concept failures):** Near-synonymous SNOMED concepts ranked equally by RRF; LLM picks wrong one. Partially addressed by ancestor path enrichment; further addressed by self-contained agent retrieval.

---

## Active Run Configuration

| Setting | Value |
|---|---|
| Model | `google/gemma-4-31B-it` (vLLM, `LANGGRAPH_LLM_BACKEND=vllm`) |
| vLLM server | `CUDA_VISIBLE_DEVICES=1,2`, TP=2, `--max-model-len 32768` |
| Embedding model | `google/embeddinggemma-300m` (`MAPPER_DEVICE=cuda:0` in splmap env) |
| BM25 index | `snomed_ct_bm25` (Elasticsearch) |
| Graph client | Neo4j at `bolt://139.52.39.81:7687` (requires `--enable-graph-client`) |
| Thinking mode | Enabled (`<|think|>` prefix in all three agent `.md` system prompts) |
| ToolNode flags | All `false` — Python-loop ReAct active |
| conda envs | `splmap` — pipeline; `vllm_env` — vLLM server |

**Run command:**
```bash
SMOKE=results/smoke_$(date +%Y%m%d_%H%M) && mkdir -p $SMOKE && \
OUTPUT_DIR=$SMOKE conda run -n splmap \
  python3 -m scripts.run_pipeline \
  --spl-list results/VO_SPL_5.txt \
  --enable-graph-client \
  2>&1 | tee $SMOKE/run.log
```

---

## Planned Work

### Phase IX — LangGraph ToolNode Conversion ⚠️ (scaffolding done, blocked on prompt rewrite)
All three ToolNode branches are implemented in `run_pipeline.py` with feature flags. Currently all `false`.

**Blocking issues before enabling:**
1. Agent `.md` prompts instruct the LLM to output JSON tool calls as text (`{"tool": ..., "args": ...}`) — incompatible with native LangGraph `tool_calls`. Prompts must be rewritten for ToolNode format (natural-language tool descriptions, plain-text final answer, no `<<END_JSON>>`).
2. Trace output parity: ToolNode path stores `{tool, args, trace}` in `direct_verify`; Python-loop stores `{direct_match, selected_id, reasoning, trace}`. Both paths must emit the same schema.

Convert order: `direct_match_node` → `focus_selector_node` → `mrcm_attribute_mapper_node`. Validate F1 ≥ baseline at each step before enabling next.
→ Full plan: `plans/plan_VLLM.md` §Phase IX

### Phase VIII — MCP Tool Server ⚠️ (files present, deps installed, verification pending)
`src/tools/snomed_mcp_server.py`, `.mcp.json`, `fastmcp`, `langchain-openai` all in place. Next: confirm Claude Code can call `search_snomed` directly via `.mcp.json` stdio registration.
→ Full plan: `plans/plan_VLLM.md` §Phase VIII

### Output Schema Restructure ⬜ (planned, not yet implemented)
Rename `results → contraindications` at SPL level. Per-item: drop `SPL_SET_ID`, `post_decision`, `fills`; rename `selected_id/term → primary_concept_id/fsn`; surface `decision`, `confidence`, `refinements` from `postcoord_pattern`. Update evaluator CSV columns: add `product_name`, `status`, `decision`, `confidence`; remove `mapping_source`, `final_concept_id`, `postcoord_expression`.
→ Full plan: `plans/OUTPUT_SCHEMA_PLAN.md`

### Optimization Roadmap

| Priority | Item | Status |
|---|---|---|
| 1 | Post-extraction precision filter (contraindication vs warning/monitoring) | ⬜ Planned |
| 2 | Slot presence detection (replace binary direct-match with slot detection) | ⬜ Planned |
| 3 | Slot-specific targeted retrieval | ✅ Superseded by self-contained agent retrieval |
| 4 | Cross-encoder reranker after RRF | ✅ Wired in (`RERANKER_MODEL_ID` env var, off by default) |
| 5 | Implicit slot inference in prompts (severity/course inferred from text) | ⬜ Planned |
| 6 | Parallelize item processing (LangGraph `Send` API) | ⬜ Planned |
| 7 | Prompt hardening for SNOMED accuracy | ⚠️ Partial — ancestor paths ✅, FSN display ⬜, negative anchoring ⬜ |

---

## Module Responsibilities

### `src/extraction/`
- **`section_parser.py`** — DailyMed XML fetching, SPL section parsing (contraindication LOINC `34070-3`); `extract_section()`, `fetch_spl_xml_by_setid()`

### `src/retrieval/`
- **`hybrid_mapper.py`** — BM25 + FAISS hybrid; `retrieve_candidates_for_item()`, `search_snomed_concepts()` (agent tool entry point), `get_cached_mapper_resources()`, `MapperResources`
- **`search_utils.py`** — `build_snomed_query`, `search_query`, `fuse_hits_rrf`, `rerank_candidates`, `get_longest_ancestor_path`
- **`embedding_utils.py`** — `load_ST_model`, `build_and_save_dense_index`, `maybe_move_index_to_gpu`
- **`es_utils.py`** — Elasticsearch connection, index creation, bulk indexing
- **`dense_ann.py`** — FAISS dense ANN wrapper

### `src/mapping/`
- **`postcoord.py`** — Legacy post-coordination worker (multi-GPU). Superseded in main pipeline by `mrcm_attribute_mapper_node`.

### `src/llm/`
- **`backends.py`** — `LocalLLM`, `load_model_local`, `extract_json`, `build_message`; stop token handling (post-processing, pending Phase VI upgrade)
- **`prompts.py`** — System/user prompt builders and LLM call wrappers for all nodes: `extract_contraindication_items`, `categorize_item_slots`, `build_direct_match_agent_user_prompt`, `build_focus_selector_user_prompt`, `build_mrcm_mapper_react_user_prompt`

### `src/snomed/`
- **`graph_client.py`** — Neo4j client: `lookup_concept`, `get_logical_definition`, `get_ancestors`, `get_siblings`, `get_domain_attributes`, `get_attribute_range`
- **`snomed_utils.py`** — RF2 flatfile loading, ECL evaluation, `get_ancestors_with_depth`, `get_range_constraints_for_attribute`, `validate_postcoord_with_mrcm`

### `src/evaluation/`
- **`evaluator.py`** — Three-level evaluation (extraction / contraindication / concept); `aggregate_agent_results`, `evaluate_aggregated_predictions`, `write_jsonl`, `write_csv_rows`, `parse_json_with_end_marker`

### `agents/` — System prompt files for ReAct agents
- `direct_match_agent.md` — Direct match ReAct agent (query formulation → search → pre-coord check → decide)
- `focus_selector.md` — Focus selector ReAct agent (identify focus component → search → verify → resolve parent)
- `postcord_agent.md` — MRCM attribute mapper ReAct agent (domain attrs → range → search per component → refinements[])
- `snomed_conventions.md` — SNOMED hierarchy taxonomy for `categorize_slots` (Agent 2)

---

## Key Invariant

> **LLM proposes; ontology/rules validate.**

- LLM calls live in `src/llm/` and are invoked by graph nodes in `scripts/run_pipeline.py`
- Ontology validation lives in `src/snomed/`
- The graph in `scripts/run_pipeline.py` orchestrates both but never mixes them within a single node

---

## LLM Backends Supported

| Backend | Class | Status | Tool calling |
|---|---|---|---|
| HuggingFace `transformers` | `HuggingFaceChatLLM` | Available | Python-loop JSON parsing |
| Azure OpenAI | `AzureChatLLM` | Available | Native (OpenAI API) |
| vLLM | `VLLMChatLLM` | **Active** | Native via `--enable-auto-tool-choice` |

Select via `LANGGRAPH_LLM_BACKEND=huggingface|azure|vllm`.

---

## Directory Layout

```
VaxMapperRepo/
├── src/                    # Library packages
│   ├── extraction/         # SPL loading and section parsing
│   ├── retrieval/          # FAISS + BM25 + RRF + reranker
│   ├── mapping/            # Legacy post-coordination worker
│   ├── llm/                # LLM backends and prompt builders
│   ├── snomed/             # SNOMED RF2 loading, ECL, Neo4j client
│   └── evaluation/         # Metrics and output utilities
├── scripts/
│   └── run_pipeline.py     # Main LangGraph pipeline entry point
├── agents/                 # ReAct agent system prompts (.md)
├── plans/                  # All planning, build, and audit documents
│   ├── AGENT_IMPLEMENTER_PLAN.md   # Master implementation plan (Phases A–VI)
│   ├── AGENT_02_BUILD_PLAN.MD      # Agent 2 (categorize_slots) build history
│   ├── AGENT_03_BUILD_PLAN.MD      # Agent 3 (mrcm_attribute_mapper) build history
│   ├── SNOMED_BUILD_PLAN.md        # Neo4j graph build plan
│   ├── SNOMED_BUILD_LOG.md         # Neo4j build execution log
│   ├── RESTRUCTURE_PLAN.md         # src/ layout migration plan
│   ├── RESTRUCTURE_LOG.md          # src/ migration log
│   ├── RESTRUCTURE_AUDIT.md        # src/ migration audit
│   ├── langgraph_notes.md          # LangGraph design notes
│   └── langgraph_runner_architecture.md
├── notebooks/dev/          # Active development notebooks
├── results/                # Gitignored run outputs (JSONL, CSV, logs)
├── CLAUDE.md               # ← This file
└── environment.yml
```
