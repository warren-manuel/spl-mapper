# Pipeline Context: SPL Contraindication Mapping

## What This Project Does

LangGraph-based pipeline that extracts contraindications from SPL (Structured Product Label) records and maps them to SNOMED concepts. Entry point: `scripts/run_pipeline.py`.

---

## Architecture

### Two Nested LangGraph Graphs

**SPL-level graph** (one per SPL record):
```
resolve_contra_section → extract_items → [loop] prepare_item → process_item → advance_item → finalize
```

**Item-level graph** (one per extracted contraindication):
```
retrieve_candidates → direct_match → prefilter → route_or_fill → normalize → validate → assemble
```

### Three Distinct LLM Calls Per Item
1. **Extraction** — pull discrete contraindication items from section text
2. **Direct match** — binary decision: does this item map to a single concept as-is?
3. **Route/fill** — choose a focus concept + optional slot fills

### Retrieval Stack
```
BM25 (multi-tier: exact → phrase → shingle → token) ─┐
                                                       ├──▶ RRF ──▶ cross-encoder re-rank ──▶ top-K ──▶ LLM
FAISS (dense, PubMed sentence-transformer)            ─┘
```
- `build_snomed_query` constructs the BM25 query (exploits `preferredTerm.keyword` and shingle sub-fields)
- Cross-encoder (`RERANKER_MODEL_ID` env var, default off) reorders RRF hits before LLM sees them
- Each candidate is enriched with an IS-A ancestor path via `get_longest_ancestor_path` (BFS on `sct2_Relationship_Snapshot`)
- Candidate format to LLM: `{id} | Concept name: {label} | Score: {score} | Ancestor path: {path}`

### State Models
- `ContraState` — SPL-level: raw record, set ID, product name, contra text, extracted items, item index, results, errors
- `ItemState` — Item-level: set ID, extracted item, candidates, direct-match decision, route/fill decision, focus concept, normalized fills, validation result, assembled result

---

## Module Responsibilities

### `src/extraction/`
- **`section_parser.py`** — DailyMed XML fetching and SPL section parsing (contraindication LOINC `34070-3`); `extract_section()`, `fetch_spl_xml_by_setid()`

### `src/retrieval/`
- **`hybrid_mapper.py`** — BM25 + FAISS hybrid orchestration; `retrieve_candidates_for_item()`, `get_cached_mapper_resources()`, `MapperResources` dataclass; SNOMED ES index settings
- **`search_utils.py`** — BM25 query building (`build_snomed_query`), FAISS dense search, RRF fusion (`fuse_hits_rrf`), cross-encoder reranking, ancestor path enrichment (`get_longest_ancestor_path`)
- **`embedding_utils.py`** — Sentence-Transformer loading (`load_ST_model`), FAISS index build/save (`build_and_save_dense_index`), GPU migration (`maybe_move_index_to_gpu`)
- **`es_utils.py`** — Elasticsearch connection, index creation, bulk indexing, container start/stop
- **`dense_ann.py`** — FAISS dense ANN wrapper (`DenseANN`); internal utility

### `src/mapping/`
- **`postcoord.py`** — Post-coordination SNOMED mapping for items where `selected_snomed_id == "N/A"`; slot-based candidate assembly (focus + causative + severity + clinical_course); multi-GPU worker orchestration

### `src/llm/`
- **`backends.py`** — HuggingFace model loading (`load_model_local`, `LocalLLM`), JSON extraction (`extract_json`), message builders (`build_message`, `build_messages_from_iter`), end-token helpers
- **`prompts.py`** — System/user prompts for all three LLM calls: `CONTRA_EXTRACT_SYSTEM_PROMPT`, `DIRECT_VERIFY_SYSTEM_PROMPT`, `ROUTE_OR_FILL_SYSTEM_PROMPT`; builder functions `extract_contraindication_items()`, `build_direct_verify_user_prompt()`, `build_route_or_fill_user_prompt()`

### `src/snomed/`
- **`snomed_utils.py`** — SNOMED RF2 file loading (`load_snomed_dataframes`), ECL evaluation (`concept_matches_ecl`), ancestor traversal (`get_ancestors_with_depth`), MRCM constraint lookup (`get_range_constraints_for_attribute`), prefilter membership (`filter_terms_by_attribute_range`), `ATTRIBUTE_TABLE` constant

### `src/evaluation/`
- **`evaluator.py`** — Three-level evaluation (extraction / contraindication / concept) using Hungarian optimal assignment + embedding similarity + tiered SNOMED hierarchy scoring; also exports pipeline utilities: `AGG_CSV_COLUMNS`, `aggregate_agent_results`, `candidate_label_by_id`, `evaluate_aggregated_predictions`, `load_spl_records_from_file`, `parse_json_with_end_marker`, `validate_postcoord_with_mrcm`, `write_csv_rows`, `write_jsonl`

---

## Key Invariant

> **LLM proposes; ontology/rules validate.**

This separation is enforced by module boundaries:
- LLM calls live exclusively in `src/llm/` and are invoked by graph nodes in `scripts/run_pipeline.py`
- Ontology validation lives exclusively in `src/snomed/snomed_utils.py`
- The graph in `scripts/run_pipeline.py` orchestrates both but never mixes them within a single node

---

## Directory Layout

```
spl-mapper/
├── src/                    # Library packages — importable from anywhere
│   ├── extraction/         # SPL loading and section parsing
│   ├── retrieval/          # FAISS + BM25 + RRF + reranker
│   ├── mapping/            # LangGraph graph, nodes, state, postcoordination
│   ├── llm/                # LLM backends and prompt strings
│   ├── snomed/             # SNOMED dataframe loading and validation
│   └── evaluation/         # Metric computation and output utilities
├── scripts/                # CLI entry points (not library code)
│   ├── run_pipeline.py     # Main LangGraph runner (was langgraph_agent_runner.py)
│   ├── run_pipeline.sh     # Isolation test runner (was run_isolation_tests.sh)
│   ├── build_es_index.py
│   ├── build_retrieval_indexes.py
│   ├── map_verify.py
│   ├── prefilter.py
│   ├── result_agg.py
│   └── multi_gpu_contra_extract.py
├── agents/                 # Planned agentic modules (see agents/README.md)
├── notebooks/              # Exploratory notebooks
│   ├── dev/                # Active development notebooks
│   └── archive/            # Historical / archived notebooks
├── tests/                  # Test stubs (one per src/ module)
├── results/                # Gitignored output artifacts
│   ├── data/               # JSONL, CSV, PKL, BIN outputs
│   └── docs/               # Presentation and analysis documents
├── CLAUDE.md
├── RESTRUCTURE_AUDIT.md
├── RESTRUCTURE_PLAN.md
├── RESTRUCTURE_LOG.md
└── environment.yml
```

---

## Agents (Planned)

See [agents/README.md](agents/README.md) for full specifications of three planned agents:
1. **SNOMED Concept Navigator** — offline pre-computation of hierarchy conventions
2. **Compositional Extractor** — structured slot decomposition at extraction time
3. **Post-Coordination Expression Agent** — pattern-finding + MRCM validation

---

## Evaluation: 3 Levels

The goal standard is **not** a full SNOMED grammatical expression. Evaluation is:

| Level | Description | Strictness |
|---|---|---|
| **Extraction** | Did the model find all contraindications per gold standard? | Per item |
| **Contraindication** | Does the model correctly map to ALL concepts in the gold annotation? | All-or-nothing |
| **Concept** | Relaxed version of contraindication level — metrics at concept level | Partial credit |

### Concept Slots (per contraindication)
- `problem_concept`
- `causative_concept`
- `severity_concept`
- `clinical_course_concept`

A correct contraindication prediction must match **all** concepts present in the gold annotation.

---

## Current Metrics (baseline, pre-optimisation — `results/20260401-02/`)

| Model | Level | Precision | Recall | F1 |
|---|---|---|---|---|
| Qwen 3.5 | Extraction | 0.871 | 0.901 | 0.886 |
| Qwen 3.5 | Contraindication | 0.409 | 0.424 | 0.416 |
| Qwen 3.5 | Concept | 0.580 | 0.524 | 0.551 |
| MedGemma | Extraction | 0.844 | 0.860 | 0.852 |
| MedGemma | Contraindication | 0.356 | 0.363 | 0.360 |
| MedGemma | Concept | 0.552 | 0.439 | 0.489 |

### Diagnosed Failure Modes
- **Failure Mode 1 (42-49% of concept failures)**: Direct-match selects a precoordinated drug-condition concept (e.g. "Allergy to ibuprofen") when the gold uses postcoordination (hypersensitivity disorder + ibuprofen). Root cause: direct-match fires on items that already carry a causative slot; prompt accepts semantic subtypes.
- **Failure Mode 2 (40-45% of concept failures)**: Same extracted text, wrong concept ID — multiple near-synonymous SNOMED concepts ranked equally by RRF; LLM picks incorrectly without hierarchy context.
- Both models perform similarly → bottleneck is pipeline design and retrieval, not model capability
- Use **concept-level F1 as the optimization target**; contraindication-level as the acceptance criterion

---

## Optimization Roadmap

### Priority 1 — Post-Extraction Precision Filter
Add a filter node after extraction to classify each item as genuine contraindication vs warning/precaution/monitoring instruction. SPL sections mix these and the LLM over-extracts.

### Priority 2 — Slot Presence Detection
Replace the binary direct-match decision with a slot detection step. Only retrieve and fill candidates for detected slots. Avoids spurious fills on simple items.

### Priority 3 — Slot-Specific Targeted Retrieval
Run separate retrieval + re-ranking per slot type with slot-appropriate queries.

### Priority 4 — Re-ranker After RRF ✅
Cross-encoder reranker wired in (`src/retrieval/search_utils.py::rerank_candidates`). Enabled via `RERANKER_MODEL_ID` env var. Off by default; backward-compatible.

### Priority 5 — Implicit Slot Inference in Prompts
Severity and clinical course are often implicit in the text. Instruct the LLM to infer from linguistic cues.

### Priority 6 — Parallelize Item Processing
Items are independent after extraction. Use LangGraph `Send` API to fan out. Watch Azure OpenAI rate limits.

### Priority 7 — Prompt Hardening for SNOMED Accuracy ⚠️ Partially done
- ✅ Ancestor path added to all candidate displays (targets Failure Mode 2)
- ✅ `DIRECT_VERIFY_SYSTEM_PROMPT` tightened: explicit subtype rejection, "when in doubt return false" default
- ⬜ FSN (Fully Specified Name) not yet shown — preferred term only
- ⬜ Negative anchoring not yet implemented

---

## Agentic Approach: Decision

A **pure ReAct/tool-calling agent** was considered and rejected: SNOMED mapping has hard ontological constraints and agents are sequential by nature. The LangGraph approach is parallelizable and keeps the ontology in control.

**Recommended hybrid**: current LangGraph flow for straightforward items; ambiguous items route to a mini ReAct agent with access to search_concepts, get_slot_candidates, and validate_expression tools.

---

## LLM Backends Supported
- Azure OpenAI (chat completions) — `AzureChatLLM` in `scripts/run_pipeline.py`
- Hugging Face local model — `HuggingFaceChatLLM` / `LocalLLM` in `src/llm/backends.py`
