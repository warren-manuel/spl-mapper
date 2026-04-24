# RESTRUCTURE AUDIT
Generated: 2026-04-24

---

## Source of Truth

**CLAUDE.md** establishes the canonical architecture:
- Two nested LangGraph graphs: SPL-level → Item-level
- Three LLM calls per item: extraction, direct-match, route/fill
- Retrieval stack: BM25 (multi-tier) + FAISS → RRF → cross-encoder → top-K → LLM
- Key invariant: **LLM proposes; ontology/rules validate**
- Active baseline results in `results/20260401-02/`

---

## Root-Level Python Files (25 files)

### KEEP_CORE — Active Pipeline

| File | Lines | Imported By | Role |
|---|---|---|---|
| `langgraph_agent_runner.py` | 1338 | `run_isolation_tests.sh`, `rewrite_agg_csv.py` | **Primary orchestrator**: SPL-level + item-level LangGraph graphs; three LLM call sites; all graph nodes defined here |
| `agent_runner.py` | 832 | `langgraph_agent_runner.py`, `rewrite_agg_csv.py` | Predecessor non-LangGraph runner; **currently acts as utility library** — exports `AGG_CSV_COLUMNS`, `aggregate_agent_results`, `candidate_label_by_id`, `evaluate_aggregated_predictions`, `load_spl_records_from_file`, `parse_json_with_end_marker`, `validate_postcoord_with_mrcm`, `write_csv_rows`, `write_jsonl` which are imported by langgraph_agent_runner.py |

### KEEP_CORE — Retrieval Entry Points

| File | Lines | Imported By | Role |
|---|---|---|---|
| `hyb_mapper.py` | 141 | none (shell entry point) | Wrapper: starts Elasticsearch, runs mapping, shuts down; calls `VaxMapper.src.utils.hyb_mapper` |
| `_hyb_mapper.py` | 411 | none | Standalone BM25+FAISS+RRF index builder and mapper; independent of LangGraph pipeline |

### KEEP_SCRIPT — Build / Infrastructure

| File | Lines | Imported By | Role |
|---|---|---|---|
| `build_retrieval_indexes.py` | 208 | none | CLI: loads SNOMED dataframes, builds ES + FAISS indexes, maps extracted items |
| `build_es_index.py` | 102 | none | CLI: rebuilds ES SNOMED index only; supports `--rebuild` and `--bm25-b` tuning |
| `elastic_es.py` | 37 | none | Docker container launcher for local Elasticsearch (dev only) |

### KEEP_SCRIPT — Candidate Validation & Post-Coordination

| File | Lines | Imported By | Role |
|---|---|---|---|
| `map_verify.py` | 383 | none | Multi-GPU SNOMED direct-match verifier; outputs `verified_hits.jsonl` |
| `postcord.py` | 688 | none | Post-coordinate mapper for items where `selected_snomed_id == "N/A"`; slot-based candidate assembly |
| `postcord_v2.py` | 727 | none | Enhanced postcord: adds ECL validation and tighter candidate filtering; appears most complete |
| `postcord1.py` | 488 | none | Earliest postcord iteration; subset of postcord.py functionality |
| `prefilter.py` | 178 | none | Precomputes ECL range-membership cache (`prefilter_cache.json`) for postcord filters |
| `result_agg.py` | 244 | none | Aggregates `verified_hits.jsonl` + `postcoord_hits.jsonl` by `SPL_SET_ID`; builds postcoord expressions; writes CSV/JSON |
| `rewrite_agg_csv.py` | 80 | none | One-time recovery: reconstructs `aggregated_hits.csv` from `postcoord_expression`; imports from `agent_runner` |

### KEEP_SCRIPT — Multi-GPU Extraction

| File | Lines | Imported By | Role |
|---|---|---|---|
| `001_llm.py` | 85 | none | Minimal extraction (IE only) → `.pkl`; loads MedGemma 27B |
| `002_llm.py` | 102 | none | Extended `001_llm.py`; adds `flatten` + `add_hits_json` post-processing |
| `multi_gpu_contra_extract.py` | 495 | none | 4-GPU parallel extraction orchestrator; splits input across GPUs; outputs `out_gpu{0-3}.jsonl` |

### KEEP_EVAL — Evaluation Suite

| File | Lines | Imported By | What changed from previous version |
|---|---|---|---|
| `evaluate_agg_results.py` | 429 | none | Baseline: concept-level exact match, attribute-level ID comparison, no embeddings |
| `evaluate_agg_results_1.py` | 748 | none | Added sentence-transformer semantic similarity scoring |
| `evaluate_agg_results_2.py` | 655 | none | Replaced greedy matching with `scipy.optimize.linear_sum_assignment` (optimal assignment) |
| `evaluate_agg_results_2[copy].py` | 565 | none | Duplicate of v2; no functional differences found |
| `evaluate_agg_results_3.py` | 858 | none | Most complete: combines optimal assignment + embedding similarity + attribute-level matching; uses `deque` for BFS |
| `semantic_eval.py` | 287 | none | Standalone semantic matching evaluation via sentence-transformers |
| `lexical_eval.py` | 240 | none | Standalone lexical/canonical mention matching at document level |

### DEPRECATED

| File | Lines | Imported By | Reason |
|---|---|---|---|
| `_langchain_agent_runner[DEP].py` | 949 | none | Predecessor LangChain runner; explicitly marked `[DEP]`; not imported by anything |

---

## VaxMapper/ Module (16 Python files)

### KEEP_CORE — LLM Abstraction

| File | Lines | Imported By | Role |
|---|---|---|---|
| `VaxMapper/src/llm.py` | 654 | `001_llm`, `002_llm`, `_hyb_mapper`, `lexical_eval`, `map_verify`, `multi_gpu_contra_extract`, `postcord*`, `semantic_eval` | Central LLM abstraction: HuggingFace model loading, message building, JSON extraction, `has_end_json_token`, `trim_after_end_json_token` |
| `VaxMapper/src/llm_runner.py` | 362 | none visible | Async/generator-based LLM runner protocol; dataclass-driven config; streaming generation support |

### KEEP_CORE — Prompts

| File | Lines | Imported By | Notes |
|---|---|---|---|
| `VaxMapper/src/utils/llm_prompt.py` | 508 | `langgraph_agent_runner` | Current prompts: `CONTRA_EXTRACT_SYSTEM_PROMPT`, `DIRECT_VERIFY_SYSTEM_PROMPT`, `ROUTE_OR_FILL_SYSTEM_PROMPT`; functions `extract_contraindication_items()`, `build_direct_verify_user_prompt()`, `build_route_or_fill_user_prompt()` |
| `VaxMapper/src/utils/_llm_prompt.py` | 418 | `agent_runner` | Earlier prompt version; imported by agent_runner.py (the non-LangGraph runner); likely superseded by `llm_prompt.py` — keep until agent_runner.py utilities are migrated |

### KEEP_CORE — SNOMED

| File | Lines | Imported By | Role |
|---|---|---|---|
| `VaxMapper/src/utils/snomed_utils.py` | 879 | `agent_runner`, `langgraph_agent_runner`, `build_es_index`, `build_retrieval_indexes`, `evaluate_agg_results`, `postcord*`, `prefilter`, `result_agg` | SNOMED RF2 loading, ECL evaluation, ancestor traversal, MRCM validation, `ATTRIBUTE_TABLE` constant |

### KEEP_CORE — Retrieval

| File | Lines | Imported By | Role |
|---|---|---|---|
| `VaxMapper/src/utils/hyb_mapper.py` | 506 | `agent_runner`, `langgraph_agent_runner`, `hyb_mapper.py`, `build_es_index`, `build_retrieval_indexes` | Hybrid BM25+FAISS orchestration: `retrieve_candidates_for_item()`, `build_or_load_faiss_index()`, `build_es_index()`, `map_terms()` |
| `VaxMapper/src/utils/search_utils.py` | 646 | `_hyb_mapper` | BM25 query building, FAISS dense search, RRF combination, cross-encoder reranking |
| `VaxMapper/src/utils/elastisearch_utils.py` | 179 | `hyb_mapper.py`, `_hyb_mapper`, `build_es_index`, `build_retrieval_indexes` | ES connection, index creation, bulk indexing, container start/stop |
| `VaxMapper/src/utils/embedding_utils.py` | 202 | `evaluate_agg_results*`, `_hyb_mapper`, `build_retrieval_indexes` | Sentence-Transformer loading, FAISS index build/save/GPU-move |
| `VaxMapper/src/utils/dense_ann.py` | 100 | none visible | FAISS dense ANN wrapper; dataclass + search interface; internal module |

### KEEP_CORE — Extraction

| File | Lines | Imported By | Role |
|---|---|---|---|
| `VaxMapper/src/utils/dailymed.py` | 186 | `agent_runner`, `langgraph_agent_runner` | DailyMed XML parsing; extracts contraindication (`CONTRA_Loinc=34070-3`) and adverse (`ADVERSE_Loinc=34084-4`) sections |

### KEEP_CORE — Ancillary

| File | Lines | Imported By | Role |
|---|---|---|---|
| `VaxMapper/src/rxnorm_term_getter.py` | 210 | none visible | RxNorm data retrieval via owlready2 + REST API; data prep utility |
| `VaxMapper/src/utils/helpers.py` | 85 | none visible | Generic pandas/requests/owlready2 helpers; not imported by active pipeline |
| `VaxMapper/main.py` | 24 | none | Package entry point wrapper; minimal |
| `VaxMapper/src/__init__.py` | 0 | — | Package marker |
| `VaxMapper/src/utils/__init__.py` | 0 | — | Package marker |

---

## Non-Python Files

### Notebooks (14 .ipynb files)

| File | Classification | Destination |
|---|---|---|
| `VaxMapper.ipynb` | NOTEBOOK | `notebooks/` |
| `VaxMapper[Archive].ipynb` | NOTEBOOK | `notebooks/archive/` |
| `VaxMapper[Archive_2].ipynb` | NOTEBOOK | `notebooks/archive/` |
| `VaxMapper_hm.ipynb` | NOTEBOOK | `notebooks/archive/` |
| `agent_runner_smoketest.ipynb` | NOTEBOOK | `notebooks/dev/` |
| `0326_langgraph_smoketest.ipynb` | NOTEBOOK | `notebooks/dev/` |
| `04_21_CI_Map_VO_Feasibility.ipynb` | NOTEBOOK | `notebooks/dev/` |
| `langgraph_debug.ipynb` | NOTEBOOK | `notebooks/dev/` |
| `RxNormINDecomp.ipynb` | NOTEBOOK | `notebooks/` |
| `SPL_IE.ipynb` | NOTEBOOK | `notebooks/` |
| `TAC17_ADR.ipynb` | NOTEBOOK | `notebooks/` |
| `umls_cfinder[Archive].ipynb` | NOTEBOOK | `notebooks/archive/` |
| `umls_cfinder_.ipynb` | NOTEBOOK | `notebooks/` |
| `umls_vaccine_expansion_v2.ipynb` | NOTEBOOK | `notebooks/` |

### Data Artifacts (move to `results/data/`, gitignore)

| File | Size | Classification |
|---|---|---|
| `inference_results.jsonl` | 1.5 MB | DATA_ARTIFACT |
| `inference_resultsgoogle_medgemma-27b-text-it.jsonl` | 1.7 MB | DATA_ARTIFACT |
| `inference_resultsgoogle_medgemma-27b-text-it_1.jsonl` | 80 KB | DATA_ARTIFACT |
| `vo_candidates.jsonl` | 19 MB | DATA_ARTIFACT |
| `runtime_audit.jsonl` | 152 KB | DATA_ARTIFACT |
| `per_doc_eval.csv` | 95 KB | DATA_ARTIFACT |
| `per_doc_eval_1.csv` | 97 KB | DATA_ARTIFACT |
| `rxnav_data.pkl` | 385 KB | DATA_ARTIFACT |
| `vo_data.pkl` | 488 KB | DATA_ARTIFACT |
| `rxnorm_faiss.bin` | 3.1 MB | DATA_ARTIFACT |
| `vo_faiss.bin` | 9.6 MB | DATA_ARTIFACT |
| `multi_gpu_contra_extract_2026-02-26_0929.log` | 2.8 KB | DATA_ARTIFACT |
| `nohup.out` | 37 KB | DATA_ARTIFACT |

### Diagrams (.mmd)

| File | Action |
|---|---|
| `langgraph_graph.mmd` | Keep in repo root (architecture reference) |
| `workflow_end_to_end.mmd` | Keep in repo root |
| `flowchart LR.mmd` | Keep in repo root |
| `Untitled-1.mmd` | Delete (draft with no content identity) |

### Markdown Docs

| File | Action |
|---|---|
| `langgraph_notes.md` | Keep (design rationale) |
| `langgraph_runner_architecture.md` | Keep (detailed architecture) |
| `presentation_content.md` | Move to `results/docs/` |
| `evaluation_presentation_content.md` | Move to `results/docs/` |
| `claude_code_instructions.md` | Move to `results/docs/` |

---

## environment.yml — Package Audit

Declared dependencies (selected):
- `python=3.10`, `pytorch=2.7.1`, `faiss-gpu`, `sentence-transformers`, `transformers`
- `jupyter=4.x`, `ollama-python`, `elasticsearch-py` (pip)
- `owlready2`, `scipy`, `numpy`, `pandas`

Packages referenced in source but **not in environment.yml**:
- `docker` (used in `elastic_es.py`) — add `docker-py`
- `langchain-core`, `langchain-openai` — only used in deprecated `_langchain_agent_runner[DEP].py`; do not add

Packages in environment.yml potentially unused after restructure:
- `ollama-python` — not found in any active `.py` import; verify before removing

---

## .gitignore — Current State vs Required

Currently excludes: `.env`, `.mmd`, `__pycache__`, `.claude/`, `.ipynb_checkpoints/`, data dirs, test outputs

Missing exclusions needed:
- `results/` (all output artifacts — dir exists but not gitignored)
- `**/*.bin` (FAISS indexes)
- `**/*.pkl` (cached data)
- `**/*.log`
- `nohup.out`
- `*.jsonl` (result files)
- `**/__pycache__/` (currently only root-level)

---

## Key Findings for Restructure

1. **`agent_runner.py` is a split module**: Half its code is the non-LangGraph pipeline (can be archived); the other half is utility functions actively imported by `langgraph_agent_runner.py`. These utilities must be extracted before `agent_runner.py` is deleted.

2. **`evaluate_agg_results_3.py` is the canonical evaluator**: It is the most complete version (858 lines) combining optimal assignment, embedding similarity, and attribute-level matching. `evaluate_agg_results_2[copy].py` is a duplicate of v2 and can be dropped.

3. **`postcord_v2.py` is the canonical postcoord**: Most complete (727 lines); adds ECL validation over `postcord.py`. `postcord1.py` is the earliest iteration and is a subset.

4. **Two prompt modules**: `llm_prompt.py` is current (used by `langgraph_agent_runner.py`); `_llm_prompt.py` is legacy (used only by `agent_runner.py`). After migrating agent_runner utilities, `_llm_prompt.py` can be removed.

5. **No `.py` files should remain in `VaxMapper/`** after restructure — all source moves to `src/`. Non-Python assets in `VaxMapper/` (requirements.txt, LICENSE, README.md) should remain or be moved to repo root.
