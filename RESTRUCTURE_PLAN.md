# RESTRUCTURE PLAN
Generated: 2026-04-24

---

## Target Directory Layout

```
VaxMapperRepo/
├── src/
│   ├── __init__.py
│   ├── extraction/
│   │   ├── __init__.py
│   │   └── section_parser.py        ← VaxMapper/src/utils/dailymed.py
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── hybrid_mapper.py         ← VaxMapper/src/utils/hyb_mapper.py
│   │   ├── search_utils.py          ← VaxMapper/src/utils/search_utils.py
│   │   ├── embedding_utils.py       ← VaxMapper/src/utils/embedding_utils.py
│   │   ├── dense_ann.py             ← VaxMapper/src/utils/dense_ann.py
│   │   └── es_utils.py              ← VaxMapper/src/utils/elastisearch_utils.py
│   ├── mapping/
│   │   ├── __init__.py
│   │   ├── graph.py                 ← graph definition extracted from langgraph_agent_runner.py
│   │   ├── nodes.py                 ← node functions extracted from langgraph_agent_runner.py
│   │   ├── state.py                 ← ContraState + ItemState TypedDicts
│   │   └── postcoord.py             ← consolidated from postcord.py + postcord_v2.py (postcord_v2 is canonical)
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── backends.py              ← VaxMapper/src/llm.py + VaxMapper/src/llm_runner.py
│   │   └── prompts.py               ← VaxMapper/src/utils/llm_prompt.py
│   ├── snomed/
│   │   ├── __init__.py
│   │   └── snomed_utils.py          ← VaxMapper/src/utils/snomed_utils.py
│   └── evaluation/
│       ├── __init__.py
│       └── evaluator.py             ← consolidated from evaluate_agg_results*.py (v3 is canonical)
│
├── scripts/
│   ├── run_pipeline.py              ← langgraph_agent_runner.py (renamed; CLI entry point)
│   ├── run_pipeline.sh              ← run_isolation_tests.sh (renamed)
│   ├── build_es_index.py            ← build_es_index.py
│   ├── build_retrieval_indexes.py   ← build_retrieval_indexes.py
│   ├── elastic_es.py                ← elastic_es.py
│   ├── map_verify.py                ← map_verify.py
│   ├── prefilter.py                 ← prefilter.py
│   ├── result_agg.py                ← result_agg.py
│   ├── rewrite_agg_csv.py           ← rewrite_agg_csv.py
│   ├── multi_gpu_contra_extract.py  ← multi_gpu_contra_extract.py
│   ├── 001_llm.py                   ← 001_llm.py
│   └── 002_llm.py                   ← 002_llm.py
│
├── agents/
│   └── README.md                    ← three planned agents documented
│
├── notebooks/
│   ├── VaxMapper.ipynb
│   ├── RxNormINDecomp.ipynb
│   ├── SPL_IE.ipynb
│   ├── TAC17_ADR.ipynb
│   ├── umls_cfinder_.ipynb
│   ├── umls_vaccine_expansion_v2.ipynb
│   ├── dev/
│   │   ├── agent_runner_smoketest.ipynb
│   │   ├── 0326_langgraph_smoketest.ipynb
│   │   ├── 04_21_CI_Map_VO_Feasibility.ipynb
│   │   └── langgraph_debug.ipynb
│   └── archive/
│       ├── VaxMapper[Archive].ipynb
│       ├── VaxMapper[Archive_2].ipynb
│       ├── VaxMapper_hm.ipynb
│       └── umls_cfinder[Archive].ipynb
│
├── tests/
│   ├── test_extraction.py           ← stub
│   ├── test_retrieval.py            ← stub
│   ├── test_mapping.py              ← stub
│   ├── test_llm.py                  ← stub
│   ├── test_snomed.py               ← stub
│   └── test_evaluation.py           ← stub
│
├── results/                         ← gitignored; all output artifacts
│   ├── data/                        ← JSONL, CSV, PKL, BIN artifacts moved here
│   └── docs/                        ← presentation docs moved here
│
├── CLAUDE.md                        ← updated to reflect new structure
├── RESTRUCTURE_AUDIT.md
├── RESTRUCTURE_PLAN.md
├── RESTRUCTURE_LOG.md
├── environment.yml
├── .gitignore                       ← updated
├── langgraph_notes.md
├── langgraph_runner_architecture.md
├── langgraph_graph.mmd
├── workflow_end_to_end.mmd
└── flowchart LR.mmd
```

---

## Module Consolidation Decisions

### `src/evaluation/evaluator.py` — from 5 versions

| Source | Status | What it added |
|---|---|---|
| `evaluate_agg_results.py` | Superseded | Baseline exact match |
| `evaluate_agg_results_1.py` | Superseded | Added embedding similarity |
| `evaluate_agg_results_2.py` | Superseded | Added optimal assignment (linear_sum_assignment) |
| `evaluate_agg_results_2[copy].py` | **Drop** (exact duplicate of v2) | Nothing |
| `evaluate_agg_results_3.py` | **Canonical** | Combines all of the above + BFS traversal with deque |

**Decision**: `evaluate_agg_results_3.py` is the canonical version. The consolidated `evaluator.py` is v3 with a header comment block documenting what was dropped from earlier versions.

`semantic_eval.py` and `lexical_eval.py` are standalone evaluation utilities preserved as separate files in `src/evaluation/`.

### `src/mapping/postcoord.py` — from 3 versions

| Source | Status | What it added |
|---|---|---|
| `postcord1.py` | Superseded | Earliest iteration; subset of postcord.py |
| `postcord.py` | Superseded | Slot-based candidate assembly; full pipeline |
| `postcord_v2.py` | **Canonical** | Adds ECL validation + tighter candidate filtering |

**Decision**: `postcord_v2.py` is canonical. Consolidated as `src/mapping/postcoord.py` with a header comment block.

### `agent_runner.py` — utility function extraction

These functions are imported by `langgraph_agent_runner.py` and must be migrated:

| Function | Target module |
|---|---|
| `AGG_CSV_COLUMNS` | `src/evaluation/evaluator.py` |
| `aggregate_agent_results()` | `src/evaluation/evaluator.py` |
| `evaluate_aggregated_predictions()` | `src/evaluation/evaluator.py` |
| `write_csv_rows()` | `src/evaluation/evaluator.py` |
| `write_jsonl()` | `src/evaluation/evaluator.py` |
| `candidate_label_by_id()` | `src/retrieval/hybrid_mapper.py` |
| `load_spl_records_from_file()` | `src/extraction/section_parser.py` |
| `parse_json_with_end_marker()` | `src/llm/backends.py` |
| `validate_postcoord_with_mrcm()` | `src/snomed/snomed_utils.py` |

After extraction, `agent_runner.py` itself is **archived to `results/archive/`** (not deleted — it contains the non-LangGraph pipeline which may be useful for ablation experiments).

### `_hyb_mapper.py` and `hyb_mapper.py` (root-level wrappers)

Both are standalone entry-point scripts that call into `VaxMapper/src/utils/hyb_mapper.py`. After that module moves to `src/retrieval/hybrid_mapper.py`, these scripts move to `scripts/` and have their imports updated.

---

## Import Rewrite Map

After restructure, all imports follow this mapping:

| Old import | New import |
|---|---|
| `from VaxMapper.src.utils.dailymed import ...` | `from src.extraction.section_parser import ...` |
| `from VaxMapper.src.utils.hyb_mapper import ...` | `from src.retrieval.hybrid_mapper import ...` |
| `from VaxMapper.src.utils.search_utils import ...` | `from src.retrieval.search_utils import ...` |
| `from VaxMapper.src.utils.embedding_utils import ...` | `from src.retrieval.embedding_utils import ...` |
| `from VaxMapper.src.utils.elastisearch_utils import ...` | `from src.retrieval.es_utils import ...` |
| `from VaxMapper.src.utils.snomed_utils import ...` | `from src.snomed.snomed_utils import ...` |
| `from VaxMapper.src.utils.llm_prompt import ...` | `from src.llm.prompts import ...` |
| `from VaxMapper.src.llm import ...` | `from src.llm.backends import ...` |
| `from VaxMapper.src.llm_runner import ...` | `from src.llm.backends import ...` |
| `from agent_runner import ...` | `from src.evaluation.evaluator import ...` (most); others per table above |

---

## Key Invariant Preservation

The restructure enforces the pipeline's architectural boundary physically:

- **`src/llm/`** — all LLM calls live here. No ontology logic.
- **`src/snomed/`** — all ontology validation lives here. No LLM calls.
- **`src/mapping/`** — LangGraph graph orchestrates the two; it calls into both but does not mix them within a single node.
- **`agents/`** — future agents call into `src/` modules; they do not bypass ontology validation.

---

## Files to Delete (from git tracking)

| File | Reason |
|---|---|
| `_langchain_agent_runner[DEP].py` | Explicitly deprecated; not imported anywhere |
| `evaluate_agg_results_2[copy].py` | Exact duplicate of `evaluate_agg_results_2.py` |
| `Untitled-1.mmd` | Draft with no identity; not referenced anywhere |

---

## Tests Stub Plan

Each stub creates the test file with the module import and one `TODO` test:

| Stub file | Tests |
|---|---|
| `tests/test_extraction.py` | `test_extract_section_returns_text` |
| `tests/test_retrieval.py` | `test_retrieve_candidates_returns_list` |
| `tests/test_mapping.py` | `test_graph_compiles` |
| `tests/test_llm.py` | `test_build_messages_returns_list` |
| `tests/test_snomed.py` | `test_load_snomed_dataframes_returns_dict` |
| `tests/test_evaluation.py` | `test_evaluate_returns_metrics_dict` |
