# RESTRUCTURE LOG
Executed: 2026-04-24

---

## Files Moved

### VaxMapper/src/ → src/ (moved, imports updated)

| Old path | New path | Import changes |
|---|---|---|
| `VaxMapper/src/llm.py` | `src/llm/backends.py` | None (no internal deps) |
| `VaxMapper/src/utils/llm_prompt.py` | `src/llm/prompts.py` | `VaxMapper.src.llm` → `src.llm.backends` |
| `VaxMapper/src/utils/snomed_utils.py` | `src/snomed/snomed_utils.py` | None |
| `VaxMapper/src/utils/hyb_mapper.py` | `src/retrieval/hybrid_mapper.py` | `VaxMapper.src.llm` → `src.llm.backends`; `VaxMapper.src.utils.*` → `src.retrieval.*` / `src.snomed.*` |
| `VaxMapper/src/utils/search_utils.py` | `src/retrieval/search_utils.py` | None |
| `VaxMapper/src/utils/embedding_utils.py` | `src/retrieval/embedding_utils.py` | None |
| `VaxMapper/src/utils/dense_ann.py` | `src/retrieval/dense_ann.py` | None |
| `VaxMapper/src/utils/elastisearch_utils.py` | `src/retrieval/es_utils.py` | None |
| `VaxMapper/src/utils/dailymed.py` | `src/extraction/section_parser.py` | None |

### Root scripts → scripts/ (moved, imports updated)

| Old path | New path |
|---|---|
| `langgraph_agent_runner.py` | `scripts/run_pipeline.py` |
| `run_isolation_tests.sh` | `scripts/run_pipeline.sh` |
| `build_es_index.py` | `scripts/build_es_index.py` |
| `build_retrieval_indexes.py` | `scripts/build_retrieval_indexes.py` |
| `map_verify.py` | `scripts/map_verify.py` |
| `prefilter.py` | `scripts/prefilter.py` |
| `result_agg.py` | `scripts/result_agg.py` |
| `rewrite_agg_csv.py` | `scripts/rewrite_agg_csv.py` |
| `multi_gpu_contra_extract.py` | `scripts/multi_gpu_contra_extract.py` |
| `001_llm.py` | `scripts/001_llm.py` |
| `002_llm.py` | `scripts/002_llm.py` |
| `hyb_mapper.py` | `scripts/hyb_mapper.py` |
| `_hyb_mapper.py` | `scripts/_hyb_mapper.py` |
| `elastic_es.py` | `scripts/elastic_es.py` |

### Notebooks → notebooks/

| Old path | New path |
|---|---|
| `VaxMapper.ipynb` | `notebooks/VaxMapper.ipynb` |
| `VaxMapper[Archive].ipynb` | `notebooks/archive/VaxMapper[Archive].ipynb` |
| `VaxMapper[Archive_2].ipynb` | `notebooks/archive/VaxMapper[Archive_2].ipynb` |
| `VaxMapper_hm.ipynb` | `notebooks/archive/VaxMapper_hm.ipynb` |
| `agent_runner_smoketest.ipynb` | `notebooks/dev/agent_runner_smoketest.ipynb` |
| `0326_langgraph_smoketest.ipynb` | `notebooks/dev/0326_langgraph_smoketest.ipynb` |
| `04_21_CI_Map_VO_Feasibility.ipynb` | `notebooks/dev/04_21_CI_Map_VO_Feasibility.ipynb` |
| `langgraph_debug.ipynb` | `notebooks/dev/langgraph_debug.ipynb` |
| `RxNormINDecomp.ipynb` | `notebooks/RxNormINDecomp.ipynb` |
| `SPL_IE.ipynb` | `notebooks/SPL_IE.ipynb` |
| `TAC17_ADR.ipynb` | `notebooks/TAC17_ADR.ipynb` |
| `umls_cfinder[Archive].ipynb` | `notebooks/archive/umls_cfinder[Archive].ipynb` |
| `umls_cfinder_.ipynb` | `notebooks/umls_cfinder_.ipynb` |
| `umls_vaccine_expansion_v2.ipynb` | `notebooks/umls_vaccine_expansion_v2.ipynb` |

### Data artifacts → results/data/ (gitignored)

`inference_results.jsonl`, `inference_resultsgoogle_medgemma-27b-text-it.jsonl`,
`inference_resultsgoogle_medgemma-27b-text-it_1.jsonl`, `vo_candidates.jsonl`,
`runtime_audit.jsonl`, `per_doc_eval.csv`, `per_doc_eval_1.csv`,
`rxnorm_faiss.bin`, `vo_faiss.bin`, `rxnav_data.pkl`, `vo_data.pkl`,
`nohup.out`, `*.log`

### Docs → results/docs/ (gitignored)

`presentation_content.md`, `evaluation_presentation_content.md`, `claude_code_instructions.md`

### Archived (not deleted — potential ablation use)

`agent_runner.py` → `results/archive/agent_runner.py`

---

## Files Consolidated

### `src/evaluation/evaluator.py`
Canonical source: `evaluate_agg_results_3.py`
Added from `agent_runner.py`: `AGG_CSV_COLUMNS`, `aggregate_agent_results`, `candidate_label_by_id`,
`evaluate_aggregated_predictions`, `load_spl_records_from_file`, `parse_json_with_end_marker`,
`validate_postcoord_with_mrcm`, `write_csv_rows`, `write_jsonl`
Dropped: `evaluate_agg_results.py`, `evaluate_agg_results_1.py`, `evaluate_agg_results_2.py`,
`evaluate_agg_results_2[copy].py` (exact duplicate of v2)

### `src/mapping/postcoord.py`
Canonical source: `postcord_v2.py`
Dropped: `postcord.py` (superseded), `postcord1.py` (earliest iteration; subset)

---

## Files Deleted

| File | Reason |
|---|---|
| `_langchain_agent_runner[DEP].py` | Explicitly deprecated; not imported anywhere |
| `evaluate_agg_results_2[copy].py` | Exact duplicate of `evaluate_agg_results_2.py` |
| `Untitled-1.mmd` | Draft diagram with no identity; not referenced |

---

## Files Created (new)

| File | Purpose |
|---|---|
| `src/__init__.py` and all package `__init__.py` files | Package markers |
| `src/mapping/postcoord.py` | Consolidated post-coordination module |
| `src/evaluation/evaluator.py` | Consolidated evaluator + pipeline utilities |
| `agents/README.md` | Three planned agents documented with interface contracts |
| `tests/test_extraction.py` | Stub |
| `tests/test_retrieval.py` | Stub |
| `tests/test_mapping.py` | Stub |
| `tests/test_llm.py` | Stub (includes one functional test for `build_message`) |
| `tests/test_snomed.py` | Stub |
| `tests/test_evaluation.py` | Stub (includes functional tests for `compute_metrics` and `AGG_CSV_COLUMNS`) |
| `RESTRUCTURE_AUDIT.md` | Full codebase audit |
| `RESTRUCTURE_PLAN.md` | Target structure definition |
| `RESTRUCTURE_LOG.md` | This file |

---

## High-Level Git Diff Summary

- **~41 Python source files** reorganized from flat root + `VaxMapper/src/` into `src/` package hierarchy
- **14 notebooks** moved to `notebooks/` with `dev/` and `archive/` subdirs
- **13+ data artifacts** (JSONL, CSV, BIN, PKL, LOG) moved to `results/data/` (gitignored)
- **3 deprecated files** deleted from git tracking
- **2 consolidations**: 5 evaluator versions → 1; 3 postcord versions → 1
- `.gitignore` expanded to cover `results/`, `**/*.bin`, `**/*.pkl`, `**/*.log`, `*.jsonl`
- `environment.yml` updated: added `scipy`, `networkx`, `lxml`, `docker`, `openai`, `langgraph`, `python-dotenv`, `requests`, `owlready2` to pip section
- `CLAUDE.md` rewritten to reflect new module layout
- Original source files in `VaxMapper/src/` and `langgraph_agent_runner.py` at root **remain in place** (not deleted) — the new `src/` layout is additive; old files can be removed after the new layout is validated on a full pipeline run
