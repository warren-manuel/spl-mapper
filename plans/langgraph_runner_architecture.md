# LangGraph Runner Architecture

## Purpose

`langgraph_agent_runner.py` is the current orchestration entrypoint for the contraindication extraction and mapping pipeline. Its goal is to preserve the behavior of the earlier `agent_runner.py` flow while making the runtime control flow explicit through LangGraph.

At a high level, the runner:

1. Loads SPL inputs from a file.
2. Resolves the contraindications section text for each SPL.
3. Uses an LLM to extract atomic contraindication items.
4. Processes each item through retrieval, direct-match verification, optional postcoordination, and validation.
5. Writes raw per-SPL outputs and aggregated outputs.
6. Optionally evaluates the aggregated predictions against a gold file.
7. Optionally writes runtime audit events and progress output.

This file is not only a graph definition. It also carries runtime bootstrap logic, backend setup, audit/progress instrumentation, retrieval wiring, and ontology prefilter cache management.

## Relationship to `agent_runner.py`

`langgraph_agent_runner.py` is not a full replacement for `agent_runner.py`. It reuses a substantial amount of functionality from it:

- input loading
- JSON parsing helpers
- aggregation logic
- CSV / JSONL writing
- postcoordination validation
- evaluation wrapper

The split today is:

- `langgraph_agent_runner.py`: orchestration, graph structure, runtime observer, LLM backend abstraction, node logic
- `agent_runner.py`: shared helper functions and the older non-LangGraph batch runner

This means the LangGraph runner depends on `agent_runner.py` as both a helper module and as a source of shared data contracts.

## High-Level Execution Flow

The end-to-end execution path is:

1. Script bootstrap
   - `.env` is loaded early.
   - CUDA visibility is set before importing components that may initialize Torch / sentence-transformers / FAISS GPU state.

2. CLI parsing
   - The runner reads input/output paths, optional gold/evaluation paths, audit options, and the backend selection.

3. LLM backend construction
   - A backend-specific chat wrapper is created for either Azure OpenAI or local Hugging Face generation.

4. Agent construction
   - `ContraLangGraphAgent` builds two graphs:
     - an SPL-level graph
     - an item-level graph

5. SPL input loading
   - SPL records are loaded via `agent_runner.load_spl_records_from_file`.

6. SPL processing
   - For each SPL record, the SPL-level graph:
     - resolves section text
     - extracts items
     - loops through extracted items
     - invokes the item-level graph for each item
     - assembles one final SPL result object

7. Aggregation
   - All item-level results across SPLs are flattened into:
     - aggregated JSONL by SPL
     - aggregated CSV by item

8. Evaluation
   - If `--gold-csv` is provided, the aggregated CSV is evaluated through `agent_runner.evaluate_aggregated_predictions`, which wraps `evaluate_agg_results_2.py`.

9. Observability
   - Progress is printed while processing.
   - Audit JSONL events are written if enabled.

## Core Components

### 1. `langgraph_agent_runner.py`: Orchestration Layer

This file owns the top-level orchestration behavior.

Responsibilities:

- early environment bootstrap, especially CUDA visibility
- LLM backend abstraction
- graph state definitions
- SPL-level graph construction
- item-level graph construction
- node instrumentation via `RunObserver`
- retrieval and prefilter wiring
- final batch run loop

Key structures:

- `ChatLLM`: minimal chat protocol expected by the runner
- `AzureLLMConfig` / `AzureChatLLM`
- `HuggingFaceLLMConfig` / `HuggingFaceChatLLM`
- `AgentRunConfig`
- `ItemState`
- `ContraState`
- `RunObserver`
- `ContraLangGraphAgent`

Important implementation detail:

- The file duplicates some helper behavior that also exists conceptually in `agent_runner.py`, especially around retrieval/prefilter wiring and config reading. That is a meaningful coupling point for future restructuring.

### 2. LLM Backend Layer

The runner supports two backends behind one `chat(...)` interface.

#### Azure path

The Azure wrapper uses `openai.AzureOpenAI` and environment-based config:

- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_DEPLOYMENT`
- `AZURE_OPENAI_API_VERSION`

Its job is only transport and response extraction. It does not know anything about contraindications.

#### Hugging Face path

The Hugging Face wrapper:

- resolves CUDA visibility before model load
- imports `VaxMapper.src.llm.load_model_local`
- loads a local chat model using environment-driven generation and loading settings

The backend-specific configuration includes model ID, device map, quantization flags, tokenizer flags, max token defaults, and decoding parameters.

Both backends feed the same prompt builders and graph logic.

### 3. SPL-Level Graph

The SPL-level graph processes one SPL record at a time.

Nodes:

- `resolve_contra_section`
- `extract_items`
- `prepare_item`
- `process_item`
- `advance_item`
- `finalize`

Behavior:

- `resolve_contra_section`
  - uses already-present section text from the record if available
  - otherwise calls `VaxMapper.src.utils.dailymed.extract_section`
- `extract_items`
  - calls `extract_contraindication_items(...)` from `VaxMapper.src.utils.llm_prompt`
  - assigns `item_index` to each extracted item
- `prepare_item`
  - moves the current item into SPL state
- `process_item`
  - invokes the item-level graph
- `advance_item`
  - increments the loop index
- `finalize`
  - assembles the final SPL result object

The SPL graph is effectively:

`resolve -> extract -> loop(item prepare/process/advance) -> finalize`

### 4. Item-Level Graph

The item-level graph processes one extracted contraindication item.

Nodes:

- `retrieve_candidates`
- `direct_match`
- `prefilter`
- `route_or_fill`
- `normalize`
- `validate`
- `assemble_direct`
- `assemble_postcoord`

Behavior:

- `retrieve_candidates`
  - runs hybrid SNOMED candidate retrieval
- `direct_match`
  - asks the LLM whether a direct semantic identity match exists among focus/direct candidates
- `prefilter`
  - filters slot candidates using SNOMED attribute-range logic
- `route_or_fill`
  - asks the LLM to choose a focus concept and optional fills
- `normalize`
  - normalizes IDs and resolves candidate labels
- `validate`
  - validates the selected focus/fills against MRCM rules
- `assemble_direct`
  - emits a `DIRECT` item result
- `assemble_postcoord`
  - emits either `POSTCOORD` or `MINIMAL`

The key control-flow branch is after `direct_match`:

- if the LLM confirms an acceptable direct match, the graph exits through `assemble_direct`
- otherwise it continues through prefilter, route-or-fill, normalize, validate, and assemble-postcoord

### 5. Prompt / Extraction Layer

`VaxMapper/src/utils/llm_prompt.py` provides the structured prompt layer used by the runner.

Responsibilities:

- contraindication extraction prompt
- direct-match verification prompt
- route-or-fill prompt
- helper functions to format candidate blocks and produce user prompts
- helper to parse structured JSON returns with the end token convention

The pipeline uses three separate LLM tasks instead of one prompt:

1. extract atomic contraindication items
2. verify a strict direct mapping candidate
3. choose a focus concept and optional fills

This is a strong architectural boundary: the prompts define task semantics, while the runner manages control flow and state.

### 6. Retrieval / Mapper Layer

`VaxMapper/src/utils/hyb_mapper.py` is the retrieval subsystem used by the item graph.

Responsibilities:

- build/load Elasticsearch and FAISS retrieval resources
- cache those resources across calls
- map extracted item text fields to candidate lists
- expose a single `retrieve_candidates_for_item(...)` call used by the runner

Important pieces:

- `MapperResources`
  - bundles the sentence-transformer model, FAISS index, concept metadata, ES client, and retrieval parameters
- `_RESOURCE_CACHE`
  - caches mapper resources by retrieval configuration
- `load_mapper_resources(...)`
  - loads SNOMED frames, builds/loads ES and FAISS assets, and creates a `MapperResources` object
- `get_cached_mapper_resources(...)`
  - memoized access to retrieval resources
- `map_item_terms(...)`
  - calls `search_query(...)` for each item term key
- `retrieve_candidates_for_item(...)`
  - transforms mapped term results into:
    - `focus_candidates`
    - `direct_candidates`
    - `causative_agent_candidates`
    - `severity_candidates`
    - `clinical_course_candidates`

The LangGraph runner treats retrieval as a black box with one input item and one structured candidate payload.

### 7. Search Layer

`VaxMapper/src/utils/search_utils.py` provides the actual hybrid retrieval operations.

Responsibilities:

- encode a query with the sentence-transformer model
- run dense search against a FAISS index
- run BM25 search against Elasticsearch
- fuse the two result lists with reciprocal rank fusion

The main helper is `search_query(...)`.

Flow inside `search_query(...)`:

1. `encode_query(...)`
2. `dense_candidates(...)`
3. `bm25_candidates(...)`
4. `fuse_hits_rrf(...)`

This means the runner’s retrieval path depends not just on a model and an index, but on a retrieval policy:

- dense semantic search
- lexical BM25 search
- fusion via RRF

### 8. Embedding / FAISS Layer

`VaxMapper/src/utils/embedding_utils.py` manages sentence-transformer loading and FAISS index handling.

Responsibilities:

- load the sentence-transformer model
- encode text batches
- build and save dense FAISS indexes
- move a CPU FAISS index to GPU for querying

Important architectural point:

- the mapper and evaluator both depend on sentence-transformer loading from this utility
- retrieval runtime stability is influenced by both this file and CUDA visibility handling in the runner

The `maybe_move_index_to_gpu(...)` function is one of the places where process-level GPU configuration and retrieval behavior meet directly.

### 9. Ontology Validation / Prefilter Layer

Two sources participate here:

- `VaxMapper/src/utils/snomed_utils.py`
- `agent_runner.validate_postcoord_with_mrcm`

The LangGraph runner uses `VaxMapper.src.utils.snomed_utils` for:

- loading SNOMED dataframes
- loading cached prefilter memberships
- filtering candidates by attribute range
- live / cached ECL-backed membership checks

The runner also maintains three in-process caches:

- `_PREFILTER_ATTR_RANGE_CACHE`
- `_PREFILTER_MEMBERSHIP_CACHE`
- `_PREFILTER_ECL_CACHE`

This is the ontology-constrained layer that narrows LLM options before final assembly and validates postcoordination after selection.

### 10. Section Resolution Layer

`VaxMapper/src/utils/dailymed.py` provides contraindications section lookup.

The runner uses:

- `CONTRA_Loinc`
- `extract_section(setid, [CONTRA_Loinc])`

This lets the SPL graph either:

- work directly from pre-supplied section text
- or fetch the contraindications section dynamically when the input is only an SPL identifier

### 11. Aggregation Layer

Aggregation is reused from `agent_runner.py`.

Key helpers:

- `aggregate_agent_results(...)`
- `write_jsonl(...)`
- `write_csv_rows(...)`
- `AGG_CSV_COLUMNS`

Role:

- convert per-item result objects from SPL outputs into normalized aggregated rows
- emit:
  - grouped JSONL by SPL
  - flat CSV by item

The aggregation step is the bridge between extraction/mapping runtime and evaluation.

### 12. Evaluation Layer

Evaluation is not implemented inside `langgraph_agent_runner.py`. It is delegated.

Call path:

- `langgraph_agent_runner.py` calls `evaluate_aggregated_predictions(...)`
- `agent_runner.py` wraps a subprocess call to `evaluate_agg_results_2.py`
- `evaluate_agg_results_2.py` performs matching and metric computation

Current evaluator responsibilities:

- load aggregated predictions and gold rows
- group rows by `SPL_SET_ID`
- compute pairwise match scores using:
  - semantic cosine between `annotation` and `query_text`
  - token Jaccard between `annotation` and `query_text`
- assign rows using either Hungarian or greedy matching
- handle ignored gold rows by matching them first and excluding matched ignored pairs from metrics
- compute:
  - extraction-level metrics
  - contraindication-level metrics
  - concept-level metrics
- write:
  - metrics JSON
  - evaluation details CSV

This means evaluation is logically downstream of the LangGraph runtime, but still part of the runner’s operational surface.

### 13. Observability Layer

`RunObserver` in `langgraph_agent_runner.py` is the observability mechanism.

Responsibilities:

- live progress line rendering
- node start / end / error events
- prompt/response logging for LLM calls
- JSONL audit output

It tracks both SPL-level and item-level context:

- SPL index and total
- SPL set ID
- item index and item total
- current node name

The observer is not just logging. It is tightly coupled to graph execution and is embedded into node wrappers for both graphs.

## State and Data Contracts

### SPL-Level State: `ContraState`

`ContraState` contains the state for one SPL as it moves through the SPL graph.

Key fields:

- `spl_record`
- `spl_set_id`
- `product_name`
- `contra_section_found`
- `contra_section_text`
- `extracted_items`
- `current_index`
- `current_item`
- `item_results`
- `final_result`
- `error`

Mutation pattern:

- `resolve_contra_section` populates SPL identity and section text
- `extract_items` populates `extracted_items`
- loop nodes update `current_item`, `current_index`, and `item_results`
- `finalize` writes `final_result`

### Item-Level State: `ItemState`

`ItemState` contains the state for one extracted contraindication item.

Key fields:

- `spl_set_id`
- `item`
- `candidates`
- `direct_match`
- `route_or_fill`
- `selected_focus_id`
- `selected_focus_term`
- `fills_norm`
- `fills_detail`
- `validation`
- `status`
- `expression`
- `item_result`

Mutation pattern:

- retrieval adds `candidates`
- direct match adds `direct_match`
- route-or-fill adds `route_or_fill`
- normalize computes selected focus and fill details
- validate may mutate selected focus/fills based on MRCM validation
- assembly writes the final item result payload

### Item Result Shape

The runtime emits three item statuses:

- `DIRECT`
- `POSTCOORD`
- `MINIMAL`

Common item result fields include:

- `SPL_SET_ID`
- `item_index`
- `query_text`
- `status`
- candidate or selection fields
- trace metadata
- extracted item payload

`DIRECT` contains a selected concept ID/term.  
`POSTCOORD` and `MINIMAL` contain a selected focus concept, fills, and an expression.

### Aggregated Output Shape

Aggregation produces:

- SPL-grouped JSONL:
  - `SPL_SET_ID`
  - `item_count`
  - `items`
- flat CSV rows:
  - `SPL_SET_ID`
  - `item_index`
  - `query_text`
  - `mapping_source`
  - `final_concept_id`
  - `final_concept_term`
  - `postcoord_expression`
  - attribute IDs/terms

This flat CSV is the evaluation input.

### Evaluation Output Shape

`evaluate_agg_results_2.py` writes:

- metrics JSON:
  - `inputs`
  - `extraction_level`
  - `contraindication_level`
  - `concept_level`
- details CSV:
  - one row per matched, unmatched, or ignored pairing outcome
  - includes assignment metadata, per-row metric contributions, and score components

## External Configuration and Runtime Dependencies

### CLI Surface

`langgraph_agent_runner.py` exposes:

- `--spl-list`
- `--out-jsonl`
- `--aggregated-jsonl`
- `--aggregated-csv`
- `--gold-csv`
- `--eval-json`
- `--eval-details-csv`
- `--audit-jsonl`
- `--disable-audit`
- `--disable-progress`
- `--backend`

### LLM Configuration

Azure:

- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_DEPLOYMENT`
- `AZURE_OPENAI_API_VERSION`

Hugging Face:

- `HF_MODEL_ID`
- `HF_DEVICE_MAP`
- `HF_DEVICE_MAP_JSON`
- `HF_TORCH_DTYPE`
- `HF_MAX_NEW_TOKENS`
- `HF_TEMPERATURE`
- `HF_TOP_P`
- `HF_TOP_K`
- `HF_REPETITION_PENALTY`
- `HF_DO_SAMPLE`
- `HF_LOAD_IN_8BIT`
- `HF_LOAD_IN_4BIT`
- `HF_TRUST_REMOTE_CODE`
- `HF_USE_FAST_TOKENIZER`
- `HF_MAX_MEMORY_JSON`
- `HF_MODEL_KWARGS_JSON`

### Agent Runtime Configuration

- `AGENT_MAX_TOKENS_EXTRACT`
- `AGENT_MAX_TOKENS_DIRECT`
- `AGENT_MAX_TOKENS_ROUTE_FILL`
- `AGENT_RETRIES`
- `LANGGRAPH_RECURSION_LIMIT`

### Retrieval / Ontology Configuration

- `SNOMED_SOURCE_DIR`
- `SNOMED_CONCEPT_PATH`
- `SNOMED_DESCRIPTION_PATH`
- `MAPPER_ES_INDEX`
- `MAPPER_DENSE_INDEX_PATH`
- `MAPPER_MODEL_NAME`
- `MAPPER_DEVICE`
- `MAPPER_K_DENSE`
- `MAPPER_K_BM25`
- `MAPPER_K_FINAL`
- `MAPPER_REBUILD_DENSE_INDEX`
- `MAPPER_REBUILD_ES_INDEX`
- `PREFILTER_CACHE_PATH`
- `PREFILTER_LIVE_FALLBACK`
- `PREFILTER_TIMEOUT`
- `PREFILTER_RETRIES`

### Evaluation Configuration

Via the wrapper / evaluator:

- `AGENT_GOLD_CSV`
- evaluator output paths
- evaluator matching parameters:
  - `alpha`
  - `beta`
  - `min_pair_score`
  - `assignment`
  - `decoupled`

### GPU / Process Configuration

The runner pays special attention to:

- `RUNNER_CUDA_VISIBLE_DEVICES`
- `CUDA_VISIBLE_DEVICES`
- `HF_CUDA_VISIBLE_DEVICES`

This is one of the most infrastructure-heavy parts of the file because model loading and FAISS GPU setup are sensitive to process-level CUDA visibility.

## Cross-File Coupling and Reuse Points

The most important couplings in the current design are:

### Runner <-> `agent_runner.py`

The LangGraph runner imports evaluation, aggregation, validation, and I/O helpers from the older runner. This means:

- orchestration is in one file
- several downstream contracts are still owned by another runner implementation

This is a strong reuse point, but it also mixes “current runtime” and “legacy/shared runtime” responsibilities.

### Runner <-> Prompt Utilities

Prompt semantics live outside the graph, but the graph is hardwired to those prompt functions and expected JSON schemas. The boundary is good conceptually, but the prompt result shapes are implicit runtime contracts.

### Runner <-> Retrieval Stack

The graph only sees `retrieve_candidates_for_item(...)`, but retrieval behavior depends on:

- sentence-transformer model loading
- FAISS asset loading and GPU placement
- Elasticsearch availability
- SNOMED frame loading
- RRF fusion policy

This makes retrieval a single call at the graph level but a deep stack operationally.

### Runner <-> Ontology Validation

Candidate prefiltering and postcoord validation are split across:

- `langgraph_agent_runner.py`
- `VaxMapper.src.utils.snomed_utils`
- `agent_runner.validate_postcoord_with_mrcm`

This is a meaningful sign that ontology-aware logic is spread across multiple files and ownership boundaries.

### Runner <-> Evaluation

Evaluation is downstream and subprocess-based, but still part of the runner’s command-line contract. The evaluation layer is not operationally independent because:

- output schema from aggregation must match evaluation expectations
- model/environment settings influence both retrieval and evaluation

## Reorganization Signals

This section is descriptive only: it points out where responsibilities currently mix.

### Orchestration and Infrastructure Live Together

`langgraph_agent_runner.py` contains both:

- domain orchestration logic
- infrastructure concerns such as backend loading, CUDA masking, progress rendering, and audit logging

These are distinct concerns today but are colocated.

### Shared Business Logic Is Split Across Two Runners

The LangGraph runner depends on `agent_runner.py` for shared contracts and helpers. As a result, the “current pipeline” is operationally spread across:

- `langgraph_agent_runner.py`
- `agent_runner.py`

This is one of the strongest existing coupling points.

### Retrieval Is Deeply Layered but Operationally Exposed as One Step

At the graph level, retrieval is one node. Underneath, it spans:

- `hyb_mapper.py`
- `search_utils.py`
- `embedding_utils.py`
- ES and FAISS runtime assets

This is useful abstraction for runtime flow, but important complexity is hidden behind one node boundary.

### Ontology Constraints Are Applied in Multiple Places

There are at least two ontology-constrained phases:

- prefiltering candidates before route-or-fill
- validating the selected postcoordination after route-or-fill

These phases rely on different helpers and ownership boundaries.

### Evaluation Is Downstream but Still Part of the Main Runtime Story

The current project shape ties prediction generation, aggregation, and evaluation closely together. Even though evaluation runs through a wrapper and subprocess, it is still a first-class part of the runner’s output contract.

## What This Document Should Help a Restructure Effort Clarify

A reader using this note should be able to identify:

- which file owns orchestration
- which files own prompt semantics
- which files own retrieval
- which files own ontology filtering and validation
- which files define output contracts
- which runtime concerns are process-level infrastructure rather than domain logic
- where cross-file coupling currently exists

The current architecture already has recognizable component boundaries. The main challenge is that those boundaries are implemented across several files with partial overlap in ownership.
