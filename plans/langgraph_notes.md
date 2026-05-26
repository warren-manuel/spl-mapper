# LangGraph Runner Notes

## Purpose

`langgraph_agent_runner.py` is a LangGraph-based batch runner for extracting and mapping contraindications from SPL records. Its intent is to preserve the behavior of the older `agent_runner.py` pipeline while making the orchestration explicit, modular, and easier to port.

At a high level, it:

1. Loads one or more SPL records.
2. Resolves or fetches the contraindications section text.
3. Uses an LLM to extract discrete contraindication items from that section.
4. For each extracted item, retrieves SNOMED candidates.
5. Either:
   - accepts a direct concept match, or
   - selects a focus concept and optional post-coordination attribute values.
6. Validates the resulting expression against MRCM rules.
7. Writes per-SPL outputs, aggregated outputs, optional evaluation outputs, and optional audit logs.

This is not just a prompt runner. It is an orchestration layer around:

- section retrieval
- LLM extraction and decision-making
- hybrid candidate retrieval
- slot prefiltering
- postcoordination validation
- run-time observability

## Core Design

The file defines two nested graphs:

### 1. SPL-level graph

This graph processes one SPL record at a time.

Nodes:

- `resolve_contra_section`
- `extract_items`
- `prepare_item`
- `process_item`
- `advance_item`
- `finalize`

Flow:

- Resolve contraindications text from the input record, or fetch it externally if needed.
- Extract contraindication items with the LLM.
- Loop over extracted items.
- For each item, invoke the item-level graph.
- Collect item results and produce one final SPL result object.

### 2. Item-level graph

This graph processes one extracted contraindication item.

Nodes:

- `retrieve_candidates`
- `direct_match`
- `prefilter`
- `route_or_fill`
- `normalize`
- `validate`
- `assemble_direct`
- `assemble_postcoord`

Flow:

- Retrieve candidate concepts and slot candidates.
- Ask the LLM whether this item is already a direct concept match.
- If yes, emit a `DIRECT` result.
- Otherwise, prefilter slot candidates using attribute-range logic.
- Ask the LLM to choose a focus concept and optional fills for:
  - `causative_agent`
  - `severity`
  - `clinical_course`
- Normalize chosen IDs and labels.
- Validate the expression against MRCM constraints.
- Emit either:
  - `POSTCOORD` if postcoordination is explicitly selected, or
  - `MINIMAL` if only a focus concept is retained.

## State Contracts

Two LangGraph state models matter:

### `ContraState`

SPL-level state contains:

- raw SPL record
- SPL set ID
- product name
- contraindications text
- extracted items
- current item index
- accumulated item results
- final result
- error field

### `ItemState`

Item-level state contains:

- SPL set ID
- extracted item payload
- retrieved candidates
- direct-match decision
- route-or-fill decision
- selected focus concept
- normalized fills
- human-readable fill detail
- validation result
- assembled item result

If this is rebuilt elsewhere, these state boundaries are worth preserving. They keep the graph understandable and make node behavior testable.

## Expected Inputs

The runner expects a file of SPL identifiers or SPL records via:

```bash
python3 langgraph_agent_runner.py --spl-list <path>
```

The input loader comes from `agent_runner.load_spl_records_from_file`, so the runtime can handle either:

- plain SPL set IDs, or
- records containing already-resolved fields such as contraindications text

If the input record already contains contraindications text, the graph uses it directly. Otherwise it attempts external section extraction.

## Main Outputs

The runner writes:

- per-SPL raw result JSONL
- aggregated JSONL
- aggregated CSV
- optional evaluation JSON
- optional evaluation details CSV
- optional runtime audit JSONL

Per-item outputs contain fields such as:

- `status`: `DIRECT`, `POSTCOORD`, or `MINIMAL`
- `selected_focus_id`
- `selected_focus_term`
- optional `fills`
- optional postcoordinated `expression`
- trace metadata for auditability

Per-SPL outputs contain:

- `SPL_SET_ID`
- `product_name`
- `contra_section_found`
- `contra_section_text`
- `n_items_in`
- `n_items_out`
- `results`

## LLM Responsibilities

The LLM is used for three distinct decisions:

1. Extract contraindication items from section text.
2. Decide whether an item directly matches a concept.
3. Choose a focus concept plus slot fills when direct match is insufficient.

This separation is important. The runner does not rely on one large prompt that does everything. It decomposes the task into narrower decisions with different prompts and token budgets.

The code supports two backends:

- Azure OpenAI chat completions
- local Hugging Face chat model wrapper

The rest of the graph is backend-agnostic through a small `ChatLLM` protocol.

## Retrieval and Validation Responsibilities

The non-LLM parts are just as important as the prompts:

- candidate retrieval uses a hybrid mapper
- slot candidates are prefiltered by SNOMED attribute-range rules
- final focus/fill combinations are validated with MRCM logic

That means the LLM is constrained by ontology-aware retrieval and validation rather than trusted blindly.

If this is ported, the strongest invariant to keep is:

`LLM proposes; ontology/rules validate.`

## Operational Features

The runner includes a lightweight `RunObserver` for:

- live progress reporting
- node start/end/error logging
- LLM call logging
- JSONL audit output

This is useful because LangGraph makes control flow explicit, and the observer turns that into traceable runtime events.

For a new project, keep some form of:

- progress display
- per-node timing
- LLM prompt/output logging
- error capture by graph node

## Configuration Surface

There are three main config areas:

### CLI

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

### Azure backend env vars

- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_DEPLOYMENT`
- `AZURE_OPENAI_API_VERSION`

### Hugging Face backend env vars

- `HF_MODEL_ID`
- `HF_DEVICE_MAP`
- `HF_DEVICE_MAP_JSON`
- `HF_TORCH_DTYPE`
- `HF_MAX_NEW_TOKENS`
- `HF_TEMPERATURE`
- `HF_LOAD_IN_8BIT`
- `HF_LOAD_IN_4BIT`
- `HF_TRUST_REMOTE_CODE`
- `HF_USE_FAST_TOKENIZER`
- `HF_MAX_MEMORY_JSON`
- `HF_MODEL_KWARGS_JSON`

### Retrieval / ontology env vars

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
- `OUTPUT_DIR`
- `LANGGRAPH_LLM_BACKEND`

### CUDA visibility

The script also explicitly manages GPU visibility using:

- `RUNNER_CUDA_VISIBLE_DEVICES`
- `CUDA_VISIBLE_DEVICES`
- `HF_CUDA_VISIBLE_DEVICES`

That is worth preserving if the new project will support multi-GPU or constrained GPU allocation.

## Behavioral Invariants To Preserve

If recreating this runner in a new repo, preserve these behaviors:

1. Separate SPL-level orchestration from item-level orchestration.
2. Treat contraindication extraction, direct match verification, and route/fill selection as separate LLM calls.
3. Always perform candidate retrieval before LLM concept selection.
4. Prefilter slot candidates before asking the LLM to fill attributes.
5. Validate postcoordination after the LLM decision.
6. Emit structured trace data so every item can be audited.
7. Keep backend-specific LLM code behind a small common interface.
8. Support batch execution and deterministic output files.

## Recommended Porting Shape

If implementing this in a new project, a clean structure would be:

- `orchestration/langgraph_runner.py`
- `orchestration/states.py`
- `orchestration/nodes/spl_nodes.py`
- `orchestration/nodes/item_nodes.py`
- `llm/backends/azure.py`
- `llm/backends/huggingface.py`
- `retrieval/candidate_retriever.py`
- `ontology/prefilter.py`
- `ontology/validation.py`
- `prompts/contraindications.py`
- `observability/run_observer.py`

The current file is workable, but it mixes orchestration, config parsing, backend setup, node definitions, and CLI entrypoint in one place.

## Minimal Mental Model

The shortest correct summary of the file is:

> For each SPL, get contraindications text, extract items, map each item to SNOMED using retrieval plus LLM decisions, validate the result, and write auditable structured outputs.

## Notes For A Fresh Chat Session

If using this document to start a new implementation elsewhere, the next chat should probably begin with:

1. defining the new repo’s target interfaces for retrieval, validation, and LLM backends
2. deciding whether LangGraph state should remain `TypedDict`-based or move to dataclasses / pydantic models
3. splitting the current monolithic file into graph nodes and infrastructure modules
4. keeping output schemas compatible if downstream evaluation tooling already exists

