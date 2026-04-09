# Pipeline Context: SPL Contraindication Mapping

## What This Project Does

LangGraph-based pipeline that extracts contraindications from SPL (Structured Product Label) records and maps them to SNOMED concepts. Built in `langgraph_agent_runner.py`.

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
Replace the binary direct-match decision with a slot detection step:
```
"For this contraindication, which slots are expressed or implied?
 - problem_concept: yes/no
 - causative_concept: yes/no
 - severity_concept: yes/no
 - clinical_course_concept: yes/no"
```
Only retrieve and fill candidates for detected slots. Avoids spurious fills on simple items.

### Priority 3 — Slot-Specific Targeted Retrieval
Run separate retrieval + re-ranking per slot type with slot-appropriate queries:
```
item: "severe hepatic impairment"
  problem query:   "hepatic impairment"   → disorder concepts
  severity query:  "severe"               → severity qualifier concepts
  causative query: [empty for this item]
  course query:    [empty for this item]
```

### Priority 4 — Re-ranker After RRF ✅
Cross-encoder reranker wired in (`search_utils.rerank_candidates`). Enabled via `RERANKER_MODEL_ID` env var (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`). Off by default; backward-compatible.

### Priority 5 — Implicit Slot Inference in Prompts
Severity and clinical course are often implicit in the text. Instruct the LLM to infer from linguistic cues ("severe", "significant", "any degree of"). Pass the full surrounding sentence, not just the extracted item string.

### Priority 6 — Parallelize Item Processing
Items are independent after extraction. Use LangGraph `Send` API to fan out:
```python
def dispatch_items(state: ContraState):
    return [Send("process_item", {..., "item": item}) for item in state["extracted_items"]]
```
Watch Azure OpenAI rate limits — add a semaphore if needed.

### Priority 7 — Prompt Hardening for SNOMED Accuracy ⚠️ Partially done
- ✅ Ancestor path added to all candidate displays (targets Failure Mode 2)
- ✅ `DIRECT_VERIFY_SYSTEM_PROMPT` tightened: explicit subtype rejection (Allergy ≠ Hypersensitivity), "when in doubt return false" default
- ⬜ FSN (Fully Specified Name) not yet shown — preferred term only
- ⬜ Negative anchoring (explain why rank-2 was rejected) not yet implemented

---

## Agentic Approach: Decision

A **pure ReAct/tool-calling agent** was considered and rejected for this task because:
- SNOMED mapping has hard ontological constraints — an agent can call validation tools but can also ignore failed results
- The current pipeline's strength is that the ontology constrains the LLM, not the other way around
- Agents are sequential by nature; the LangGraph approach is parallelizable

**Recommended hybrid**: use the current LangGraph flow for straightforward items; route ambiguous items (e.g. conjunctions, negations, multiple clinical concepts in one item) to a mini ReAct agent with access to search_concepts, get_slot_candidates, and validate_expression tools.

---

## LLM Backends Supported
- Azure OpenAI (chat completions)
- Hugging Face local model (via `ChatLLM` protocol)

## Key Invariant
> LLM proposes; ontology/rules validate.

Keep this separation even as the pipeline evolves.
