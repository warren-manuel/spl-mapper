# Claude Code Instructions: Presentation Content Extraction

## Your Task

Read the specified files from the codebase and produce a **single output file** called
`presentation_content.md`. This file will be fed into a separate chat to generate the
final slide deck. Follow the output format **exactly** — structure, section headers, and
placeholder tokens must be preserved as written.

---

## Output File

Write everything to: `presentation_content.md`

---

## Files to Read

Your first priority is to **primarily** these files, in this order:

1. `langgraph_agent_runner.py`
2. `VaxMapper/src/utils/dailymed.py`
3. `VaxMapper/src/utils/hyb_mapper.py`
4. `VaxMapper/src/utils/llm_prompt.py`
5. `VaxMapper/src/utils/snomed_utils.py`

If you encounter any additional relevant files required for this task read them as you deem necessary
---

## Output Format

Produce `presentation_content.md` with the following sections **in order**.
Each section maps to one slide. Do not add extra sections or reorder them.

---

### SECTION 1 — Title Slide

```
## SLIDE_01: Title
- Title: [extract or infer the project/pipeline name from file headers, class names, or docstrings]
- Subtitle: Extracting and Normalizing Drug Contraindications to SNOMED CT Using LLMs
- Author placeholder: [AUTHOR]
- Affiliation placeholder: [AFFILIATION]
- Date placeholder: [DATE]
```

---

### SECTION 2 — Motivation

```
## SLIDE_02: Motivation
[Do not extract from code. Leave the following block verbatim:]

- SPL contraindication sections exist as regulatory free text — clinically meaningful but
  computationally opaque
- Normalizing to SNOMED CT enables: pharmacovigilance, clinical decision support (CDS),
  and cross-drug comparison at scale
- Prior approaches relied on rule-based NLP or manual curation — neither scales to the
  full DailyMed corpus
- Key informatics gap: no reproducible, ontology-grounded pipeline exists for this task
```

---

### SECTION 3 — Problem Formulation

```
## SLIDE_03: Problem Formulation
[Do not extract from code. Leave the following block verbatim:]

- Task: Information Extraction + Concept Normalization
- Input: Free-text contraindication section from an SPL document
- Output: Set of SNOMED CT concept IDs with labels, grounded to the Clinical Finding
  or Disorder hierarchy
- What makes it hard:
  * Negation and modifier scope (e.g., "not recommended in patients with...")
  * Non-standard clinical phrasing across manufacturers
  * SNOMED CT breadth — millions of candidate concepts
  * Contraindication granularity varies widely across drugs
```

---

### SECTION 4 — Data & Knowledge Sources

```
## SLIDE_04: Data & Knowledge Sources

[From snomed_utils.py:]
- Describe how SNOMED CT is loaded (source files, format — e.g., RF2 txt files)
- List the key dataframes or data structures built (column names or structure if visible)
- Note any ECL filtering or hierarchy restriction used (e.g., Clinical Finding subtree)
- Paste the most informative 5–10 lines of code (function signature + core logic),
  truncated if needed. Label it: [SNOMED_UTILS_SNIPPET]

[From dailymed.py:]
- What DailyMed API endpoint(s) are called
- What SPL section(s) are targeted (section codes or names)
- Paste the most informative 5–10 lines (API call + section extraction), truncated.
  Label it: [DAILYMED_SNIPPET]
```

---

### SECTION 5 — Pipeline Architecture

```
## SLIDE_05: Pipeline Architecture (LangGraph)

[From langgraph_agent_runner.py:]

NODES:
- List every LangGraph node name defined (e.g., added via add_node()). 
  Format: one node per line as: NODE: <name>

EDGES:
- List every edge defined (add_edge, add_conditional_edges).
  Format: one edge per line as: EDGE: <source> --> <target> [CONDITIONAL: <condition_name>]
  Mark conditional edges clearly.

ENTRY POINT:
- ENTRY: <entry node name>

DATACLASSES / STATE SCHEMA:
- List the names of all dataclasses or TypedDicts used as LangGraph state
- For each, list its fields (name + type only, one per line)
- Label this block: [STATE_SCHEMA]

BATCH DESIGN NOTE:
- Describe in 1–2 sentences how SPL-level state relates to item-level state
  (i.e., how one SPL state contains multiple item states)
```

---

### SECTION 6 — Component 1: SPL Ingestion

```
## SLIDE_06: Component 1 — SPL Ingestion via DailyMed

[From dailymed.py:]
- What input triggers ingestion (e.g., NDC code, drug name, set ID)
- What the function returns (data structure / fields)
- Any preprocessing applied to extracted text before passing downstream
- [DAILYMED_SNIPPET] — reuse snippet from SLIDE_04 or paste a different
  representative one if more appropriate here
```

---

### SECTION 7 — Component 2: LLM Extraction

```
## SLIDE_07: Component 2 — LLM Extraction of Contraindication Spans

[From llm_prompt.py — extraction call only:]
- What is the LLM asked to extract? (summarize from the system/user prompt)
- What format is the output expected in? (list, JSON, spans, etc.)
- Any few-shot examples present? (yes/no, how many)
- Paste the extraction prompt template (system message or key instruction block),
  truncated to ~10 lines max. Label it: [EXTRACTION_PROMPT_SNIPPET]
- Note the model name / API used if visible in this file or in langgraph_agent_runner.py
```

---

### SECTION 8 — Component 3: Hybrid Candidate Retrieval

```
## SLIDE_08: Component 3 — Hybrid Candidate Retrieval (BM25 + Dense + RRF)

[From hyb_mapper.py:]
- What is the index built over? (SNOMED CT descriptions, preferred terms, synonyms?)
- BM25 leg: library used, any preprocessing (tokenization, stemming?)
- Dense leg: embedding model name/source, vector store or search method used
- RRF fusion: formula or k parameter used if visible
- How many candidates are returned before re-ranking / passing to LLM?
- Paste the RRF or fusion logic, ~5–10 lines. Label it: [RRF_SNIPPET]
- Paste the dense retrieval call, ~5–10 lines. Label it: [DENSE_SNIPPET]
```

---

### SECTION 9 — Component 4: LLM Verification & Slot-filling

```
## SLIDE_09: Component 4 — LLM Verification & Slot-filling

[From llm_prompt.py — verification and slot-filling calls:]
- Direct verify call: what is the LLM asked to do? (e.g., confirm candidate is correct,
  binary yes/no, confidence score?)
- Slot-filling call: what slots are being filled? (e.g., condition name, severity,
  population qualifier, negation flag?)
- How do these two calls relate — are they sequential, conditional?
- Paste the verification prompt template, ~10 lines max.
  Label it: [VERIFY_PROMPT_SNIPPET]
- Paste the slot-filling prompt template, ~10 lines max.
  Label it: [SLOTFILL_PROMPT_SNIPPET]
```

---

### SECTION 10 — Evaluation & Results

```
## SLIDE_10: Evaluation & Results

[Do not extract from code. Use placeholders exactly as written:]

- Total contraindication spans processed: [N_SPANS]
- Total SPL documents processed: [N_SPLS]
- Precision at concept level: [PRECISION]
- Recall at concept level: [RECALL]
- F1 at concept level: [F1]
- Baseline comparison (TBD): [BASELINE_NAME] — Precision: [BL_PRECISION], Recall: [BL_RECALL], F1: [BL_F1]
- Notes on evaluation methodology: [EVAL_NOTES]
```

---

### SECTION 11 — Limitations & Future Work

```
## SLIDE_11: Limitations & Future Work

[From your reading of the full codebase, infer honest limitations. Suggest 3–5 from:]
- Any hardcoded assumptions (e.g., SNOMED hierarchy scope, section targeting)
- Any TODO comments or incomplete functions
- Any noted edge cases in prompts or retrieval logic
- Known failure modes (e.g., highly specific compound conditions)

[Future work — leave as placeholders for the presenter to fill:]
- FUTURE_1: [e.g., extension to other SPL sections such as Warnings]
- FUTURE_2: [e.g., generalization beyond vaccines]
- FUTURE_3: [e.g., automated evaluation with larger annotated corpus]
```

---

### SECTION 12 — Conclusion

```
## SLIDE_12: Conclusion

[Do not extract from code. Leave the following block verbatim:]

- We present a modular, reproducible pipeline for ontology-grounded contraindication
  normalization from SPL free text
- Core informatics contribution: multi-step LLM reasoning + hybrid retrieval grounds
  clinical language to SNOMED CT at scale
- Pipeline is stateful (LangGraph), auditable (slot-filling), and extensible beyond
  vaccines
- [CLOSING_STATEMENT — presenter to fill]
```

---

## Additional Instructions

- **Do not summarize or paraphrase prompt templates** — paste them as close to verbatim
  as possible (truncate long examples, but preserve instruction wording)
- **Do not hallucinate** — if something is not present in the code, write `[NOT FOUND]`
- **Keep snippets short** — 5–10 lines per snippet, add `# ... (truncated)` if cutting
- **Preserve all placeholder tokens** like `[N_SPANS]`, `[BASELINE_NAME]`, etc. exactly
- **Flag ambiguity** — if a component's role is unclear from the code, add a note:
  `[CLAUDE NOTE: ...]` so the presenter can clarify

---

## Final Check Before Writing Output

Before writing `presentation_content.md`, confirm:
- [ ] All 5 files have been read
- [ ] All 12 sections are present in output
- [ ] NODE/EDGE/ENTRY block is complete for SLIDE_05
- [ ] All `[PLACEHOLDER]` tokens are preserved
- [ ] No section is missing or merged with another