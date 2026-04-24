# Agents (Planned)

This directory is a placeholder for agentic modules that extend the core LangGraph pipeline in `src/mapping/`. Each agent operates within the architectural invariant:

> **LLM proposes; ontology/rules validate.**

Agents call into `src/` modules — they do not bypass ontology validation.

---

## Planned Agents

### 1. SNOMED Concept Navigator (`snomed_concept_reader`)

**Purpose:** Load and structure the top-level SNOMED CT hierarchy (Clinical Finding, Body Structure, Substance, Morphologic Abnormality, Pharmaceutical/Biological Product, etc.) as navigational context for LLM prompts.

**Interface:**
- Reads from `src/snomed/snomed_utils.py` (`load_snomed_dataframes`, IS-A graph)
- Produces `snomed_conventions.md` as a stable artifact documenting the hierarchy and naming conventions the LLM should respect
- **NOT a runtime agent** — runs once per SNOMED release as offline pre-computation
- Output feeds into prompt construction in `src/llm/prompts.py`

**Status:** Not started

---

### 2. Compositional Extractor (`compositional_extractor`)

**Purpose:** Extract contraindications from SPL text AND simultaneously decompose each into SNOMED slots (clinical_finding, body_site, substance, severity, morphology). Replaces the current monolithic extraction → mapping boundary with structured slot output.

**Interface:**
- Extends `src/extraction/section_parser.py` (section fetching) and `src/mapping/state.py` (once created: `ItemState` gains slot decomposition fields)
- Output feeds directly into slot-specific retrieval in `src/retrieval/hybrid_mapper.py`
- Connects to Priority 2 (Slot Presence Detection) and Priority 3 (Slot-Specific Targeted Retrieval) from the optimization roadmap

**Status:** Not started

---

### 3. Post-Coordination Expression Agent (`postcord_agent`)

**Purpose:** Given slot-mapped candidates, produce a valid SNOMED CT post-coordinated expression. Consists of two sub-agents:

**a. Pattern Finder**
- Queries Neo4j (or SNOMED RF2 relationship graph) for similar expressed concepts
- Extracts expression pattern by analogy from structurally similar pre-coordinated concepts
- Reads from `src/snomed/snomed_utils.py` (relationship graph, ECL evaluation)

**b. Expression Validator**
- Verifies candidate expression against MRCM rules (domain/range constraints, required roles)
- Operates against pre-computed OWL closure from `src/snomed/snomed_utils.py`
- Routes low-confidence results to human review queue

**Interface:**
- Wraps `src/mapping/postcoord.py` (current postcord_v2 logic)
- Input: slot-mapped candidates from `src/retrieval/hybrid_mapper.py`
- Output: validated SNOMED CT post-coordinated expression or human-review flag
- Connects to Priority 7 (Prompt Hardening) and the hybrid ReAct/LangGraph approach described in CLAUDE.md

**Status:** Not started

---

## Integration Contract

All agents must:
1. Call into `src/` modules for ontology access — never duplicate SNOMED logic
2. Return structured output compatible with `ContraState` or `ItemState` in `src/mapping/`
3. Pass results through the validation layer in `src/snomed/snomed_utils.py` before committing
4. Be parallelizable via LangGraph `Send` API when processing multiple items
