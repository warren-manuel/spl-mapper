## SLIDE_01: Title
- Title: VaxMapper: SPL Contraindication Normalization Pipeline
- Subtitle: Extracting and Normalizing Drug Contraindications to SNOMED CT Using LLMs
- Author placeholder: [AUTHOR]
- Affiliation placeholder: [AFFILIATION]
- Date placeholder: [DATE]

---

## SLIDE_02: Motivation

- SPL contraindication sections exist as regulatory free text — clinically meaningful but
  computationally opaque
- Normalizing to SNOMED CT enables: pharmacovigilance, clinical decision support (CDS),
  and cross-drug comparison at scale
- Prior approaches relied on rule-based NLP or manual curation — neither scales to the
  full DailyMed corpus
- Key informatics gap: no reproducible, ontology-grounded pipeline exists for this task

---

## SLIDE_03: Problem Formulation

- Task: Information Extraction + Concept Normalization
- Input: Free-text contraindication section from an SPL document
- Output: Set of SNOMED CT concept IDs with labels, grounded to the Clinical Finding
  or Disorder hierarchy
- What makes it hard:
  * Negation and modifier scope (e.g., "not recommended in patients with...")
  * Non-standard clinical phrasing across manufacturers
  * SNOMED CT breadth — millions of candidate concepts
  * Contraindication granularity varies widely across drugs

---

## SLIDE_04: Data & Knowledge Sources

### SNOMED CT (from snomed_utils.py)

- Loaded from RF2 tab-separated `.txt` snapshot files stored under `snomed_source/`
- File patterns resolved at runtime:
  - `sct2_Concept_Snapshot_*.txt`
  - `sct2_Description_Snapshot-en_*.txt`
  - `sct2_Relationship_Snapshot_*.txt`
  - `der2_sssssssRefset_MRCMDomainSnapshot_*.txt`
  - `der2_cissccRefset_MRCMAttributeDomainSnapshot_*.txt`
  - `der2_ssccRefset_MRCMAttributeRangeSnapshot_*.txt`
- Key dataframes built:
  - `concept_df` — columns: `conceptId`, `term`, `semantic_tag`
  - `synonym_df` — columns: `conceptId`, `term`
  - `terms_df` — columns: `conceptId`, `term_text`, `term_type` (`preferred` | `synonym`)
  - `snomed_complete_df` — columns: `conceptId`, `term`, `semantic_tag`, `synonyms` (list)
  - `enriched_terms_df` — columns: `conceptId`, `term_text` (pipe-delimited: preferred | synonyms | semantic_tag | attr=val pairs)
  - `rel_df` — columns: `sourceId`, `destinationId`, `typeId`, `relationshipGroup`
- Hierarchy restriction: IS-A relationships (typeId=`116680003`); ECL validation via Snowstorm REST API
- Attribute slots hardcoded: `causative_agent` (246075003), `severity` (246112005), `clinical_course` (263502005)

[SNOMED_UTILS_SNIPPET]
```python
def _load_rf2_df(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")

def create_concept_df(
    concept_snapshot_path: Optional[str] = None,
    description_snapshot_path: Optional[str] = None,
    snomed_source_dir: str = "snomed_source",
) -> pd.DataFrame:
    concept_rf2 = _load_rf2_df(concept_path)
    desc_rf2 = _load_rf2_df(desc_path)
    snomed_active_con = concept_rf2[concept_rf2["active"] == 1].copy()
    snomed_des_df = desc_rf2[
        (desc_rf2["conceptId"].isin(snomed_active_con["id"])) & (desc_rf2["active"] == 1)
    ][["conceptId", "term", "typeId"]].copy()
    concept_df = snomed_des_df[snomed_des_df["typeId"] == int(FSN_TYPE_ID)][["conceptId", "term"]].copy()
    concept_df["semantic_tag"] = concept_df["term"].apply(extract_semantic_tag)
    # ... (truncated)
```

### DailyMed SPL (from dailymed.py)

- API endpoint: `https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/{setid}.xml`
- SPL sections targeted by LOINC code:
  - `34070-3` — Contraindications
  - `34084-4` — Adverse Reactions
- Section located via XPath on the parsed HL7 XML tree

[DAILYMED_SNIPPET]
```python
DAILYMED_BASE = "https://dailymed.nlm.nih.gov/dailymed/services/v2"
CONTRA_Loinc = "34070-3"
ADVERSE_Loinc = "34084-4"

def fetch_spl_xml_by_setid(setid: str, timeout: int = 30) -> bytes:
    url = f"{DAILYMED_BASE}/spls/{setid}.xml"  # returns the full SPL document (v2)
    resp = requests.get(url, timeout=timeout)
    if resp.status_code != 200 or not resp.content.strip():
        raise DailyMedError(f"Failed to fetch SPL XML for setid={setid}: {resp.status_code}")
    return resp.content

def find_section_by_loinc(root: etree._Element, loinc_code: str) -> etree._Element | None:
    ns = {"hl7": get_default_ns(root)}
    xpath = ".//hl7:component[hl7:section/hl7:code[@code=$code]]/hl7:section"
    results = root.xpath(xpath, namespaces=ns, code=loinc_code)
    return results[0] if results else None
```

---

## SLIDE_05: Pipeline Architecture (LangGraph)

### SPL-Level Graph

NODES:
- NODE: resolve_contra_section
- NODE: extract_items
- NODE: prepare_item
- NODE: process_item
- NODE: advance_item
- NODE: finalize

EDGES:
- ENTRY: resolve_contra_section
- EDGE: resolve_contra_section --> extract_items [CONDITIONAL: route_after_resolve]
- EDGE: resolve_contra_section --> finalize [CONDITIONAL: route_after_resolve]
- EDGE: extract_items --> prepare_item [CONDITIONAL: route_after_extract]
- EDGE: extract_items --> finalize [CONDITIONAL: route_after_extract]
- EDGE: prepare_item --> process_item
- EDGE: process_item --> advance_item
- EDGE: advance_item --> prepare_item [CONDITIONAL: continue_or_finish]
- EDGE: advance_item --> finalize [CONDITIONAL: continue_or_finish]
- EDGE: finalize --> END

### Item-Level Graph

NODES:
- NODE: retrieve_candidates
- NODE: direct_match
- NODE: prefilter
- NODE: route_or_fill
- NODE: normalize
- NODE: validate
- NODE: assemble_direct
- NODE: assemble_postcoord

EDGES:
- ENTRY: retrieve_candidates
- EDGE: retrieve_candidates --> direct_match
- EDGE: direct_match --> assemble_direct [CONDITIONAL: route_after_direct_match]
- EDGE: direct_match --> prefilter [CONDITIONAL: route_after_direct_match]
- EDGE: prefilter --> route_or_fill
- EDGE: route_or_fill --> normalize
- EDGE: normalize --> validate
- EDGE: validate --> assemble_postcoord
- EDGE: assemble_direct --> END
- EDGE: assemble_postcoord --> END

[STATE_SCHEMA]

ItemState (TypedDict, total=False):
- spl_set_id: str
- item: Dict[str, Any]
- candidates: Dict[str, List[Dict[str, Any]]]
- direct_match: Dict[str, Any]
- route_or_fill: Dict[str, Any]
- selected_problem_id: str
- selected_focus_term: str
- fills_norm: Dict[str, str]
- fills_detail: Dict[str, Dict[str, str]]
- validation: Dict[str, Any]
- status: str
- expression: str
- item_result: Dict[str, Any]

ContraState (TypedDict, total=False):
- spl_record: Dict[str, Any]
- spl_set_id: str
- product_name: Optional[str]
- contra_section_found: bool
- contra_section_text: str
- extracted_items: List[Dict[str, Any]]
- current_index: int
- current_item: Dict[str, Any]
- item_results: List[Dict[str, Any]]
- final_result: Dict[str, Any]
- error: str

BATCH DESIGN NOTE:
ContraState holds the full `extracted_items` list plus a `current_index` counter. The SPL-level graph iterates sequentially: `prepare_item` copies `extracted_items[current_index]` into `current_item`, `process_item` invokes the entire item-level graph for that single item and appends the result to `item_results`, and `advance_item` increments `current_index` or routes to `finalize`. One SPL state thus drives N sequential item-graph invocations, collecting all results before finalization.

---

## SLIDE_06: Component 1 — SPL Ingestion via DailyMed

- Input: SPL Set ID (string) — the unique DailyMed identifier for a drug/vaccine record
- Returns:
  ```
  {
    "setid": str,
    "product_name": str | None,
    "found": bool,
    "sections": {
      "<loinc_code>": {
        "section_xml": str,
        "section_text": str
      }
    }
  }
  ```
- Preprocessing applied by `_normalize_narrative_text()` before passing downstream:
  - Replace non-breaking spaces (`\xa0` → ` `) and encoding artifacts (`â€¢` → `•`)
  - Normalize line endings (CRLF/CR → LF)
  - Collapse runs of whitespace (spaces, tabs, form-feeds)
  - Collapse 3+ consecutive newlines to 2
  - Normalize bullet list formatting and punctuation spacing

[DAILYMED_SNIPPET]
```python
def fetch_spl_xml_by_setid(setid: str, timeout: int = 30) -> bytes:
    url = f"{DAILYMED_BASE}/spls/{setid}.xml"  # returns the full SPL document (v2)
    resp = requests.get(url, timeout=timeout)
    if resp.status_code != 200 or not resp.content.strip():
        raise DailyMedError(f"Failed to fetch SPL XML for setid={setid}: {resp.status_code}")
    return resp.content

def find_section_by_loinc(root: etree._Element, loinc_code: str) -> etree._Element | None:
    ns = {"hl7": get_default_ns(root)}
    xpath = ".//hl7:component[hl7:section/hl7:code[@code=$code]]/hl7:section"
    results = root.xpath(xpath, namespaces=ns, code=loinc_code)
    return results[0] if results else None
```

---

## SLIDE_07: Component 2 — LLM Extraction of Contraindication Spans

- What the LLM extracts: All atomic contraindications from the SPL contraindication section text. "Atomic" means the smallest semantically complete contraindicated condition — coordinated phrases (e.g., "A or B") must be split into separate items.
- Output format: Single-line minified JSON object `{"items":[...]}` terminated with `<<END_JSON>>`. Each item has 5 fields: `ci_text`, `contraindication_state_text`, `substance_text`, `severity_span`, `clinical_course_span`.
- Few-shot examples: None explicitly; 3 inline worked examples embedded in the system prompt instructions (coordination splitting and non-splitting cases).
- Model / API: Azure OpenAI or HuggingFace local model, selected via `LANGGRAPH_LLM_BACKEND` environment variable (default: `huggingface`).

[EXTRACTION_PROMPT_SNIPPET]
```
You are a biomedical NLP assistant that identifies CONTRAINDICATIONS in regulatory drug or vaccine documents.

Your overall job has TWO STRICT SUBTASKS:

- TASK 1: Identify and list all ATOMIC contraindication in the text.
- TASK 2: Convert those ATOMIC contraindications into a structured JSON output with the specified fields.

You must complete TASK 1 fully and accurately before starting TASK 2.

A contraindication is a condition or situation where the product SHOULD NOT be used or administered.
An ATOMIC contraindication is the smallest SEMANTICALLY COMPLETE contraindicated condition or situation.
# ... (truncated)
```

---

## SLIDE_08: Component 3 — Hybrid Candidate Retrieval (BM25 + Dense + RRF)

- Index built over: SNOMED CT enriched terms (`enriched_terms_df`) — preferred terms, synonyms, semantic tags, and MRCM attribute relationships, all pipe-delimited into a single `term_text` field per concept.
- BM25 leg:
  - Library: Elasticsearch
  - BM25 parameters: `k1=1.2`, `b=0.5`
  - Fields: `preferredTerm.exact` (keyword), `preferredTerm.phrase`, `preferredTerm.shingle` (2–3 word n-grams), `all_terms`
  - Analyzer: lowercase + ASCII folding
- Dense leg:
  - Embedding model: `tavakolih/all-MiniLM-L6-v2-pubmed-full` (PubMed-fine-tuned MiniLM)
  - Vector store: FAISS (CPU, optionally GPU via `maybe_move_index_to_gpu`)
  - Vectors are L2-normalized at index build time
- RRF fusion + candidate count:
  - `k_bm25=50` candidates from Elasticsearch
  - `k_dense=50` candidates from FAISS
  - Fused to `k_final=20` via RRF in `search_utils.search_query()`
  - Optional cross-encoder re-ranking if `cross_encoder` resource is loaded

[RRF_SNIPPET]
```python
hits = search_query(
    query_text=query,
    model=resources.st_model,
    faiss_index=resources.faiss_index,
    concept_meta_df=resources.concept_meta_df,
    es=resources.es,
    bm25_index=resources.bm25_index,
    label_column="term",
    bm25_text_field="preferredTerm",
    bm25_id_field="conceptId",
    bm25_label_field="preferredTerm",
    k_dense=resources.k_dense,
    k_bm25=resources.k_bm25,
    k_final=resources.k_final,
    # ... (truncated)
)
```

[DENSE_SNIPPET]
```python
def build_or_load_faiss_index(
    terms_df: pd.DataFrame,
    model_name: str,
    device: str,
    index_path: str,
    rebuild_index: bool,
):
    st_model = load_ST_model(model_name, device=device)
    if rebuild_index or not os.path.exists(index_path):
        _ = build_and_save_dense_index(
            df=terms_df, model=st_model, text_column="term_text",
            id_column="conceptId", batch_size=256, normalize=True,
            use_gpu_for_queries=True, save_index=True, index_filename=index_path,
        )
    cpu_index = faiss.read_index(index_path)
    faiss_index = maybe_move_index_to_gpu(cpu_index)
    return st_model, faiss_index
```

---

## SLIDE_09: Component 4 — LLM Verification & Slot-filling

- Direct verify call: The LLM is given the `contraindication_state_text` and a ranked candidate list. It must decide if a single candidate is an exact lexical-semantic match — identical clinical meaning, identical ontological level. Returns binary `direct_match: true/false` plus the winning `selected_id` and `selected_term`. Strict rules: subtypes are rejected ("Allergy ≠ Hypersensitivity"), supertypes rejected, temporal qualifiers rejected.
- Slot-filling call (`route_or_fill`): Fills 4 attribute slots — `focus_problem` (from FOCUS_CANDIDATES), `causative_agent`, `severity`, `clinical_course` (each from their own candidate lists). Also outputs a `post_decision` field ("YES" if the minimal 4-slot model is sufficient, "N/A" if clinically important meaning falls outside it).
- Relationship: Sequential. Direct verify runs first; if it returns `false`, the item proceeds to `prefilter` then `route_or_fill` for slot-level mapping. Both calls operate on different candidate sets retrieved by the hybrid stack.

[VERIFY_PROMPT_SNIPPET]
```
You are a strict biomedical terminology validator.
Your role is to determine if a candidate is a LEXICAL and SEMANTIC identity match for the contraindication query.

Your ONLY task:
- Identify if a DIRECT MATCH exists among the provided candidates.
- If a match exists, select exactly ONE candidate from the list.
- Otherwise, return no match.

When in doubt, return no match. A false negative here is recoverable; a false positive is not.

A candidate is a DIRECT MATCH only if its label is an exact lexical-semantic equivalent to the query —
identical clinical meaning, identical ontological level, identical primary term.
# ... (truncated)
```

[SLOTFILL_PROMPT_SNIPPET]
```
You are a SNOMED CT minimal representation assistant for SNOMED CT concepts.

Your job has two parts:
1) Decide whether a given expression can be sufficiently represented using ONLY:
   - one problem concept
   - optional causative_agent
   - optional severity
   - optional clinical_course
2) Regardless of that decision, extract the best available minimal concept representation from the provided candidates.

Rules:
- Always select the best focus concept you can from FOCUS_CANDIDATES, unless no credible focus exists.
- Always select the best value for each attribute from its candidate list when supported by the text.
- If no supported value exists for an attribute, output "N/A" for that attribute.
# ... (truncated)
```

---

## SLIDE_10: Evaluation & Results

- Total contraindication spans processed: [N_SPANS]
- Total SPL documents processed: [N_SPLS]
- Precision at concept level: [PRECISION]
- Recall at concept level: [RECALL]
- F1 at concept level: [F1]
- Baseline comparison (TBD): [BASELINE_NAME] — Precision: [BL_PRECISION], Recall: [BL_RECALL], F1: [BL_F1]
- Notes on evaluation methodology: [EVAL_NOTES]

---

## SLIDE_11: Limitations & Future Work

Limitations inferred from codebase:

1. **Hardcoded Snowstorm server IP** (`139.52.39.136:8080`) in `snomed_utils.py` — not parameterized via environment variable; breaks portability across deployments.
2. **Fixed attribute slot set** — pipeline maps only to `causative_agent`, `severity`, and `clinical_course`; other clinically relevant SNOMED attributes (e.g., finding site, associated morphology) are out of scope.
3. **Sequential item processing** — items within an SPL are processed one at a time in a loop; LangGraph `Send` API fan-out (parallelization) is noted as a roadmap item but not yet implemented.
4. **Stop token disabled in extraction** — `stop=None` workaround in `extract_items_node` due to model prematurely emitting the end token; may allow runaway generation on edge cases.
5. **Dense index path hardcoded** — `results/snomed_terms_dense_test.bin` in `hyb_mapper.py`; must be manually updated when the index is rebuilt or relocated.

Future work:
- FUTURE_1: [e.g., extension to other SPL sections such as Warnings]
- FUTURE_2: [e.g., generalization beyond vaccines]
- FUTURE_3: [e.g., automated evaluation with larger annotated corpus]

---

## SLIDE_12: Conclusion

- We present a modular, reproducible pipeline for ontology-grounded contraindication
  normalization from SPL free text
- Core informatics contribution: multi-step LLM reasoning + hybrid retrieval grounds
  clinical language to SNOMED CT at scale
- Pipeline is stateful (LangGraph), auditable (slot-filling), and extensible beyond
  vaccines
- [CLOSING_STATEMENT — presenter to fill]
