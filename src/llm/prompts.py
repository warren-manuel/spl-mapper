from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.llm.backends import (
    build_message,
    extract_json,
    has_end_json_token,
    trim_after_end_json_token,
)

END_JSON_TOKEN = "<<END_JSON>>"

# ---------------------------------------------
# Prompt 01: CONTRAINDICATION EXTRACTION
# ---------------------------------------------
CONTRA_EXTRACT_SYSTEM_PROMPT = """
You are a biomedical NLP assistant that identifies CONTRAINDICATIONS in regulatory drug or vaccine documents.

Your overall job has TWO STRICT SUBTASKS:

- TASK 1: Identify and list all ATOMIC contraindication in the text.
- TASK 2: Convert those ATOMIC contraindications into a structured JSON output with the specified fields.

You must complete TASK 1 fully and accurately before starting TASK 2.

--------------------
DEFINITIONS
--------------------
A contraindication is a condition or situation where the product SHOULD NOT be used or administered.

An ATOMIC contraindication is the smallest SEMANTICALLY COMPLETE contraindicated condition or situation.
CRITICAL:
- An atomic contraindication must be a COMPLETE clinical statement.
- It must be understandable on its own without missing context.
- It must include the clinical head (e.g., hypersensitivity, coadministration, disease, condition).


--------------------
TASK 1: IDENTIFY ATOMIC CONTRAINDICATIONS
--------------------
Given the input TEXT:

1) Carefully read the TEXT and identify EVERY text span that expresses a contraindication.

2) For each span split any coordinated phrase into multiple ATOMIC contraindications ONLY when coordination represents DISTINCT standalone contraindications:
   - If a contraindication statement contains coordination (e.g., "or", "and", commas, bullet lists),
   you MUST ALWAYS split it into MULTIPLE atomic contraindications whenever each item could stand alone as a distinct contraindication.
   Example:
     "known hypersensitivity to drug A or to any of the other ingredients in drug A, or
     with known hypersensitivity to drug B analogs, including or such as drug C"
   You MUST extract THREE separate contraindications:
   1) hypersensitivity to drug A
   2) hypersensitivity to ingredients of drug A
   3) hypersensitivity to drug B
   DO NOT add a separate contraindication for the example drug (drug C).

   BUT:

    DO NOT split when coordination defines a SINGLE condition:
    Example:
    "drugs that prolong QT interval and are metabolized via CYP3A4"
    
    KEEP AS ONE atomic contraindication

   Do NOT merge coordinated items into a single generalized condition.
   Do NOT abstract away specific substances or categories.
   - For lists such as "A, B, and C", create one atomic contraindication for each of A, B, and C.

3) Atomic span construction rule (CRITICAL): Each atomic contraindication must be grounded in the TEXT using only words that appear in the TEXT, but it does NOT need to be a single contiguous substring when splitting coordination.
   - When a shared head is coordinated (e.g., "moderate or severe condition X is a contraindication"), you MUST create separate atomic items by recombining the shared head with each coordinated modifier using exact words from the TEXT:
     - moderate condition X
     - severe condition X
   - You may drop coordinating glue ("or", "and") and repeated filler words, but you MUST NOT paraphrase or introduce new clinical terms not present in the TEXT.

For the purpose of TASK 2, internally represent the result of TASK 1 as a list of atomic contraindications called ATOMIC_LIST.

--------------------
TASK 1 SELF-CHECK: COORDINATION SPLITTING
--------------------
Before producing the final JSON:
Review each item in ATOMIC_LIST and verify that:
1) It a COMPLETE clinical statement
2) It contains no unsplit coordination that should have been separated.
3) It is the smallest standalone contraindicated condition supported by the text.

If not, fix it.

--------------------
TASK 2: STRUCTURED OUTPUT
--------------------
For each final atomic contraindication in ATOMIC_LIST, output one JSON object with these fields:

- "ci_text":
  - The atomic contraindication text in the ATOMIC_LIST.

-"contraindication_state_text":
  - A CORE clinical problem (state, condition, procedure, or situation) that makes use of the drug unsafe, with all modifiers removed.
    Examples:
    "hypersensitivity disposition"
    "drug administration"
    "hepatic impairment"
  - Normalize wording to a general clinical formulation rather than copying the original phrasing.
  - This may be a disease/disorder, clinical finding, procedure, or clinical situation.
    Do NOT include:
        - Drug or substance names.
        - Standalone modifiers such as severity, clinical course, temporality, or laboratory thresholds.
        - Remove population framing and keep only the underlying clinical state.

- "substance_text":
  - The exact substance, ingredient, product, or drug that is the causative agent for the constraindicated state.
  - If absent, set to null.

- "severity_span":
  - Exact severity wording from the text, if present.
  - Examples: mild, moderate, severe
  - If absent, set to null.

- "clinical_course_span":
  - Exact clinical course wording from the text, if present.
  - Examples: acute, chronic, recurrent, subacute
  - If absent, set to null.

Rules:
- Do NOT invent substances, severities, or courses that are not in the text.
- If a field is not explicitly supported by the text, you MUST set it to null.

--------------------
OUTPUT FORMAT
--------------------
You MUST return ONLY a single line of MINIFIED JSON.
Do NOT include markdown code fences.
Do NOT include any explanations, reasoning, or extra text.

The JSON must end with the exact token:
<<END_JSON>>

The JSON object MUST have the form:
{"items":[...]}

Each element of "items" MUST be an object with these fields:
- "ci_text"
- "contraindication_state_text"
- "substance_text"
- "severity_span"
- "clinical_course_span"


If there are no contraindications in the text, return: {"items":[]}
"""

CONTRA_EXTRACT_USER_PROMPT = """
Here is the CONTRAINDICATIONS section from a vaccine SPL document:
{text}
"""

# ---------------------------------------------
# Prompt 01b: SIMPLE IDENTIFICATION (step 1 of two-step extraction)
# Prompt contents to be supplied — must return {"items":[{"ci_text":"..."},...]}<<END_JSON>>
# ---------------------------------------------
# Prompt loaded from agents/extract_agent.md at first call (see _load_extract_agent below).
SIMPLE_EXTRACT_SYSTEM_PROMPT = ""  # nulled — content moved to agents/extract_agent.md
SIMPLE_EXTRACT_USER_PROMPT = """
Here is the section of text from the drug label:
{text}
{ingredient_block}
Please identify any contraindications mentioned in the text and list them clearly.
"""
# ---------------------------------------------
# Prompt 05: DECOMPOSE COORDINATIONS (step 2 of two-step extraction)
# Prompt contents to be supplied — must return {"items":[...]}<<END_JSON>> per item
# Full item schema: ci_text, contraindication_state_text, substance_text,
#                   severity_span, clinical_course_span
# ---------------------------------------------
# Prompt loaded from agents/decompose_agent.md at first call (see _load_decompose_agent below).
DECOMPOSE_SYSTEM_PROMPT = ""  # nulled — content moved to agents/decompose_agent.md
DECOMPOSE_USER_PROMPT = """
Here is a contraindication span extracted from the text.
contraindication span: {ci_text}
Please apply the decomposition rules as specified.
"""

# ---------------------------------------------
# Markdown prompt loaders — extraction and decomposition
# Follow the same lazy-cache pattern as _load_focus_selector (line ~748).
# ---------------------------------------------

_EXTRACT_AGENT_CACHE: str = ""


def _load_extract_agent() -> str:
    global _EXTRACT_AGENT_CACHE
    if not _EXTRACT_AGENT_CACHE:
        p = Path(__file__).parent.parent.parent / "agents" / "extract_agent.md"
        if p.exists():
            _EXTRACT_AGENT_CACHE = p.read_text(encoding="utf-8")
    return _EXTRACT_AGENT_CACHE


_DECOMPOSE_AGENT_CACHE: str = ""


def _load_decompose_agent() -> str:
    global _DECOMPOSE_AGENT_CACHE
    if not _DECOMPOSE_AGENT_CACHE:
        p = Path(__file__).parent.parent.parent / "agents" / "decompose_agent.md"
        if p.exists():
            _DECOMPOSE_AGENT_CACHE = p.read_text(encoding="utf-8")
    return _DECOMPOSE_AGENT_CACHE


# ---------------------------------------------
# Prompt 02: MAPPING VERIFICATION
# ---------------------------------------------
# Original prompt (pre-20260407). Used when USE_STRICT_PROMPTS=0.
DIRECT_VERIFY_SYSTEM_PROMPT_ORIGINAL = """
You are a strict biomedical terminology validator.
Your role is to determine if a candidate is a LEXICAL and SEMANTIC identity match for the contraindication query.

Your ONLY task:
- Identify if a DIRECT MATCH exists among the provided candidates.
- If a match exists, select exactly ONE candidate from the list.
- Otherwise, return no match.

--------------------
DIRECT MATCH (STRICT)
--------------------
A candidate is a DIRECT MATCH only if it is an exact semantic equivalent. Do NOT bridge concepts even if they are clinically related.

1) Lexical-Semantic Alignment:
- The candidate must match the specificity and naming of the query.
- Do NOT bridge concepts that use different primary terms even if they are clinically related.
- If the query text and candidate label belong to different levels of the hierarchy, return no match.
2) Temporal/Contextual Scope:
- "Post-X" is NOT a match for "X".
- "History of X" is NOT a match for "X".
- If the query implies a state after an event and the candidate is only the event, return no match.
3) Meaning Completeness:
- Do NOT select a candidate that represents only part of the query.
4) Semantic Type Guardrail:
- If the query implies a condition/state, do NOT select a procedure concept unless the query explicitly describes a procedure.
5) Hierarchical Granularity:
- A match must exist at the same level of specificity as the query.
- A broader parent or narrower child is NOT a direct match.

--------------------
CANDIDATE ELIGIBILITY FILTERS
--------------------
Unless the query explicitly describes a procedure or product, do NOT select candidates whose label indicates:
- procedure
- administration
- vaccination
- other overly broad parent concepts when the query is specific

--------------------
OUTPUT FORMAT
--------------------
Return ONLY minified JSON followed immediately by <<END_JSON>>.
Do NOT use markdown fences.
Do NOT include explanations or extra fields.

Schema:
{"direct_match":true/false,"selected_id":"<candidate id or N/A>","selected_term":"<candidate term or N/A>"}<<END_JSON>>

Rules:
- If direct_match=true, selected_id MUST be from the candidate list.
- If direct_match=false, selected_id and selected_term MUST be N/A.
"""

# Stricter prompt added 20260407: explicit subtype rejection, "when in doubt return false" default.

DIRECT_VERIFY_SYSTEM_PROMPT = """
You are a strict biomedical terminology validator.
Your role is to determine if a candidate is a LEXICAL and SEMANTIC identity match for the contraindication query.

Your ONLY task:
- Identify if a DIRECT MATCH exists among the provided candidates.
- If a match exists, select exactly ONE candidate from the list.
- Otherwise, return no match.

When in doubt, return no match. A false negative here is recoverable; a false positive is not.

--------------------
DIRECT MATCH (STRICT)
--------------------
A candidate is a DIRECT MATCH only if its label is an exact lexical-semantic equivalent to the query — identical clinical meaning, identical ontological level, identical primary term.

1) Primary Term Identity:
- The root condition word used in the query MUST be the same root condition word in the candidate label.
- Clinically related terms that are NOT interchangeable in SNOMED CT:
    * "Hypersensitivity" ≠ "Allergy"  (Allergy is a subtype of Hypersensitivity — different concepts)
    * "Hypersensitivity" ≠ "Sensitivity"
    * "Disorder" ≠ "Finding"
    * "Reaction" ≠ "Hypersensitivity"
- If the query says "hypersensitivity to X" and the candidate says "allergy to X", return no match even if they seem clinically equivalent. They occupy different positions in the SNOMED CT hierarchy.

2) Hierarchical Granularity — subtypes and supertypes are both wrong:
- A narrower (more specific) concept is NOT a match: "Allergy to ibuprofen" is NOT a match for "Hypersensitivity to ibuprofen".
- A broader (more general) concept is NOT a match: "Hypersensitivity disorder" is NOT a match for "Hypersensitivity to ibuprofen".
- The candidate must encode exactly the same meaning — not more specific, not more general.

3) Meaning Completeness:
- Do NOT select a candidate that represents only part of the query meaning.
- Do NOT select a candidate that adds qualifiers not present in the query (e.g., "known", "documented", "acute").

4) Temporal/Contextual Scope:
- "Post-X" is NOT a match for "X".
- "History of X" is NOT a match for "X".
- If the query implies a state after an event and the candidate is only the event, return no match.

5) Semantic Type Guardrail:
- If the query implies a condition/state, do NOT select a procedure concept unless the query explicitly describes a procedure.

--------------------
CANDIDATE ELIGIBILITY FILTERS
--------------------
Unless the query explicitly describes a procedure or product, do NOT select candidates whose label indicates:
- procedure
- administration
- vaccination
- other overly broad parent concepts when the query is specific

--------------------
OUTPUT FORMAT
--------------------
Return ONLY minified JSON followed immediately by <<END_JSON>>.
Do NOT use markdown fences.
Do NOT include explanations or extra fields.

Schema:
{"direct_match":true/false,"selected_id":"<candidate id or N/A>","selected_term":"<candidate term or N/A>"}<<END_JSON>>

Rules:
- If direct_match=true, selected_id MUST be from the candidate list.
- If direct_match=false, selected_id and selected_term MUST be N/A.
"""

DIRECT_VERIFY_USER_TEMPLATE = """QUERY:
"{ci_text}"

CANDIDATES (choose from these only):
{candidate_block}
"""


SPLIT_SYSTEM_PROMPT = """You split coordinated contraindication text into ATOMIC items.

Return ONLY minified JSON followed by <<END_JSON>>. No fences. No explanations.

Schema:
{"atomic_spans":["...","..."]}<<END_JSON>>

Rules:
- Split lists joined by "or", "and", commas when each item can stand alone.
- Keep shared context phrases needed to preserve meaning (e.g., "after taking aspirin").
- Do not invent info not in the text.
"""

SPLIT_USER_TEMPLATE = """TEXT:
{ci_text}
"""


def format_candidate_block(candidates: List[Dict[str, Any]], max_n: int = 10) -> str:
    if not candidates:
        return "NONE"
    lines = []
    for i, candidate in enumerate(candidates[:max_n], 1):
        cid = candidate.get("id")
        label = candidate.get("label") or candidate.get("term") or ""
        score = candidate.get("rerank_score") or candidate.get("fused") or 0.0
        ancestor_path = candidate.get("ancestor_path", "")
        line = f"{i}) {cid} | Concept name: {label} | Score: {score:.3f}"
        if ancestor_path:
            line += f" | Ancestor path: {ancestor_path}"
        lines.append(line)
    return "\n".join(lines)


def build_direct_verify_user_prompt(
    ci_text: str,
    candidates: List[Dict[str, Any]],
    *,
    max_n: int = 10,
) -> str:
    return DIRECT_VERIFY_USER_TEMPLATE.format(
        ci_text=ci_text,
        candidate_block=format_candidate_block(candidates, max_n=max_n),
    )


def build_split_user_prompt(ci_text: str) -> str:
    return SPLIT_USER_TEMPLATE.format(ci_text=ci_text)


def split_atomic_if_needed(
    item: Dict[str, Any],
    *,
    call_llm_json: Any,
    looks_coordinated_fn: Any,
    max_tokens: int,
    retries: int,
    backoff_s: float,
    sleep_fn: Any,
) -> List[Dict[str, Any]]:
    ci_text = item.get("ci_text", "")
    if not looks_coordinated_fn(ci_text):
        return [item]

    user = build_split_user_prompt(ci_text)
    for attempt in range(retries + 1):
        parsed, _raw = call_llm_json(SPLIT_SYSTEM_PROMPT, user, max_tokens=max_tokens)
        if parsed and isinstance(parsed.get("atomic_spans"), list) and parsed["atomic_spans"]:
            out = []
            for span in parsed["atomic_spans"]:
                next_item = dict(item)
                next_item["ci_text"] = span
                out.append(next_item)
            return out
        sleep_fn(backoff_s * (attempt + 1))

    return [item]


ROUTE_OR_FILL_SYSTEM_PROMPT = """
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
- Use ONLY IDs from the provided candidate lists.
- Do NOT invent IDs.

post_decision rules:
- "YES" if the contraindication is sufficiently represented by the minimal model above.
- "N/A" if clinically important meaning remains outside that model, or if the representation is incomplete.

Output ONLY minified JSON followed by <<END_JSON>>.
Do NOT use markdown fences.
Do NOT include explanations or extra fields.

Schema:
{"post_decision":"YES|N/A","selected_problem_id":"<id or N/A>","fills":{"causative_agent":"<id or N/A>","severity":"<id or N/A>","clinical_course":"<id or N/A>"}}<<END_JSON>>
"""


ROUTE_OR_FILL_USER_TEMPLATE = """QUERY:
{ci_text}

EXTRACTED_FIELDS:
contraindication_state_text={contraindication_state_text}
substance_text={substance_text}
severity_span={severity_span}
course_span={course_span}

ATTRIBUTE_TABLE:
{attribute_table_json}

FOCUS_CANDIDATES:
{focus_candidates_block}

CAUSATIVE_AGENT_CANDIDATES:
{agent_candidates_block}

SEVERITY_CANDIDATES:
{severity_candidates_block}

CLINICAL_COURSE_CANDIDATES:
{course_candidates_block}
"""


def build_route_or_fill_user_prompt(
    item: Dict[str, Any],
    attribute_table_json: str,
    cands: Dict[str, List[Dict[str, Any]]],
    *,
    max_n: int = 10,
) -> str:
    return ROUTE_OR_FILL_USER_TEMPLATE.format(
        ci_text=item.get("ci_text"),
        contraindication_state_text=item.get("contraindication_state_text"),
        substance_text=item.get("substance_text"),
        severity_span=item.get("severity_span"),
        course_span=item.get("course_span"),
        attribute_table_json=attribute_table_json,
        focus_candidates_block=format_candidate_block(cands.get("focus_candidates", []), max_n=max_n),
        agent_candidates_block=format_candidate_block(cands.get("causative_agent_candidates", []), max_n=max_n),
        severity_candidates_block=format_candidate_block(cands.get("severity_candidates", []), max_n=max_n),
        course_candidates_block=format_candidate_block(cands.get("clinical_course_candidates", []), max_n=max_n),
    )


def normalize_contra_extraction_item(item: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(item, dict):
        return {}

    ci_text = item.get("ci_text")
    if ci_text is None:
        ci_text = item.get("span_text")

    substance_text = item.get("substance_text")
    if substance_text is None:
        substance_text = item.get("substance_span")

    return {
        "ci_text": ci_text,
        "contraindication_state_text": item.get("contraindication_state_text"),
        "substance_text": substance_text,
        "severity_span": item.get("severity_span"),
        "course_span": item.get("course_span"),
    }


def parse_contra_extraction_output(text: str) -> List[Dict[str, Any]]:
    cleaned = trim_after_end_json_token(text, token=END_JSON_TOKEN, include_token=False)
    parsed = extract_json(cleaned)
    if not isinstance(parsed, dict):
        return []

    items = parsed.get("items")
    if not isinstance(items, list):
        return []

    normalized_items: List[Dict[str, Any]] = []
    for item in items:
        normalized = normalize_contra_extraction_item(item)
        if normalized.get("ci_text") or normalized.get("contraindication_state_text"):
            normalized_items.append(normalized)
    return normalized_items


def _build_ingredient_block(ingredients: Optional[Dict[str, List[str]]]) -> str:
    """Return an 'Available Ingredients:' context line for injection into the extraction prompt.

    Uses a plain pipe-delimited format (not JSON) so the LLM can copy ingredient names
    verbatim into ci_text without any JSON escaping concerns.

    Returns an empty string when ingredients is None or both lists are empty so that callers
    can safely substitute it into any template with ``{ingredient_block}``.
    """
    if not ingredients:
        return ""
    active = ingredients.get("active", [])
    inactive = ingredients.get("inactive", [])
    all_ingredients = [*active, *inactive]
    if not all_ingredients:
        return ""
    return "Available Ingredients: " + " | ".join(all_ingredients)


def extract_contraindication_items(
    chat_fn: Any,
    text: str,
    *,
    max_tokens: int = 512,
    stop: Optional[List[str]] = None,
    retries: int = 1,
    retry_token_increment: int = 256,
    system_prompt: str = CONTRA_EXTRACT_SYSTEM_PROMPT,
    user_prompt_template: str = CONTRA_EXTRACT_USER_PROMPT,
    ingredients: Optional[Dict[str, List[str]]] = None,
) -> Tuple[List[Dict[str, Any]], str]:
    ingredient_block = _build_ingredient_block(ingredients)
    try:
        user_prompt = user_prompt_template.format(text=text, ingredient_block=ingredient_block)
    except KeyError:
        # Template does not have {ingredient_block} — format without it
        user_prompt = user_prompt_template.format(text=text)
    messages = build_message(system_prompt, user_prompt)

    last_raw = ""
    attempts = max(1, retries + 1)
    for attempt in range(attempts):
        run_tokens = max_tokens + (attempt * retry_token_increment)
        last_raw = chat_fn(messages, max_tokens=run_tokens, stop=stop)
        items = parse_contra_extraction_output(last_raw)
        if items or has_end_json_token(last_raw, token=END_JSON_TOKEN, require_terminal=False):
            return items, last_raw

    return [], last_raw


def build_decompose_user_prompt(item: Dict[str, Any]) -> str:
    return DECOMPOSE_USER_PROMPT.format(ci_text=item.get("ci_text", ""))


def decompose_contraindication_item(
    chat_fn: Any,
    item: Dict[str, Any],
    *,
    max_tokens: int = 512,
    stop: Optional[List[str]] = None,
    retries: int = 1,
    retry_token_increment: int = 256,
    system_prompt: str = "",
) -> Tuple[List[Dict[str, Any]], str]:
    effective_system = system_prompt or DECOMPOSE_SYSTEM_PROMPT or _load_decompose_agent()
    messages = build_message(effective_system, build_decompose_user_prompt(item))
    last_raw = ""
    attempts = max(1, retries + 1)
    for attempt in range(attempts):
        run_tokens = max_tokens + (attempt * retry_token_increment)
        last_raw = chat_fn(messages, max_tokens=run_tokens, stop=stop)
        items = parse_contra_extraction_output(last_raw)
        if items or has_end_json_token(last_raw, token=END_JSON_TOKEN, require_terminal=False):
            return items if items else [item], last_raw
    return [item], last_raw  # fallback: return original item unchanged


# ---------------------------------------------
# Prompt 06: CATEGORIZE SLOTS (Agent 2 — Compositional Extractor)
# System prompt loaded from agents/snomed_conventions.md at first call.
# ---------------------------------------------

_CONVENTIONS_CACHE: str = ""


def _load_snomed_conventions() -> str:
    global _CONVENTIONS_CACHE
    if not _CONVENTIONS_CACHE:
        p = Path(__file__).parent.parent.parent / "agents" / "snomed_conventions.md"
        if p.exists():
            _CONVENTIONS_CACHE = p.read_text(encoding="utf-8")
    return _CONVENTIONS_CACHE


CATEGORIZE_SLOTS_USER_PROMPT = """Contraindication span:
{ci_text}

Identify each clinical component and tag it with its SNOMED CT hierarchy.
Return minified JSON with key "components" followed by <<END_JSON>>.
Example: {{"components": [{{"text": "ibuprofen", "hierarchy": "Substance", "role": "causative_agent"}}]}}<<END_JSON>>
"""


def parse_categorize_slots_output(text: str) -> List[Dict[str, Any]]:
    cleaned = trim_after_end_json_token(text, token=END_JSON_TOKEN, include_token=False)
    parsed = extract_json(cleaned)
    if not isinstance(parsed, dict):
        return []
    # Accept both "components" (canonical) and "segments" (LLM variant)
    components = parsed.get("components") or parsed.get("segments") or []
    if not isinstance(components, list):
        return []
    return [
        c for c in components
        if isinstance(c, dict)
        # Accept both "text" (canonical) and "span" (LLM variant)
        and (c.get("text") or c.get("span"))
        and c.get("hierarchy")
    ]


def categorize_item_slots(
    chat_fn: Any,
    item: Dict[str, Any],
    *,
    max_tokens: int = 256,
    stop: Optional[List[str]] = None,
    retries: int = 1,
    retry_token_increment: int = 128,
) -> Tuple[List[Dict[str, Any]], str]:
    system = _load_snomed_conventions()
    if not system:
        return [], ""
    messages = build_message(system, CATEGORIZE_SLOTS_USER_PROMPT.format(
        ci_text=item.get("ci_text", "")
    ))
    last_raw = ""
    attempts = max(1, retries + 1)
    for attempt in range(attempts):
        run_tokens = max_tokens + (attempt * retry_token_increment)
        last_raw = chat_fn(messages, max_tokens=run_tokens, stop=stop)
        components = parse_categorize_slots_output(last_raw)
        if components or has_end_json_token(last_raw, token=END_JSON_TOKEN, require_terminal=False):
            return components, last_raw
    return [], last_raw


# ---------------------------------------------
# Prompt 07: FOCUS SELECTOR (Agent 2.5 — ReAct focus concept selection)
# System prompt loaded from agents/focus_selector.md at first call.
# ---------------------------------------------

_FOCUS_SELECTOR_CACHE: str = ""


def _load_focus_selector() -> str:
    global _FOCUS_SELECTOR_CACHE
    if not _FOCUS_SELECTOR_CACHE:
        p = Path(__file__).parent.parent.parent / "agents" / "focus_selector.md"
        if p.exists():
            _FOCUS_SELECTOR_CACHE = p.read_text(encoding="utf-8")
    return _FOCUS_SELECTOR_CACHE


def build_focus_selector_user_prompt(
    item: Dict[str, Any],
    slot_hierarchies: Dict[str, Any],
) -> str:
    lines: List[str] = []
    lines.append(f"FULL CONTRAINDICATION: {item.get('ci_text', '')}")
    lines.append("\nAGENT 2 COMPONENTS (use these to identify the focus component):")
    for text, meta in slot_hierarchies.items():
        hier = meta.get("hierarchy", "?")
        resolved = meta.get("resolved_preferred_term") or text
        lines.append(f"  [{hier}] {resolved}")
    lines.append(
        "\nCall search_snomed on the focus component text to retrieve candidates. "
        "Verify abstract/pre-coordinated status with get_logical_definition. "
        "Return final answer with <<END_JSON>>."
    )
    return "\n".join(lines)


def parse_focus_selector_output(raw: str) -> Dict[str, Any]:
    """
    Parses either a tool call or a final answer from the focus selector LLM output.

    Tool call:  {"tool": "...", "args": {...}}
    Final answer: {"focus_sctid": "...", "reasoning": "..."}<<END_JSON>>
    """
    cleaned = trim_after_end_json_token(raw, token=END_JSON_TOKEN, include_token=False)
    # Try final answer first (has END_JSON_TOKEN)
    if has_end_json_token(raw, token=END_JSON_TOKEN, require_terminal=False):
        parsed = extract_json(cleaned)
        if isinstance(parsed, dict) and "focus_sctid" in parsed:
            return parsed

    # Try tool call (no END_JSON_TOKEN — raw JSON object)
    parsed = extract_json(raw)
    if isinstance(parsed, dict) and "tool" in parsed:
        return parsed

    return {}


# ---------------------------------------------
# Prompt 07b: DIRECT MATCH AGENT (ReAct upgrade of direct_match_node)
# System prompt loaded from agents/direct_match_agent.md at first call.
# ---------------------------------------------

_DIRECT_MATCH_AGENT_CACHE: str = ""


def _load_direct_match_agent() -> str:
    global _DIRECT_MATCH_AGENT_CACHE
    if not _DIRECT_MATCH_AGENT_CACHE:
        p = Path(__file__).parent.parent.parent / "agents" / "direct_match_agent.md"
        if p.exists():
            _DIRECT_MATCH_AGENT_CACHE = p.read_text(encoding="utf-8")
    return _DIRECT_MATCH_AGENT_CACHE


def build_direct_match_agent_user_prompt(ci_text: str) -> str:
    return (
        f"CONTRAINDICATION TEXT: {ci_text}\n\n"
        "Formulate a normalized search query, call search_snomed to retrieve candidates, "
        "verify the top result with get_logical_definition, then decide direct match or not. "
        "Return final answer with <<END_JSON>>."
    )


def parse_direct_match_agent_output(raw: str) -> Dict[str, Any]:
    """
    Parses either a tool call or a final answer from the direct match agent.

    Tool call:    {"tool": "...", "args": {...}}
    Final answer: {"direct_match": true/false, "selected_id": "...", ...}<<END_JSON>>
    """
    cleaned = trim_after_end_json_token(raw, token=END_JSON_TOKEN, include_token=False)
    if has_end_json_token(raw, token=END_JSON_TOKEN, require_terminal=False):
        parsed = extract_json(cleaned)
        if isinstance(parsed, dict) and "direct_match" in parsed:
            return parsed

    parsed = extract_json(raw)
    if isinstance(parsed, dict) and "tool" in parsed:
        return parsed

    return {}


# ---------------------------------------------
# Prompt 08: PATTERN FINDER (Agent 3 — Post-Coordination Expression Agent)
# System prompt loaded from agents/postcord_agent.md at first call.
# ---------------------------------------------

_POSTCORD_AGENT_CACHE: str = ""


def _load_postcord_agent() -> str:
    global _POSTCORD_AGENT_CACHE
    if not _POSTCORD_AGENT_CACHE:
        p = Path(__file__).parent.parent.parent / "agents" / "postcord_agent.md"
        if p.exists():
            _POSTCORD_AGENT_CACHE = p.read_text(encoding="utf-8")
    return _POSTCORD_AGENT_CACHE


def build_pattern_finder_user_prompt(
    item: Dict[str, Any],
    analogues_context: Dict[str, Any],
    fills_norm: Dict[str, str],
    selected_problem_id: str,
    fills_detail: Optional[Dict[str, Any]] = None,
) -> str:
    lines: List[str] = []

    # Structural context block (from Neo4j)
    if analogues_context:
        lines.append("=== STRUCTURAL CONTEXT (from SNOMED CT ontology) ===")

        ancestors = analogues_context.get("ancestors", {})
        if ancestors:
            lines.append(f"\nANCESTORS of focus concept {selected_problem_id}:")
            for anc_id, depth in list(ancestors.items())[:8]:
                lines.append(f"  [{depth} hops] {anc_id}")

        focus_roles = analogues_context.get("focus_roles", [])
        if focus_roles:
            lines.append(f"\nROLE RELATIONSHIPS of focus concept {selected_problem_id}:")
            for r in focus_roles[:10]:
                lines.append(f"  {r.get('type_fsn','?')} → {r.get('destination_preferred_term','?')} ({r.get('destination_sctid','?')})")

        for slot_key in ("causative_agent", "severity", "clinical_course"):
            slot_roles = analogues_context.get(f"{slot_key}_roles", [])
            slot_id = fills_norm.get(slot_key, "N/A")
            if slot_roles:
                lines.append(f"\nROLE RELATIONSHIPS of {slot_key} concept {slot_id}:")
                for r in slot_roles[:6]:
                    lines.append(f"  {r.get('type_fsn','?')} → {r.get('destination_preferred_term','?')}")
    else:
        lines.append("=== STRUCTURAL CONTEXT: none available ===")

    lines.append("\n=== CURRENT MAPPING ===")
    focus_term = (fills_detail or {}).get("focus", {}).get("term", "N/A") if fills_detail else "N/A"
    lines.append(f"FOCUS: {selected_problem_id} | {focus_term}")
    for slot_key in ("causative_agent", "severity", "clinical_course"):
        slot_id = fills_norm.get(slot_key, "N/A")
        slot_term = (fills_detail or {}).get(slot_key, {}).get("term", "N/A") if fills_detail else "N/A"
        lines.append(f"{slot_key.upper()}: {slot_id} | {slot_term}")

    lines.append(f"\n=== CONTRAINDICATION TEXT ===\n{item.get('ci_text', '')}")
    lines.append("\nPropose refined post-coordinated expression. Return minified JSON followed by <<END_JSON>>.")
    return "\n".join(lines)


def parse_pattern_finder_output(text: str) -> Dict[str, Any]:
    cleaned = trim_after_end_json_token(text, token=END_JSON_TOKEN, include_token=False)
    parsed = extract_json(cleaned)
    if not isinstance(parsed, dict):
        return {}
    if "proposed_focus_id" not in parsed and "decision" not in parsed:
        return {}
    return parsed


def run_pattern_finder(
    chat_fn: Any,
    item: Dict[str, Any],
    analogues_context: Dict[str, Any],
    fills_norm: Dict[str, str],
    selected_problem_id: str,
    fills_detail: Optional[Dict[str, Any]] = None,
    *,
    max_tokens: int = 256,
    stop: Optional[List[str]] = None,
    retries: int = 1,
    retry_token_increment: int = 128,
) -> Tuple[Dict[str, Any], str]:
    system = _load_postcord_agent()
    if not system:
        return {}, ""
    user = build_pattern_finder_user_prompt(
        item, analogues_context, fills_norm, selected_problem_id, fills_detail
    )
    messages = build_message(system, user)
    last_raw = ""
    attempts = max(1, retries + 1)
    for attempt in range(attempts):
        run_tokens = max_tokens + (attempt * retry_token_increment)
        last_raw = chat_fn(messages, max_tokens=run_tokens, stop=stop)
        parsed = parse_pattern_finder_output(last_raw)
        if parsed or has_end_json_token(last_raw, token=END_JSON_TOKEN, require_terminal=False):
            return parsed, last_raw
    return {}, last_raw


# ---------------------------------------------
# Prompt 09: MRCM ATTRIBUTE MAPPER (Agent 3 v2)
# System prompt loaded from agents/postcord_agent.md at first call.
# Replaces pattern_finder when MRCM_MAPPER_ENABLED=1.
# ---------------------------------------------

def build_mrcm_mapper_user_prompt(
    item: Dict[str, Any],
    slot_hierarchies: Dict[str, Any],
    component_candidates: List[Dict[str, Any]],
    mrcm_rules: List[Dict[str, Any]],
    focus_sctid: str,
    focus_term: str,
) -> str:
    lines: List[str] = []
    lines.append(f"FOCUS CONCEPT: {focus_sctid} | {focus_term}")
    lines.append(f"CONTRAINDICATION: {item.get('ci_text', '')}")

    lines.append("\nMRCM ALLOWED ATTRIBUTES:")
    if mrcm_rules:
        for rule in mrcm_rules:
            attr_sctid = rule.get("attribute_sctid", "?")
            attr_name  = rule.get("attribute_name", "?")
            range_ecl  = rule.get("range_constraint", "")[:80]
            lines.append(f"  {attr_sctid} | {attr_name} | range: {range_ecl}")
    else:
        lines.append("  (no MRCM rules available — use standard post-coordination attributes)")

    focus_tags = {"clinical finding", "procedure", "disorder", "finding", "regime/therapy"}
    lines.append("\nNON-FOCUS COMPONENTS (from Agent 2):")
    has_non_focus = False
    for text, meta in slot_hierarchies.items():
        if meta.get("hierarchy", "").lower() not in focus_tags:
            resolved = meta.get("resolved_preferred_term") or text
            lines.append(f"  [{meta.get('hierarchy', '?')}] {resolved}")
            has_non_focus = True
    if not has_non_focus:
        lines.append("  (none identified)")

    lines.append(f"\nCANDIDATE POOL ({len(component_candidates)} candidates):")
    for i, c in enumerate(component_candidates[:15], 1):
        cid   = c.get("id", "?")
        label = c.get("label") or c.get("term", "?")
        src   = c.get("source_hierarchy", "?")
        lines.append(f"  {i}) {cid} | {label} | source_hierarchy={src}")

    lines.append(
        "\nAssign each non-focus component to an allowed MRCM attribute and select "
        "the best matching value from the candidate pool. "
        "Return minified JSON followed by <<END_JSON>>."
    )
    return "\n".join(lines)


def build_mrcm_mapper_react_user_prompt(
    item: Dict[str, Any],
    focus_sctid: str,
    focus_term: str,
    slot_hierarchies: Dict[str, Any],
) -> str:
    """Initial prompt for the ReAct mrcm_attribute_mapper loop.
    The agent discovers MRCM attributes and searches for values via tools.
    """
    lines: List[str] = []
    lines.append(f"FOCUS CONCEPT: {focus_sctid} | {focus_term}")
    lines.append(f"CONTRAINDICATION: {item.get('ci_text', '')}")

    focus_tags = {"clinical finding", "procedure", "disorder", "finding", "regime/therapy"}
    lines.append("\nNON-FOCUS COMPONENTS (from Agent 2):")
    has_non_focus = False
    for text, meta in slot_hierarchies.items():
        if meta.get("hierarchy", "").lower() not in focus_tags:
            resolved = meta.get("resolved_preferred_term") or text
            lines.append(f"  [{meta.get('hierarchy', '?')}] {resolved}")
            has_non_focus = True
    if not has_non_focus:
        lines.append("  (none identified)")

    lines.append(
        "\nStart by calling get_domain_attributes to discover valid MRCM attributes "
        "for this focus concept. Then use search_snomed to find values for each "
        "non-focus component and return refinements[]."
    )
    return "\n".join(lines)


def parse_mrcm_mapper_output(raw: str) -> Dict[str, Any]:
    """
    Parses either a tool call or a final answer from the mrcm_attribute_mapper agent.

    Tool call:    {"tool": "...", "args": {...}}
    Final answer: {"decision": "...", "refinements": [...], "confidence": ...}<<END_JSON>>
    """
    cleaned = trim_after_end_json_token(raw, token=END_JSON_TOKEN, include_token=False)
    if has_end_json_token(raw, token=END_JSON_TOKEN, require_terminal=False):
        parsed = extract_json(cleaned)
        if isinstance(parsed, dict) and ("refinements" in parsed or "decision" in parsed):
            return parsed

    # Tool call (no END_JSON_TOKEN)
    parsed = extract_json(raw)
    if isinstance(parsed, dict) and "tool" in parsed:
        return parsed

    return {}


def run_mrcm_mapper(
    chat_fn: Any,
    item: Dict[str, Any],
    slot_hierarchies: Dict[str, Any],
    component_candidates: List[Dict[str, Any]],
    mrcm_rules: List[Dict[str, Any]],
    focus_sctid: str,
    focus_term: str,
    *,
    max_tokens: int = 512,
    stop: Optional[List[str]] = None,
    retries: int = 1,
    retry_token_increment: int = 128,
) -> Tuple[Dict[str, Any], str]:
    system = _load_postcord_agent()
    if not system:
        return {}, ""
    user = build_mrcm_mapper_user_prompt(
        item, slot_hierarchies, component_candidates, mrcm_rules, focus_sctid, focus_term
    )
    messages = build_message(system, user)
    last_raw = ""
    attempts = max(1, retries + 1)
    for attempt in range(attempts):
        run_tokens = max_tokens + (attempt * retry_token_increment)
        last_raw = chat_fn(messages, max_tokens=run_tokens, stop=stop)
        parsed = parse_mrcm_mapper_output(last_raw)
        if parsed or has_end_json_token(last_raw, token=END_JSON_TOKEN, require_terminal=False):
            return parsed, last_raw
    return {}, last_raw