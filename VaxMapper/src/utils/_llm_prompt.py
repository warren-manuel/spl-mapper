from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from VaxMapper.src.llm import (
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
   you MUST ALWAYS split it into MULTIPLE atomic contraindications. whenever each item could stand alone.
   Examples:
   - Text:
     "known hypersensitivity to drug A or to any of the other ingredients in drug A, or
     with known hypersensitivity to drug B analogs, including or such as drug C"
   You MUST extract THREE separate contraindications:
   1) hypersensitivity to drug A
   2) hypersensitivity to ingredients of drug A
   3) hypersensitivity to drug B
   DO NOT add a separate contraindication for the example drug (drug C).

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
1) Review ATOMIC_LIST and verify that no item still contains unsplit coordination that should have been separated.
2) If any item still contains multiple standalone contraindications joined by "and", "or", commas, or list structure, split it further.
3) Ensure each final item is the smallest standalone contraindicated condition supported by the text.

--------------------
TASK 2: STRUCTURED OUTPUT
--------------------
For each final atomic contraindication in ATOMIC_LIST, output one JSON object with these fields:

- "ci_text":
  - The atomic contraindication text.
  - It should be grounded in the original text and contain enough context to preserve meaning.

-"contraindication_state_text":
  - A concise, normalized description of the core clinical problem (state, condition, procedure, or situation) that makes use of the drug unsafe, with all modifiers removed.
  - Normalize wording to a general clinical formulation rather than copying the original phrasing.
  - This may be a disease/disorder, clinical finding, procedure, or clinical situation.
    Do NOT return:
        - Drug or substance names.
        - Standalone modifiers such as severity, clinical course, temporality, or laboratory thresholds.
        - Remove population framing and keep only the underlying clinical state.

- "substance_text":
  - The exact substance, ingredient, product, or drug that is the causative agent for the constraindicated state.
  - If absent, set to null.

- "severity_span":
  - Exact severity wording from the text, if present.
  - If absent, set to null.

- "course_span":
  - Exact clinical course wording from the text, if present.
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
- "course_span"


If there are no contraindications in the text, return: {"items":[]}
"""

CONTRA_EXTRACT_USER_PROMPT = """
Here is the CONTRAINDICATIONS section from a vaccine SPL document:
{text}
"""

# ---------------------------------------------
# Prompt 02: MAPPING VERIFICATION
# ---------------------------------------------

DIRECT_VERIFY_SYSTEM_PROMPT = """
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
        lines.append(f"{i}) {cid} |{label}|")
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
You are a SNOMED CT minimal representation assistant for contraindications.

Your job has two parts:
1) Decide whether the contraindication can be sufficiently represented using ONLY:
   - one focus concept
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
{"post_decision":"YES|N/A","selected_focus_id":"<id or N/A>","fills":{"causative_agent":"<id or N/A>","severity":"<id or N/A>","clinical_course":"<id or N/A>"}}<<END_JSON>>
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
) -> Tuple[List[Dict[str, Any]], str]:
    messages = build_message(system_prompt, user_prompt_template.format(text=text))

    last_raw = ""
    attempts = max(1, retries + 1)
    for attempt in range(attempts):
        run_tokens = max_tokens + (attempt * retry_token_increment)
        last_raw = chat_fn(messages, max_tokens=run_tokens, stop=stop)
        items = parse_contra_extraction_output(last_raw)
        if items or has_end_json_token(last_raw, token=END_JSON_TOKEN, require_terminal=False):
            return items, last_raw

    return [], last_raw