#!/usr/bin/env python3
"""
LangChain-based contraindication runner.

Goals:
- Mirror the high-level behavior of agent_runner.py with less orchestration code.
- Make contraindication extraction fields configurable instead of hardcoded.
- Keep retrieval pluggable so BM25/FAISS/reranking and graph traversal can be added incrementally.

This first version intentionally reuses the existing SNOMED retrieval and prompt logic where
that preserves behavior, while moving LLM orchestration onto LangChain.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

try:
    from langchain_openai import AzureChatOpenAI
except ImportError:  # pragma: no cover
    AzureChatOpenAI = None

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:  # pragma: no cover
    HuggingFaceEmbeddings = None

try:
    import networkx as nx
except ImportError:  # pragma: no cover
    nx = None

from VaxMapper.src.utils.dailymed import CONTRA_Loinc, extract_section
from VaxMapper.src.utils.hyb_mapper import (
    DEFAULT_ITEM_TERM_KEYS,
    get_cached_mapper_resources,
    retrieve_candidates_for_item as retrieve_candidates_for_item_hybrid,
)
from VaxMapper.src.utils._llm_prompt import (
    DIRECT_VERIFY_SYSTEM_PROMPT,
    ROUTE_OR_FILL_SYSTEM_PROMPT,
    build_direct_verify_user_prompt,
    build_route_or_fill_user_prompt,
)
from VaxMapper.src.utils.snomed_utils import (
    DEFAULT_PREFILTER_CONTENT_TYPE,
    filter_terms_by_attribute_range,
    load_prefilter_memberships,
    load_snomed_dataframes,
)


END = "<<END_JSON>>"
DEFAULT_ATTRIBUTE_TABLE = {
    "causative_agent": 246075003,
    "severity": 246112005,
    "clinical_course": 263502005,
}
DEFAULT_EXTRACTION_FIELDS = (
    "ci_text",
    "contraindication_state_text",
    "substance_text",
    "severity_span",
    "course_span",
)

_PREFILTER_ATTR_RANGE_CACHE: Dict[str, Any] = {}
_PREFILTER_MEMBERSHIP_CACHE: Dict[str, Dict[str, Dict[int, bool]]] = {}
_PREFILTER_ECL_CACHE: Dict[Tuple[int, str], bool] = {}


@dataclass(frozen=True)
class ExtractionFieldSpec:
    name: str
    description: str
    nullable: bool = True
    exact_span: bool = False


FIELD_SPECS: Dict[str, ExtractionFieldSpec] = {
    "ci_text": ExtractionFieldSpec(
        name="ci_text",
        description="The atomic contraindication text grounded in the source text.",
        nullable=False,
        exact_span=False,
    ),
    "contraindication_state_text": ExtractionFieldSpec(
        name="contraindication_state_text",
        description="A concise normalized description of the clinical state or situation.",
        nullable=True,
    ),
    "substance_text": ExtractionFieldSpec(
        name="substance_text",
        description="The causative substance, ingredient, product, or drug if explicitly stated.",
        nullable=True,
    ),
    "severity_span": ExtractionFieldSpec(
        name="severity_span",
        description="Exact severity wording from the text, if present.",
        nullable=True,
        exact_span=True,
    ),
    "course_span": ExtractionFieldSpec(
        name="course_span",
        description="Exact clinical course wording from the text, if present.",
        nullable=True,
        exact_span=True,
    ),
    "age_constraint": ExtractionFieldSpec(
        name="age_constraint",
        description="A brief normalized age restriction if explicitly stated.",
        nullable=True,
    ),
    "population_span": ExtractionFieldSpec(
        name="population_span",
        description="Exact population wording from the text if present.",
        nullable=True,
        exact_span=True,
    ),
    "other_modifiers": ExtractionFieldSpec(
        name="other_modifiers",
        description="Short free-text summary of additional clinically relevant modifiers.",
        nullable=True,
    ),
}


def parse_json_with_end_marker(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    if END in text:
        text = text.split(END)[0].strip()
    if text.strip().startswith("```"):
        lines = [ln for ln in text.splitlines() if not ln.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except Exception:
        return None


def looks_coordinated(ci_text: str) -> bool:
    if not ci_text:
        return False
    lowered = ci_text.lower()
    return (" or " in lowered) or (" and " in lowered) or ("," in lowered and "such as" not in lowered)


def safe_sleep(backoff_s: float) -> None:
    time.sleep(backoff_s)


def candidate_label_by_id(cands: List[Dict[str, Any]], concept_id: str) -> str:
    target = str(concept_id or "")
    if not target or target == "N/A":
        return "N/A"
    for cand in cands or []:
        if str(cand.get("id")) == target:
            return str(cand.get("label") or cand.get("term") or "N/A")
    return "N/A"


def format_json_schema_snippet(field_names: Sequence[str]) -> str:
    parts = []
    for name in field_names:
        spec = FIELD_SPECS[name]
        if spec.nullable:
            parts.append(f'"{name}":"string or null"')
        else:
            parts.append(f'"{name}":"string"')
    return "{" + ",".join(parts) + "}"


def build_dynamic_extraction_system_prompt(field_names: Sequence[str]) -> str:
    field_lines = []
    for name in field_names:
        spec = FIELD_SPECS[name]
        nullable_text = "Use null if absent." if spec.nullable else "This field is required."
        span_text = " Copy exact wording when present." if spec.exact_span else ""
        field_lines.append(f'- "{name}": {spec.description} {nullable_text}{span_text}')

    schema = format_json_schema_snippet(field_names)
    return f"""
You are a biomedical NLP assistant that identifies CONTRAINDICATIONS in regulatory drug or vaccine documents.

Your job has two steps:
1) Identify every atomic contraindication in the text.
2) Return one JSON object per atomic contraindication using the requested fields.

Atomic contraindications must be aggressively split:
- Split coordinated items joined by "and", "or", commas, semicolons, or bullet-list structure when they can stand alone.
- Preserve shared context needed to keep each item meaningful.
- Do not invent clinical information not grounded in the text.

Return ONLY minified JSON followed immediately by {END}.
No markdown fences. No explanations.

Output schema:
{{"items":[{schema}]}}{END}

Requested fields for each item:
{chr(10).join(field_lines)}
""".strip()


SPLIT_SYSTEM = """You split coordinated contraindication text into ATOMIC items.

Return ONLY minified JSON followed by <<END_JSON>>. No fences. No explanations.

Schema:
{"atomic_spans":["...","..."]}<<END_JSON>>

Rules:
- Split lists joined by "or", "and", commas when each item can stand alone.
- Keep shared context phrases needed to preserve meaning.
- Do not invent info not in the text.
"""


class LangChainChatAdapter:
    def __init__(self, llm: Any):
        self.llm = llm
        self.parser = StrOutputParser()

    def invoke_text(self, system_prompt: str, user_prompt: str, *, max_tokens: int) -> str:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("user", "{user_prompt}"),
            ]
        )
        chain = prompt | self.llm.bind(max_completion_tokens=max_tokens) | self.parser
        return chain.invoke({"user_prompt": user_prompt})

    def chat(self, messages: List[Dict[str, str]], max_tokens: int = 512, **_: Any) -> str:
        normalized_messages = []
        for message in messages:
            role = str(message.get("role", "user"))
            content = str(message.get("content", ""))
            normalized_messages.append((role, content))
        prompt = ChatPromptTemplate.from_messages(normalized_messages)
        chain = prompt | self.llm.bind(max_completion_tokens=max_tokens) | self.parser
        return chain.invoke({})


class ConceptRetriever(Protocol):
    def retrieve(self, item: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        ...


@dataclass
class LangChainEmbeddingFactory:
    model_name: str
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    encode_kwargs: Dict[str, Any] = field(default_factory=dict)

    def build(self) -> Any:
        if HuggingFaceEmbeddings is None:
            raise ImportError("langchain_huggingface is required for Hugging Face embedding backends.")
        return HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs=self.model_kwargs,
            encode_kwargs=self.encode_kwargs,
        )


@dataclass
class HybridMapperRetriever:
    item_term_keys: Tuple[str, ...]

    def __post_init__(self) -> None:
        self.resources = get_cached_mapper_resources(
            snomed_source_dir=os.environ.get("SNOMED_SOURCE_DIR", "snomed_us_source"),
            concept_path=os.environ.get("SNOMED_CONCEPT_PATH"),
            description_path=os.environ.get("SNOMED_DESCRIPTION_PATH"),
            es_index=os.environ.get("MAPPER_ES_INDEX", "snomed_ct_es_index"),
            dense_index_path=os.environ.get("MAPPER_DENSE_INDEX_PATH", "results/snomed_terms_dense_test.bin"),
            model_name=os.environ.get("MAPPER_MODEL_NAME", "tavakolih/all-MiniLM-L6-v2-pubmed-full"),
            device=os.environ.get("MAPPER_DEVICE", "cuda"),
            k_dense=int(os.environ.get("MAPPER_K_DENSE", "50")),
            k_bm25=int(os.environ.get("MAPPER_K_BM25", "50")),
            k_final=int(os.environ.get("MAPPER_K_FINAL", "20")),
            rebuild_dense_index=os.environ.get("MAPPER_REBUILD_DENSE_INDEX", "").lower() in {"1", "true", "yes"},
            item_term_keys=self.item_term_keys,
        )

    def retrieve(self, item: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        return retrieve_candidates_for_item_hybrid(item, self.resources)


@dataclass
class SnomedGraphNavigator:
    graph: Any | None = None

    @classmethod
    def from_edge_csv(
        cls,
        path: str,
        source_column: str = "source",
        target_column: str = "target",
    ) -> "SnomedGraphNavigator":
        if nx is None:
            raise ImportError("networkx is required for graph traversal support.")
        graph = nx.DiGraph()
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                source = str(row.get(source_column, "")).strip()
                target = str(row.get(target_column, "")).strip()
                if source and target:
                    graph.add_edge(source, target)
        return cls(graph=graph)

    def ancestors(self, concept_id: str, max_depth: Optional[int] = None) -> List[str]:
        if not self.graph or not concept_id or concept_id not in self.graph:
            return []
        if max_depth is None:
            return sorted(nx.ancestors(self.graph, concept_id))
        visited = set()
        frontier = [(concept_id, 0)]
        while frontier:
            node, depth = frontier.pop(0)
            if depth >= max_depth:
                continue
            for parent in self.graph.predecessors(node):
                if parent not in visited:
                    visited.add(parent)
                    frontier.append((parent, depth + 1))
        return sorted(visited)

    def shortest_path(self, source: str, target: str) -> List[str]:
        if not self.graph or not source or not target:
            return []
        try:
            return list(nx.shortest_path(self.graph, source=source, target=target))
        except Exception:
            return []


def prefilter_slot_candidates(cands: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
    snomed_source_dir = os.environ.get("SNOMED_SOURCE_DIR", "snomed_us_source")
    prefilter_cache_path = os.environ.get("PREFILTER_CACHE_PATH", "").strip()
    prefilter_live_fallback = os.environ.get("PREFILTER_LIVE_FALLBACK", "1").lower() not in {"0", "false", "no"}
    prefilter_timeout = int(os.environ.get("PREFILTER_TIMEOUT", "10"))
    prefilter_retries = int(os.environ.get("PREFILTER_RETRIES", "1"))

    if snomed_source_dir not in _PREFILTER_ATTR_RANGE_CACHE:
        _PREFILTER_ATTR_RANGE_CACHE[snomed_source_dir] = load_snomed_dataframes(
            snomed_source_dir=snomed_source_dir
        )["attr_range"]
    attr_range_df = _PREFILTER_ATTR_RANGE_CACHE[snomed_source_dir]

    membership_maps: Dict[str, Dict[int, bool]] = {}
    if prefilter_cache_path:
        if prefilter_cache_path not in _PREFILTER_MEMBERSHIP_CACHE:
            _PREFILTER_MEMBERSHIP_CACHE[prefilter_cache_path] = load_prefilter_memberships(prefilter_cache_path)
        membership_maps = _PREFILTER_MEMBERSHIP_CACHE[prefilter_cache_path]

    filtered = dict(cands)
    for key in ("causative_agent", "severity", "clinical_course"):
        filtered[f"{key}_candidates"] = filter_terms_by_attribute_range(
            cands.get(f"{key}_candidates", []) or [],
            key,
            attr_range_df=attr_range_df,
            ecl_cache=_PREFILTER_ECL_CACHE,
            membership_map=membership_maps.get(key),
            live_fallback=prefilter_live_fallback,
            content_type_id=DEFAULT_PREFILTER_CONTENT_TYPE.get(key),
            timeout=prefilter_timeout,
            retries=prefilter_retries,
        )
    return filtered


def validate_postcoord_with_mrcm(selected_focus_id: str, fills: Dict[str, str]) -> Tuple[bool, List[str]]:
    fail_reasons: List[str] = []
    if not selected_focus_id or str(selected_focus_id) == "N/A":
        fail_reasons.append("FOCUS_MISSING")
    elif not str(selected_focus_id).isdigit():
        fail_reasons.append("FOCUS_NOT_NUMERIC")

    for key, value in (fills or {}).items():
        if value in (None, "", "N/A"):
            continue
        if not str(value).isdigit():
            fail_reasons.append(f"{key.upper()}_NOT_NUMERIC")
    return (len(fail_reasons) == 0), fail_reasons


@dataclass
class AgentRunConfig:
    extraction_fields: Tuple[str, ...] = DEFAULT_EXTRACTION_FIELDS
    attribute_table: Dict[str, int] = field(default_factory=lambda: dict(DEFAULT_ATTRIBUTE_TABLE))
    max_llm_tokens_short: int = 384
    max_llm_tokens_mid: int = 512
    retries: int = 2
    backoff_s: float = 1.0


class ContraLangChainAgent:
    def __init__(
        self,
        llm: LangChainChatAdapter,
        retriever: ConceptRetriever,
        cfg: AgentRunConfig,
        graph_navigator: Optional[SnomedGraphNavigator] = None,
    ):
        self.llm = llm
        self.retriever = retriever
        self.cfg = cfg
        self.graph_navigator = graph_navigator
        self.extraction_system_prompt = build_dynamic_extraction_system_prompt(cfg.extraction_fields)
        self.extract_chain = (
            {
                "user_prompt": RunnableLambda(
                    lambda payload: f"Here is the CONTRAINDICATIONS section from a vaccine SPL document:\n{payload['text']}"
                )
            }
            | ChatPromptTemplate.from_messages(
                [
                    ("system", self.extraction_system_prompt),
                    ("user", "{user_prompt}"),
                ]
            )
            | self.llm.llm.bind(max_completion_tokens=self.cfg.max_llm_tokens_mid)
            | RunnableLambda(self._extract_message_text)
        )

    @staticmethod
    def _extract_message_text(message: Any) -> str:
        if isinstance(message, str):
            return message
        if isinstance(message, AIMessage):
            return str(message.content or "")
        return str(getattr(message, "content", "") or "")

    def _call_llm_json(self, system: str, user: str, max_tokens: int) -> Tuple[Optional[Dict[str, Any]], str]:
        raw = self.llm.invoke_text(system, user, max_tokens=max_tokens)
        return parse_json_with_end_marker(raw), raw

    def extract_items_for_spl(self, spl_record: Dict[str, Any]) -> List[Dict[str, Any]]:
        contra_text = (
            spl_record.get("contra_text")
            or spl_record.get("contra_section_text")
            or spl_record.get("section_text")
            or ""
        )
        contra_text = str(contra_text).strip()
        if not contra_text:
            return []

        last_raw = ""
        for attempt in range(self.cfg.retries + 1):
            chain = self.extract_chain.with_config({"run_name": "extract_contraindications"})
            last_raw = chain.invoke({"text": contra_text})
            parsed = parse_json_with_end_marker(last_raw)
            items = parsed.get("items") if isinstance(parsed, dict) else None
            if isinstance(items, list):
                normalized = []
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    normalized_item = {}
                    for field_name in self.cfg.extraction_fields:
                        value = item.get(field_name)
                        normalized_item[field_name] = value
                    if normalized_item.get("ci_text"):
                        normalized.append(normalized_item)
                return normalized
            safe_sleep(self.cfg.backoff_s * (attempt + 1))
        return []

    def split_atomic_if_needed(self, item: Dict[str, Any]) -> List[Dict[str, Any]]:
        ci_text = str(item.get("ci_text") or "")
        if not looks_coordinated(ci_text):
            return [item]
        parsed, _raw = self._call_llm_json(SPLIT_SYSTEM, f"TEXT:\n{ci_text}", self.cfg.max_llm_tokens_short)
        atomic_spans = parsed.get("atomic_spans") if isinstance(parsed, dict) else None
        if not isinstance(atomic_spans, list) or not atomic_spans:
            return [item]
        out = []
        for span in atomic_spans:
            new_item = dict(item)
            new_item["ci_text"] = span
            out.append(new_item)
        return out

    def verify_direct_match(self, ci_text: str, focus_candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        user = build_direct_verify_user_prompt(ci_text, focus_candidates, max_n=10)
        parsed, raw = self._call_llm_json(
            DIRECT_VERIFY_SYSTEM_PROMPT,
            user,
            max_tokens=self.cfg.max_llm_tokens_short,
        )
        if not parsed:
            return {"direct_match": False, "selected_id": "N/A", "selected_term": "N/A", "raw": raw, "parse_failed": True}
        return {**parsed, "raw": raw}

    def route_or_fill(self, item: Dict[str, Any], cands: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        for required_key in DEFAULT_EXTRACTION_FIELDS:
            item.setdefault(required_key, None)
        user = build_route_or_fill_user_prompt(
            item,
            json.dumps(self.cfg.attribute_table, separators=(",", ":")),
            cands,
            max_n=10,
        )
        parsed, raw = self._call_llm_json(
            ROUTE_OR_FILL_SYSTEM_PROMPT,
            user,
            max_tokens=self.cfg.max_llm_tokens_mid,
        )
        if not parsed:
            focus_fallback = cands.get("focus_candidates", [])
            selected_focus_id = str((focus_fallback[0].get("id") if focus_fallback else "N/A"))
            return {
                "post_decision": "N/A",
                "selected_focus_id": selected_focus_id,
                "fills": {"causative_agent": "N/A", "severity": "N/A", "clinical_course": "N/A"},
                "raw": raw,
                "parse_failed": True,
            }
        return {**parsed, "raw": raw}

    def maybe_add_graph_trace(self, selected_focus_id: str) -> Dict[str, Any]:
        if not self.graph_navigator or selected_focus_id in ("", "N/A"):
            return {}
        return {
            "focus_ancestors": self.graph_navigator.ancestors(str(selected_focus_id), max_depth=2)[:25],
        }

    def process_item(self, spl_set_id: str, item: Dict[str, Any]) -> Dict[str, Any]:
        atomic_items = self.split_atomic_if_needed(item)
        results = []

        for ai in atomic_items:
            ci_text = str(ai.get("ci_text") or "")
            item_index = ai.get("item_index")
            cands = self.retriever.retrieve(ai)

            direct = self.verify_direct_match(ci_text, cands.get("direct_candidates", cands.get("focus_candidates", [])))
            if direct.get("direct_match") is True and direct.get("selected_id") not in (None, "", "N/A"):
                results.append(
                    {
                        "SPL_SET_ID": spl_set_id,
                        "item_index": item_index,
                        "query_text": ci_text,
                        "status": "DIRECT",
                        "selected_id": direct["selected_id"],
                        "selected_term": direct.get("selected_term", "N/A"),
                        "trace": {"direct_verify": direct},
                        "extracted_item": ai,
                    }
                )
                continue

            cands_pref = prefilter_slot_candidates(cands)
            route_fill = self.route_or_fill(ai, cands_pref)
            selected_focus_id = str(route_fill.get("selected_focus_id", "N/A"))
            fills = route_fill.get("fills", {}) or {}
            post_decision = str(route_fill.get("post_decision", "N/A"))

            fills_norm: Dict[str, str] = {}
            for key in ("causative_agent", "severity", "clinical_course"):
                value = fills.get(key, "N/A")
                if isinstance(value, dict):
                    value = value.get("id", "N/A")
                fills_norm[key] = str(value)

            fills_detail = {
                "causative_agent": {
                    "id": fills_norm.get("causative_agent", "N/A"),
                    "term": candidate_label_by_id(
                        cands_pref.get("causative_agent_candidates", []),
                        fills_norm.get("causative_agent", "N/A"),
                    ),
                },
                "severity": {
                    "id": fills_norm.get("severity", "N/A"),
                    "term": candidate_label_by_id(cands_pref.get("severity_candidates", []), fills_norm.get("severity", "N/A")),
                },
                "clinical_course": {
                    "id": fills_norm.get("clinical_course", "N/A"),
                    "term": candidate_label_by_id(
                        cands_pref.get("clinical_course_candidates", []),
                        fills_norm.get("clinical_course", "N/A"),
                    ),
                },
            }
            selected_focus_term = candidate_label_by_id(cands_pref.get("focus_candidates", []), selected_focus_id)

            ok, fail_reasons = validate_postcoord_with_mrcm(selected_focus_id, fills_norm)
            if not ok:
                if selected_focus_id != "N/A" and not selected_focus_id.isdigit():
                    selected_focus_id = "N/A"
                    selected_focus_term = "N/A"
                for key in ("causative_agent", "severity", "clinical_course"):
                    if fills_detail[key]["id"] != "N/A" and not str(fills_detail[key]["id"]).isdigit():
                        fills_detail[key]["id"] = "N/A"
                        fills_detail[key]["term"] = "N/A"
                        fills_norm[key] = "N/A"
                post_decision = "N/A"

            ax_pairs = []
            for key, attr_id in self.cfg.attribute_table.items():
                value = fills_norm.get(key, "N/A")
                if value != "N/A":
                    ax_pairs.append(f"{attr_id}={value}")
            expression = f"{selected_focus_id}:{{{','.join(ax_pairs)}}}" if ax_pairs else selected_focus_id

            results.append(
                {
                    "SPL_SET_ID": spl_set_id,
                    "item_index": item_index,
                    "query_text": ci_text,
                    "status": "POSTCOORD" if post_decision == "YES" else "MINIMAL",
                    "post_decision": post_decision,
                    "selected_focus_id": selected_focus_id,
                    "selected_focus_term": selected_focus_term,
                    "fills": fills_detail,
                    "expression": expression,
                    "trace": {
                        "direct_verify": direct,
                        "route_or_fill": route_fill,
                        "validation": {"ok": ok, "fail_reasons": fail_reasons},
                        "graph": self.maybe_add_graph_trace(selected_focus_id),
                    },
                    "extracted_item": ai,
                }
            )
        return {"item_results": results}

    def process_spl(self, spl_record: Dict[str, Any]) -> Dict[str, Any]:
        spl_set_id = spl_record.get("SPL_SET_ID") or spl_record.get("spl_set_id") or str(uuid.uuid4())
        spl_context = dict(spl_record)
        spl_context["SPL_SET_ID"] = spl_set_id

        contra_text = (
            spl_context.get("contra_text")
            or spl_context.get("contra_section_text")
            or spl_context.get("section_text")
            or ""
        )
        if not str(contra_text).strip():
            try:
                contra_section = extract_section(str(spl_set_id), [CONTRA_Loinc])
            except Exception as exc:
                return {
                    "SPL_SET_ID": spl_set_id,
                    "n_items_in": 0,
                    "n_items_out": 0,
                    "results": [],
                    "contra_section_found": False,
                    "error": f"extract_section_failed: {exc}",
                }
            section_payload = (contra_section.get("sections") or {}).get(CONTRA_Loinc, {})
            spl_context["contra_section_text"] = section_payload.get("section_text") or ""
            spl_context["contra_section_xml"] = section_payload.get("section_xml")
            spl_context["product_name"] = contra_section.get("product_name")
            spl_context["contra_section_found"] = bool(section_payload.get("section_text"))
        else:
            spl_context["contra_section_text"] = str(contra_text).strip()
            spl_context["contra_section_found"] = True

        items = self.extract_items_for_spl(spl_context)
        all_results: List[Dict[str, Any]] = []
        for item_index, item in enumerate(items):
            indexed_item = dict(item)
            indexed_item["item_index"] = item_index
            out = self.process_item(spl_set_id, indexed_item)
            all_results.extend(out["item_results"])

        return {
            "SPL_SET_ID": spl_set_id,
            "product_name": spl_context.get("product_name"),
            "contra_section_found": bool(spl_context.get("contra_section_found")),
            "contra_section_text": spl_context.get("contra_section_text"),
            "n_items_in": len(items),
            "n_items_out": len(all_results),
            "results": all_results,
            "extraction_fields": list(self.cfg.extraction_fields),
        }


AGG_CSV_COLUMNS = [
    "SPL_SET_ID",
    "item_index",
    "query_text",
    "mapping_source",
    "final_concept_id",
    "final_concept_term",
    "postcoord_expression",
    "causative_agent_id",
    "causative_agent_term",
    "severity_id",
    "severity_term",
    "clinical_course_id",
    "clinical_course_term",
]


def aggregate_result_item(result: Dict[str, Any]) -> Dict[str, Any]:
    status = str(result.get("status", ""))
    fills = result.get("fills") or {}

    def fill_value(attr_key: str, field_name: str) -> str:
        return str(((fills.get(attr_key) or {}).get(field_name)) or "N/A")

    if status == "DIRECT":
        final_concept_id = str(result.get("selected_id", "N/A"))
        final_concept_term = str(result.get("selected_term", "N/A"))
        return {
            "SPL_SET_ID": str(result.get("SPL_SET_ID", "")),
            "item_index": int(result.get("item_index", 0)),
            "query_text": str(result.get("query_text", "")),
            "mapping_source": "verified",
            "final_concept_id": final_concept_id,
            "final_concept_term": final_concept_term,
            "postcoord_expression": final_concept_id,
            "attributes": {},
        }

    if status in {"POSTCOORD", "MINIMAL"}:
        attributes = {}
        for key in ("causative_agent", "severity", "clinical_course"):
            value_id = fill_value(key, "id")
            if value_id == "N/A":
                continue
            attributes[key] = {
                "value_id": value_id,
                "value_term": fill_value(key, "term"),
            }
        return {
            "SPL_SET_ID": str(result.get("SPL_SET_ID", "")),
            "item_index": int(result.get("item_index", 0)),
            "query_text": str(result.get("query_text", "")),
            "mapping_source": "postcoord" if status == "POSTCOORD" else "minimal",
            "final_concept_id": str(result.get("selected_focus_id", "N/A")),
            "final_concept_term": str(result.get("selected_focus_term", "N/A")),
            "postcoord_expression": str(result.get("expression", "N/A")),
            "attributes": attributes,
        }

    return {
        "SPL_SET_ID": str(result.get("SPL_SET_ID", "")),
        "item_index": int(result.get("item_index", 0)),
        "query_text": str(result.get("query_text", "")),
        "mapping_source": status.lower() if status else "unmapped",
        "final_concept_id": "N/A",
        "final_concept_term": "N/A",
        "postcoord_expression": "N/A",
        "attributes": {},
    }


def aggregated_item_to_csv_row(item: Dict[str, Any]) -> Dict[str, Any]:
    attributes = item.get("attributes") or {}

    def value(attr_key: str, field_name: str) -> str:
        return str(((attributes.get(attr_key) or {}).get(field_name)) or "")

    return {
        "SPL_SET_ID": item["SPL_SET_ID"],
        "item_index": item["item_index"],
        "query_text": item.get("query_text", ""),
        "mapping_source": item.get("mapping_source", ""),
        "final_concept_id": item.get("final_concept_id", ""),
        "final_concept_term": item.get("final_concept_term", ""),
        "postcoord_expression": item.get("postcoord_expression", ""),
        "causative_agent_id": value("causative_agent", "value_id"),
        "causative_agent_term": value("causative_agent", "value_term"),
        "severity_id": value("severity", "value_id"),
        "severity_term": value("severity", "value_term"),
        "clinical_course_id": value("clinical_course", "value_id"),
        "clinical_course_term": value("clinical_course", "value_term"),
    }


def aggregate_agent_results(run_rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        spl_set_id = str(row.get("SPL_SET_ID", ""))
        for result in row.get("results") or []:
            grouped[spl_set_id].append(aggregate_result_item(result))

    aggregated_rows: List[Dict[str, Any]] = []
    csv_rows: List[Dict[str, Any]] = []
    for spl_set_id in sorted(grouped):
        ordered_items = sorted(grouped[spl_set_id], key=lambda item: int(item.get("item_index", 0)))
        aggregated_rows.append({"SPL_SET_ID": spl_set_id, "item_count": len(ordered_items), "items": ordered_items})
        csv_rows.extend(aggregated_item_to_csv_row(item) for item in ordered_items)
    return aggregated_rows, csv_rows


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv_rows(path: str, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_spl_records_from_file(path: str) -> List[Dict[str, Any]]:
    if not path:
        raise ValueError("--spl-list is required")

    lower = path.lower()
    if lower.endswith(".csv"):
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return []
            fieldnames = {name.strip(): name for name in reader.fieldnames if name}
            spl_key = (
                fieldnames.get("SPL_SET_ID")
                or fieldnames.get("spl_set_id")
                or fieldnames.get("setid")
                or fieldnames.get("SETID")
            )
            if spl_key is None:
                raise ValueError(
                    f"CSV must contain one of: SPL_SET_ID, spl_set_id, setid, SETID. Found: {reader.fieldnames}"
                )
            records = []
            for row in reader:
                spl_set_id = str(row.get(spl_key, "")).strip()
                if spl_set_id:
                    records.append({"SPL_SET_ID": spl_set_id})
            return records

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            spl_set_id = line.strip()
            if spl_set_id and not spl_set_id.startswith("#"):
                records.append({"SPL_SET_ID": spl_set_id})
    return records


def parse_field_list(value: str) -> Tuple[str, ...]:
    fields = tuple(dict.fromkeys(part.strip() for part in value.split(",") if part.strip()))
    if not fields:
        raise ValueError("At least one extraction field is required.")
    unknown = [field for field in fields if field not in FIELD_SPECS]
    if unknown:
        raise ValueError(f"Unsupported extraction fields: {unknown}. Supported: {sorted(FIELD_SPECS)}")
    return fields


def build_langchain_llm() -> LangChainChatAdapter:
    if AzureChatOpenAI is None:
        raise ImportError("langchain_openai is required. Install it to use this runner.")
    llm = AzureChatOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        temperature=1.0,
        timeout=30,
    )
    return LangChainChatAdapter(llm)


def build_graph_navigator(graph_edges_path: str) -> Optional[SnomedGraphNavigator]:
    if not graph_edges_path:
        return None
    return SnomedGraphNavigator.from_edge_csv(graph_edges_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spl-list", required=True, help="Path to text or CSV file containing SPL_SET_ID values.")
    parser.add_argument("--out-jsonl", default="langchain_agent_results.jsonl")
    parser.add_argument("--aggregated-jsonl", default="langchain_aggregated_hits.jsonl")
    parser.add_argument("--aggregated-csv", default="langchain_aggregated_hits.csv")
    parser.add_argument(
        "--extract-fields",
        default=",".join(DEFAULT_EXTRACTION_FIELDS),
        help=f"Comma-separated extraction fields. Supported: {','.join(sorted(FIELD_SPECS))}",
    )
    parser.add_argument(
        "--retrieval-item-keys",
        default=",".join(DEFAULT_ITEM_TERM_KEYS),
        help="Comma-separated item keys to send into hybrid retrieval.",
    )
    parser.add_argument(
        "--snomed-graph-edges",
        default="",
        help="Optional CSV edge list for NetworkX-based SNOMED traversal.",
    )
    args = parser.parse_args()

    extraction_fields = parse_field_list(args.extract_fields)
    retrieval_item_keys = tuple(part.strip() for part in args.retrieval_item_keys.split(",") if part.strip())

    llm = build_langchain_llm()
    retriever = HybridMapperRetriever(item_term_keys=retrieval_item_keys)
    graph_navigator = build_graph_navigator(args.snomed_graph_edges)

    agent = ContraLangChainAgent(
        llm=llm,
        retriever=retriever,
        cfg=AgentRunConfig(extraction_fields=extraction_fields),
        graph_navigator=graph_navigator,
    )

    spl_records = load_spl_records_from_file(args.spl_list)
    run_rows: List[Dict[str, Any]] = []
    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for spl in spl_records:
            result = agent.process_spl(spl)
            run_rows.append(result)
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    aggregated_rows, csv_rows = aggregate_agent_results(run_rows)
    write_jsonl(args.aggregated_jsonl, aggregated_rows)
    write_csv_rows(args.aggregated_csv, csv_rows, AGG_CSV_COLUMNS)

    print(f"Wrote {len(spl_records)} SPL results to: {args.out_jsonl}")
    print(f"Wrote {len(aggregated_rows)} aggregated SPL groups to: {args.aggregated_jsonl}")
    print(f"Wrote {len(csv_rows)} aggregated item rows to: {args.aggregated_csv}")


if __name__ == "__main__":
    main()
