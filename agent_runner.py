#!/usr/bin/env python3
"""
agent_runner.py

Option A: Plain Python orchestrator (agentic controller) that processes ONE SPL at a time.
- Maintains state for each SPL (reduces error propagation).
- Uses Azure GPT-5 for bounded decisions:
  1) atomic split (optional)
  2) direct match verify
  3) route-or-fill for minimal representation + post_decision
- Uses deterministic tools for:
  - candidate retrieval (BM25+dense RRF)
  - MRCM domain/attribute/range lookup
  - Snowstorm ECL membership/expansion
  - validation + fallback

"""

from __future__ import annotations
import argparse
import csv
import os
import json
import subprocess
import time
import uuid
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable

from openai import AzureOpenAI

from VaxMapper.src.utils.dailymed import CONTRA_Loinc, extract_section
from VaxMapper.src.utils.hyb_mapper import (
    DEFAULT_ITEM_TERM_KEYS,
    get_cached_mapper_resources,
    retrieve_candidates_for_item as retrieve_candidates_for_item_hybrid,
)
from VaxMapper.src.utils._llm_prompt import (
    DIRECT_VERIFY_SYSTEM_PROMPT,
    build_direct_verify_user_prompt,
    build_route_or_fill_user_prompt,
    extract_contraindication_items,
    ROUTE_OR_FILL_SYSTEM_PROMPT,
)
from VaxMapper.src.utils.snomed_utils import (
    DEFAULT_PREFILTER_CONTENT_TYPE,
    filter_terms_by_attribute_range,
    load_prefilter_memberships,
    load_snomed_dataframes,
)


# -----------------------------
# Azure OpenAI client (minimal, requests-based)
# -----------------------------
@dataclass
class AzureOpenAIConfig:
    endpoint: str                 # e.g., "https://<resource>.openai.azure.com"
    api_key: str
    deployment: str               # e.g., "gpt-5"
    api_version: str              # set to your supported version, e.g. "2024-02-15-preview"
    timeout_s: int = 30
    # max_retries: int = 2 # not supported by AzureOpenAI client, implement retries in caller logic if needed

class AzureChatLLM:
    """
    Minimal Azure OpenAI SDK wrapper that preserves the existing chat(...) interface.
    """
    def __init__(self, cfg: AzureOpenAIConfig):
        self.cfg = cfg
        self.client = AzureOpenAI(
            api_key=cfg.api_key,
            api_version=cfg.api_version,
            azure_endpoint=cfg.endpoint,
            timeout=cfg.timeout_s,
        )

    def chat(self, messages: List[Dict[str, str]], max_tokens: int = 512, temperature: float = 1.0,
             stop: Optional[List[str]] = None) -> str:
        payload: Dict[str, Any] = {
            "model": self.cfg.deployment,
            "messages": messages,
            "max_completion_tokens": max_tokens,
            "temperature": temperature,
        }

        response = self.client.chat.completions.create(**payload)
        content = response.choices[0].message.content
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if hasattr(part, "text") and getattr(part, "text", None):
                    text_parts.append(part.text)
            return "".join(text_parts)
        return content or ""


# -----------------------------
# Shared helpers
# -----------------------------
END = "<<END_JSON>>"
_PREFILTER_ATTR_RANGE_CACHE: Dict[str, Any] = {}
_PREFILTER_MEMBERSHIP_CACHE: Dict[str, Dict[str, Dict[int, bool]]] = {}
_PREFILTER_ECL_CACHE: Dict[Tuple[int, str], bool] = {}

def parse_json_with_end_marker(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    if END in text:
        text = text.split(END)[0].strip()
    # strip accidental fences
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
    t = ci_text.lower()
    # heuristic triggers for under-splitting
    return (" or " in t) or (" and " in t) or ("," in t and "such as" not in t)

def safe_sleep(backoff_s: float):
    time.sleep(backoff_s)


def candidate_label_by_id(cands: List[Dict[str, Any]], concept_id: str) -> str:
    target = str(concept_id or "")
    if not target or target == "N/A":
        return "N/A"
    for cand in cands or []:
        if str(cand.get("id")) == target:
            return str(cand.get("label") or cand.get("term") or "N/A")
    return "N/A"


# -----------------------------
# Hook signatures you must wire to your code
# -----------------------------
def extract_items_for_spl(
    spl_record: Dict[str, Any],
    *,
    chat_fn: Optional[Callable[..., str]] = None,
    max_tokens: int = 512,
    stop: Optional[List[str]] = None,
    retries: int = 1,
    retry_token_increment: int = 256,
) -> List[Dict[str, Any]]:
    """
    Return list of extracted contraindication items for ONE SPL (Step 1 output items[]).
    Must return list of dicts with keys:
      ci_text, contraindication_state_text, substance_text, severity_span, clinical_course_span
    """
    if chat_fn is None:
        raise ValueError("extract_items_for_spl requires chat_fn")

    contra_text = (
        spl_record.get("contra_text")
        or spl_record.get("contra_section_text")
        or spl_record.get("section_text")
        or ""
    )
    contra_text = str(contra_text).strip()
    if not contra_text:
        return []

    items, _raw = extract_contraindication_items(
        chat_fn,
        contra_text,
        max_tokens=max_tokens,
        stop=stop,
        retries=retries,
        retry_token_increment=retry_token_increment,
    )
    return items

def retrieve_candidates_for_item(item: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Replace with hyb_mapper.py output for this item.
    Return candidates dict with keys:
      focus_candidates, causative_agent_candidates, severity_candidates, clinical_course_candidates
    Each is list[{id,label,score,...}]
    """
    # TODO(step8): Add timeout/retry handling around mapper resource loading and retrieval.
    # TODO(step8): Add fallback behavior when Elasticsearch, FAISS assets, or retrieval resources are unavailable.
    # TODO(step8): Add structured trace logging for retrieval attempts/failures at SPL/item granularity.
    resources = get_cached_mapper_resources(
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
        item_term_keys=DEFAULT_ITEM_TERM_KEYS,
    )
    return retrieve_candidates_for_item_hybrid(item, resources)

def prefilter_slot_candidates(cands: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Replace with your prefilter.py logic using ECL membership for each attribute.
    Input/Output same as retrieve_candidates_for_item.
    """
    # TODO(step8): Add timeout/retry handling for Snowstorm-backed live ECL checks.
    # TODO(step8): Add fallback behavior when prefilter cache files or Snowstorm are unavailable.
    # TODO(step8): Add structured trace logging for prefilter cache hits/misses and live validation failures.
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
    filtered["causative_agent_candidates"] = filter_terms_by_attribute_range(
        cands.get("causative_agent_candidates", []) or [],
        "causative_agent",
        attr_range_df=attr_range_df,
        ecl_cache=_PREFILTER_ECL_CACHE,
        membership_map=membership_maps.get("causative_agent"),
        live_fallback=prefilter_live_fallback,
        content_type_id=DEFAULT_PREFILTER_CONTENT_TYPE.get("causative_agent"),
        timeout=prefilter_timeout,
        retries=prefilter_retries,
    )
    filtered["severity_candidates"] = filter_terms_by_attribute_range(
        cands.get("severity_candidates", []) or [],
        "severity",
        attr_range_df=attr_range_df,
        ecl_cache=_PREFILTER_ECL_CACHE,
        membership_map=membership_maps.get("severity"),
        live_fallback=prefilter_live_fallback,
        content_type_id=DEFAULT_PREFILTER_CONTENT_TYPE.get("severity"),
        timeout=prefilter_timeout,
        retries=prefilter_retries,
    )
    filtered["clinical_course_candidates"] = filter_terms_by_attribute_range(
        cands.get("clinical_course_candidates", []) or [],
        "clinical_course",
        attr_range_df=attr_range_df,
        ecl_cache=_PREFILTER_ECL_CACHE,
        membership_map=membership_maps.get("clinical_course"),
        live_fallback=prefilter_live_fallback,
        content_type_id=DEFAULT_PREFILTER_CONTENT_TYPE.get("clinical_course"),
        timeout=prefilter_timeout,
        retries=prefilter_retries,
    )
    return filtered

def validate_postcoord_with_mrcm(selected_focus_id: str, fills: Dict[str, str]) -> Tuple[bool, List[str]]:
    """
    Lightweight validation only.

    Full MRCM/expression validation is intentionally skipped for now because
    slot candidates are already range-prefiltered before the LLM sees them.
    This guard only catches obviously malformed outputs.
    """
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

# -----------------------------
# Agent controller
# -----------------------------
@dataclass
class AgentRunConfig:
    attribute_table: Dict[str, int] = field(default_factory=lambda: {
        "causative_agent": 246075003,
        "severity": 246112005,
        "clinical_course": 263502005,
    })
    max_llm_tokens_short: int = 384
    max_llm_tokens_mid: int = 512
    retries: int = 2
    backoff_s: float = 1.0
    stop: List[str] = field(default_factory=lambda: [END])

class ContraAgent:
    def __init__(self, llm: AzureChatLLM, cfg: AgentRunConfig):
        self.llm = llm
        self.cfg = cfg

    def _call_llm_json(self, system: str, user: str, max_tokens: int) -> Tuple[Optional[Dict[str, Any]], str]:
        raw = self.llm.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=max_tokens,
            temperature=1.0,
            stop=self.cfg.stop
        )
        parsed = parse_json_with_end_marker(raw)
        return parsed, raw

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
                "fills": {
                    "causative_agent": "N/A",
                    "severity": "N/A",
                    "clinical_course": "N/A",
                },
                "raw": raw,
                "parse_failed": True,
            }
        return {**parsed, "raw": raw}

    def process_item(self, spl_set_id: str, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process ONE contraindication item agentically with retries/fallbacks.
        """
        # TODO(step8): Add item-level deterministic dependency tracing so retrieval/prefilter failures
        # are captured alongside the existing LLM trace.
        # 0) Use extracted contraindication items as-is.
        atomic_items = [item]

        results = []
        for ai in atomic_items:
            ci_text = ai.get("ci_text", "")
            item_index = ai.get("item_index")
            cands = retrieve_candidates_for_item(ai)

            # 1) Direct match verify
            direct = self.verify_direct_match(ci_text, cands.get("direct_candidates", cands.get("focus_candidates", [])))
            if direct.get("direct_match") is True and direct.get("selected_id") not in (None, "", "N/A"):
                results.append({
                    "SPL_SET_ID": spl_set_id,
                    "item_index": item_index,
                    "query_text": ci_text,
                    "status": "DIRECT",
                    "selected_id": direct["selected_id"],
                    "selected_term": direct.get("selected_term", "N/A"),
                    "trace": {"direct_verify": direct},
                    "extracted_item": ai,
                })
                continue

            # 2) Prefilter for slot attributes (ECL membership)
            cands_pref = prefilter_slot_candidates(cands)

            # 3) Route-or-fill for minimal representation
            route_fill = self.route_or_fill(ai, cands_pref)
            selected_focus_id = str(route_fill.get("selected_focus_id", "N/A"))
            fills = route_fill.get("fills", {}) or {}
            post_decision = str(route_fill.get("post_decision", "N/A"))

            # normalize fills to string ids
            fills_norm = {}
            for k in ("causative_agent", "severity", "clinical_course"):
                v = fills.get(k, "N/A")
                if isinstance(v, dict):
                    v = v.get("id", "N/A")
                fills_norm[k] = str(v)

            fills_detail = {
                "causative_agent": {
                    "id": fills_norm.get("causative_agent", "N/A"),
                    "term": candidate_label_by_id(cands_pref.get("causative_agent_candidates", []), fills_norm.get("causative_agent", "N/A")),
                },
                "severity": {
                    "id": fills_norm.get("severity", "N/A"),
                    "term": candidate_label_by_id(cands_pref.get("severity_candidates", []), fills_norm.get("severity", "N/A")),
                },
                "clinical_course": {
                    "id": fills_norm.get("clinical_course", "N/A"),
                    "term": candidate_label_by_id(cands_pref.get("clinical_course_candidates", []), fills_norm.get("clinical_course", "N/A")),
                },
            }
            selected_focus_term = candidate_label_by_id(cands_pref.get("focus_candidates", []), selected_focus_id)

            # 4) Lightweight validation / normalization
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

            # 5) Compose expression (simple group 0)
            # You can include only filled slots != N/A
            ax_pairs = []
            for key, attr_id in self.cfg.attribute_table.items():
                val = fills_norm.get(key, "N/A")
                if val != "N/A":
                    ax_pairs.append(f"{attr_id}={val}")
            expr = f"{selected_focus_id}:{{{','.join(ax_pairs)}}}" if ax_pairs else selected_focus_id

            results.append({
                "SPL_SET_ID": spl_set_id,
                "item_index": item_index,
                "query_text": ci_text,
                "status": "POSTCOORD" if post_decision == "YES" else "MINIMAL",
                "post_decision": post_decision,
                "selected_focus_id": selected_focus_id,
                "selected_focus_term": selected_focus_term,
                "fills": fills_detail,
                "expression": expr,
                "trace": {
                    "direct_verify": direct,
                    "route_or_fill": route_fill,
                    "validation": {"ok": ok, "fail_reasons": fail_reasons},
                },
                "extracted_item": ai,
            })

        # If atomic split produced multiple results, return as a list wrapper or flatten upstream
        return {"item_results": results}

    def process_spl(self, spl_record: Dict[str, Any]) -> Dict[str, Any]:
        spl_set_id = spl_record.get("SPL_SET_ID") or spl_record.get("spl_set_id") or str(uuid.uuid4())
        # TODO(step8): Add SPL-level trace logging for DailyMed fetch, retrieval initialization,
        # cache usage, and deterministic dependency failures/retries.
        spl_context = dict(spl_record)
        spl_context["SPL_SET_ID"] = spl_set_id

        contra_section = None
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

        items = extract_items_for_spl(
            spl_context,
            chat_fn=self.llm.chat,
            max_tokens=self.cfg.max_llm_tokens_mid,
            stop=self.cfg.stop,
            retries=self.cfg.retries,
        )
        all_results = []

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
        }


# -----------------------------
# Aggregation + evaluation
# -----------------------------
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

    def fill_value(attr_key: str, field: str) -> str:
        return str(((fills.get(attr_key) or {}).get(field)) or "N/A")

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

    def val(attr_key: str, field: str) -> str:
        return str(((attributes.get(attr_key) or {}).get(field)) or "")

    return {
        "SPL_SET_ID": item["SPL_SET_ID"],
        "item_index": item["item_index"],
        "query_text": item.get("query_text", ""),
        "mapping_source": item.get("mapping_source", ""),
        "final_concept_id": item.get("final_concept_id", ""),
        "final_concept_term": item.get("final_concept_term", ""),
        "postcoord_expression": item.get("postcoord_expression", ""),
        "causative_agent_id": val("causative_agent", "value_id"),
        "causative_agent_term": val("causative_agent", "value_term"),
        "severity_id": val("severity", "value_id"),
        "severity_term": val("severity", "value_term"),
        "clinical_course_id": val("clinical_course", "value_id"),
        "clinical_course_term": val("clinical_course", "value_term"),
    }


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
        aggregated_rows.append(
            {
                "SPL_SET_ID": spl_set_id,
                "item_count": len(ordered_items),
                "items": ordered_items,
            }
        )
        csv_rows.extend(aggregated_item_to_csv_row(item) for item in ordered_items)
    return aggregated_rows, csv_rows


def evaluate_aggregated_predictions(
    pred_csv: str,
    gold_csv: str,
    out_json: str,
    out_details_csv: str,
    *,
    discard_na_gold_expression: bool = True,
    st_model_id: Optional[str] = None,
    st_device: str = "cuda",
    alpha: float = 0.85,
    beta: float = 0.15,
    min_pair_score: float = 0.5,
    require_semantic: bool = True,
    decoupled: bool = True,
) -> Dict[str, Any]:
    resolved_st_model_id = st_model_id or os.environ.get("MAPPER_MODEL_NAME", "tavakolih/all-MiniLM-L6-v2-pubmed-full")
    cmd = [
        "python3",
        "evaluate_agg_results_2.py",
        "--pred-csv", pred_csv,
        "--gold-csv", gold_csv,
        "--out-json", out_json,
        "--out-details-csv", out_details_csv,
        "--st-model-id", resolved_st_model_id,
        "--st-device", st_device,
        "--alpha", str(alpha),
        "--beta", str(beta),
        "--min-pair-score", str(min_pair_score),
    ]
    if discard_na_gold_expression:
        cmd.append("--discard-na-gold-expression")
    if require_semantic:
        cmd.append("--require-semantic")
    if not decoupled:
        cmd.append("--no-decoupled")

    subprocess.run(cmd, check=True)
    with open(out_json, "r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------------
# Minimal CLI runner
# -----------------------------
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

            records: List[Dict[str, Any]] = []
            for row in reader:
                spl_set_id = str(row.get(spl_key, "")).strip()
                if spl_set_id:
                    records.append({"SPL_SET_ID": spl_set_id})
            return records

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            spl_set_id = line.strip()
            if not spl_set_id or spl_set_id.startswith("#"):
                continue
            records.append({"SPL_SET_ID": spl_set_id})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spl-list",
        required=True,
        help="Path to text or CSV file containing SPL_SET_ID values.",
    )
    parser.add_argument(
        "--out-jsonl",
        default=os.environ.get("AGENT_OUT_JSONL", "agent_results.jsonl"),
        help="Path to write raw per-SPL agent JSONL results.",
    )
    parser.add_argument(
        "--aggregated-jsonl",
        default=os.environ.get("AGGREGATED_OUT_JSONL", "aggregated_hits.jsonl"),
        help="Path to write aggregated JSONL results.",
    )
    parser.add_argument(
        "--aggregated-csv",
        default=os.environ.get("AGGREGATED_OUT_CSV", "aggregated_hits.csv"),
        help="Path to write aggregated CSV results.",
    )
    parser.add_argument(
        "--gold-csv",
        default=os.environ.get("AGENT_GOLD_CSV", ""),
        help="Optional gold CSV path for evaluation.",
    )
    parser.add_argument(
        "--eval-json",
        default=os.environ.get("AGENT_EVAL_JSON", "eval_metrics.json"),
        help="Path to write evaluation metrics JSON when --gold-csv is provided.",
    )
    parser.add_argument(
        "--eval-details-csv",
        default=os.environ.get("AGENT_EVAL_DETAILS_CSV", "eval_details.csv"),
        help="Path to write evaluation details CSV when --gold-csv is provided.",
    )
    args = parser.parse_args()

    # Azure config from env
    cfg = AzureOpenAIConfig(
        endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],     # e.g. gpt-5 deployment name
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],   # set to your supported version
    )
    llm = AzureChatLLM(cfg)
    agent = ContraAgent(llm, AgentRunConfig())

    spl_records = load_spl_records_from_file(args.spl_list)
    run_rows: List[Dict[str, Any]] = []

    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for spl in spl_records:
            res = agent.process_spl(spl)
            run_rows.append(res)
            f.write(json.dumps(res, ensure_ascii=False) + "\n")

    aggregated_rows, csv_rows = aggregate_agent_results(run_rows)
    write_jsonl(args.aggregated_jsonl, aggregated_rows)
    write_csv_rows(args.aggregated_csv, csv_rows, AGG_CSV_COLUMNS)

    print(f"Wrote {len(spl_records)} SPL results to: {args.out_jsonl}")
    print(f"Wrote {len(aggregated_rows)} aggregated SPL groups to: {args.aggregated_jsonl}")
    print(f"Wrote {len(csv_rows)} aggregated item rows to: {args.aggregated_csv}")

    if args.gold_csv:
        metrics = evaluate_aggregated_predictions(
            pred_csv=args.aggregated_csv,
            gold_csv=args.gold_csv,
            out_json=args.eval_json,
            out_details_csv=args.eval_details_csv,
        )
        print(json.dumps(metrics, indent=2))
        print(f"Wrote evaluation metrics to: {args.eval_json}")
        print(f"Wrote evaluation details to: {args.eval_details_csv}")


if __name__ == "__main__":
    main()
