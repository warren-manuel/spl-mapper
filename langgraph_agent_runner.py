#!/usr/bin/env python3
"""
LangGraph-based contraindication runner with pluggable LLM backends.

This module aims for behavioral parity with agent_runner.py while moving the
orchestration layer onto LangGraph.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, TypedDict, Union
from dotenv import load_dotenv


def _early_resolve_runner_cuda_visible_devices() -> Optional[str]:
    for env_name in ("RUNNER_CUDA_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "HF_CUDA_VISIBLE_DEVICES"):
        value = os.environ.get(env_name, "").strip()
        if value:
            return value
    return None


# In script mode, apply the GPU visibility mask before importing modules that may
# transitively initialize torch / sentence-transformers / FAISS GPU state.
load_dotenv(override=True)
_early_visible_devices = _early_resolve_runner_cuda_visible_devices()
if _early_visible_devices:
    os.environ["CUDA_VISIBLE_DEVICES"] = _early_visible_devices

from langgraph.graph import END, StateGraph
from openai import AzureOpenAI

from agent_runner import (
    AGG_CSV_COLUMNS,
    END as END_MARKER,
    aggregate_agent_results,
    candidate_label_by_id,
    evaluate_aggregated_predictions,
    load_spl_records_from_file,
    parse_json_with_end_marker,
    validate_postcoord_with_mrcm,
    write_csv_rows,
    write_jsonl,
)
from VaxMapper.src.utils.dailymed import CONTRA_Loinc, extract_section
from VaxMapper.src.utils.hyb_mapper import (
    DEFAULT_ITEM_TERM_KEYS,
    get_cached_mapper_resources,
    retrieve_candidates_for_item as retrieve_candidates_for_item_hybrid,
)
from VaxMapper.src.utils.llm_prompt import (
    CONTRA_EXTRACT_SYSTEM_PROMPT,
    CONTRA_EXTRACT_USER_PROMPT,
    DIRECT_VERIFY_SYSTEM_PROMPT,
    DIRECT_VERIFY_SYSTEM_PROMPT_ORIGINAL,
    ROUTE_OR_FILL_SYSTEM_PROMPT,
    build_direct_verify_user_prompt,
    build_route_or_fill_user_prompt,
    extract_contraindication_items,
)
from VaxMapper.src.utils.snomed_utils import (
    DEFAULT_PREFILTER_CONTENT_TYPE,
    filter_terms_by_attribute_range,
    load_prefilter_memberships,
    load_snomed_dataframes,
)


_PREFILTER_ATTR_RANGE_CACHE: Dict[str, Any] = {}
_PREFILTER_MEMBERSHIP_CACHE: Dict[str, Dict[str, Dict[int, bool]]] = {}
_PREFILTER_ECL_CACHE: Dict[Tuple[int, str], bool] = {}


class ConfigError(ValueError):
    pass


class ChatLLM(Protocol):
    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 1.0,
        stop: Optional[List[str]] = None,
    ) -> str:
        ...


_CUDA_VISIBILITY_CONFIGURED = False


def resolve_runner_cuda_visible_devices() -> Optional[str]:
    for env_name in ("RUNNER_CUDA_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "HF_CUDA_VISIBLE_DEVICES"):
        value = os.environ.get(env_name, "").strip()
        if value:
            return value
    return None


def configure_process_cuda_visibility() -> Optional[str]:
    global _CUDA_VISIBILITY_CONFIGURED
    if _CUDA_VISIBILITY_CONFIGURED:
        return os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() or None
    visible_devices = resolve_runner_cuda_visible_devices()
    if visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
    _CUDA_VISIBILITY_CONFIGURED = True
    return visible_devices


@dataclass
class AzureLLMConfig:
    endpoint: str
    api_key: str
    deployment: str
    api_version: str
    timeout_s: int = 30

    @classmethod
    def from_env(cls) -> "AzureLLMConfig":
        required = {
            "endpoint": "AZURE_OPENAI_ENDPOINT",
            "api_key": "AZURE_OPENAI_API_KEY",
            "deployment": "AZURE_OPENAI_DEPLOYMENT",
            "api_version": "AZURE_OPENAI_API_VERSION",
        }
        values: Dict[str, str] = {}
        missing: List[str] = []
        for field_name, env_name in required.items():
            value = os.environ.get(env_name, "").strip()
            if not value:
                missing.append(env_name)
            values[field_name] = value
        if missing:
            raise ConfigError(
                "Missing Azure OpenAI configuration. Set: " + ", ".join(missing)
            )
        return cls(**values)


class AzureChatLLM:
    def __init__(self, cfg: AzureLLMConfig):
        self.cfg = cfg
        self.client = AzureOpenAI(
            api_key=cfg.api_key,
            api_version=cfg.api_version,
            azure_endpoint=cfg.endpoint,
            timeout=cfg.timeout_s,
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 1.0,
        stop: Optional[List[str]] = None,
    ) -> str:
        payload: Dict[str, Any] = {
            "model": self.cfg.deployment,
            "messages": messages,
            "max_completion_tokens": max_tokens,
            "temperature": temperature,
        }
        if stop:
            payload["stop"] = stop

        response = self.client.chat.completions.create(**payload)
        content = response.choices[0].message.content
        if isinstance(content, list):
            text_parts: List[str] = []
            for part in content:
                if hasattr(part, "text") and getattr(part, "text", None):
                    text_parts.append(part.text)
            return "".join(text_parts)
        return content or ""


@dataclass
class HuggingFaceLLMConfig:
    model_id: str
    device_map: Union[str, Dict[str, Any]] = "auto"
    max_new_tokens: int = 512
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None
    do_sample: Optional[bool] = None
    torch_dtype: Optional[str] = "auto"
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    trust_remote_code: bool = True
    use_fast_tokenizer: bool = True
    max_memory: Optional[Dict[str, Any]] = None
    model_kwargs: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "HuggingFaceLLMConfig":
        model_id = os.environ.get("HF_MODEL_ID", "").strip()
        if not model_id:
            raise ConfigError("Missing Hugging Face configuration. Set: HF_MODEL_ID")

        def parse_json_env(env_name: str) -> Optional[Any]:
            raw = os.environ.get(env_name, "").strip()
            if not raw:
                return None
            try:
                return json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ConfigError(f"{env_name} must be valid JSON.") from exc

        def parse_bool_env(env_name: str, default: bool) -> bool:
            raw = os.environ.get(env_name, "").strip().lower()
            if not raw:
                return default
            return raw in {"1", "true", "yes", "on"}

        def parse_optional_bool_env(env_name: str) -> Optional[bool]:
            raw = os.environ.get(env_name, "").strip().lower()
            if not raw:
                return None
            return raw in {"1", "true", "yes", "on"}

        def parse_optional_float_env(env_name: str) -> Optional[float]:
            raw = os.environ.get(env_name, "").strip()
            if not raw:
                return None
            return float(raw)

        def parse_optional_int_env(env_name: str) -> Optional[int]:
            raw = os.environ.get(env_name, "").strip()
            if not raw:
                return None
            return int(raw)

        device_map_json = parse_json_env("HF_DEVICE_MAP_JSON")
        device_map = device_map_json if device_map_json is not None else (
            os.environ.get("HF_DEVICE_MAP", "auto").strip() or "auto"
        )

        max_memory = parse_json_env("HF_MAX_MEMORY_JSON")
        model_kwargs = parse_json_env("HF_MODEL_KWARGS_JSON")
        if model_kwargs is None:
            model_kwargs = {}
        if not isinstance(model_kwargs, dict):
            raise ConfigError("HF_MODEL_KWARGS_JSON must decode to a JSON object.")
        if max_memory is not None and not isinstance(max_memory, dict):
            raise ConfigError("HF_MAX_MEMORY_JSON must decode to a JSON object.")

        return cls(
            model_id=model_id,
            device_map=device_map,
            max_new_tokens=int(os.environ.get("HF_MAX_NEW_TOKENS", "512")),
            temperature=parse_optional_float_env("HF_TEMPERATURE"),
            top_p=parse_optional_float_env("HF_TOP_P"),
            top_k=parse_optional_int_env("HF_TOP_K"),
            repetition_penalty=parse_optional_float_env("HF_REPETITION_PENALTY"),
            do_sample=parse_optional_bool_env("HF_DO_SAMPLE"),
            torch_dtype=os.environ.get("HF_TORCH_DTYPE", "auto").strip() or "auto",
            load_in_8bit=parse_bool_env("HF_LOAD_IN_8BIT", False),
            load_in_4bit=parse_bool_env("HF_LOAD_IN_4BIT", False),
            trust_remote_code=parse_bool_env("HF_TRUST_REMOTE_CODE", True),
            use_fast_tokenizer=parse_bool_env("HF_USE_FAST_TOKENIZER", True),
            max_memory=max_memory,
            model_kwargs=model_kwargs,
        )


class HuggingFaceChatLLM:
    def __init__(self, cfg: HuggingFaceLLMConfig):
        configure_process_cuda_visibility()
        self.cfg = cfg
        from VaxMapper.src.llm import load_model_local

        model_kwargs = dict(cfg.model_kwargs)
        if cfg.max_memory is not None:
            model_kwargs["max_memory"] = cfg.max_memory
        self.model = load_model_local(
            model_id=cfg.model_id,
            device_map=cfg.device_map,
            torch_dtype=cfg.torch_dtype,
            load_in_8bit=cfg.load_in_8bit,
            load_in_4bit=cfg.load_in_4bit,
            trust_remote_code=cfg.trust_remote_code,
            use_fast_tokenizer=cfg.use_fast_tokenizer,
            **model_kwargs,
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: float = 1.0,
        stop: Optional[List[str]] = None,
    ) -> str:
        temperature_value = self.cfg.temperature if self.cfg.temperature is not None else temperature
        effective_max_tokens = max_tokens if max_tokens is not None else self.cfg.max_new_tokens
        generate_kwargs: Dict[str, Any] = {}
        if self.cfg.do_sample is not None:
            generate_kwargs["do_sample"] = self.cfg.do_sample
        if self.cfg.top_p is not None:
            generate_kwargs["top_p"] = self.cfg.top_p
        if self.cfg.top_k is not None:
            generate_kwargs["top_k"] = self.cfg.top_k
        if self.cfg.repetition_penalty is not None:
            generate_kwargs["repetition_penalty"] = self.cfg.repetition_penalty
        response = self.model.generate(
            messages,
            max_new_tokens=effective_max_tokens or self.model.tokenizer.model_max_length,
            temperature=temperature_value,
            stop=stop,
            **generate_kwargs,
        )
        return str(response or "")

    def resolve_effective_max_tokens(self, max_tokens: Optional[int]) -> int:
        return max_tokens if max_tokens is not None else self.cfg.max_new_tokens


def build_llm(backend: str) -> ChatLLM:
    configure_process_cuda_visibility()
    if backend == "azure":
        return AzureChatLLM(AzureLLMConfig.from_env())
    if backend == "huggingface":
        return HuggingFaceChatLLM(HuggingFaceLLMConfig.from_env())
    raise ConfigError(f"Unsupported backend '{backend}'. Expected 'azure' or 'huggingface'.")


@dataclass
class AgentRunConfig:
    attribute_table: Dict[str, int] = field(
        default_factory=lambda: {
            "causative_agent": 246075003,
            "severity": 246112005,
            "clinical_course": 263502005,
        }
    )
    extraction_max_tokens: int = 512
    direct_match_max_tokens: int = 384
    route_or_fill_max_tokens: int = 512
    retries: int = 2
    recursion_limit: int = 256
    stop: List[str] = field(default_factory=lambda: [END_MARKER])
    use_strict_prompts: bool = True

    @classmethod
    def from_env(cls) -> "AgentRunConfig":
        return cls(
            extraction_max_tokens=int(os.environ.get("AGENT_MAX_TOKENS_EXTRACT", "512")),
            direct_match_max_tokens=int(os.environ.get("AGENT_MAX_TOKENS_DIRECT", "384")),
            route_or_fill_max_tokens=int(os.environ.get("AGENT_MAX_TOKENS_ROUTE_FILL", "512")),
            retries=int(os.environ.get("AGENT_RETRIES", "2")),
            recursion_limit=int(os.environ.get("LANGGRAPH_RECURSION_LIMIT", "256")),
            use_strict_prompts=os.environ.get("USE_STRICT_PROMPTS", "1").lower() not in {"0", "false", "no"},
        )


class ItemState(TypedDict, total=False):
    spl_set_id: str
    item: Dict[str, Any]
    candidates: Dict[str, List[Dict[str, Any]]]
    direct_match: Dict[str, Any]
    route_or_fill: Dict[str, Any]
    selected_problem_id: str
    selected_focus_term: str
    fills_norm: Dict[str, str]
    fills_detail: Dict[str, Dict[str, str]]
    validation: Dict[str, Any]
    status: str
    expression: str
    item_result: Dict[str, Any]


class ContraState(TypedDict, total=False):
    spl_record: Dict[str, Any]
    spl_set_id: str
    product_name: Optional[str]
    contra_section_found: bool
    contra_section_text: str
    extracted_items: List[Dict[str, Any]]
    current_index: int
    current_item: Dict[str, Any]
    item_results: List[Dict[str, Any]]
    final_result: Dict[str, Any]
    error: str


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunObserver:
    def __init__(
        self,
        *,
        audit_path: Optional[str] = None,
        audit_enabled: bool = True,
        progress_enabled: bool = True,
    ):
        self.audit_enabled = audit_enabled and bool(audit_path)
        self.progress_enabled = progress_enabled
        self.audit_path = Path(audit_path) if audit_path else None
        self._is_tty = bool(getattr(sys.stdout, "isatty", lambda: False)())
        self._last_progress_len = 0
        self._progress_state: Dict[str, Any] = {
            "spl_index": None,
            "spl_total": None,
            "spl_set_id": None,
            "item_index": None,
            "item_total": None,
            "node_name": None,
        }
        if self.audit_enabled and self.audit_path is not None:
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)

    def set_spl_context(self, *, spl_index: Optional[int] = None, spl_total: Optional[int] = None, spl_set_id: Optional[str] = None) -> None:
        if spl_index is not None:
            self._progress_state["spl_index"] = spl_index
        if spl_total is not None:
            self._progress_state["spl_total"] = spl_total
        if spl_set_id is not None:
            self._progress_state["spl_set_id"] = spl_set_id
        self._render_progress()

    def set_item_context(self, *, item_index: Optional[int] = None, item_total: Optional[int] = None) -> None:
        if item_index is not None:
            self._progress_state["item_index"] = item_index
        if item_total is not None:
            self._progress_state["item_total"] = item_total
        self._render_progress()

    def set_node(self, node_name: str) -> None:
        self._progress_state["node_name"] = node_name
        self._render_progress()

    def clear_progress(self) -> None:
        if self.progress_enabled and self._is_tty:
            sys.stdout.write("\r" + (" " * self._last_progress_len) + "\r")
            sys.stdout.flush()
            self._last_progress_len = 0

    def _render_progress(self) -> None:
        if not self.progress_enabled:
            return
        spl_idx = self._progress_state.get("spl_index")
        spl_total = self._progress_state.get("spl_total")
        spl_set_id = self._progress_state.get("spl_set_id") or "N/A"
        item_idx = self._progress_state.get("item_index")
        item_total = self._progress_state.get("item_total")
        node_name = self._progress_state.get("node_name") or "N/A"

        spl_part = f"SPL {spl_idx}/{spl_total}" if spl_idx is not None and spl_total is not None else "SPL ?/?"
        if item_idx is not None and item_total is not None:
            item_part = f"item {item_idx + 1}/{item_total}"
        else:
            item_part = "item -/-"
        line = f"{spl_part} | {spl_set_id} | {item_part} | node={node_name}"

        if self._is_tty:
            padded = line
            if len(line) < self._last_progress_len:
                padded = line + (" " * (self._last_progress_len - len(line)))
            sys.stdout.write("\r" + padded)
            sys.stdout.flush()
            self._last_progress_len = len(line)
        else:
            print(line)

    def _write_event(self, event: Dict[str, Any]) -> None:
        if not self.audit_enabled or self.audit_path is None:
            return
        with self.audit_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def log_event(self, event_type: str, **payload: Any) -> None:
        event = {
            "event_type": event_type,
            "timestamp": _now_iso(),
            "spl_index": self._progress_state.get("spl_index"),
            "spl_total": self._progress_state.get("spl_total"),
            "spl_set_id": self._progress_state.get("spl_set_id"),
            "item_index": self._progress_state.get("item_index"),
            "item_total": self._progress_state.get("item_total"),
            "node_name": self._progress_state.get("node_name"),
            **payload,
        }
        self._write_event(event)


def retrieve_candidates_for_item(item: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    configure_process_cuda_visibility()

    use_bm25_tuning    = os.environ.get("USE_BM25_TUNING",    "1").lower() not in {"0", "false", "no"}
    use_ancestor_paths = os.environ.get("USE_ANCESTOR_PATHS", "1").lower() not in {"0", "false", "no"}

    bm25_b = 0.5 if use_bm25_tuning else 0.75
    # MAPPER_ES_INDEX wins if set explicitly; otherwise derive from the BM25 flag.
    # Tuning ON  → preserve existing default index name (backward compat).
    # Tuning OFF → point at the pre-built original-b index.
    es_index = os.environ.get(
        "MAPPER_ES_INDEX",
        "snomed_ct_es_index" if use_bm25_tuning else "snomed_ct_es_index_original",
    )

    resources = get_cached_mapper_resources(
        snomed_source_dir=os.environ.get("SNOMED_SOURCE_DIR", "snomed_us_source"),
        concept_path=os.environ.get("SNOMED_CONCEPT_PATH"),
        description_path=os.environ.get("SNOMED_DESCRIPTION_PATH"),
        es_index=es_index,
        dense_index_path=os.environ.get("MAPPER_DENSE_INDEX_PATH", "results/snomed_terms_dense_test.bin"),
        model_name=os.environ.get("MAPPER_MODEL_NAME", "tavakolih/all-MiniLM-L6-v2-pubmed-full"),
        device=os.environ.get("MAPPER_DEVICE", "cuda:0"),
        k_dense=int(os.environ.get("MAPPER_K_DENSE", "50")),
        k_bm25=int(os.environ.get("MAPPER_K_BM25", "50")),
        k_final=int(os.environ.get("MAPPER_K_FINAL", "25")),
        n_final=int(os.environ.get("MAPPER_N_FINAL", "15")),
        rebuild_dense_index=os.environ.get("MAPPER_REBUILD_DENSE_INDEX", "").lower() in {"1", "true", "yes"},
        rebuild_es_index=os.environ.get("MAPPER_REBUILD_ES_INDEX", "").lower() in {"1", "true", "yes"},
        item_term_keys=DEFAULT_ITEM_TERM_KEYS,
        reranker_model_id=os.environ.get("RERANKER_MODEL_ID", ""),
        reranker_device=os.environ.get("RERANKER_DEVICE", "cuda:0"),
        bm25_b=bm25_b,
    )

    if not use_ancestor_paths:
        resources = replace(resources, is_a_graph=None)  # shallow copy — never mutate the cache

    return retrieve_candidates_for_item_hybrid(item, resources)


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


class ContraLangGraphAgent:
    def __init__(self, llm: ChatLLM, cfg: Optional[AgentRunConfig] = None, observer: Optional[RunObserver] = None):
        self.llm = llm
        self.cfg = cfg or AgentRunConfig()
        self.observer = observer
        self.item_graph = self._build_item_graph()
        self.spl_graph = self._build_spl_graph()

    def _log_llm_call(
        self,
        *,
        call_name: str,
        system: str,
        user: str,
        max_tokens: Optional[int],
        effective_max_tokens: int,
        raw: str,
        parsed: Optional[Dict[str, Any]],
        duration_s: float,
    ) -> None:
        if self.observer is None:
            return
        self.observer.log_event(
            "llm_call",
            call_name=call_name,
            duration_s=duration_s,
            max_tokens=max_tokens,
            effective_max_tokens=effective_max_tokens,
            system_prompt=system,
            user_prompt=user,
            raw_output=raw,
            parsed_output=parsed,
            parse_success=parsed is not None,
        )

    def _call_llm_json(self, system: str, user: str, max_tokens: Optional[int], *, call_name: str) -> Tuple[Optional[Dict[str, Any]], str]:
        started = time.perf_counter()
        effective_max_tokens = max_tokens if max_tokens is not None else getattr(self.llm, "resolve_effective_max_tokens", lambda mt: 512)(max_tokens)
        raw = self.llm.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=max_tokens,
            temperature=1.0,
            stop=self.cfg.stop,
        )
        parsed = parse_json_with_end_marker(raw)
        self._log_llm_call(
            call_name=call_name,
            system=system,
            user=user,
            max_tokens=max_tokens,
            effective_max_tokens=effective_max_tokens,
            raw=raw,
            parsed=parsed,
            duration_s=time.perf_counter() - started,
        )
        return parsed, raw

    def _verify_direct_match(self, ci_text: str, focus_candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        system_prompt = (DIRECT_VERIFY_SYSTEM_PROMPT if self.cfg.use_strict_prompts
                         else DIRECT_VERIFY_SYSTEM_PROMPT_ORIGINAL)
        user = build_direct_verify_user_prompt(ci_text, focus_candidates, max_n=10)
        parsed, raw = self._call_llm_json(
            system_prompt,
            user,
            max_tokens=self.cfg.direct_match_max_tokens,
            call_name="direct_match",
        )
        if not parsed:
            return {
                "direct_match": False,
                "selected_id": "N/A",
                "selected_term": "N/A",
                "raw": raw,
                "parse_failed": True,
            }
        return {**parsed, "raw": raw}

    def _route_or_fill(self, item: Dict[str, Any], cands: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        user = build_route_or_fill_user_prompt(
            item,
            json.dumps(self.cfg.attribute_table, separators=(",", ":")),
            cands,
            max_n=10,
        )
        parsed, raw = self._call_llm_json(
            ROUTE_OR_FILL_SYSTEM_PROMPT,
            user,
            max_tokens=self.cfg.route_or_fill_max_tokens,
            call_name="route_or_fill",
        )
        if not parsed:
            focus_fallback = cands.get("focus_candidates", [])
            selected_problem_id = str((focus_fallback[0].get("id") if focus_fallback else "N/A"))
            return {
                "post_decision": "N/A",
                "selected_problem_id": selected_problem_id,
                "fills": {
                    "causative_agent": "N/A",
                    "severity": "N/A",
                    "clinical_course": "N/A",
                },
                "raw": raw,
                "parse_failed": True,
            }
        return {**parsed, "raw": raw}

    def _build_item_graph(self):
        graph = StateGraph(ItemState)

        def instrument_item_node(node_name: str, fn: Callable[[ItemState], ItemState]) -> Callable[[ItemState], ItemState]:
            def wrapped(state: ItemState) -> ItemState:
                if self.observer is not None:
                    item = state.get("item", {})
                    item_index = item.get("item_index")
                    self.observer.set_item_context(item_index=item_index)
                    self.observer.set_node(node_name)
                    self.observer.log_event("node_start", node_name=node_name, graph_level="item")
                started = time.perf_counter()
                status = "ok"
                try:
                    result = fn(state)
                    return result
                except Exception as exc:
                    status = "error"
                    if self.observer is not None:
                        self.observer.log_event(
                            "node_error",
                            node_name=node_name,
                            graph_level="item",
                            duration_s=time.perf_counter() - started,
                            error=str(exc),
                        )
                    raise
                finally:
                    if self.observer is not None and status == "ok":
                        self.observer.log_event(
                            "node_end",
                            node_name=node_name,
                            graph_level="item",
                            duration_s=time.perf_counter() - started,
                            status=status,
                        )
            return wrapped

        def retrieve_candidates_node(state: ItemState) -> ItemState:
            return {**state, "candidates": retrieve_candidates_for_item(state["item"])}

        def direct_match_node(state: ItemState) -> ItemState:
            direct = self._verify_direct_match(
                state["item"].get("ci_text", ""),
                state["candidates"].get("direct_candidates", state["candidates"].get("focus_candidates", [])),
            )
            return {**state, "direct_match": direct}

        def route_after_direct_match(state: ItemState) -> str:
            direct = state.get("direct_match", {})
            if direct.get("direct_match") is True and direct.get("selected_id") not in (None, "", "N/A"):
                return "assemble_direct"
            return "prefilter"

        def prefilter_node(state: ItemState) -> ItemState:
            return {**state, "candidates": prefilter_slot_candidates(state["candidates"])}

        def route_or_fill_node(state: ItemState) -> ItemState:
            return {
                **state,
                "route_or_fill": self._route_or_fill(state["item"], state["candidates"]),
            }

        def normalize_node(state: ItemState) -> ItemState:
            route_fill = state.get("route_or_fill", {})
            selected_problem_id = str(route_fill.get("selected_problem_id", "N/A"))
            fills = route_fill.get("fills", {}) or {}

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
                        state["candidates"].get("causative_agent_candidates", []),
                        fills_norm.get("causative_agent", "N/A"),
                    ),
                },
                "severity": {
                    "id": fills_norm.get("severity", "N/A"),
                    "term": candidate_label_by_id(
                        state["candidates"].get("severity_candidates", []),
                        fills_norm.get("severity", "N/A"),
                    ),
                },
                "clinical_course": {
                    "id": fills_norm.get("clinical_course", "N/A"),
                    "term": candidate_label_by_id(
                        state["candidates"].get("clinical_course_candidates", []),
                        fills_norm.get("clinical_course", "N/A"),
                    ),
                },
            }
            selected_focus_term = candidate_label_by_id(
                state["candidates"].get("focus_candidates", []),
                selected_problem_id,
            )
            return {
                **state,
                "selected_problem_id": selected_problem_id,
                "selected_focus_term": selected_focus_term,
                "fills_norm": fills_norm,
                "fills_detail": fills_detail,
            }

        def validate_node(state: ItemState) -> ItemState:
            selected_problem_id = state.get("selected_problem_id", "N/A")
            selected_focus_term = state.get("selected_focus_term", "N/A")
            fills_norm = dict(state.get("fills_norm", {}))
            fills_detail = dict(state.get("fills_detail", {}))
            route_fill = dict(state.get("route_or_fill", {}))
            post_decision = str(route_fill.get("post_decision", "N/A"))

            ok, fail_reasons = validate_postcoord_with_mrcm(selected_problem_id, fills_norm)
            if not ok:
                if selected_problem_id != "N/A" and not selected_problem_id.isdigit():
                    selected_problem_id = "N/A"
                    selected_focus_term = "N/A"
                for key in ("causative_agent", "severity", "clinical_course"):
                    if fills_detail[key]["id"] != "N/A" and not str(fills_detail[key]["id"]).isdigit():
                        fills_detail[key]["id"] = "N/A"
                        fills_detail[key]["term"] = "N/A"
                        fills_norm[key] = "N/A"
                post_decision = "N/A"

            route_fill["post_decision"] = post_decision
            return {
                **state,
                "selected_problem_id": selected_problem_id,
                "selected_focus_term": selected_focus_term,
                "fills_norm": fills_norm,
                "fills_detail": fills_detail,
                "route_or_fill": route_fill,
                "validation": {"ok": ok, "fail_reasons": fail_reasons},
            }

        def assemble_direct_node(state: ItemState) -> ItemState:
            direct = state.get("direct_match", {})
            item = state["item"]
            return {
                **state,
                "status": "DIRECT",
                "item_result": {
                    "SPL_SET_ID": state["spl_set_id"],
                    "item_index": item.get("item_index"),
                    "query_text": item.get("ci_text", ""),
                    "status": "DIRECT",
                    "selected_id": direct.get("selected_id", "N/A"),
                    "selected_term": direct.get("selected_term", "N/A"),
                    "trace": {"direct_verify": direct},
                    "extracted_item": item,
                },
            }

        def assemble_postcoord_node(state: ItemState) -> ItemState:
            ax_pairs: List[str] = []
            for key, attr_id in self.cfg.attribute_table.items():
                val = state["fills_norm"].get(key, "N/A")
                if val != "N/A":
                    ax_pairs.append(f"{attr_id}={val}")
            expression = (
                f"{state['selected_problem_id']}:{{{','.join(ax_pairs)}}}"
                if ax_pairs
                else state["selected_problem_id"]
            )
            post_decision = str(state["route_or_fill"].get("post_decision", "N/A"))
            status = "POSTCOORD" if post_decision == "YES" else "MINIMAL"
            item = state["item"]
            return {
                **state,
                "expression": expression,
                "status": status,
                "item_result": {
                    "SPL_SET_ID": state["spl_set_id"],
                    "item_index": item.get("item_index"),
                    "query_text": item.get("ci_text", ""),
                    "status": status,
                    "post_decision": post_decision,
                    "selected_problem_id": state["selected_problem_id"],
                    "selected_focus_term": state["selected_focus_term"],
                    "fills": state["fills_detail"],
                    "expression": expression,
                    "trace": {
                        "direct_verify": state.get("direct_match", {}),
                        "route_or_fill": state.get("route_or_fill", {}),
                        "validation": state.get("validation", {}),
                    },
                    "extracted_item": item,
                },
            }

        graph.add_node("retrieve_candidates", instrument_item_node("retrieve_candidates", retrieve_candidates_node))
        graph.add_node("direct_match", instrument_item_node("direct_match", direct_match_node))
        graph.add_node("prefilter", instrument_item_node("prefilter", prefilter_node))
        graph.add_node("route_or_fill", instrument_item_node("route_or_fill", route_or_fill_node))
        graph.add_node("normalize", instrument_item_node("normalize", normalize_node))
        graph.add_node("validate", instrument_item_node("validate", validate_node))
        graph.add_node("assemble_direct", instrument_item_node("assemble_direct", assemble_direct_node))
        graph.add_node("assemble_postcoord", instrument_item_node("assemble_postcoord", assemble_postcoord_node))

        graph.set_entry_point("retrieve_candidates")
        graph.add_edge("retrieve_candidates", "direct_match")
        graph.add_conditional_edges(
            "direct_match",
            route_after_direct_match,
            {"assemble_direct": "assemble_direct", "prefilter": "prefilter"},
        )
        graph.add_edge("prefilter", "route_or_fill")
        graph.add_edge("route_or_fill", "normalize")
        graph.add_edge("normalize", "validate")
        graph.add_edge("validate", "assemble_postcoord")
        graph.add_edge("assemble_direct", END)
        graph.add_edge("assemble_postcoord", END)

        return graph.compile()

    def _build_spl_graph(self):
        graph = StateGraph(ContraState)

        def instrument_spl_node(node_name: str, fn: Callable[[ContraState], ContraState]) -> Callable[[ContraState], ContraState]:
            def wrapped(state: ContraState) -> ContraState:
                if self.observer is not None:
                    spl_set_id = state.get("spl_set_id")
                    extracted_items = state.get("extracted_items", [])
                    current_index = state.get("current_index")
                    self.observer.set_spl_context(spl_set_id=spl_set_id)
                    self.observer.set_item_context(
                        item_index=current_index if current_index is not None else None,
                        item_total=len(extracted_items) if extracted_items else None,
                    )
                    self.observer.set_node(node_name)
                    self.observer.log_event("node_start", node_name=node_name, graph_level="spl")
                started = time.perf_counter()
                status = "ok"
                try:
                    result = fn(state)
                    if self.observer is not None:
                        self.observer.set_spl_context(spl_set_id=result.get("spl_set_id"))
                        if "extracted_items" in result:
                            self.observer.set_item_context(item_total=len(result.get("extracted_items", [])))
                    return result
                except Exception as exc:
                    status = "error"
                    if self.observer is not None:
                        self.observer.log_event(
                            "node_error",
                            node_name=node_name,
                            graph_level="spl",
                            duration_s=time.perf_counter() - started,
                            error=str(exc),
                        )
                    raise
                finally:
                    if self.observer is not None and status == "ok":
                        self.observer.log_event(
                            "node_end",
                            node_name=node_name,
                            graph_level="spl",
                            duration_s=time.perf_counter() - started,
                            status=status,
                        )
            return wrapped

        def resolve_contra_section_node(state: ContraState) -> ContraState:
            spl_record = dict(state["spl_record"])
            spl_set_id = spl_record.get("SPL_SET_ID") or spl_record.get("spl_set_id") or str(uuid.uuid4())
            spl_record["SPL_SET_ID"] = spl_set_id

            contra_text = (
                spl_record.get("contra_text")
                or spl_record.get("contra_section_text")
                or spl_record.get("section_text")
                or ""
            )
            if str(contra_text).strip():
                return {
                    **state,
                    "spl_record": spl_record,
                    "spl_set_id": spl_set_id,
                    "product_name": spl_record.get("product_name"),
                    "contra_section_found": True,
                    "contra_section_text": str(contra_text).strip(),
                }

            try:
                contra_section = extract_section(str(spl_set_id), [CONTRA_Loinc])
            except Exception as exc:
                return {
                    **state,
                    "spl_record": spl_record,
                    "spl_set_id": spl_set_id,
                    "contra_section_found": False,
                    "contra_section_text": "",
                    "item_results": [],
                    "error": f"extract_section_failed: {exc}",
                }

            section_payload = (contra_section.get("sections") or {}).get(CONTRA_Loinc, {})
            spl_record["contra_section_text"] = section_payload.get("section_text") or ""
            spl_record["contra_section_xml"] = section_payload.get("section_xml")
            spl_record["product_name"] = contra_section.get("product_name")
            spl_record["contra_section_found"] = bool(section_payload.get("section_text"))

            return {
                **state,
                "spl_record": spl_record,
                "spl_set_id": spl_set_id,
                "product_name": spl_record.get("product_name"),
                "contra_section_found": bool(spl_record.get("contra_section_found")),
                "contra_section_text": spl_record.get("contra_section_text", ""),
            }

        def route_after_resolve(state: ContraState) -> str:
            if state.get("error"):
                return "finalize"
            return "extract_items"

        def extract_items_node(state: ContraState) -> ContraState:
            spl_context = dict(state["spl_record"])
            spl_context["contra_section_text"] = state.get("contra_section_text", "")
            extraction_started = time.perf_counter()
            items, raw = extract_contraindication_items(
                self.llm.chat,
                spl_context.get("contra_section_text", ""),
                max_tokens=self.cfg.extraction_max_tokens,
                # stop=self.cfg.stop,
                stop = None, # early termination due to model spitting out end token
                retries=self.cfg.retries,
            )
            parsed_payload: Optional[Dict[str, Any]] = {"items": items} if items else None
            self._log_llm_call(
                call_name="extract_contraindications",
                system=CONTRA_EXTRACT_SYSTEM_PROMPT,
                user=CONTRA_EXTRACT_USER_PROMPT.format(text=spl_context.get("contra_section_text", "")),
                max_tokens=self.cfg.extraction_max_tokens,
                effective_max_tokens=self.cfg.extraction_max_tokens,
                raw=raw,
                parsed=parsed_payload,
                duration_s=time.perf_counter() - extraction_started,
            )
            indexed_items: List[Dict[str, Any]] = []
            for item_index, item in enumerate(items):
                indexed_item = dict(item)
                indexed_item["item_index"] = item_index
                indexed_items.append(indexed_item)
            return {
                **state,
                "extracted_items": indexed_items,
                "current_index": 0,
                "item_results": [],
            }

        def route_after_extract(state: ContraState) -> str:
            if not state.get("extracted_items"):
                return "finalize"
            return "prepare_item"

        def prepare_item_node(state: ContraState) -> ContraState:
            idx = state.get("current_index", 0)
            items = state.get("extracted_items", [])
            if idx >= len(items):
                return state
            return {**state, "current_item": items[idx]}

        def process_item_node(state: ContraState) -> ContraState:
            item_state: ItemState = {
                "spl_set_id": state["spl_set_id"],
                "item": state["current_item"],
            }
            result = self.item_graph.invoke(item_state)
            item_results = list(state.get("item_results", []))
            item_results.append(result["item_result"])
            return {**state, "item_results": item_results}

        def advance_item_node(state: ContraState) -> ContraState:
            return {**state, "current_index": state.get("current_index", 0) + 1}

        def continue_or_finish(state: ContraState) -> str:
            if state.get("current_index", 0) >= len(state.get("extracted_items", [])):
                return "finalize"
            return "prepare_item"

        def finalize_node(state: ContraState) -> ContraState:
            if state.get("error"):
                return {
                    **state,
                    "final_result": {
                        "SPL_SET_ID": state.get("spl_set_id"),
                        "n_items_in": 0,
                        "n_items_out": 0,
                        "results": [],
                        "contra_section_found": False,
                        "error": state["error"],
                    },
                }

            extracted_items = state.get("extracted_items", [])
            item_results = state.get("item_results", [])
            return {
                **state,
                "final_result": {
                    "SPL_SET_ID": state["spl_set_id"],
                    "product_name": state.get("product_name"),
                    "contra_section_found": bool(state.get("contra_section_found")),
                    "contra_section_text": state.get("contra_section_text", ""),
                    "n_items_in": len(extracted_items),
                    "n_items_out": len(item_results),
                    "results": item_results,
                },
            }

        graph.add_node("resolve_contra_section", instrument_spl_node("resolve_contra_section", resolve_contra_section_node))
        graph.add_node("extract_items", instrument_spl_node("extract_items", extract_items_node))
        graph.add_node("prepare_item", instrument_spl_node("prepare_item", prepare_item_node))
        graph.add_node("process_item", instrument_spl_node("process_item", process_item_node))
        graph.add_node("advance_item", instrument_spl_node("advance_item", advance_item_node))
        graph.add_node("finalize", instrument_spl_node("finalize", finalize_node))

        graph.set_entry_point("resolve_contra_section")
        graph.add_conditional_edges(
            "resolve_contra_section",
            route_after_resolve,
            {"extract_items": "extract_items", "finalize": "finalize"},
        )
        graph.add_conditional_edges(
            "extract_items",
            route_after_extract,
            {"prepare_item": "prepare_item", "finalize": "finalize"},
        )
        graph.add_edge("prepare_item", "process_item")
        graph.add_edge("process_item", "advance_item")
        graph.add_conditional_edges(
            "advance_item",
            continue_or_finish,
            {"prepare_item": "prepare_item", "finalize": "finalize"},
        )
        graph.add_edge("finalize", END)

        return graph.compile()

    def process_spl(self, spl_record: Dict[str, Any]) -> Dict[str, Any]:
        init_state: ContraState = {"spl_record": dict(spl_record)}
        result = self.spl_graph.invoke(
            init_state,
            config={"recursion_limit": self.cfg.recursion_limit},
        )
        return result["final_result"]


def main() -> None:
    configure_process_cuda_visibility()
    output_dir = Path(os.environ.get("OUTPUT_DIR", "results/langgraph_run"))

    def output_path(filename: str) -> str:
        return str(output_dir / filename)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spl-list", required=True, help="Path to text or CSV file containing SPL_SET_ID values.")
    parser.add_argument(
        "--out-jsonl",
        default=output_path("agent_results.jsonl"),
        help="Path to write raw per-SPL JSONL results.",
    )
    parser.add_argument(
        "--aggregated-jsonl",
        default=output_path("aggregated_results.jsonl"),
        help="Path to write aggregated JSONL results.",
    )
    parser.add_argument(
        "--aggregated-csv",
        default=output_path("aggregated_hits.csv"),
        help="Path to write aggregated CSV results.",
    )
    parser.add_argument(
        "--gold-csv",
        default=os.environ.get("AGENT_GOLD_CSV", ""),
        help="Optional gold CSV path for evaluation.",
    )
    parser.add_argument(
        "--eval-json",
        default=output_path("eval_metrics.json"),
        help="Path to write evaluation metrics JSON when --gold-csv is provided.",
    )
    parser.add_argument(
        "--eval-details-csv",
        default=output_path("evaluation_details.csv"),
        help="Path to write evaluation details CSV when --gold-csv is provided.",
    )
    parser.add_argument(
        "--audit-jsonl",
        default=output_path("runtime_audit.jsonl"),
        help="Path to write runtime audit JSONL.",
    )
    parser.add_argument(
        "--disable-audit",
        action="store_true",
        help="Disable runtime audit JSONL logging.",
    )
    parser.add_argument(
        "--disable-progress",
        action="store_true",
        help="Disable live single-line progress display.",
    )
    parser.add_argument(
        "--backend",
        default=os.environ.get("LANGGRAPH_LLM_BACKEND", "huggingface"),
        choices=("azure", "huggingface"),
        help="LLM backend to use.",
    )
    args = parser.parse_args()

    llm = build_llm(args.backend)
    observer = RunObserver(
        audit_path=args.audit_jsonl,
        audit_enabled=not args.disable_audit,
        progress_enabled=not args.disable_progress,
    )
    print(f"Agent Config: {AgentRunConfig.from_env()}")
    agent = ContraLangGraphAgent(llm=llm, cfg=AgentRunConfig.from_env(), observer=observer)

    spl_records = load_spl_records_from_file(args.spl_list)
    run_rows: List[Dict[str, Any]] = []

    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for spl_index, spl in enumerate(spl_records, 1):
            observer.set_spl_context(spl_index=spl_index, spl_total=len(spl_records))
            result = agent.process_spl(spl)
            run_rows.append(result)
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    observer.clear_progress()

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
