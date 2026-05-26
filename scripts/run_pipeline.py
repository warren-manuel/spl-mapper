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

from src.evaluation.evaluator import (
    AGG_CSV_COLUMNS,
    END_MARKER as END_MARKER,
    aggregate_agent_results,
    candidate_label_by_id,
    evaluate_aggregated_predictions,
    load_spl_records_from_file,
    parse_json_with_end_marker,
    validate_postcoord_with_mrcm,
    write_csv_rows,
    write_jsonl,
)
from src.extraction.section_parser import CONTRA_Loinc, extract_section
from src.retrieval.hybrid_mapper import (
    DEFAULT_ITEM_TERM_KEYS,
    get_cached_mapper_resources,
    retrieve_candidates_for_item as retrieve_candidates_for_item_hybrid,
    search_snomed_concepts,
)
from src.llm.prompts import (
    CATEGORIZE_SLOTS_USER_PROMPT,
    CONTRA_EXTRACT_SYSTEM_PROMPT,
    CONTRA_EXTRACT_USER_PROMPT,
    DECOMPOSE_SYSTEM_PROMPT,
    DECOMPOSE_USER_PROMPT,
    DIRECT_VERIFY_SYSTEM_PROMPT,
    DIRECT_VERIFY_SYSTEM_PROMPT_ORIGINAL,
    ROUTE_OR_FILL_SYSTEM_PROMPT,
    SIMPLE_EXTRACT_SYSTEM_PROMPT,
    SIMPLE_EXTRACT_USER_PROMPT,
    _load_direct_match_agent,
    _load_focus_selector,
    _load_postcord_agent,
    _load_snomed_conventions,
    build_decompose_user_prompt,
    build_direct_match_agent_user_prompt,
    build_direct_verify_user_prompt,
    build_focus_selector_user_prompt,
    build_mrcm_mapper_react_user_prompt,
    build_route_or_fill_user_prompt,
    parse_mrcm_mapper_output,
    categorize_item_slots,
    decompose_contraindication_item,
    extract_contraindication_items,
    parse_direct_match_agent_output,
    parse_focus_selector_output,
    run_mrcm_mapper,
    run_pattern_finder,
)
from src.snomed.snomed_utils import (
    DEFAULT_PREFILTER_CONTENT_TYPE,
    filter_terms_by_attribute_range,
    load_prefilter_memberships,
    load_snomed_dataframes,
)


_PREFILTER_ATTR_RANGE_CACHE: Dict[str, Any] = {}
_PREFILTER_MEMBERSHIP_CACHE: Dict[str, Dict[str, Dict[int, bool]]] = {}
_PREFILTER_ECL_CACHE: Dict[Tuple[int, str], bool] = {}


def append_extraction_cache(path: str, entry: Dict[str, Any]) -> None:
    """Append a single SPL extraction entry to the cache file (crash-safe)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def load_extraction_cache(path: str) -> Dict[str, Dict[str, Any]]:
    """Load extraction cache JSONL into a dict keyed by spl_set_id."""
    cache: Dict[str, Dict[str, Any]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entry = json.loads(line)
                spl_set_id = entry.get("spl_set_id", "")
                if spl_set_id:
                    cache[spl_set_id] = entry
    return cache


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
        from src.llm.backends import load_model_local

        model_kwargs = dict(cfg.model_kwargs)
        if cfg.max_memory is not None:
            # JSON keys are always strings; accelerate requires integer keys for GPU devices
            model_kwargs["max_memory"] = {
                int(k) if isinstance(k, str) and k.lstrip("-").isdigit() else k: v
                for k, v in cfg.max_memory.items()
            }
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


# ---------------------------------------------------------------------------
# vLLM backend (OpenAI-compatible endpoint)
# ---------------------------------------------------------------------------

@dataclass
class VLLMLLMConfig:
    base_url: str = "http://localhost:8000/v1"
    model_id: str = "google/gemma-4-31B-it"
    api_key: str = "token"
    max_new_tokens: int = 2048
    temperature: float = 0.0
    timeout: int = 120

    @classmethod
    def from_env(cls) -> "VLLMLLMConfig":
        return cls(
            base_url=os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"),
            model_id=os.environ.get("VLLM_MODEL_ID", "google/gemma-4-31B-it"),
            api_key=os.environ.get("VLLM_API_KEY", "token"),
            max_new_tokens=int(os.environ.get("HF_MAX_NEW_TOKENS", "2048")),
            temperature=float(os.environ.get("HF_TEMPERATURE", "0.0")),
            timeout=int(os.environ.get("VLLM_TIMEOUT_SECONDS", "120")),
        )


class VLLMChatLLM:
    def __init__(self, cfg: VLLMLLMConfig):
        import openai as _openai
        self.cfg = cfg
        self._client = _openai.OpenAI(
            base_url=cfg.base_url,
            api_key=cfg.api_key,
            timeout=cfg.timeout,
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: float = 1.0,
        stop: Optional[List[str]] = None,
    ) -> str:
        response = self._client.chat.completions.create(
            model=self.cfg.model_id,
            messages=messages,
            max_tokens=max_tokens or self.cfg.max_new_tokens,
            temperature=self.cfg.temperature,
            stop=stop or None,
        )
        return response.choices[0].message.content or ""

    def get_langchain_llm(self):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            base_url=self.cfg.base_url,
            api_key=self.cfg.api_key,
            model=self.cfg.model_id,
            temperature=self.cfg.temperature,
        )

    def resolve_effective_max_tokens(self, max_tokens: Optional[int]) -> int:
        return max_tokens if max_tokens is not None else self.cfg.max_new_tokens


def build_llm(backend: str) -> ChatLLM:
    configure_process_cuda_visibility()
    if backend == "azure":
        return AzureChatLLM(AzureLLMConfig.from_env())
    if backend == "huggingface":
        return HuggingFaceChatLLM(HuggingFaceLLMConfig.from_env())
    if backend == "vllm":
        return VLLMChatLLM(VLLMLLMConfig.from_env())
    raise ConfigError(f"Unsupported backend '{backend}'. Expected 'azure', 'huggingface', or 'vllm'.")


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
    decompose_max_tokens: int = 512
    categorize_slots_max_tokens: int = 256
    direct_match_max_tokens: int = 384
    direct_match_max_tool_calls: int = 8
    route_or_fill_max_tokens: int = 512
    pattern_finder_max_tokens: int = 256
    focus_selector_max_tokens: int = 512
    focus_selector_max_tool_calls: int = 10
    mrcm_mapper_max_tokens: int = 512
    mrcm_mapper_max_tool_calls: int = 12
    retries: int = 2
    recursion_limit: int = 256
    stop: List[str] = field(default_factory=lambda: [END_MARKER])
    use_strict_prompts: bool = True
    extraction_only: bool = False
    categorize_slots_enabled: bool = True
    pattern_finder_enabled: bool = False   # deprecated: superseded by mrcm_attribute_mapper
    route_or_fill_enabled: bool = False    # deprecated: superseded by focus_selector + mrcm_attribute_mapper
    focus_selector_enabled: bool = True
    mrcm_mapper_enabled: bool = True
    direct_match_agent_enabled: bool = True
    # Phase IX: ToolNode conversion flags (require LANGGRAPH_LLM_BACKEND=vllm)
    direct_match_use_toolnode: bool = False
    focus_selector_use_toolnode: bool = False
    mrcm_mapper_use_toolnode: bool = False

    @classmethod
    def from_env(cls) -> "AgentRunConfig":
        def _flag(name: str, default: str = "0") -> bool:
            return os.environ.get(name, default).lower() not in {"0", "false", "no"}
        return cls(
            extraction_max_tokens=int(os.environ.get("AGENT_MAX_TOKENS_EXTRACT", "512")),
            decompose_max_tokens=int(os.environ.get("AGENT_MAX_TOKENS_DECOMPOSE", "512")),
            categorize_slots_max_tokens=int(os.environ.get("AGENT_MAX_TOKENS_CATEGORIZE", "256")),
            direct_match_max_tokens=int(os.environ.get("AGENT_MAX_TOKENS_DIRECT", "384")),
            direct_match_max_tool_calls=int(os.environ.get("DIRECT_MATCH_MAX_TOOL_CALLS", "8")),
            route_or_fill_max_tokens=int(os.environ.get("AGENT_MAX_TOKENS_ROUTE_FILL", "512")),
            pattern_finder_max_tokens=int(os.environ.get("AGENT_MAX_TOKENS_PATTERN_FINDER", "256")),
            focus_selector_max_tokens=int(os.environ.get("AGENT_MAX_TOKENS_FOCUS_SELECTOR", "512")),
            focus_selector_max_tool_calls=int(os.environ.get("FOCUS_SELECTOR_MAX_TOOL_CALLS", "10")),
            mrcm_mapper_max_tokens=int(os.environ.get("AGENT_MAX_TOKENS_MRCM_MAPPER", "512")),
            mrcm_mapper_max_tool_calls=int(os.environ.get("MRCM_MAPPER_MAX_TOOL_CALLS", "12")),
            retries=int(os.environ.get("AGENT_RETRIES", "2")),
            recursion_limit=int(os.environ.get("LANGGRAPH_RECURSION_LIMIT", "256")),
            use_strict_prompts=_flag("USE_STRICT_PROMPTS", "1"),
            categorize_slots_enabled=_flag("CATEGORIZE_SLOTS_ENABLED", "1"),
            pattern_finder_enabled=_flag("PATTERN_FINDER_ENABLED", "0"),
            route_or_fill_enabled=_flag("ROUTE_OR_FILL_ENABLED", "0"),
            focus_selector_enabled=_flag("FOCUS_SELECTOR_ENABLED", "1"),
            mrcm_mapper_enabled=_flag("MRCM_MAPPER_ENABLED", "1"),
            direct_match_agent_enabled=_flag("DIRECT_MATCH_AGENT_ENABLED", "1"),
            direct_match_use_toolnode=_flag("DIRECT_MATCH_USE_TOOLNODE"),
            focus_selector_use_toolnode=_flag("FOCUS_SELECTOR_USE_TOOLNODE"),
            mrcm_mapper_use_toolnode=_flag("MRCM_MAPPER_USE_TOOLNODE"),
        )


class ItemState(TypedDict, total=False):
    spl_set_id: str
    item: Dict[str, Any]
    candidates: Dict[str, List[Dict[str, Any]]]
    direct_match: Dict[str, Any]
    slot_hierarchies: Dict[str, Any]
    route_or_fill: Dict[str, Any]
    selected_problem_id: str
    selected_focus_term: str
    fills_norm: Dict[str, str]
    fills_detail: Dict[str, Dict[str, str]]
    postcoord_pattern: Dict[str, Any]
    focus_selector_trace: List[Dict[str, Any]]
    mrcm_mapper_trace: List[Dict[str, Any]]
    component_candidates: List[Dict[str, Any]]
    refinements: List[Dict[str, Any]]
    review_flag: bool
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

    use_bm25_tuning = os.environ.get("USE_BM25_TUNING", "1").lower() not in {"0", "false", "no"}
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

    return retrieve_candidates_for_item_hybrid(item, resources)


def search_snomed_for_agent(
    query: str,
    hierarchy_filter: Optional[str] = None,
    k: int = 10,
    graph_client: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Wrapper around search_snomed_concepts using the same resource bundle as the pipeline."""
    configure_process_cuda_visibility()
    use_bm25_tuning = os.environ.get("USE_BM25_TUNING", "1").lower() not in {"0", "false", "no"}
    bm25_b = 0.5 if use_bm25_tuning else 0.75
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
        graph_client=graph_client,
    )
    return search_snomed_concepts(query, resources, hierarchy_filter=hierarchy_filter, k=k)


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
    def __init__(self, llm: ChatLLM, cfg: Optional[AgentRunConfig] = None, observer: Optional[RunObserver] = None, extraction_cache: Optional[Dict[str, Dict[str, Any]]] = None, graph_client: Optional[Any] = None):
        self.llm = llm
        self.cfg = cfg or AgentRunConfig()
        self.observer = observer
        self.extraction_cache = extraction_cache  # None = disabled; {} = enabled but empty
        self.graph_client = graph_client          # Optional SnomedGraphClient for Phase 2 lookups
        # Phase IX: LangChain tools for ToolNode conversion (None when all flags off or backend lacks get_langchain_llm)
        self._agent_tools: Optional[list] = None
        if (
            any([self.cfg.direct_match_use_toolnode, self.cfg.focus_selector_use_toolnode, self.cfg.mrcm_mapper_use_toolnode])
            and hasattr(self.llm, "get_langchain_llm")
            and graph_client is not None
        ):
            from src.tools.schemas import make_agent_tools
            self._agent_tools = make_agent_tools(graph_client, search_snomed_for_agent)
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
            temperature=0.0,
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

    def _verify_direct_match(self, ci_text: str, focus_candidates: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        # ReAct path: self-contained — agent does its own search_snomed retrieval
        if self.cfg.direct_match_agent_enabled:
            return self._verify_direct_match_react(ci_text)

        # Simple chain fallback (agent disabled) — requires pre-fetched candidates
        candidates = focus_candidates or []
        system_prompt = (DIRECT_VERIFY_SYSTEM_PROMPT if self.cfg.use_strict_prompts
                         else DIRECT_VERIFY_SYSTEM_PROMPT_ORIGINAL)
        user = build_direct_verify_user_prompt(ci_text, candidates, max_n=10)
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

    def _verify_direct_match_react(self, ci_text: str) -> Dict[str, Any]:
        system = _load_direct_match_agent()
        if not system:
            return {
                "direct_match": False,
                "selected_id": "N/A",
                "selected_term": "N/A",
                "parse_failed": True,
            }

        user = build_direct_match_agent_user_prompt(ci_text)

        # Phase IX: ToolNode path — structured tool calls via LangGraph create_react_agent
        if self.cfg.direct_match_use_toolnode and self._agent_tools is not None:
            from langgraph.prebuilt import create_react_agent
            from langchain_core.messages import SystemMessage, HumanMessage
            dm_tools = [t for t in self._agent_tools if t.name in {"search_snomed", "get_logical_definition", "get_ancestors"}]
            agent = create_react_agent(
                model=self.llm.get_langchain_llm().bind_tools(dm_tools),
                tools=dm_tools,
                prompt=SystemMessage(content=system),
            )
            dm_started = time.perf_counter()
            result = agent.invoke({"messages": [HumanMessage(content=user)]})
            final_content = result["messages"][-1].content or ""
            parsed = parse_direct_match_agent_output(final_content)
            trace = [{"raw": getattr(m, "content", ""), "parsed": {}} for m in result["messages"]]
            self._log_llm_call(
                call_name="direct_match",
                system=system,
                user=user,
                max_tokens=self.cfg.direct_match_max_tokens,
                effective_max_tokens=self.cfg.direct_match_max_tokens,
                raw=final_content,
                parsed=parsed or None,
                duration_s=time.perf_counter() - dm_started,
            )
            if not parsed:
                return {"direct_match": False, "selected_id": "N/A", "selected_term": "N/A", "raw": final_content, "parse_failed": True, "trace": trace}
            return {**parsed, "trace": trace}

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        trace: List[Dict[str, Any]] = []
        parsed: Dict[str, Any] = {}
        dm_started = time.perf_counter()

        for _ in range(self.cfg.direct_match_max_tool_calls):
            raw = self.llm.chat(
                messages,
                max_tokens=self.cfg.direct_match_max_tokens,
                stop=None,
            )
            call = parse_direct_match_agent_output(raw)
            trace.append({"raw": raw, "parsed": call})

            if "direct_match" in call:  # final answer
                parsed = call
                break
            if "tool" in call:
                result = self._run_ontology_tool(call, self.graph_client)
                trace[-1]["result"] = result
                messages = self._extend_react_messages(messages, raw, result)

        self._log_llm_call(
            call_name="direct_match",
            system=system,
            user=user,
            max_tokens=self.cfg.direct_match_max_tokens,
            effective_max_tokens=self.cfg.direct_match_max_tokens,
            raw=trace[-1]["raw"] if trace else "",
            parsed=parsed or None,
            duration_s=time.perf_counter() - dm_started,
        )

        if not parsed:
            return {
                "direct_match": False,
                "selected_id": "N/A",
                "selected_term": "N/A",
                "raw": trace[-1]["raw"] if trace else "",
                "parse_failed": True,
                "trace": trace,
            }
        return {**parsed, "trace": trace}

    def _verify_direct_match_simple(self, ci_text: str, focus_candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
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

    # ------------------------------------------------------------------
    # Shared ReAct helpers — used by both _verify_direct_match_react and
    # focus_selector_node (inner functions in _build_item_graph delegate here)
    # ------------------------------------------------------------------

    @staticmethod
    def _run_ontology_tool(call: Dict[str, Any], graph_client: Any) -> Dict[str, Any]:
        tool = call.get("tool", "")
        args = call.get("args", {})
        try:
            if tool == "lookup_concept":
                results = graph_client.lookup_concept(
                    args.get("text", ""), args.get("hierarchy_filter")
                )
                return {"results": [r.__dict__ for r in results]}
            if tool == "get_logical_definition":
                results = graph_client.get_logical_definition(args.get("sctid", ""))
                return {"roles": [r.__dict__ for r in results]}
            if tool == "get_ancestors":
                results = graph_client.get_ancestors(
                    args.get("sctid", ""), max_depth=int(args.get("max_depth", 2))
                )
                return {"ancestors": [r.__dict__ for r in results]}
            if tool == "get_siblings":
                results = graph_client.get_siblings(
                    args.get("sctid", ""), limit=int(args.get("limit", 5))
                )
                return {"siblings": [r.__dict__ for r in results]}
            if tool == "get_domain_attributes":
                results = graph_client.get_domain_attributes(args.get("focus_sctid", ""))
                return {"attributes": results}
            if tool == "get_attribute_range":
                results = graph_client.get_attribute_range(args.get("attr_sctid", ""))
                return {"ranges": [r.__dict__ for r in results]}
            if tool == "search_snomed":
                results = search_snomed_for_agent(
                    args.get("query", ""),
                    hierarchy_filter=args.get("hierarchy_filter"),
                    k=int(args.get("k", 10)),
                    graph_client=graph_client,
                )
                return {"candidates": results}
            if tool == "validate_ecl":
                from src.snomed.snomed_utils import concept_matches_ecl, base as _SNOW_BASE
                sctid = args.get("sctid", "")
                range_ecl = args.get("range_ecl", "")
                if not sctid or not range_ecl:
                    return {"valid": False, "error": "missing sctid or range_ecl"}
                valid = concept_matches_ecl(
                    concept_id=int(sctid), ecl=range_ecl,
                    base=_SNOW_BASE, timeout=30, retries=1,
                )
                return {"valid": bool(valid)}
        except Exception as exc:
            return {"error": str(exc)}
        return {"error": f"unknown tool: {tool}"}

    @staticmethod
    def _extend_react_messages(
        messages: List[Dict[str, str]], tool_call_raw: str, result: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        return messages + [
            {"role": "assistant", "content": tool_call_raw},
            {"role": "user", "content": f"Tool result: {json.dumps(result, ensure_ascii=False)}"},
        ]

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
            direct = self._verify_direct_match(state["item"].get("ci_text", ""))
            return {**state, "direct_match": direct}

        def route_after_direct_match(state: ItemState) -> str:
            direct = state.get("direct_match", {})
            if direct.get("direct_match") is True and direct.get("selected_id") not in (None, "", "N/A"):
                return "assemble_direct"
            return "categorize_slots"

        # Hierarchy → item slot field mapping.
        # Keyed lowercase — LLM may return the hierarchy section name OR its semantic tags.
        _HIERARCHY_TO_SLOT: Dict[str, str] = {
            # Hierarchy section names (from conventions.md headers)
            "clinical finding":                  "contraindication_state_text",
            "procedure":                         "contraindication_state_text",
            "substance":                         "substance_text",
            "pharmaceutical/biological product": "substance_text",
            "organism":                          "substance_text",
            "qualifier value":                   "severity_span",
            # Semantic tags (what the LLM often outputs instead of the full hierarchy name)
            "disorder":                          "contraindication_state_text",
            "finding":                           "contraindication_state_text",
            "regime/therapy":                    "contraindication_state_text",
            "product":                           "substance_text",
            "medicinal product":                 "substance_text",
            "clinical drug":                     "substance_text",
        }

        def categorize_slots_node(state: ItemState) -> ItemState:
            if not self.cfg.categorize_slots_enabled:
                return state
            item = dict(state["item"])
            cat_started = time.perf_counter()
            components, raw = categorize_item_slots(
                self.llm.chat,
                item,
                max_tokens=self.cfg.categorize_slots_max_tokens,
                stop=None,
                retries=self.cfg.retries,
            )
            self._log_llm_call(
                call_name="categorize_slots",
                system=_load_snomed_conventions(),
                user=CATEGORIZE_SLOTS_USER_PROMPT.format(ci_text=item.get("ci_text", "")),
                max_tokens=self.cfg.categorize_slots_max_tokens,
                effective_max_tokens=self.cfg.categorize_slots_max_tokens,
                raw=raw,
                parsed={"components": components} if components else None,
                duration_s=time.perf_counter() - cat_started,
            )
            # Phase 2: resolve low-confidence components via graph_client lookup
            if self.graph_client is not None:
                for comp in components:
                    if comp.get("lookup_needed"):
                        try:
                            results = self.graph_client.lookup_concept(
                                comp["text"], hierarchy_filter=comp.get("hierarchy")
                            )
                            if results:
                                comp["resolved_preferred_term"] = results[0].preferred_term
                                comp["resolved_sctid"] = results[0].sctid
                        except Exception:
                            pass  # lookup failure is non-fatal

            slot_hierarchies: Dict[str, Any] = {}
            for comp in components:
                text = comp.get("text") or comp.get("span", "")
                resolved_text = comp.get("resolved_preferred_term") or text
                hierarchy = comp.get("hierarchy", "")
                slot_hierarchies[text] = {
                    "hierarchy": hierarchy,
                    "confidence": comp.get("confidence", ""),
                    "lookup_needed": comp.get("lookup_needed", False),
                    "resolved_preferred_term": comp.get("resolved_preferred_term"),
                    "resolved_sctid": comp.get("resolved_sctid"),
                }
                target_field = _HIERARCHY_TO_SLOT.get(hierarchy.lower())
                if target_field and not item.get(target_field):
                    item[target_field] = resolved_text
            return {**state, "item": item, "slot_hierarchies": slot_hierarchies}

        def retrieve_component_candidates_node(state: ItemState) -> ItemState:
            slot_hierarchies = state.get("slot_hierarchies", {})
            all_candidates: List[Dict[str, Any]] = []
            for text, meta in slot_hierarchies.items():
                query_text = meta.get("resolved_preferred_term") or text
                if not query_text:
                    continue
                component_item = {
                    "ci_text": query_text,
                    "contraindication_state_text": None,
                    "substance_text": None,
                    "severity_span": None,
                    "course_span": None,
                }
                try:
                    cands = retrieve_candidates_for_item(component_item)
                    focus_cands = cands.get("focus_candidates", [])
                    for c in focus_cands:
                        c["source_component"] = text
                        c["source_hierarchy"] = meta.get("hierarchy", "")
                    all_candidates.extend(focus_cands)
                except Exception:
                    pass  # per-component retrieval failure is non-fatal
            return {**state, "component_candidates": all_candidates}

        def _find_focus_component_text(slot_hierarchies: Dict[str, Any]) -> str:
            focus_hierarchies = {"clinical finding", "procedure", "disorder", "finding", "regime/therapy"}
            for text, meta in slot_hierarchies.items():
                if str(meta.get("hierarchy", "")).lower() in focus_hierarchies:
                    return str(meta.get("resolved_preferred_term") or text)
            return ""

        def _execute_focus_tool(call: Dict[str, Any], graph_client: Any) -> Dict[str, Any]:
            tool = call.get("tool", "")
            args = call.get("args", {})
            try:
                if tool == "lookup_concept":
                    results = graph_client.lookup_concept(
                        args.get("text", ""), args.get("hierarchy_filter")
                    )
                    return {"results": [r.__dict__ for r in results]}
                if tool == "get_logical_definition":
                    results = graph_client.get_logical_definition(args.get("sctid", ""))
                    return {"roles": [r.__dict__ for r in results]}
                if tool == "get_ancestors":
                    results = graph_client.get_ancestors(
                        args.get("sctid", ""), max_depth=int(args.get("max_depth", 2))
                    )
                    return {"ancestors": [r.__dict__ for r in results]}
                if tool == "get_siblings":
                    results = graph_client.get_siblings(
                        args.get("sctid", ""), limit=int(args.get("limit", 5))
                    )
                    return {"siblings": [r.__dict__ for r in results]}
            except Exception as exc:
                return {"error": str(exc)}
            return {"error": f"unknown tool: {tool}"}

        def _append_tool_result(
            messages: List[Dict[str, str]], tool_call_raw: str, result: Dict[str, Any]
        ) -> List[Dict[str, str]]:
            return messages + [
                {"role": "assistant", "content": tool_call_raw},
                {"role": "user", "content": f"Tool result: {json.dumps(result, ensure_ascii=False)}"},
            ]

        def _pick_top_focus_candidate(
            pool: List[Dict[str, Any]], slot_hierarchies: Dict[str, Any]
        ) -> Dict[str, Any]:
            focus_tags = {"clinical finding", "procedure", "disorder", "finding", "regime/therapy"}
            for c in pool:
                if str(c.get("source_hierarchy", "")).lower() in focus_tags:
                    return c
            return pool[0] if pool else {}

        def focus_selector_node(state: ItemState) -> ItemState:
            slot_hierarchies = state.get("slot_hierarchies", {})

            # Fallback: agent disabled — use a quick BM25+FAISS search on the focus text
            if not self.cfg.focus_selector_enabled:
                focus_text = _find_focus_component_text(slot_hierarchies)
                if focus_text:
                    hits = search_snomed_for_agent(focus_text, hierarchy_filter="Clinical Finding", k=5, graph_client=self.graph_client)
                    if hits:
                        top = hits[0]
                        return {
                            **state,
                            "selected_problem_id": str(top.get("id", "N/A")),
                            "selected_focus_term": str(top.get("label") or top.get("term", "N/A")),
                            "focus_selector_trace": [],
                        }
                return {**state, "selected_problem_id": "N/A", "selected_focus_term": "N/A", "focus_selector_trace": []}

            system = _load_focus_selector()
            user = build_focus_selector_user_prompt(state["item"], slot_hierarchies)

            # Phase IX: ToolNode path
            if self.cfg.focus_selector_use_toolnode and self._agent_tools is not None:
                from langgraph.prebuilt import create_react_agent
                from langchain_core.messages import SystemMessage, HumanMessage
                fs_tools = [t for t in self._agent_tools if t.name in {"search_snomed", "get_logical_definition", "get_ancestors", "get_siblings"}]
                agent = create_react_agent(
                    model=self.llm.get_langchain_llm().bind_tools(fs_tools),
                    tools=fs_tools,
                    prompt=SystemMessage(content=system),
                )
                fs_started = time.perf_counter()
                result = agent.invoke({"messages": [HumanMessage(content=user)]})
                final_content = result["messages"][-1].content or ""
                parsed = parse_focus_selector_output(final_content)
                trace: List[Dict[str, Any]] = [{"raw": getattr(m, "content", ""), "parsed": {}} for m in result["messages"]]
                self._log_llm_call(
                    call_name="focus_selector",
                    system=system,
                    user=user,
                    max_tokens=self.cfg.focus_selector_max_tokens,
                    effective_max_tokens=self.cfg.focus_selector_max_tokens,
                    raw=final_content,
                    parsed=parsed or None,
                    duration_s=time.perf_counter() - fs_started,
                )
            else:
                messages: List[Dict[str, str]] = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ]
                trace = []
                parsed = {}
                fs_started = time.perf_counter()

                for _ in range(self.cfg.focus_selector_max_tool_calls):
                    raw = self.llm.chat(
                        messages,
                        max_tokens=self.cfg.focus_selector_max_tokens,
                        stop=None,
                    )
                    call = parse_focus_selector_output(raw)
                    trace.append({"raw": raw, "parsed": call})

                    if "focus_sctid" in call:  # final answer
                        parsed = call
                        break
                    if "tool" in call:
                        result = self._run_ontology_tool(call, self.graph_client)
                        trace[-1]["result"] = result
                        messages = self._extend_react_messages(messages, raw, result)

                self._log_llm_call(
                    call_name="focus_selector",
                    system=system,
                    user=user,
                    max_tokens=self.cfg.focus_selector_max_tokens,
                    effective_max_tokens=self.cfg.focus_selector_max_tokens,
                    raw=trace[-1]["raw"] if trace else "",
                    parsed=parsed or None,
                    duration_s=time.perf_counter() - fs_started,
                )

            selected_id = str(parsed.get("focus_sctid", "N/A"))

            # Resolve term: scan search_snomed candidates in trace, then ancestor results
            selected_term = "N/A"
            for step in trace:
                for cand in step.get("parsed", {}).get("candidates", []):
                    if str(cand.get("id", "")) == selected_id:
                        selected_term = str(cand.get("label") or cand.get("term", "N/A"))
                        break
                for anc in step.get("parsed", {}).get("ancestors", []):
                    if str(anc.get("sctid", "")) == selected_id:
                        selected_term = str(anc.get("preferred_term", "N/A"))
                        break
                if selected_term != "N/A":
                    break
            if selected_term == "N/A":
                selected_term = str(parsed.get("reasoning", "N/A"))[:80]

            # Fallback: direct graph lookup when trace scan didn't find the term
            if selected_term == "N/A" and selected_id != "N/A" and self.graph_client is not None:
                try:
                    results = self.graph_client.lookup_concept(selected_id)
                    if results:
                        selected_term = (
                            getattr(results[0], "preferred_term", None)
                            or getattr(results[0], "fsn", None)
                            or "N/A"
                        )
                except Exception:
                    pass

            return {
                **state,
                "selected_problem_id": selected_id,
                "selected_focus_term": selected_term,
                "focus_selector_trace": trace,
            }

        def prefilter_node(state: ItemState) -> ItemState:
            return {**state, "candidates": prefilter_slot_candidates(state["candidates"])}

        def route_or_fill_node(state: ItemState) -> ItemState:
            if not self.cfg.route_or_fill_enabled:
                # v1 path disabled: pass state through with an empty route_or_fill stub
                # so downstream nodes (normalize v1 fallback) don't KeyError
                return {**state, "route_or_fill": {"post_decision": "N/A", "fills": {}}}
            result = self._route_or_fill(state["item"], state["candidates"])
            # Preserve focus_selector's selected_problem_id if it was set (Agent 2.5 output
            # takes priority over route_or_fill's own focus pick)
            if state.get("selected_problem_id") not in (None, "", "N/A"):
                result["selected_problem_id"] = state["selected_problem_id"]
            return {**state, "route_or_fill": result}

        def normalize_node(state: ItemState) -> ItemState:
            selected_problem_id = state.get("selected_problem_id", "N/A")
            selected_focus_term = state.get("selected_focus_term", "N/A")
            refinements = state.get("refinements", [])
            fills_norm:   Dict[str, str]            = {}
            fills_detail: Dict[str, Dict[str, str]] = {}

            if refinements:
                # v2 path: build SCTID-keyed fills from mrcm_attribute_mapper refinements[]
                for ref in refinements:
                    attr_sctid = str(ref.get("attribute_sctid", ""))
                    val_sctid  = str(ref.get("value_sctid", "N/A"))
                    val_term   = str(ref.get("value_term", "N/A"))
                    if attr_sctid and val_sctid != "N/A":
                        fills_norm[attr_sctid]   = val_sctid
                        fills_detail[attr_sctid] = {
                            "id": val_sctid,
                            "term": val_term,
                            "attribute_fsn": ref.get("attribute_fsn", ""),
                        }
            else:
                # v1 fallback: 3-slot fills from route_or_fill
                route_fill = state.get("route_or_fill", {})
                selected_problem_id = str(route_fill.get("selected_problem_id", selected_problem_id))
                fills = route_fill.get("fills", {}) or {}
                for key in ("causative_agent", "severity", "clinical_course"):
                    value = fills.get(key, "N/A")
                    if isinstance(value, dict):
                        value = value.get("id", "N/A")
                    fills_norm[key] = str(value)
                    fills_detail[key] = {
                        "id": fills_norm[key],
                        "term": candidate_label_by_id(
                            state.get("candidates", {}).get(f"{key}_candidates", []),
                            fills_norm[key],
                        ),
                    }
            return {
                **state,
                "selected_problem_id": selected_problem_id,
                "selected_focus_term": selected_focus_term,
                "fills_norm": fills_norm,
                "fills_detail": fills_detail,
            }

        def pattern_finder_node(state: ItemState) -> ItemState:
            if not self.cfg.pattern_finder_enabled:
                return {**state, "postcoord_pattern": {}}
            selected_problem_id = state.get("selected_problem_id", "N/A")
            fills_norm = state.get("fills_norm", {})
            fills_detail = state.get("fills_detail", {})
            item = state["item"]
            analogues_context: Dict[str, Any] = {}
            if self.graph_client is not None and selected_problem_id.isdigit():
                try:
                    roles = self.graph_client.get_logical_definition(selected_problem_id)
                    analogues_context["focus_roles"] = [r.__dict__ for r in roles]
                    analogues_context["ancestors"] = self.graph_client.get_ancestors(
                        selected_problem_id, max_depth=2
                    )
                    for slot_key in ("causative_agent", "severity", "clinical_course"):
                        val_id = fills_norm.get(slot_key, "N/A")
                        if val_id != "N/A" and val_id.isdigit():
                            slot_roles = self.graph_client.get_logical_definition(val_id)
                            analogues_context[f"{slot_key}_roles"] = [r.__dict__ for r in slot_roles]
                except Exception:
                    pass  # graph queries are non-fatal
            pf_started = time.perf_counter()
            parsed, raw = run_pattern_finder(
                self.llm.chat,
                item,
                analogues_context,
                fills_norm,
                selected_problem_id,
                fills_detail,
                max_tokens=self.cfg.pattern_finder_max_tokens,
                stop=None,
                retries=self.cfg.retries,
            )
            self._log_llm_call(
                call_name="pattern_finder",
                system=system,
                user=f"FOCUS:{selected_problem_id} CI:{item.get('ci_text','')}",
                max_tokens=self.cfg.pattern_finder_max_tokens,
                effective_max_tokens=self.cfg.pattern_finder_max_tokens,
                raw=raw,
                parsed=parsed or None,
                duration_s=time.perf_counter() - pf_started,
            )
            return {**state, "postcoord_pattern": parsed}

        def mrcm_attribute_mapper_node(state: ItemState) -> ItemState:
            if not self.cfg.mrcm_mapper_enabled:
                return {**state, "refinements": [], "review_flag": False}

            focus_sctid      = state.get("selected_problem_id", "N/A")
            focus_term       = state.get("selected_focus_term", "N/A")
            slot_hierarchies = state.get("slot_hierarchies", {})

            # If focus_term looks like reasoning text (too long) or is missing,
            # resolve the preferred term directly from graph_client
            if focus_sctid != "N/A" and (focus_term == "N/A" or len(focus_term) > 60) \
                    and self.graph_client is not None:
                try:
                    _res = self.graph_client.lookup_concept(focus_sctid)
                    if _res:
                        focus_term = getattr(_res[0], "preferred_term", None) \
                                  or getattr(_res[0], "fsn", None) \
                                  or focus_term
                except Exception:
                    pass

            system = _load_postcord_agent()
            user = build_mrcm_mapper_react_user_prompt(
                state["item"], focus_sctid, focus_term, slot_hierarchies
            )

            # Phase IX: ToolNode path
            if self.cfg.mrcm_mapper_use_toolnode and self._agent_tools is not None:
                from langgraph.prebuilt import create_react_agent
                from langchain_core.messages import SystemMessage, HumanMessage
                mm_tools = [t for t in self._agent_tools if t.name in {"get_domain_attributes", "get_attribute_range", "search_snomed", "get_logical_definition", "validate_ecl"}]
                agent = create_react_agent(
                    model=self.llm.get_langchain_llm().bind_tools(mm_tools),
                    tools=mm_tools,
                    prompt=SystemMessage(content=system),
                )
                mapper_started = time.perf_counter()
                result = agent.invoke({"messages": [HumanMessage(content=user)]})
                final_content = result["messages"][-1].content or ""
                parsed = parse_mrcm_mapper_output(final_content)
                # Build trace from LangGraph messages for parity with Python-loop path
                trace: List[Dict[str, Any]] = []
                for msg in result.get("messages", []):
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            trace.append({"tool": tc["name"], "args": tc["args"], "result": None})
                    elif getattr(msg, "name", None) and trace and trace[-1]["result"] is None:
                        trace[-1]["result"] = msg.content
                self._log_llm_call(
                    call_name="mrcm_attribute_mapper",
                    system=system,
                    user=user,
                    max_tokens=self.cfg.mrcm_mapper_max_tokens,
                    effective_max_tokens=self.cfg.mrcm_mapper_max_tokens,
                    raw=final_content,
                    parsed=parsed or None,
                    duration_s=time.perf_counter() - mapper_started,
                )
            else:
                messages: List[Dict[str, str]] = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ]
                trace: List[Dict[str, Any]] = []
                parsed = {}
                mapper_started = time.perf_counter()

                for _ in range(self.cfg.mrcm_mapper_max_tool_calls):
                    raw = self.llm.chat(
                        messages,
                        max_tokens=self.cfg.mrcm_mapper_max_tokens,
                        stop=None,
                    )
                    call = parse_mrcm_mapper_output(raw)
                    trace.append({"raw": raw, "parsed": call})

                    if "refinements" in call or "decision" in call:  # final answer
                        parsed = call
                        break
                    if "tool" in call:
                        result = self._run_ontology_tool(call, self.graph_client)
                        trace[-1]["result"] = result
                        messages = self._extend_react_messages(messages, raw, result)

                self._log_llm_call(
                    call_name="mrcm_attribute_mapper",
                    system=system,
                    user=user,
                    max_tokens=self.cfg.mrcm_mapper_max_tokens,
                    effective_max_tokens=self.cfg.mrcm_mapper_max_tokens,
                    raw=trace[-1]["raw"] if trace else "",
                    parsed=parsed or None,
                    duration_s=time.perf_counter() - mapper_started,
                )

            refinements = parsed.get("refinements", []) if parsed else []
            review_flag = parsed.get("confidence", 1.0) < 0.5 if parsed else True
            return {**state, "refinements": refinements, "postcoord_pattern": parsed or {},
                    "mrcm_mapper_trace": trace, "review_flag": review_flag}

        def expression_validator_node(state: ItemState) -> ItemState:
            selected_problem_id = state.get("selected_problem_id", "N/A")
            selected_focus_term = state.get("selected_focus_term", "N/A")
            fills_norm = dict(state.get("fills_norm", {}))
            fills_detail = dict(state.get("fills_detail", {}))
            route_fill = dict(state.get("route_or_fill", {}))
            postcoord_pattern = state.get("postcoord_pattern", {})

            # fills_norm is already populated by normalize_node (from refinements[] or route_or_fill)
            # No override needed here — expression_validator only validates, not fills

            # Sanitise non-digit IDs (preserves previous validate_node behaviour)
            if selected_problem_id != "N/A" and not selected_problem_id.isdigit():
                selected_problem_id = "N/A"
                selected_focus_term = "N/A"
            for key in ("causative_agent", "severity", "clinical_course"):
                if fills_detail.get(key, {}).get("id", "N/A") != "N/A" and not str(fills_detail[key]["id"]).isdigit():
                    fills_detail[key] = {"id": "N/A", "term": "N/A"}
                    fills_norm[key] = "N/A"

            # MRCM range check — iterate refinements[] natively (informational; does not block)
            range_errors: List[str] = []
            if self.graph_client is not None:
                for ref in state.get("refinements", []):
                    attr_sctid = str(ref.get("attribute_sctid", ""))
                    val_sctid  = str(ref.get("value_sctid", "N/A"))
                    if attr_sctid and val_sctid != "N/A" and val_sctid.isdigit():
                        try:
                            ranges = self.graph_client.get_attribute_range(attr_sctid)
                            if not ranges:
                                range_errors.append(
                                    f"{ref.get('attribute_fsn', attr_sctid)}: no MRCM range"
                                )
                        except Exception:
                            pass

            # Final gate: existing MRCM validation (unchanged)
            ok, fail_reasons = validate_postcoord_with_mrcm(selected_problem_id, fills_norm)
            if not ok:
                route_fill["post_decision"] = "N/A"

            review_flag = (not ok) or (postcoord_pattern.get("confidence", 1.0) < 0.5)
            return {
                **state,
                "selected_problem_id": selected_problem_id,
                "selected_focus_term": selected_focus_term,
                "fills_norm": fills_norm,
                "fills_detail": fills_detail,
                "route_or_fill": route_fill,
                "validation": {"ok": ok, "fail_reasons": fail_reasons, "range_errors": range_errors},
                "review_flag": review_flag,
            }

        def route_after_mapper(state: ItemState) -> str:
            return "assemble_review" if state.get("review_flag", False) else "assemble_postcoord"

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
            for attr_key, val_id in state["fills_norm"].items():
                if val_id == "N/A":
                    continue
                # attr_key is a SCTID string (v2) or a slot name like "causative_agent" (v1)
                attr_sctid = attr_key if attr_key.isdigit() else str(
                    self.cfg.attribute_table.get(attr_key, "")
                )
                if attr_sctid:
                    ax_pairs.append(f"{attr_sctid}={val_id}")
            expression = (
                f"{state['selected_problem_id']}:{{{','.join(ax_pairs)}}}"
                if ax_pairs
                else state["selected_problem_id"]
            )
            post_decision = str(state.get("route_or_fill", {}).get("post_decision", "N/A"))
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
                        "focus_selector": state.get("focus_selector_trace", []),
                        "mrcm_mapper": state.get("mrcm_mapper_trace", []),
                        "postcoord_pattern": state.get("postcoord_pattern", {}),
                    },
                    "extracted_item": item,
                },
            }

        def assemble_review_node(state: ItemState) -> ItemState:
            ax_pairs: List[str] = []
            for attr_key, val_id in state["fills_norm"].items():
                if val_id == "N/A":
                    continue
                attr_sctid = attr_key if attr_key.isdigit() else str(
                    self.cfg.attribute_table.get(attr_key, "")
                )
                if attr_sctid:
                    ax_pairs.append(f"{attr_sctid}={val_id}")
            expression = (
                f"{state['selected_problem_id']}:{{{','.join(ax_pairs)}}}"
                if ax_pairs
                else state["selected_problem_id"]
            )
            item = state["item"]
            return {
                **state,
                "expression": expression,
                "status": "REVIEW",
                "item_result": {
                    "SPL_SET_ID": state["spl_set_id"],
                    "item_index": item.get("item_index"),
                    "query_text": item.get("ci_text", ""),
                    "status": "REVIEW",
                    "review_flag": True,
                    "post_decision": str(state.get("route_or_fill", {}).get("post_decision", "N/A")),
                    "selected_problem_id": state["selected_problem_id"],
                    "selected_focus_term": state["selected_focus_term"],
                    "fills": state["fills_detail"],
                    "expression": expression,
                    "trace": {
                        "direct_verify": state.get("direct_match", {}),
                        "focus_selector": state.get("focus_selector_trace", []),
                        "mrcm_mapper": state.get("mrcm_mapper_trace", []),
                        "postcoord_pattern": state.get("postcoord_pattern", {}),
                    },
                    "extracted_item": item,
                },
            }

        graph.add_node("direct_match", instrument_item_node("direct_match", direct_match_node))
        graph.add_node("categorize_slots", instrument_item_node("categorize_slots", categorize_slots_node))
        graph.add_node("focus_selector", instrument_item_node("focus_selector", focus_selector_node))
        graph.add_node("route_or_fill", instrument_item_node("route_or_fill", route_or_fill_node))
        graph.add_node("normalize", instrument_item_node("normalize", normalize_node))
        graph.add_node("pattern_finder", instrument_item_node("pattern_finder", pattern_finder_node))
        graph.add_node("mrcm_attribute_mapper", instrument_item_node("mrcm_attribute_mapper", mrcm_attribute_mapper_node))
        graph.add_node("assemble_direct", instrument_item_node("assemble_direct", assemble_direct_node))
        graph.add_node("assemble_postcoord", instrument_item_node("assemble_postcoord", assemble_postcoord_node))
        graph.add_node("assemble_review", instrument_item_node("assemble_review", assemble_review_node))

        graph.set_entry_point("direct_match")
        graph.add_conditional_edges(
            "direct_match",
            route_after_direct_match,
            {"assemble_direct": "assemble_direct", "categorize_slots": "categorize_slots"},
        )
        graph.add_edge("categorize_slots", "focus_selector")
        if self.cfg.route_or_fill_enabled:
            graph.add_edge("focus_selector", "route_or_fill")
            graph.add_edge("route_or_fill", "normalize")
        else:
            graph.add_edge("focus_selector", "normalize")
        graph.add_edge("normalize", "mrcm_attribute_mapper")
        graph.add_conditional_edges(
            "mrcm_attribute_mapper",
            route_after_mapper,
            {"assemble_postcoord": "assemble_postcoord", "assemble_review": "assemble_review"},
        )
        graph.add_edge("assemble_direct", END)
        graph.add_edge("assemble_postcoord", END)
        graph.add_edge("assemble_review", END)

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

        def bootstrap_node(state: ContraState) -> ContraState:
            if self.extraction_cache is None:  # None = caching disabled; {} = enabled but empty
                return state
            spl_record = state.get("spl_record", {})
            spl_set_id = (spl_record.get("SPL_SET_ID") or spl_record.get("spl_set_id") or "").strip()
            entry = self.extraction_cache.get(spl_set_id)
            if entry is None:
                return state
            return {
                **state,
                "spl_set_id": spl_set_id,
                "product_name": entry.get("product_name"),
                "contra_section_found": entry.get("contra_section_found", False),
                "contra_section_text": entry.get("contra_section_text", ""),
                "extracted_items": entry.get("extracted_items", []),
                "current_index": 0,
                "item_results": [],
            }

        def route_after_bootstrap(state: ContraState) -> str:
            if "extracted_items" not in state:   # no cache hit — run full extraction
                return "resolve_contra_section"
            if not state.get("extracted_items") or self.cfg.extraction_only:
                return "finalize"
            return "prepare_item"                 # cache hit, items ready

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
            section_text = spl_context.get("contra_section_text", "")
            extraction_started = time.perf_counter()
            items, raw = extract_contraindication_items(
                self.llm.chat,
                section_text,
                max_tokens=self.cfg.extraction_max_tokens,
                stop=None,
                retries=self.cfg.retries,
                system_prompt=SIMPLE_EXTRACT_SYSTEM_PROMPT or CONTRA_EXTRACT_SYSTEM_PROMPT,
                user_prompt_template=SIMPLE_EXTRACT_USER_PROMPT or CONTRA_EXTRACT_USER_PROMPT,
            )
            active_system = SIMPLE_EXTRACT_SYSTEM_PROMPT or CONTRA_EXTRACT_SYSTEM_PROMPT
            active_user_tmpl = SIMPLE_EXTRACT_USER_PROMPT or CONTRA_EXTRACT_USER_PROMPT
            parsed_payload: Optional[Dict[str, Any]] = {"items": items} if items else None
            self._log_llm_call(
                call_name="extract_contraindications",
                system=active_system,
                user=active_user_tmpl.format(text=section_text),
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

        def decompose_coordinations_node(state: ContraState) -> ContraState:
            items = state.get("extracted_items", [])
            if not items:
                return state
            decomposed_all: List[Dict[str, Any]] = []
            for item in items:
                decompose_started = time.perf_counter()
                results, raw = decompose_contraindication_item(
                    self.llm.chat,
                    item,
                    max_tokens=self.cfg.decompose_max_tokens,
                    stop=None,
                    retries=self.cfg.retries,
                )
                self._log_llm_call(
                    call_name="decompose_coordination",
                    system=DECOMPOSE_SYSTEM_PROMPT,
                    user=build_decompose_user_prompt(item),
                    max_tokens=self.cfg.decompose_max_tokens,
                    effective_max_tokens=self.cfg.decompose_max_tokens,
                    raw=raw,
                    parsed={"items": results} if results else None,
                    duration_s=time.perf_counter() - decompose_started,
                )
                decomposed_all.extend(results)
            indexed = [{**it, "item_index": i} for i, it in enumerate(decomposed_all)]
            return {**state, "extracted_items": indexed}

        def route_after_decompose(state: ContraState) -> str:
            if not state.get("extracted_items") or self.cfg.extraction_only:
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
            final: Dict[str, Any] = {
                "SPL_SET_ID": state["spl_set_id"],
                "product_name": state.get("product_name"),
                "contra_section_found": bool(state.get("contra_section_found")),
                "contra_section_text": state.get("contra_section_text", ""),
                "n_items_in": len(extracted_items),
                "n_items_out": len(item_results),
                "results": item_results,
            }
            if self.cfg.extraction_only:
                final["extracted_items"] = extracted_items
            return {**state, "final_result": final}

        graph.add_node("bootstrap", instrument_spl_node("bootstrap", bootstrap_node))
        graph.add_node("resolve_contra_section", instrument_spl_node("resolve_contra_section", resolve_contra_section_node))
        graph.add_node("extract_items", instrument_spl_node("extract_items", extract_items_node))
        graph.add_node("decompose_coordinations", instrument_spl_node("decompose_coordinations", decompose_coordinations_node))
        graph.add_node("prepare_item", instrument_spl_node("prepare_item", prepare_item_node))
        graph.add_node("process_item", instrument_spl_node("process_item", process_item_node))
        graph.add_node("advance_item", instrument_spl_node("advance_item", advance_item_node))
        graph.add_node("finalize", instrument_spl_node("finalize", finalize_node))

        graph.set_entry_point("bootstrap")
        graph.add_conditional_edges(
            "bootstrap",
            route_after_bootstrap,
            {"resolve_contra_section": "resolve_contra_section", "prepare_item": "prepare_item", "finalize": "finalize"},
        )
        graph.add_conditional_edges(
            "resolve_contra_section",
            route_after_resolve,
            {"extract_items": "extract_items", "finalize": "finalize"},
        )
        graph.add_edge("extract_items", "decompose_coordinations")
        graph.add_conditional_edges(
            "decompose_coordinations",
            route_after_decompose,
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
        choices=("azure", "huggingface", "vllm"),
        help="LLM backend to use.",
    )
    parser.add_argument(
        "--extraction-only",
        action="store_true",
        help="Stop after extraction — skip candidate retrieval and mapping. Output includes extracted_items per SPL.",
    )
    parser.add_argument(
        "--enable-graph-client",
        action="store_true",
        help="Enable SnomedGraphClient lookups in categorize_slots for low-confidence components (Phase 2).",
    )
    parser.add_argument(
        "--extracted-items-cache",
        default="",
        help=(
            "Path to extraction cache JSONL. On first run the file is created and "
            "populated as each SPL completes. On subsequent runs the file is loaded "
            "so the extract_items LLM call is skipped entirely."
        ),
    )
    args = parser.parse_args()

    # Load extraction cache if it already exists; otherwise we'll build it during this run.
    extraction_cache: Optional[Dict[str, Dict[str, Any]]] = None
    cache_was_loaded = False
    if args.extracted_items_cache and Path(args.extracted_items_cache).exists():
        extraction_cache = load_extraction_cache(args.extracted_items_cache)
        cache_was_loaded = True
        print(f"Loaded extraction cache ({len(extraction_cache)} entries): {args.extracted_items_cache}")

    llm = build_llm(args.backend)
    observer = RunObserver(
        audit_path=args.audit_jsonl,
        audit_enabled=not args.disable_audit,
        progress_enabled=not args.disable_progress,
    )
    cfg = replace(AgentRunConfig.from_env(), extraction_only=args.extraction_only)
    print(f"Agent Config: {cfg}")

    graph_client = None
    if args.enable_graph_client:
        from src.snomed.graph_client import SnomedGraphClient
        graph_client = SnomedGraphClient()
        print("SnomedGraphClient: connected (Phase 2 lookups enabled)")

    agent = ContraLangGraphAgent(llm=llm, cfg=cfg, observer=observer, extraction_cache=extraction_cache, graph_client=graph_client)

    spl_records = load_spl_records_from_file(args.spl_list)
    run_rows: List[Dict[str, Any]] = []

    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for spl_index, spl in enumerate(spl_records, 1):
            observer.set_spl_context(spl_index=spl_index, spl_total=len(spl_records))
            result = agent.process_spl(spl)
            run_rows.append(result)
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            # Append to cache immediately after each SPL so progress survives a crash.
            if args.extracted_items_cache and not cache_was_loaded and not result.get("error"):
                append_extraction_cache(args.extracted_items_cache, {
                    "spl_set_id": result["SPL_SET_ID"],
                    "product_name": result.get("product_name"),
                    "contra_section_found": result.get("contra_section_found", False),
                    "contra_section_text": result.get("contra_section_text", ""),
                    "extracted_items": [r["extracted_item"] for r in result.get("results", [])],
                })
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
