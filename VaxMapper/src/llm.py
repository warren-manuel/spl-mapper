
"""
Features
--------
1) load_model_local(model_id, **kwargs)
   - Loads a local Hugging Face causal LLM + tokenizer with sensible defaults.
   - Returns a LocalLLM instance with a .generate(messages, **gen_kwargs) method.

2) message builders
   - build_message(system, user) -> [{"role":"system",...}, {"role":"user",...}]
   - build_messages_from_iter(system_template, user_template, iterable, **fmt_kwargs)
     where iterable yields dicts (or (name, text) tuples). Useful for e.g., iterating
     over XML sections and generating a prompt per section.

3) XML helpers (optional)
   - xml_section_iter(xml_source, xpaths=None, tags=None, strip=True)
     Yields dictionaries like {"section": "<name>", "text": "<content>"}.
     Uses lxml if available; otherwise falls back to ElementTree with simple tag search.

Examples
--------
>>> from llm_runner import load_model_local, build_messages_from_iter, xml_section_iter
>>> llm = load_model_local("meta-llama/Llama-3.1-8B-Instruct")
>>> system = "You extract adverse reactions. Reply JSON only."
>>> sections = xml_section_iter("TENIVAC.xml", tags=["ADVERSE REACTIONS", "CONTRAINDICATIONS"])
>>> msgs = build_messages_from_iter(system, "Section: {section}\n\n{text}", sections)
>>> out = [llm.generate(m, max_new_tokens=256) for m in msgs]
>>> print(out[0]["text"])
"""

from __future__ import annotations

import os
import json
import copy
import inspect
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Generator, List, Optional, Tuple, Union

# ---- Optional torch import (only required if you pass a torch_dtype) ----
try:
    import torch  # type: ignore
    _TORCH_AVAILABLE = True
except Exception:
    _TORCH_AVAILABLE = False

# ---- Transformers (required) ----
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer,
)


# ==============================
# LLM Wrapper
# ==============================

@dataclass
class LocalLLM:
    model: Any
    tokenizer: Any
    chat_template_fallback: str = field(default="\n\n")
    enable_thinking: Optional[bool] = False  
    def _format_chat(self, messages: List[Dict[str, str]]) -> str:
        """
        Use tokenizer.apply_chat_template if available; otherwise do a simple join.
        """
        apply = getattr(self.tokenizer, "apply_chat_template", None)
        if callable(apply):
            return apply(messages, add_generation_prompt=True, tokenize=False, enable_thinking=self.enable_thinking)
        # Fallback: concatenate system + user+assistant lines
        parts = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                parts.append(f"[SYSTEM]\n{content}")
            elif role == "user":
                parts.append(f"[USER]\n{content}")
            else:
                parts.append(f"[{role.upper()}]\n{content}")
        parts.append("[ASSISTANT]\n")
        return self.chat_template_fallback.join(parts)

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.95,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        do_sample: Optional[bool] = None,
        stop: Optional[List[str]] = None,
        **gen_kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Single-turn generation. Pass a list of {'role','content'} dicts.

        Returns a dict: {'text': <str>, 'raw': <transformers output>}
        """
        if do_sample is None:
            # Default to sampling only if temperature > ~0
            do_sample = temperature > 0.0

        # Format input
        prompt = self._format_chat(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt")

        # Move to model device if necessary
        if hasattr(self.model, "device") and self.model.device.type != "meta":
            for k, v in inputs.items():
                if hasattr(v, "to"):
                    inputs[k] = v.to(self.model.device)

        eos_token_ids = gen_kwargs.pop("eos_token_id", None)

        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "eos_token_id": eos_token_ids,
            "pad_token_id": (
                getattr(self.tokenizer, "pad_token_id", None)
                or getattr(self.tokenizer, "eos_token_id", None)
            ),
            **gen_kwargs,
        }
        if do_sample:
            # Sampling-only params; avoid passing when do_sample=False to suppress warnings.
            generate_kwargs["temperature"] = temperature
            generate_kwargs["top_p"] = top_p
            if top_k is not None:
                generate_kwargs["top_k"] = top_k
        elif hasattr(self.model, "generation_config") and self.model.generation_config is not None:
            # Some model configs carry default sampling params (e.g., top_p/top_k),
            # which can trigger warnings during greedy decoding unless cleared.
            gen_cfg = copy.deepcopy(self.model.generation_config)
            for k in ("top_p", "top_k", "temperature"):
                if hasattr(gen_cfg, k):
                    setattr(gen_cfg, k, None)
            generate_kwargs["generation_config"] = gen_cfg
        if repetition_penalty is not None:
            generate_kwargs["repetition_penalty"] = repetition_penalty

        output = self.model.generate(
            **inputs,
            **generate_kwargs,
        )

        # text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        # # Try to strip the prompt if present in decode
        # if text.startswith(prompt):
        #     text = text[len(prompt):].lstrip()
        response = self.tokenizer.decode(output[0][len(inputs[0]):], skip_special_tokens=True)
        if stop:
            for marker in stop:
                if marker and marker in response:
                    response = response.split(marker)[0]
                    break
        return response
    
# ==============================
# Loader
# ==============================

def load_model_local(
    model_id: str,
    device_map: Union[str, Dict[str, int]] = "auto",
    torch_dtype: Union[str, "torch.dtype", None] = "auto",
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    trust_remote_code: bool = True,
    use_fast_tokenizer: bool = True,
    **model_kwargs: Any,
) -> LocalLLM:
    """
    Load a local CausalLM + tokenizer by model_id.

    Parameters
    ----------
    model_id: HF model id or local path.
    device_map: "auto" or explicit device map.
    torch_dtype: "auto", None, or a torch.dtype (e.g., torch.bfloat16).
    load_in_8bit / load_in_4bit: BitsAndBytes quantization flags.
    trust_remote_code: Allow custom model code.
    use_fast_tokenizer: Use fast tokenizers if available.
    **model_kwargs: Forwarded to from_pretrained (e.g., attn_implementation="flash_attention_2").

    Returns
    -------
    LocalLLM
    """
    tok = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=use_fast_tokenizer,
        trust_remote_code=trust_remote_code,
    )

    # Safety: ensure a pad token exists if missing
    if tok.pad_token is None:
        if tok.eos_token:
            tok.pad_token = tok.eos_token
        else:
            tok.add_special_tokens({"pad_token": "<|pad|>"})

    # dtype handling
    dtype_arg = None
    if torch_dtype == "auto":
        dtype_arg = "auto"
    elif _TORCH_AVAILABLE and isinstance(torch_dtype, torch.dtype):
        dtype_arg = torch_dtype
    else:
        dtype_arg = None if torch_dtype is None else torch_dtype  # pass-through string if user really wants to

    from_pretrained_kwargs = {
        "device_map": device_map,
        # "load_in_8bit": load_in_8bit,
        # "load_in_4bit": load_in_4bit,
        "trust_remote_code": trust_remote_code,
        **model_kwargs,
    }
    from_pretrained_sig = inspect.signature(AutoModelForCausalLM.from_pretrained)
    if "dtype" in from_pretrained_sig.parameters:
        from_pretrained_kwargs["dtype"] = dtype_arg
    else:
        from_pretrained_kwargs["torch_dtype"] = dtype_arg

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        **from_pretrained_kwargs,
    )

    # Only resize when special token insertion actually changed tokenizer length.
    if hasattr(model, "resize_token_embeddings"):
        input_embeddings = getattr(model, "get_input_embeddings", lambda: None)()
        current_size = getattr(input_embeddings, "num_embeddings", None)
        target_size = len(tok)
        if current_size is not None and current_size != target_size:
            model.resize_token_embeddings(target_size, mean_resizing=False)

    return LocalLLM(model=model, tokenizer=tok)

# ==============================
# Message Builders
# ==============================

def build_message(system: str, user: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

def build_messages_from_iter(
    system_template: str,
    user_template: str,
    iterable: Iterable[Union[Dict[str, Any], Tuple[str, str]]],
    **fmt_kwargs: Any,
) -> List[List[Dict[str, str]]]:
    """
    Given a system + user template and an iterable that yields dicts or (name, text) tuples,
    return a list of chat message lists.
    """
    chats: List[List[Dict[str, str]]] = []

    # Format the system once (allow late-binding with fmt_kwargs).
    def _render_system(payload: Dict[str, Any]) -> str:
        data = {**fmt_kwargs, **payload}
        return system_template.format(**data)

    for item in iterable:
        if isinstance(item, tuple):
            # Assume (name, text)
            payload = {"section": item[0], "text": item[1]}
        elif isinstance(item, dict):
            payload = item
        else:
            payload = {"text": str(item)}

        # Ensure keys for common placeholders
        payload.setdefault("section", payload.get("name", ""))
        payload.setdefault("text", payload.get("content", ""))

        # print(payload)

        sys_msg = _render_system(payload)
        usr_msg = user_template.format(**{**fmt_kwargs, **payload})
        chats.append(build_message(sys_msg, usr_msg))
    return chats



# ==============================
# Convenience batch runner
# ==============================

def generate_over_iterable(
    llm: LocalLLM,
    system_template: str,
    user_template: str,
    iterable: Iterable[Union[Dict[str, Any], Tuple[str, str]]],
    stop: Optional[List[str]] = None,
    **gen_kwargs: Any,
) -> List[Dict[str, Any]]:
    """
    Build messages for each item from iterable and call LLM.generate on each.
    Returns a list of dicts with {'input': payload, 'messages': messages, 'output': text}.
    """
    results: List[Dict[str, Any]] = []
    chats = build_messages_from_iter(system_template, user_template, iterable)
    for messages, payload in zip(chats, iterable):
        out = llm.generate(messages, stop=stop, **gen_kwargs)
        results.append({"input": payload, "messages": messages, "output": out["text"]})
    return results


# ==============================
# LLM Message JSON Parsing
# ==============================

def _find_json_span(text: str) -> Optional[Tuple[int, int]]:
    """
    Heuristically find the first top-level {...} or [...] span in the text
    using a simple stack-based scanner. Returns (start, end) indices or None.
    """
    for open_ch, close_ch in [("{", "}"), ("[", "]")]:
        start = text.find(open_ch)
        while start != -1:
            depth = 0
            for i, ch in enumerate(text[start:], start):
                if ch == open_ch:
                    depth += 1
                elif ch == close_ch:
                    depth -= 1
                    if depth == 0:
                        return start, i + 1  # end index is exclusive
            # If we got here, parentheses didn't balance; try another starting point
            start = text.find(open_ch, start + 1)
    return None

### NEW: 02/18/2026

def _find_outer_json_span(text: str) -> Optional[Tuple[int, int]]:
    """
    Find a balanced top-level JSON span ONLY if it starts at the first '{' or '['.
    If the outer JSON is truncated/unbalanced, return None (do not fall back to inner objects).
    """
    i = 0
    n = len(text)
    # find first brace/bracket
    while i < n and text[i].isspace():
        i += 1
    # allow some leading junk like ```json\n
    first_curly = text.find("{")
    first_square = text.find("[")
    if first_curly == -1 and first_square == -1:
        return None
    start = first_curly if (first_square == -1 or (first_curly != -1 and first_curly < first_square)) else first_square
    open_ch = text[start]
    close_ch = "}" if open_ch == "{" else "]"

    depth = 0
    for j in range(start, n):
        ch = text[j]
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return start, j + 1  # end exclusive

    # unbalanced => truncated outer JSON
    return None

def _strip_code_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        # drop first fence line (``` or ```json)
        if lines:
            lines = lines[1:]
        # drop trailing fence line if present
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    return t

def _collect_balanced_objects(s: str) -> List[str]:
    """
    Collect balanced {...} substrings from s (top-level objects in a stream).
    Useful for salvaging items if output is truncated.
    """
    objs = []
    i, n = 0, len(s)
    while i < n:
        if s[i] != "{":
            i += 1
            continue
        depth = 0
        start = i
        j = i
        while j < n:
            ch = s[j]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    objs.append(s[start:j+1])
                    i = j + 1
                    break
            j += 1
        else:
            # ran out => truncated object
            break
    return objs

def _salvage_items_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Salvage complete item objects that look like {"span_text":..., "condition_text":...}
    even when the overall JSON is truncated.
    """
    # Try to anchor to the items array if possible
    items_idx = text.find('"items"')
    if items_idx == -1:
        items_idx = text.find("'items'")
    if items_idx != -1:
        bracket = text.find("[", items_idx)
        if bracket != -1:
            stream = text[bracket+1:]
            obj_strs = _collect_balanced_objects(stream)
            items = []
            for o in obj_strs:
                oo = o.replace(",}", "}")  # mild repair
                try:
                    d = json.loads(oo)
                    if isinstance(d, dict) and ("condition_text" in d or "span_text" in d):
                        items.append(d)
                except Exception:
                    continue
            if items:
                return {"items": items}

    # Fallback: collect any complete objects in the whole text
    obj_strs = _collect_balanced_objects(text)
    items = []
    for o in obj_strs:
        oo = o.replace(",}", "}")
        try:
            d = json.loads(oo)
            if isinstance(d, dict) and ("condition_text" in d or "span_text" in d):
                items.append(d)
        except Exception:
            continue
    if items:
        return {"items": items}
    return None

# def extract_json(text: str) -> Optional[Any]:
#     """
#     Extract and parse JSON from model output.
#     - Strips ``` fences
#     - Parses full JSON if present
#     - If truncated, salvages complete items from the partial output
#     """
#     if text is None:
#         return None

#     text = _strip_code_fences(text)
#     print(text)

#     # 1) Try direct JSON parse
#     try:
#         return json.loads(text)
#     except Exception:
#         pass

#     # 2) Try to locate a balanced top-level JSON span
#     span = _find_json_span(text)
#     if span:
#         start, end = span
#         json_like = text[start:end].strip()

#         # Try parsing extracted span
#         try:
#             return json.loads(json_like)
#         except Exception:
#             # mild repairs
#             repaired = json_like.replace(",]", "]").replace(",}", "}")
#             try:
#                 return json.loads(repaired)
#             except Exception:
#                 # salvage from inside json_like
#                 salvaged = _salvage_items_from_text(json_like)
#                 if salvaged is not None:
#                     return salvaged

#     # 3) If no balanced span (likely truncation), salvage complete items from whole text
#     salvaged = _salvage_items_from_text(text)
#     if salvaged is not None:
#         return salvaged

#     return None

def extract_json(text: str) -> Optional[Any]:
    text = _strip_code_fences(text)

    # 1) direct parse
    try:
        return json.loads(text)
    except Exception:
        pass

    # 2) parse balanced OUTER JSON only
    span = _find_outer_json_span(text)
    if span:
        start, end = span
        json_like = text[start:end].strip()
        try:
            return json.loads(json_like)
        except Exception:
            repaired = json_like.replace(",]", "]").replace(",}", "}")
            try:
                return json.loads(repaired)
            except Exception:
                pass

    # 3) outer JSON is likely truncated => salvage complete items
    salvaged = _salvage_items_from_text(text)
    if salvaged is not None:
        return salvaged

    return None


def has_end_json_token(
    text: Optional[str],
    token: str = "<<END_JSON>>",
    require_terminal: bool = True,
) -> bool:
    """
    Check whether model output includes the expected end token.

    Parameters
    ----------
    text : Optional[str]
        Raw model output text.
    token : str
        Sentinel token indicating completion.
    require_terminal : bool
        If True, token must be the final non-whitespace content.
        If False, token may appear anywhere in text.
    """
    if not text:
        return False
    if require_terminal:
        return text.rstrip().endswith(token)
    return token in text


def trim_after_end_json_token(
    text: Optional[str],
    token: str = "<<END_JSON>>",
    include_token: bool = True,
) -> str:
    """
    Trim any trailing content that appears after the first end token.
    Returns the original text when token is absent.
    """
    if not text:
        return ""
    idx = text.find(token)
    if idx < 0:
        return text
    end = idx + (len(token) if include_token else 0)
    return text[:end]
### NEW: 02/18/2026 <END>
### PREV EXTRACT_JSON: 02/18/2026 

# def extract_json(text: str) -> Optional[Any]:
#     """
#     Extract and parse a JSON object or array from a model's output.
#     Returns Python data (dict/list) or None if parsing fails.
#     """
#     text = text.strip()

#     # 1. Try direct JSON parse first
#     try:
#         return json.loads(text)
#     except Exception:
#         pass

#     # 2. Try to locate a {...} or [...] span
#     span = _find_json_span(text)
#     if not span:
#         return None

#     start, end = span
#     json_like = text[start:end].strip()
    

#     # 3. Try parsing the extracted slice
#     try:
#         return json.loads(json_like)
#     except Exception:
#         pass

#     # 4. Last-resort mild "repair": common minor issues
#     repaired = (
#         json_like
#         .replace(",]", "]")
#         .replace(",}", "}")
#     )
#     try:
#         return json.loads(repaired)
#     except Exception:
#         return None
    

# ==============================
# JSONL PARSING HELPERS
# ==============================

def flatten(data):
    return [item for sub in data for item in sub]


def add_hits_json(records: List[Dict[str, Any]], output: List[str]) -> List[Dict[str, Any]]:
    """
    Add a JSON-serialized 'hits_json' field to each record, containing
    only the specified keys from each hit.

    Example:
      output = ["id", "label"]
      hits_json -> [{"id": 123, "label": "Pregnancy (finding)"}, ...]
    """
    payloads = []
    for r in records:
        hits = r.get("hits", []) or []

        # Keep only requested keys per hit
        filtered_hits = [
            {k: h.get(k) for k in output if k in h}
            for h in hits
        ]

        payloads.append({
            **r,
            "hits_json": json.dumps(
                filtered_hits,
                indent=2,
                ensure_ascii=False
            )
        })
    return payloads