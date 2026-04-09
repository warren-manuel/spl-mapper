
"""
llm_runner.py — Reusable helpers for running local LLMs and building chat messages.

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

    def _format_chat(self, messages: List[Dict[str, str]]) -> str:
        """
        Use tokenizer.apply_chat_template if available; otherwise do a simple join.
        """
        apply = getattr(self.tokenizer, "apply_chat_template", None)
        if callable(apply):
            return apply(messages, add_generation_prompt=True, tokenize=False)
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
        if stop:
            # If tokenizer knows these tokens, map to IDs; otherwise pass strings via stopping_criteria in other libs.
            token_ids = []
            for s in stop:
                tok = self.tokenizer.encode(s, add_special_tokens=False)
                if tok:
                    token_ids.extend(tok)
            if token_ids:
                eos_token_ids = token_ids

        output = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            eos_token_id=eos_token_ids,
            pad_token_id=getattr(self.tokenizer, "pad_token_id", None) or getattr(self.tokenizer, "eos_token_id", None),
            **gen_kwargs,
        )

        text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        # Try to strip the prompt if present in decode
        if text.startswith(prompt):
            text = text[len(prompt):].lstrip()
        return {"text": text, "raw": output}

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

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device_map,
        torch_dtype=dtype_arg,
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit,
        trust_remote_code=trust_remote_code,
        **model_kwargs,
    )

    # If we added a pad token, resize embeddings.
    if hasattr(model, "resize_token_embeddings"):
        model.resize_token_embeddings(len(tok))

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

        sys_msg = _render_system(payload)
        usr_msg = user_template.format(**{**fmt_kwargs, **payload})
        chats.append(build_message(sys_msg, usr_msg))
    return chats



# ==============================
# XML Helpers
# ==============================

def _clean_text(s: str, strip: bool = True) -> str:
    if s is None:
        return ""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    if strip:
        s = "\n".join(line.strip() for line in s.splitlines()).strip()
    return s

def xml_section_iter(
    xml_source: Union[str, os.PathLike],
    xpaths: Optional[List[str]] = None,
    tags: Optional[List[str]] = None,
    strip: bool = True,
    section_key: str = "section",
    text_key: str = "text",
) -> Generator[Dict[str, str], None, None]:
    """
    Iterate selected sections from an SPL-like XML.

    Parameters
    ----------
    xml_source : path to XML file or XML string.
    xpaths     : list of xpath expressions (requires lxml).
    tags       : list of tag names (exact or case-insensitive match on tag text) for stdlib fallback.
    strip      : strip whitespace from lines.
    section_key, text_key : output keys

    Yields
    ------
    dict like {"section": "<NAME>", "text": "<content>"}
    """
    xml_text: Optional[str] = None
    if Path(str(xml_source)).exists():
        xml_text = Path(str(xml_source)).read_text(encoding="utf-8", errors="ignore")
    else:
        xml_text = str(xml_source)

    # Try lxml first for full XPath power
    try:
        from lxml import etree  # type: ignore

        root = etree.fromstring(xml_text.encode("utf-8", "ignore"))
        if xpaths:
            for xp in xpaths:
                for node in root.xpath(xp):
                    sec_name = node.get("title") or node.tag
                    content = "".join(node.itertext())
                    yield {section_key: str(sec_name), text_key: _clean_text(content, strip)}
            return  # done if xpaths provided

        # Otherwise, try tags by case-insensitive contains on element titles/headings
        if tags:
            wanted = {t.lower() for t in tags}
            for el in root.iter():
                title = (el.get("title") or el.get("heading") or el.tag or "").strip()
                title_l = title.lower()
                if any(t == title_l or t in title_l for t in wanted):
                    content = "".join(el.itertext())
                    yield {section_key: title or el.tag, text_key: _clean_text(content, strip)}
            return

    except Exception:
        # Fallback to stdlib ElementTree with simple tag search
        import xml.etree.ElementTree as ET  # type: ignore
        root = ET.fromstring(xml_text.encode("utf-8", "ignore"))
        if tags:
            wanted = {t.lower() for t in tags}
            for el in root.iter():
                tag_name = (el.tag or "").split("}")[-1]
                title = el.attrib.get("title") or tag_name
                t_lower = (title or tag_name or "").lower()
                if any(t == t_lower or t in t_lower for t in wanted):
                    content = "".join(el.itertext())
                    yield {section_key: title or tag_name, text_key: _clean_text(content, strip)}
            return

        # No tags/xpaths: yield top-level text
        content = "".join(root.itertext())
        yield {section_key: root.tag, text_key: _clean_text(content, strip)}

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