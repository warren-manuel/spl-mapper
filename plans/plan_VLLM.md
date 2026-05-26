# vLLM Pipeline Fixes and Phases VII–IX

---

## Fixes (post-vLLM bring-up, 2026-05-13 — all implemented ✅)

Three issues surfaced from `results/20260513_vllm/agent_results.jsonl`.

### Fix 1 — `--max-model-len` for vLLM server (server restart only)

vLLM 0.19.1 does **not** support `--max-model-len auto`. Use an explicit value.

**Memory budget (A100 80GB × 2, TP=2):**
- Gemma 4 31B weights in BF16 ≈ 62 GB total → ~31 GB per GPU
- Free for KV cache: ~49 GB per GPU
- KV cache at 32768 tokens ≈ 12 GB per GPU — well within budget

**Updated server command:**
```bash
CUDA_VISIBLE_DEVICES=1,2 conda run -n vllm_env \
  python -m vllm.entrypoints.openai.api_server \
  --model google/gemma-4-31B-it \
  --tensor-parallel-size 2 \
  --port 8000 \
  --max-model-len 32768 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.90 \
  --tool-call-parser pythonic \
  --enable-auto-tool-choice
```

Verification: re-run pipeline; confirm no context overflow errors in run.log.

---

### Fix 2 — Surface `mrcm_attribute_mapper` tool call trace

**File:** `scripts/run_pipeline.py`

**Root cause:** `mrcm_attribute_mapper_node` builds a local `trace: List[Dict]` (~line 1530) that records each tool call step but never writes it to state. `assemble_postcoord_node`/`assemble_review_node` read `state["postcoord_pattern"]` which contains only the final parsed answer. Additionally, `focus_selector_trace` is in `ItemState` and returned from the node but NOT included in the assembled `item_result.trace`.

**2a. Add field to `ItemState` (~line 527), after `focus_selector_trace`:**
```python
mrcm_mapper_trace: List[Dict[str, Any]]
```

**2b. Return trace from `mrcm_attribute_mapper_node` (~line 1563), add to return dict:**
```python
"mrcm_mapper_trace": trace,
```

**2c. Update trace dict in both `assemble_postcoord_node` and `assemble_review_node`:**
```python
"trace": {
    "direct_verify": state.get("direct_match", {}),
    "focus_selector": state.get("focus_selector_trace", []),
    "mrcm_mapper": state.get("mrcm_mapper_trace", []),
    "postcoord_pattern": state.get("postcoord_pattern", {}),
},
```
Drop `route_or_fill` and `validation` keys — both disabled, always empty.

Verification: run pipeline on test_5spl.jsonl; `agent_results.jsonl` `trace.mrcm_mapper` is a list of step dicts with `tool`, `args`, `result`.

---

### Fix 3 — `selected_focus_term` always "N/A"

**File:** `scripts/run_pipeline.py`

**Root cause:** `focus_selector_node` term resolution scans trace steps for the SCTID in `candidates` and `ancestors` result keys. When focus SCTID came from a `get_ancestors` result (abstract parent of a pre-coordinated concept), key name mismatch causes miss → falls back to reasoning text → "N/A". `normalize_node` secondary fix via `focus_candidates` is dead — that list is never populated since `retrieve_candidates_node` was removed in Phase III.

**Impact:** `selected_focus_term` is passed to `build_mrcm_mapper_react_user_prompt` — "N/A" here degrades mapper prompt quality (functional regression, not display only).

**Add to `focus_selector_node`** (~line 1355, after existing trace scan loop):
```python
if selected_term == "N/A" and selected_id != "N/A" and self.graph_client is not None:
    try:
        results = self.graph_client.lookup_concept(selected_id)
        if results:
            selected_term = getattr(results[0], "preferred_term", None) \
                         or getattr(results[0], "fsn", None) \
                         or "N/A"
    except Exception:
        pass
```

**Also remove dead code in `normalize_node`**: the `candidate_label_by_id(state.get("candidates", {}).get("focus_candidates", []), ...)` call always no-ops; remove or guard it.

Verification: run pipeline; `selected_focus_term` shows preferred term (e.g. `"Hypersensitivity condition"`) rather than `"N/A"` for postcoord items.

---

## Execution order (completed ✅)

```
1. Fix 1: restarted vLLM server with --max-model-len 32768
2. Fix 2 + Fix 3: implemented in scripts/run_pipeline.py
3. Smoke tested
```

---

## Post-vLLM Pipeline Fixes (2026-05-19/21 — all implemented ✅)

### Fix 4 — `categorize_slots` parse robustness (`src/llm/prompts.py`)
LLM was outputting `"segments"` key and `"span"` field due to "Segment this span" wording in user prompt. Parser and node builder now accept both variants. User prompt reworded to "Identify each clinical component…" with explicit JSON schema example.
- `parse_categorize_slots_output`: accepts `parsed.get("components") or parsed.get("segments")`; `comp.get("text") or comp.get("span", "")`
- `categorize_slots_node`: `comp.get("text") or comp.get("span", "")`
- **Impact:** `NON-FOCUS COMPONENTS: (none identified)` resolved; Agent 2 slots now reach Agent 3

### Fix 5 — Focus term sanitization (`scripts/run_pipeline.py`)
`mrcm_attribute_mapper_node` now sanitizes `focus_term` before prompt builder: if >60 chars (reasoning text fallback) or "N/A", calls `graph_client.lookup_concept(focus_sctid)` for clean preferred term.
- **Impact:** `FOCUS CONCEPT: 419076005 | Allergic reaction (disorder)` instead of reasoning text

### Fix 6 — MRCM agent `reasoning` output parity (`agents/postcord_agent.md`)
Added `reasoning` field to postcord_agent output format — matches `direct_match_agent.md` and `focus_selector.md`.

### Fix 7 — Graph client OPTIONAL MATCH (`src/snomed/graph_client.py`)
- `get_domain_attributes` and `get_attribute_range`: inner `MATCH (attr:Concept)` → `OPTIONAL MATCH`; fallback to SCTID string when Concept node missing (attribute concepts not yet loaded)
- New `get_hierarchy_map()`: bulk query returning `{sctid_int: top_level_hierarchy}` for all concepts

### Fix 8 — Retrieval enrichment: `top_level_hierarchy` + `semantic_tag` in hits
- `MapperResources` (hybrid_mapper.py): new `hierarchy_map: Optional[Dict[int, str]]` field
- `get_cached_mapper_resources`: new `graph_client` param; calls `get_hierarchy_map()` once and caches; `USE_ANCESTOR_PATHS` env var check moved here
- `search_query` (search_utils.py): enriches all hits with `top_level_hierarchy` + `semantic_tag`
- `hierarchy_filter` in `search_snomed_concepts` now works (was always returning empty before)
- `search_snomed_for_agent`: new `graph_client=None` kwarg; call sites in `ContraLangGraphAgent` pass `self.graph_client`

### Fix 9 — SNOMED graph: Attribute hierarchy added + rebuilt ✅ (`scripts/build_snomed_graph.py`)
`246061005` (Attribute) added to `TOP_LEVEL_HIERARCHIES`, `CONTRAINDICATION_RELEVANT`, `HIERARCHY_PRIORITY`. Graph rebuilt 2026-05-26 — `get_domain_attributes` now returns proper names ("Causative agent") instead of bare SCTIDs.

### Fix 10 — Code quality
- `_call_llm_json`: `temperature=1.0` → `0.0`
- All `_log_llm_call` sites: removed `system[:200] + "…"` truncation (full system prompt now logged)
- ToolNode `create_react_agent`: `state_modifier=` → `prompt=` (LangGraph 1.0 API rename)
- `.env`: duplicate `HF_TEMPERATURE=0.1` block commented out; clean `.env` created

### Phase IX ToolNode — issue identified, flags reset to `false`
ToolNode path produces `{tool, args, trace}` in `direct_verify` instead of `{direct_match, selected_id, reasoning, trace}` because agent `.md` prompts are written for Python-loop JSON-text output, not native LangGraph tool calls. All flags reset to `false` pending prompt rewrite + trace output parity fix.

---

# Phases VII–IX: vLLM Backend, MCP Tool Server, ToolNode Conversion

## Context

Replaces the HuggingFace `transformers` local backend with a vLLM serving endpoint. This
solves Phase VI (stop token noise) for free, enables native structured tool calling, and
creates the MCP server seam needed to expose SNOMED tools to any external orchestrator.

Three sequential phases:
- **Phase VII** — vLLM backend, same Python-loop ReAct (drop-in swap)
- **Phase VIII** — MCP server wrapping SNOMED tools (external-orchestrator-ready)
- **Phase IX** — Progressive ToolNode conversion (replaces Python-loop ReAct per agent)

Phase VI (`PostThinkingStoppingCriteria`) is **cancelled** — vLLM handles stop tokens and
thinking-mode correctly natively.

---

## Phase VII — vLLM Backend (drop-in swap)

### Goal
Same pipeline behavior, different serving layer. Python-loop ReAct continues to work
unchanged. Stop token noise disappears. Model swap = restart vLLM server.

### VII.1 — vLLM server deployment

```bash
vllm serve google/gemma-4-31B-it \
  --port 8000 \
  --tensor-parallel-size 3 \
  --gpu-memory-utilization 0.90 \
  --enable-auto-tool-choice \
  --tool-call-parser pythonic \
  --max-model-len 8192
```

Tool-call-parser per model:
| Model | `--tool-call-parser` |
|---|---|
| Gemma 4 | `pythonic` |
| Qwen 3.x | `hermes` |
| Llama 3.1/3.3 | `llama3_json` |
| Azure GPT | N/A (cloud) |

FAISS + embedding model still run in the same Python process (unchanged).
Main model GPUs freed — vLLM owns them, Python process owns only FAISS GPU.

### VII.2 — `VLLMLLMConfig` dataclass in `scripts/run_pipeline.py`

```python
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
```

### VII.3 — `VLLMChatLLM` class in `scripts/run_pipeline.py`

Implements the same `.chat()` interface as `HuggingFaceChatLLM`. Wraps `openai.OpenAI`.

```python
class VLLMChatLLM:
    def __init__(self, cfg: VLLMLLMConfig):
        import openai
        self.cfg = cfg
        self._client = openai.OpenAI(
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
            temperature=self.cfg.temperature if self.cfg.temperature is not None else temperature,
            stop=stop or None,
        )
        return response.choices[0].message.content or ""
```

**Key detail:** `stop=["<<END_JSON>>"]` is passed directly to vLLM's completions API.
vLLM handles stop tokens natively — no post-processing truncation needed.
Thinking mode (`<|think|>`) works; vLLM stops after `<<END_JSON>>` even when it appears
in the thinking block because vLLM's generation is token-level, not string-splitting.

### VII.4 — Extend `build_llm()` in `scripts/run_pipeline.py`

```python
def build_llm(backend: str) -> ChatLLM:
    configure_process_cuda_visibility()
    if backend == "azure":
        return AzureChatLLM(AzureLLMConfig.from_env())
    if backend == "huggingface":
        return HuggingFaceChatLLM(HuggingFaceLLMConfig.from_env())
    if backend == "vllm":
        return VLLMChatLLM(VLLMLLMConfig.from_env())
    raise ConfigError(f"Unsupported backend '{backend}'.")
```

Add `"vllm"` to argparse `choices` tuple.

### VII.5 — New env vars

| Var | Default | Purpose |
|---|---|---|
| `VLLM_BASE_URL` | `http://localhost:8000/v1` | vLLM server endpoint |
| `VLLM_MODEL_ID` | `google/gemma-4-31B-it` | Model name sent in API calls |
| `VLLM_API_KEY` | `token` | Auth token (vLLM accepts any non-empty string) |
| `VLLM_TIMEOUT_SECONDS` | `120` | Per-request timeout |
| `LANGGRAPH_LLM_BACKEND` | `huggingface` | Add `vllm` as new valid choice |

### VII.6 — Cancel Phase VI

`PostThinkingStoppingCriteria` is not needed. Remove Phase VI TODO items from
`AGENT_IMPLEMENTER_PLAN.md` status table. The 7 `stop=None` overrides in
`run_pipeline.py` can stay for now — vLLM handles stop correctly regardless.

### VII.7 — Files changed

| File | Change |
|---|---|
| `scripts/run_pipeline.py` | `VLLMLLMConfig`, `VLLMChatLLM`, extend `build_llm()`, argparse choices |
| `.env` | Add `VLLM_BASE_URL`, `VLLM_MODEL_ID`, set `LANGGRAPH_LLM_BACKEND=vllm` |

### VII Verification

```bash
# Start vLLM server, then:
LANGGRAPH_LLM_BACKEND=vllm python3 -m scripts.run_pipeline \
  --input results/VO_RUN/test_5spl.jsonl \
  --output-dir results/vllm_smoke/
```

Check `runtime_audit.jsonl`:
- `raw_output` ends at `<<END_JSON>>` — no trailing noise
- `duration_s` for extraction/decomposition decreases vs transformers baseline
- All 5 SPLs complete without crash

---

## Phase VIII — MCP Tool Server

### Goal
Expose all 7 SNOMED tools as MCP servers. `_run_ontology_tool` in `run_pipeline.py`
can call them via MCP client instead of direct Python function calls. Any external
orchestrator (Claude, LangChain, future frameworks) can then use the same tools.

### VIII.1 — New file: `src/tools/snomed_mcp_server.py`

Uses `fastmcp` (or `mcp` package). Wraps `src/retrieval/hybrid_mapper.py` and
`src/snomed/graph_client.py`.

```python
from fastmcp import FastMCP
from src.retrieval.hybrid_mapper import search_snomed_concepts, get_cached_mapper_resources
from src.snomed.graph_client import SnomedGraphClient

mcp = FastMCP("snomed-tools")
_graph_client: Optional[SnomedGraphClient] = None

@mcp.tool()
def search_snomed(query: str, hierarchy_filter: str = "", k: int = 10) -> list:
    """BM25+FAISS hybrid search over SNOMED CT concepts."""
    return search_snomed_concepts(query, hierarchy_filter or None, k)

@mcp.tool()
def get_logical_definition(sctid: str) -> dict:
    """Return HAS_ROLE edges (defining relationships) for a concept."""
    results = _graph_client.get_logical_definition(sctid)
    return {"roles": [r.__dict__ for r in results]}

@mcp.tool()
def get_ancestors(sctid: str, max_depth: int = 2) -> dict:
    """Return IS-A ancestors up to max_depth."""
    results = _graph_client.get_ancestors(sctid, max_depth=max_depth)
    return {"ancestors": [r.__dict__ for r in results]}

@mcp.tool()
def get_siblings(sctid: str, limit: int = 5) -> dict:
    """Return sibling concepts (same parent)."""
    results = _graph_client.get_siblings(sctid, limit=limit)
    return {"siblings": [r.__dict__ for r in results]}

@mcp.tool()
def get_domain_attributes(focus_sctid: str) -> dict:
    """Return MRCM-valid attributes for a focus concept's domain."""
    results = _graph_client.get_domain_attributes(focus_sctid)
    return {"attributes": results}

@mcp.tool()
def get_attribute_range(attr_sctid: str) -> dict:
    """Return MRCM range constraint and names for an attribute."""
    results = _graph_client.get_attribute_range(attr_sctid)
    return {"ranges": [r.__dict__ for r in results]}

@mcp.tool()
def lookup_concept(text: str, hierarchy_filter: str = "") -> dict:
    """Lookup a SNOMED concept by preferred term text."""
    results = _graph_client.lookup_concept(text, hierarchy_filter or None)
    return {"results": [r.__dict__ for r in results]}

if __name__ == "__main__":
    mcp.run()
```

### VIII.2 — MCP transport options

**HTTP (recommended for pipeline integration):**
```bash
fastmcp run src/tools/snomed_mcp_server.py --transport http --port 8001
```

**Stdio (for Claude Code / external agent harnesses):**
Register in `.mcp.json`:
```json
{
  "mcpServers": {
    "snomed-tools": {
      "command": "python3",
      "args": ["-m", "src.tools.snomed_mcp_server"],
      "env": {"SNOMED_DB_PATH": "results/snomed_graph.db", "NEO4J_URI": "..."}
    }
  }
}
```

### VIII.3 — Optional MCP dispatch in `_run_ontology_tool`

Gate with `SNOMED_TOOLS_BACKEND` env var. Default: `"direct"` (current behavior unchanged).

```python
@staticmethod
def _run_ontology_tool(call, graph_client, mcp_client=None) -> Dict[str, Any]:
    if mcp_client is not None:
        # MCP path — same tool names, same arg schemas
        return mcp_client.call_tool(call["tool"], call.get("args", {}))
    # Direct path (current behavior)
    tool = call.get("tool", "")
    args = call.get("args", {})
    ...  # existing dispatch unchanged
```

### VIII.4 — New env vars

| Var | Default | Purpose |
|---|---|---|
| `SNOMED_TOOLS_BACKEND` | `direct` | `direct` = Python calls; `mcp` = MCP client |
| `SNOMED_MCP_URL` | `http://localhost:8001` | MCP server URL (when `mcp` backend) |

### VIII.5 — Files changed

| File | Change |
|---|---|
| `src/tools/snomed_mcp_server.py` | New file — FastMCP server |
| `scripts/run_pipeline.py` | Optional MCP client path in `_run_ontology_tool` |
| `.mcp.json` | Register server for Claude Code / external agents |
| `environment.yml` | Add `fastmcp` dependency |

### VIII Verification

```bash
# Start MCP server
fastmcp run src/tools/snomed_mcp_server.py --transport http --port 8001

# Run pipeline with MCP backend
SNOMED_TOOLS_BACKEND=mcp SNOMED_MCP_URL=http://localhost:8001 \
  python3 -m scripts.run_pipeline --input results/VO_RUN/test_5spl.jsonl
```

Outputs should be identical to direct backend. Test with Claude Code by pointing
at `.mcp.json` and manually calling `search_snomed("hypersensitivity disorder")`.

---

## Phase IX — LangGraph ToolNode Conversion

### Goal
Replace Python-loop ReAct with LangGraph native `ToolNode` + `create_react_agent`.
Requires Phase VII (vLLM with `--enable-auto-tool-choice`). Agents read their `.md`
system prompts and call tools via structured `ToolCall` objects.

Convert in order: `direct_match_node` → `focus_selector_node` → `mrcm_attribute_mapper_node`.

### IX.1 — Tool schema definitions

```python
# src/tools/schemas.py
from langchain_core.tools import tool

@tool
def search_snomed(query: str, hierarchy_filter: str = "", k: int = 10) -> list:
    """BM25+FAISS hybrid search over SNOMED CT concepts."""
    return search_snomed_for_agent(query, hierarchy_filter or None, k)

@tool
def get_logical_definition(sctid: str) -> dict:
    """Return HAS_ROLE defining relationships for a concept."""
    ...

# etc for all 7 tools
```

Tool objects are passed to both `bind_tools()` (for the LLM) and `ToolNode` (for execution).

### IX.2 — `VLLMChatLLM` tool-calling extension

Add `bind_tools()` support:
```python
class VLLMChatLLM:
    def get_langchain_llm(self) -> ChatOpenAI:
        """Return a LangChain ChatOpenAI pointed at the vLLM endpoint."""
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            base_url=self.cfg.base_url,
            api_key=self.cfg.api_key,
            model=self.cfg.model_id,
            temperature=self.cfg.temperature,
        )
```

### IX.3 — Per-agent conversion pattern

For each agent, replace:
```python
# Before: Python-loop ReAct
for _ in range(max_tool_calls):
    raw = self.llm.chat(messages, ...)
    call = parse_output(raw)
    if "final_answer" in call: break
    result = self._run_ontology_tool(call, self.graph_client)
    messages = self._extend_react_messages(messages, raw, result)
```

With:
```python
# After: LangGraph create_react_agent subgraph
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(
    model=self.llm.get_langchain_llm().bind_tools(AGENT_TOOLS),
    tools=AGENT_TOOLS,
    state_modifier=SystemMessage(content=_load_agent_md()),
)
result = agent.invoke({"messages": [HumanMessage(content=user_prompt)]})
```

### IX.4 — Conversion order and flags

Each conversion is gated by a config flag so fallback to Python-loop is always available:

| Flag | Default | Effect |
|---|---|---|
| `DIRECT_MATCH_USE_TOOLNODE` | `false` | When `true`, uses ToolNode for direct_match |
| `FOCUS_SELECTOR_USE_TOOLNODE` | `false` | When `true`, uses ToolNode for focus_selector |
| `MRCM_MAPPER_USE_TOOLNODE` | `false` | When `true`, uses ToolNode for mrcm_attribute_mapper |

Set to `true` one at a time. Validate metrics at each step before enabling the next.

### IX.5 — Files changed

| File | Change |
|---|---|
| `src/tools/schemas.py` | New — LangChain `@tool` definitions for all 7 tools |
| `scripts/run_pipeline.py` | `VLLMChatLLM.get_langchain_llm()`; per-agent ToolNode branch behind feature flags |
| `environment.yml` | Add `langchain-openai` dependency |

### IX Verification

Per agent, after enabling ToolNode:
- `runtime_audit.jsonl` shows structured tool calls with typed args (not raw JSON strings)
- Agent `.md` system prompts are unchanged and still loaded
- Concept-level F1 ≥ Python-loop baseline on 5-SPL test

---

## Summary

| Phase | What | Blocks Phase VI? | Key Seam |
|---|---|---|---|
| VII | vLLM backend (drop-in) | Cancels VI | `build_llm()` + `VLLMChatLLM` |
| VIII | MCP server (SNOMED tools) | No | `_run_ontology_tool` + `.mcp.json` |
| IX | ToolNode conversion (per agent) | Needs VII | `get_langchain_llm().bind_tools()` |

## Dependencies

```
Phase VII ──→ Phase VIII (optional, parallel)
Phase VII ──→ Phase IX (required)
Phase VIII ──→ Phase IX (optional — ToolNode can call direct Python or MCP)
```

Phase VIII is not a hard dependency of Phase IX — ToolNode can call the Python functions
directly. MCP is the transport for external orchestrators, not for internal LangGraph use.
