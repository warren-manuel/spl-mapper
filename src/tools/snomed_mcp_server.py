"""
FastMCP server exposing all 7 SNOMED CT ontology tools.

Phase VIII: allows external orchestrators (Claude Code, LangChain, etc.) to
call SNOMED tools via MCP protocol without needing the pipeline Python env.

Run modes:
  HTTP (pipeline integration):
    python -m src.tools.snomed_mcp_server --transport http --port 8001

  Stdio (Claude Code / harness):
    registered in .mcp.json — Claude Code spawns this as a subprocess

Pipeline integration (optional):
  Set SNOMED_TOOLS_BACKEND=mcp + SNOMED_MCP_URL=http://localhost:8001
  in .env to route _run_ontology_tool dispatch through this server.
"""
from __future__ import annotations

import os
import sys
from typing import Optional

from fastmcp import FastMCP

# Add repo root to path when run as __main__ from any working directory
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.retrieval.hybrid_mapper import search_snomed_concepts  # noqa: E402
from src.snomed.graph_client import SnomedGraphClient  # noqa: E402

mcp = FastMCP("snomed-tools")

_graph_client: Optional[SnomedGraphClient] = None


def _get_client() -> SnomedGraphClient:
    global _graph_client
    if _graph_client is None:
        _graph_client = SnomedGraphClient(
            uri=os.environ["NEO4J_URI"],
            user=os.environ["NEO4J_USER"],
            password=os.environ["NEO4J_PASSWORD"],
        )
    return _graph_client


@mcp.tool()
def search_snomed(query: str, hierarchy_filter: str = "", k: int = 10) -> list:
    """BM25+FAISS hybrid search over SNOMED CT concepts. Returns candidates with id, label, score, ancestor_path."""
    return search_snomed_concepts(query, hierarchy_filter or None, k)


@mcp.tool()
def get_logical_definition(sctid: str) -> dict:
    """Return HAS_ROLE defining relationships for a concept. Empty roles means abstract concept."""
    results = _get_client().get_logical_definition(sctid)
    return {"roles": [r.__dict__ for r in results]}


@mcp.tool()
def get_ancestors(sctid: str, max_depth: int = 2) -> dict:
    """Return IS-A ancestors of a concept up to max_depth levels."""
    results = _get_client().get_ancestors(sctid, max_depth=max_depth)
    return {"ancestors": [r.__dict__ for r in results]}


@mcp.tool()
def get_siblings(sctid: str, limit: int = 5) -> dict:
    """Return sibling concepts sharing the same parent."""
    results = _get_client().get_siblings(sctid, limit=limit)
    return {"siblings": [r.__dict__ for r in results]}


@mcp.tool()
def get_domain_attributes(focus_sctid: str) -> dict:
    """Return MRCM-valid attributes for a focus concept's domain."""
    return {"attributes": _get_client().get_domain_attributes(focus_sctid)}


@mcp.tool()
def get_attribute_range(attr_sctid: str) -> dict:
    """Return MRCM range constraint ECL and attribute names."""
    results = _get_client().get_attribute_range(attr_sctid)
    return {"ranges": [r.__dict__ for r in results]}


@mcp.tool()
def lookup_concept(text: str, hierarchy_filter: str = "") -> dict:
    """Lookup a SNOMED concept by preferred term text."""
    results = _get_client().lookup_concept(text, hierarchy_filter or None)
    return {"results": [r.__dict__ for r in results]}


@mcp.tool()
def validate_ecl(sctid: str, range_ecl: str) -> dict:
    """Check whether a SNOMED concept satisfies an MRCM range ECL via Snowstorm.
    Returns {valid: true/false}."""
    from src.snomed.snomed_utils import concept_matches_ecl, base as _SNOW_BASE
    try:
        valid = concept_matches_ecl(int(sctid), range_ecl, base=_SNOW_BASE, timeout=30, retries=1)
        return {"valid": bool(valid)}
    except Exception as exc:
        return {"valid": False, "error": str(exc)}


if __name__ == "__main__":
    mcp.run()
