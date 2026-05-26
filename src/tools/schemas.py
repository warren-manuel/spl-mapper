"""
LangChain tool definitions for SNOMED CT ontology access.

Phase IX: create bound tool instances via make_agent_tools() for use with
LangGraph ToolNode + create_react_agent. The same logical tools as
_run_ontology_tool dispatch in run_pipeline.py — different transport.
"""
from __future__ import annotations
from typing import Any, Callable, List, Optional

from langchain_core.tools import tool


def make_agent_tools(
    graph_client: Any,
    search_fn: Callable,
    tool_subset: Optional[List[str]] = None,
) -> list:
    """
    Return LangChain tool instances with graph_client and search_fn bound via closure.

    tool_subset: if given, only return tools whose .name is in this list.
    """

    @tool
    def search_snomed(query: str, hierarchy_filter: str = "", k: int = 10) -> dict:
        """BM25+FAISS hybrid search over SNOMED CT concepts. Returns candidates with id, label, score, ancestor_path."""
        return {"candidates": search_fn(query, hierarchy_filter or None, k)}

    @tool
    def get_logical_definition(sctid: str) -> dict:
        """Return HAS_ROLE defining relationships for a SNOMED concept. Empty roles means abstract (preferred for focus)."""
        results = graph_client.get_logical_definition(sctid)
        return {"roles": [r.__dict__ for r in results]}

    @tool
    def get_ancestors(sctid: str, max_depth: int = 2) -> dict:
        """Return IS-A ancestors of a concept up to max_depth levels. Use to find abstract parent of a pre-coordinated hit."""
        results = graph_client.get_ancestors(sctid, max_depth=max_depth)
        return {"ancestors": [r.__dict__ for r in results]}

    @tool
    def get_siblings(sctid: str, limit: int = 5) -> dict:
        """Return sibling concepts that share the same parent. Use to verify concept placement in hierarchy."""
        results = graph_client.get_siblings(sctid, limit=limit)
        return {"siblings": [r.__dict__ for r in results]}

    @tool
    def get_domain_attributes(focus_sctid: str) -> dict:
        """Return MRCM-valid attribute SCTIDs and names for the domain of a focus concept."""
        return {"attributes": graph_client.get_domain_attributes(focus_sctid)}

    @tool
    def get_attribute_range(attr_sctid: str) -> dict:
        """Return MRCM range constraint ECL string and attribute names for a given attribute SCTID."""
        results = graph_client.get_attribute_range(attr_sctid)
        return {"ranges": [r.__dict__ for r in results]}

    @tool
    def lookup_concept(text: str, hierarchy_filter: str = "") -> dict:
        """Lookup a SNOMED concept by preferred term text with optional hierarchy filter."""
        results = graph_client.lookup_concept(text, hierarchy_filter or None)
        return {"results": [r.__dict__ for r in results]}

    @tool
    def validate_ecl(sctid: str, range_ecl: str) -> dict:
        """Check whether a SNOMED concept satisfies an MRCM range ECL via Snowstorm.
        Returns {valid: true/false}. Call after search_snomed to confirm MRCM validity before accepting a value."""
        from src.snomed.snomed_utils import concept_matches_ecl, base as _SNOW_BASE
        try:
            valid = concept_matches_ecl(int(sctid), range_ecl, base=_SNOW_BASE, timeout=30, retries=1)
            return {"valid": bool(valid)}
        except Exception as exc:
            return {"valid": False, "error": str(exc)}

    all_tools = [
        search_snomed,
        get_logical_definition,
        get_ancestors,
        get_siblings,
        get_domain_attributes,
        get_attribute_range,
        lookup_concept,
        validate_ecl,
    ]
    if tool_subset:
        name_set = set(tool_subset)
        all_tools = [t for t in all_tools if t.name in name_set]
    return all_tools
