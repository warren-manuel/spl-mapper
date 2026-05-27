"""
SnomedGraphClient — Neo4j-backed SNOMED CT lookup for the spl-mapper pipeline.

Credentials from environment (or .env):
  NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE

Capabilities:
  lookup_concept        — exact/synonym/substring search with optional hierarchy filter
  get_top_level_hierarchy — sctid → hierarchy name
  get_ancestors         — IS_A* traversal up to max_depth
  get_logical_definition — all HAS_ROLE edges for a concept (SNOMED logical definition)
  get_attribute_range   — MRCMRange constraints for an attribute sctid
  get_attribute_domain  — MRCMAttributeDomain rules for an attribute sctid
  check_concept_exists  — boolean existence check
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv


@dataclass
class ConceptMatch:
    sctid: str
    preferred_term: str
    fsn: str
    semantic_tag: str
    top_level_hierarchy: str
    match_type: str  # "exact" | "synonym" | "substring" | "fulltext"


@dataclass
class RoleTriple:
    """A single SNOMED role relationship (part of a logical definition)."""
    type_sctid: str
    type_fsn: str
    destination_sctid: str
    destination_preferred_term: str
    rel_group: int


@dataclass
class MRCMRange:
    attribute_sctid: str
    attribute_fsn: str
    range_constraint: str


@dataclass
class MRCMAttributeDomain:
    attribute_sctid: str
    domain_id: str
    grouped: bool
    attr_cardinality: str
    rule_strength_id: str
    content_type_id: str


class SnomedGraphClient:
    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
    ) -> None:
        load_dotenv(override=True)
        from neo4j import GraphDatabase

        _uri      = uri      or os.environ.get("NEO4J_URI", "")
        _user     = user     or os.environ.get("NEO4J_USER", "")
        _password = password or os.environ.get("NEO4J_PASSWORD", "")
        self._database = database or os.environ.get("NEO4J_DATABASE", "neo4j")

        missing = [k for k, v in {"NEO4J_URI": _uri, "NEO4J_USER": _user, "NEO4J_PASSWORD": _password}.items() if not v]
        if missing:
            raise ValueError(f"Missing Neo4j credentials: {missing}")

        self._driver = GraphDatabase.driver(_uri, auth=(_user, _password))

        # In-memory read caches — keyed by method arguments.
        # All four methods are pure reads; results never change within a run.
        self._cache_ancestors: Dict[Any, List] = {}
        self._cache_logical_def: Dict[str, List] = {}
        self._cache_attr_range: Dict[str, List] = {}
        self._cache_domain_attrs: Dict[str, List] = {}

    def close(self) -> None:
        self._driver.close()

    def __enter__(self) -> "SnomedGraphClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _concept_match(self, record: Any, match_type: str) -> ConceptMatch:
        return ConceptMatch(
            sctid=record["sctid"] or "",
            preferred_term=record["preferred_term"] or "",
            fsn=record["fsn"] or "",
            semantic_tag=record["semantic_tag"] or "",
            top_level_hierarchy=record["top_level_hierarchy"] or "",
            match_type=match_type,
        )

    _CONCEPT_RETURN = (
        "c.sctid AS sctid, c.preferred_term AS preferred_term, "
        "c.fsn AS fsn, c.semantic_tag AS semantic_tag, "
        "c.top_level_hierarchy AS top_level_hierarchy"
    )

    # ── 1. lookup_concept ────────────────────────────────────────────────────

    def lookup_concept(
        self,
        text: str,
        hierarchy_filter: Optional[str] = None,
    ) -> List[ConceptMatch]:
        """
        Priority order:
          1. Exact preferred_term (case-insensitive)
          2. Exact FSN (case-insensitive)
          3. Exact synonym match (ANY s IN c.synonyms)
          4. Fulltext search on preferred_term / fsn
          5. Substring preferred_term CONTAINS
        Returns up to 5 results.
        """
        results: List[ConceptMatch] = []
        seen: List[str] = []

        with self._driver.session(database=self._database) as session:
            hier_clause = "AND ($h IS NULL OR c.top_level_hierarchy = $h)"

            # 1. Exact preferred_term
            for r in session.run(
                f"MATCH (c:Concept) WHERE toLower(c.preferred_term) = toLower($t) "
                f"{hier_clause} RETURN {self._CONCEPT_RETURN} LIMIT 5",
                t=text, h=hierarchy_filter,
            ):
                results.append(self._concept_match(r, "exact"))
                seen.append(r["sctid"])

            if len(results) >= 5:
                return results

            # 2. Exact FSN
            for r in session.run(
                f"MATCH (c:Concept) WHERE toLower(c.fsn) = toLower($t) "
                f"AND NOT c.sctid IN $seen {hier_clause} "
                f"RETURN {self._CONCEPT_RETURN} LIMIT 5",
                t=text, h=hierarchy_filter, seen=seen,
            ):
                results.append(self._concept_match(r, "exact"))
                seen.append(r["sctid"])

            if len(results) >= 5:
                return results[:5]

            # 3. Exact synonym
            for r in session.run(
                f"MATCH (c:Concept) WHERE any(s IN c.synonyms WHERE toLower(s) = toLower($t)) "
                f"AND NOT c.sctid IN $seen {hier_clause} "
                f"RETURN {self._CONCEPT_RETURN} LIMIT 5",
                t=text, h=hierarchy_filter, seen=seen,
            ):
                results.append(self._concept_match(r, "synonym"))
                seen.append(r["sctid"])

            if len(results) >= 5:
                return results[:5]

            # 4. Fulltext search
            try:
                ft_query = f'"{text}"' if " " in text else text
                for r in session.run(
                    f"CALL db.index.fulltext.queryNodes('concept_text_search', $q) "
                    f"YIELD node AS c, score WHERE NOT c.sctid IN $seen "
                    f"AND ($h IS NULL OR c.top_level_hierarchy = $h) "
                    f"RETURN {self._CONCEPT_RETURN} ORDER BY score DESC LIMIT 5",
                    q=ft_query, h=hierarchy_filter, seen=seen,
                ):
                    results.append(self._concept_match(r, "fulltext"))
                    seen.append(r["sctid"])
            except Exception:
                pass  # fulltext index may not exist yet

            if len(results) >= 5:
                return results[:5]

            # 5. Substring preferred_term
            for r in session.run(
                f"MATCH (c:Concept) WHERE toLower(c.preferred_term) CONTAINS toLower($t) "
                f"AND NOT c.sctid IN $seen {hier_clause} "
                f"RETURN {self._CONCEPT_RETURN} LIMIT 5",
                t=text, h=hierarchy_filter, seen=seen,
            ):
                results.append(self._concept_match(r, "substring"))

        return results[:5]

    # ── 2. get_top_level_hierarchy ───────────────────────────────────────────

    def get_top_level_hierarchy(self, sctid: str) -> Optional[str]:
        """Return top_level_hierarchy for a given SCTID."""
        with self._driver.session(database=self._database) as session:
            r = session.run(
                "MATCH (c:Concept {sctid: $sctid}) RETURN c.top_level_hierarchy AS h",
                sctid=sctid,
            ).single()
            return r["h"] if r else None

    def get_hierarchy_map(self) -> Dict[int, str]:
        """Return {sctid_int: top_level_hierarchy} for all concepts in the graph."""
        with self._driver.session(database=self._database) as session:
            rows = session.run(
                "MATCH (c:Concept) WHERE c.top_level_hierarchy IS NOT NULL "
                "RETURN c.sctid AS sctid, c.top_level_hierarchy AS h"
            )
            return {int(r["sctid"]): r["h"] for r in rows}

    # ── 3. get_ancestors ─────────────────────────────────────────────────────

    def get_ancestors(self, sctid: str, max_depth: int = 5) -> List[ConceptMatch]:
        """Traverse IS_A edges upward up to max_depth hops."""
        key = (sctid, max_depth)
        if key not in self._cache_ancestors:
            with self._driver.session(database=self._database) as session:
                results = session.run(
                    f"MATCH (c:Concept {{sctid: $sctid}})-[:IS_A*1..{int(max_depth)}]->(a:Concept) "
                    f"RETURN DISTINCT {self._CONCEPT_RETURN.replace('c.', 'a.')}",
                    sctid=sctid,
                )
                self._cache_ancestors[key] = [self._concept_match(r, "exact") for r in results]
        return self._cache_ancestors[key]

    # ── 4. get_logical_definition ─────────────────────────────────────────────

    def get_logical_definition(self, sctid: str) -> List[RoleTriple]:
        """
        Return all HAS_ROLE edges from this concept (its SNOMED logical definition).
        Grouped by rel_group; group 0 = ungrouped attributes.
        """
        if sctid not in self._cache_logical_def:
            with self._driver.session(database=self._database) as session:
                rows = session.run(
                    "MATCH (c:Concept {sctid: $sctid})-[r:HAS_ROLE]->(d:Concept) "
                    "RETURN r.type_sctid AS type_sctid, r.type_fsn AS type_fsn, "
                    "       d.sctid AS dst_sctid, d.preferred_term AS dst_term, "
                    "       r.group AS rel_group "
                    "ORDER BY r.group, r.type_sctid",
                    sctid=sctid,
                )
                self._cache_logical_def[sctid] = [
                    RoleTriple(
                        type_sctid=r["type_sctid"] or "",
                        type_fsn=r["type_fsn"] or "",
                        destination_sctid=r["dst_sctid"] or "",
                        destination_preferred_term=r["dst_term"] or "",
                        rel_group=r["rel_group"] or 0,
                    )
                    for r in rows
                ]
        return self._cache_logical_def[sctid]

    # ── 5. get_attribute_range ───────────────────────────────────────────────

    def get_attribute_range(self, attribute_sctid: str) -> List[MRCMRange]:
        """Return MRCM range constraints for a given attribute SCTID.
        When multiple rules exist, prefers content_type_id=723594008 (precoordinated)."""
        if attribute_sctid not in self._cache_attr_range:
            with self._driver.session(database=self._database) as session:
                rows = list(session.run(
                    "MATCH (r:MRCMRange {attribute_sctid: $attr}) "
                    "OPTIONAL MATCH (attr:Concept {sctid: $attr}) "
                    "RETURN r.range_constraint AS rc, r.content_type_id AS ct, "
                    "       attr.fsn AS fsn",
                    attr=attribute_sctid,
                ))
            # Prefer precoordinated rule (723594008) when multiple rows exist
            if len(rows) > 1:
                filtered = [r for r in rows if r["ct"] == "723594008"]
                if filtered:
                    rows = filtered
            self._cache_attr_range[attribute_sctid] = [
                MRCMRange(
                    attribute_sctid=attribute_sctid,
                    attribute_fsn=r["fsn"] or attribute_sctid,
                    range_constraint=r["rc"] or "",
                )
                for r in rows
            ]
        return self._cache_attr_range[attribute_sctid]

    # ── 6. get_attribute_domain_rules ────────────────────────────────────────

    def get_attribute_domain_rules(self, attribute_sctid: str) -> List[MRCMAttributeDomain]:
        """Return MRCM attribute-domain rules for a given attribute SCTID."""
        with self._driver.session(database=self._database) as session:
            rows = session.run(
                "MATCH (d:MRCMAttributeDomain {attribute_sctid: $attr}) "
                "RETURN d.domain_id AS domain_id, d.grouped AS grouped, "
                "       d.attr_cardinality AS ac, d.rule_strength_id AS rs, "
                "       d.content_type_id AS ct",
                attr=attribute_sctid,
            )
            return [
                MRCMAttributeDomain(
                    attribute_sctid=attribute_sctid,
                    domain_id=r["domain_id"] or "",
                    grouped=bool(r["grouped"]),
                    attr_cardinality=r["ac"] or "",
                    rule_strength_id=r["rs"] or "",
                    content_type_id=r["ct"] or "",
                )
                for r in rows
            ]

    # ── 7. get_domain_attributes ─────────────────────────────────────────────

    def get_domain_attributes(self, focus_sctid: str) -> List[Dict[str, str]]:
        """
        Return applicable attributes for a given focus concept, enriched with names.
        Finds all MRCMAttributeDomain entries whose domain_id is an ancestor of
        (or equal to) the focus concept. Returns list of dicts with
        attribute_sctid, preferred_term, and fsn.
        """
        if focus_sctid not in self._cache_domain_attrs:
            with self._driver.session(database=self._database) as session:
                rows = session.run(
                    "MATCH (focus:Concept {sctid: $sctid})-[:IS_A*0..30]->(ancestor:Concept) "
                    "MATCH (d:MRCMAttributeDomain) WHERE d.domain_id = ancestor.sctid "
                    "OPTIONAL MATCH (attr:Concept {sctid: d.attribute_sctid}) "
                    "RETURN DISTINCT d.attribute_sctid AS attribute_sctid, attr.fsn AS fsn",
                    sctid=focus_sctid,
                )
                seen: set = set()
                result: List[Dict[str, str]] = []
                for r in rows:
                    attr = r["attribute_sctid"]
                    if attr and attr not in seen:
                        seen.add(attr)
                        result.append({
                            "attribute_sctid": attr,
                            "fsn": r["fsn"] or attr,
                        })
            self._cache_domain_attrs[focus_sctid] = result
        return self._cache_domain_attrs[focus_sctid]

    # ── 8. get_siblings ──────────────────────────────────────────────────────

    def get_siblings(self, sctid: str, limit: int = 10) -> List[ConceptMatch]:
        """Return concepts sharing at least one IS_A parent with sctid."""
        with self._driver.session(database=self._database) as session:
            results = session.run(
                "MATCH (n:Concept {sctid: $sctid})-[:IS_A]->(parent:Concept)"
                "<-[:IS_A]-(sibling:Concept) "
                "WHERE sibling.sctid <> $sctid "
                f"RETURN DISTINCT {self._CONCEPT_RETURN.replace('c.', 'sibling.')} "
                "LIMIT $limit",
                sctid=sctid,
                limit=limit,
            )
            return [self._concept_match(r, "sibling") for r in results]

    # ── 8. check_concept_exists ──────────────────────────────────────────────

    def check_concept_exists(self, sctid: str) -> bool:
        with self._driver.session(database=self._database) as session:
            r = session.run(
                "MATCH (c:Concept {sctid: $sctid}) RETURN count(c) > 0 AS exists",
                sctid=sctid,
            ).single()
            return bool(r["exists"]) if r else False


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    client = SnomedGraphClient()
    print("\n--- SnomedGraphClient Smoke Test ---\n")

    # 1. lookup_concept
    cases = [
        ("Nitrates",            "Substance"),
        ("Hepatic failure",     "Clinical Finding"),
        ("severe",              "Qualifier Value"),
        ("structure of liver",  "Body Structure"),
        ("administration of drug", "Procedure"),
        ("Nitric oxide",        "Substance"),
    ]
    print("[ lookup_concept ]")
    for term, expected in cases:
        results = client.lookup_concept(term)
        if not results:
            print(f"  FAIL  '{term}' → no results (expected {expected})")
        elif results[0].top_level_hierarchy != expected:
            print(f"  WARN  '{term}' → {results[0].top_level_hierarchy} (expected {expected}) | {results[0].preferred_term}")
        else:
            print(f"  OK    '{term}' → {results[0].preferred_term} [{results[0].semantic_tag}]")

    # 2. get_logical_definition for "Allergy to aspirin" (387480006)
    print("\n[ get_logical_definition: 387480006 Allergy to aspirin ]")
    ld = client.get_logical_definition("387480006")
    if ld:
        for triple in ld:
            print(f"  group={triple.rel_group}  {triple.type_fsn} ({triple.type_sctid}) → {triple.destination_preferred_term} ({triple.destination_sctid})")
    else:
        print("  (not in loaded graph or no role rels)")

    # 3. get_attribute_range for causative_agent (246075003)
    print("\n[ get_attribute_range: 246075003 causative_agent ]")
    for r in client.get_attribute_range("246075003"):
        print(f"  range_constraint: {r.range_constraint[:120]}")
        print(f"  content_type_id:  {r.content_type_id}")

    # 4. get_ancestors
    print("\n[ get_ancestors: 387480006 depth=3 ]")
    for a in client.get_ancestors("387480006", max_depth=3):
        print(f"  {a.sctid} | {a.preferred_term} | {a.top_level_hierarchy}")

    # 5. get_siblings — find siblings of "Allergy to aspirin" (387480006)
    print("\n[ get_siblings: 387480006 limit=5 ]")
    for s in client.get_siblings("387480006", limit=5):
        print(f"  {s.sctid} | {s.preferred_term}")

    client.close()
    print("\n--- Smoke test complete ---")
