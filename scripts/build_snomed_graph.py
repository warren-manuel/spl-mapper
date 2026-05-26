#!/usr/bin/env python3
"""
Build SNOMED CT knowledge graph in Neo4j for Agent 1.

Capabilities loaded:
  1. Hierarchy traversal     — IS_A edges
  2. Synonym / label lookup  — synonyms[] array property + fulltext index
  3. Logical definitions     — HAS_ROLE edges (type_sctid, type_fsn, rel_group)
  4. Attribute-aware expl.   — MRCMRange / MRCMAttributeDomain nodes + links to Concept
  5. Convention checking     — MRCMDomain nodes + top_level_hierarchy + semantic_tag

Usage:
  conda run -n splmap python scripts/build_snomed_graph.py
"""
from __future__ import annotations

import os
import sys
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.snomed.snomed_utils import _strip_semantic_tag, load_snomed_dataframes

# ── Hierarchy constants ────────────────────────────────────────────────────────

TOP_LEVEL_HIERARCHIES: Dict[str, str] = {
    "404684003": "Clinical Finding",
    "71388002":  "Procedure",
    "363787002": "Observable Entity",
    "123037004": "Body Structure",
    "410607006": "Organism",
    "105590001": "Substance",
    "373873005": "Pharmaceutical/Biological Product",
    "362981000": "Qualifier Value",
    "243796009": "Situation with Explicit Context",
    "246061005": "Attribute",
}

CONTRAINDICATION_RELEVANT: Set[str] = {
    "404684003", "71388002", "105590001",
    "123037004", "362981000", "363787002", "410607006",
    "246061005",
}

HIERARCHY_PRIORITY: List[str] = [
    "404684003", "71388002", "105590001",
    "123037004", "362981000", "363787002", "410607006",
    "246061005",
]

LOG_PATH = Path("SNOMED_BUILD_LOG.md")
CONVENTIONS_PATH = Path("agents/snomed_conventions.md")
BATCH_CONCEPTS = 500
BATCH_EDGES   = 1000
BATCH_MRCM    = 200


# ── Logging ────────────────────────────────────────────────────────────────────

def _log(msg: str) -> None:
    print(msg)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


# ── Stage A: Load and enrich dataframes ───────────────────────────────────────

def build_hierarchy_map(isa_df: pd.DataFrame) -> Dict[int, str]:
    _log("  Building parent→children index for hierarchy BFS...")
    children_of: Dict[int, Set[int]] = defaultdict(set)
    for parent, child in zip(
        isa_df["destinationId"].astype(int).tolist(),
        isa_df["sourceId"].astype(int).tolist(),
    ):
        children_of[parent].add(child)

    hierarchy_map: Dict[int, str] = {}
    for hier_id in HIERARCHY_PRIORITY:
        root, hier_name = int(hier_id), TOP_LEVEL_HIERARCHIES[hier_id]
        queue: deque = deque([root])
        visited: Set[int] = set()
        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            if node not in hierarchy_map:
                hierarchy_map[node] = hier_name
            for child in children_of.get(node, set()):
                if child not in hierarchy_map:
                    queue.append(child)

    _log(f"  Hierarchy map: {len(hierarchy_map)} concepts assigned")
    return hierarchy_map


def load_and_enrich(snomed_source_dir: str) -> Tuple[
    pd.DataFrame,   # concepts_to_load (CONTRAINDICATION_RELEVANT)
    pd.DataFrame,   # isa_df
    pd.DataFrame,   # role_df (non-IS-A active rels)
    Dict[int, str], # sctid -> preferred_term lookup (for type_fsn on edges)
    pd.DataFrame,   # attr_range
    pd.DataFrame,   # attr_domain
    pd.DataFrame,   # mrcm_domain
]:
    _log("  Calling load_snomed_dataframes()...")
    t0 = time.perf_counter()
    dfs = load_snomed_dataframes(snomed_source_dir=snomed_source_dir)
    _log(f"  Completed in {time.perf_counter() - t0:.1f}s")

    concept_df       = dfs["concept_df"].copy()
    synonym_df       = dfs["synonym_df"]
    snomed_complete  = dfs["snomed_complete_df"]
    rel_df           = dfs["rel_df"]
    attr_range       = dfs["attr_range"]
    attr_domain      = dfs["attr_domain"]
    mrcm_domain      = dfs["domain"]

    _log(f"  concept_df: {len(concept_df)} rows")
    _log(f"  rel_df: {len(rel_df)} rows")

    # Derive preferred_term (FSN stripped of semantic tag)
    concept_df["preferred_term"] = concept_df["term"].apply(_strip_semantic_tag)
    concept_df["sctid"] = concept_df["conceptId"].astype(str)

    # Fast lookup: int sctid → preferred_term  (used for type_fsn on HAS_ROLE)
    id_to_preferred: Dict[int, str] = dict(
        zip(concept_df["conceptId"].astype(int), concept_df["preferred_term"])
    )

    # Attach aggregated synonyms from snomed_complete
    synonym_map: Dict[int, List[str]] = {}
    for row in snomed_complete.itertuples(index=False):
        syns = row.synonyms if isinstance(row.synonyms, list) else []
        synonym_map[int(row.conceptId)] = [str(s) for s in syns if s]
    concept_df["synonyms"] = concept_df["conceptId"].astype(int).map(
        lambda cid: synonym_map.get(cid, [])
    )

    # IS-A and role split
    isa_df  = rel_df[rel_df["typeId"] == 116680003].copy()
    role_df = rel_df[rel_df["typeId"] != 116680003].copy()
    _log(f"  IS-A rels: {len(isa_df)}   Role rels: {len(role_df)}")

    # Hierarchy assignment
    hierarchy_map = build_hierarchy_map(isa_df)
    concept_df["top_level_hierarchy"] = (
        concept_df["conceptId"].astype(int).map(hierarchy_map)
    )

    # Filter to CONTRAINDICATION_RELEVANT
    relevant_names = {TOP_LEVEL_HIERARCHIES[k] for k in CONTRAINDICATION_RELEVANT}
    concepts_to_load = concept_df[
        concept_df["top_level_hierarchy"].isin(relevant_names)
    ].copy()
    _log(f"  Concepts in CONTRAINDICATION_RELEVANT: {len(concepts_to_load)}")
    for h in HIERARCHY_PRIORITY:
        n = TOP_LEVEL_HIERARCHIES[h]
        if n not in relevant_names:
            continue
        cnt = len(concepts_to_load[concepts_to_load["top_level_hierarchy"] == n])
        _log(f"    {n}: {cnt}")

    return concepts_to_load, isa_df, role_df, id_to_preferred, attr_range, attr_domain, mrcm_domain


# ── Stage B: Schema ────────────────────────────────────────────────────────────

_SCHEMA_CYPHER = """
CREATE CONSTRAINT concept_sctid IF NOT EXISTS
  FOR (c:Concept) REQUIRE c.sctid IS UNIQUE;

CREATE CONSTRAINT mrcm_range_uid IF NOT EXISTS
  FOR (r:MRCMRange) REQUIRE r.uid IS UNIQUE;

CREATE CONSTRAINT mrcm_attr_domain_uid IF NOT EXISTS
  FOR (d:MRCMAttributeDomain) REQUIRE d.uid IS UNIQUE;

CREATE CONSTRAINT mrcm_domain_uid IF NOT EXISTS
  FOR (d:MRCMDomain) REQUIRE d.uid IS UNIQUE;

CREATE INDEX concept_preferred_term IF NOT EXISTS
  FOR (c:Concept) ON (c.preferred_term);

CREATE INDEX concept_semantic_tag IF NOT EXISTS
  FOR (c:Concept) ON (c.semantic_tag);

CREATE INDEX concept_hierarchy IF NOT EXISTS
  FOR (c:Concept) ON (c.top_level_hierarchy);

CREATE INDEX mrcm_range_attr IF NOT EXISTS
  FOR (r:MRCMRange) ON (r.attribute_sctid);

CREATE INDEX mrcm_attr_domain_attr IF NOT EXISTS
  FOR (d:MRCMAttributeDomain) ON (d.attribute_sctid);

CREATE INDEX mrcm_domain_concept IF NOT EXISTS
  FOR (d:MRCMDomain) ON (d.concept_sctid);
"""

_FULLTEXT_CYPHER = """
CREATE FULLTEXT INDEX concept_text_search IF NOT EXISTS
  FOR (c:Concept) ON EACH [c.preferred_term, c.fsn]
"""


def setup_schema(session: Any) -> None:
    for stmt in _SCHEMA_CYPHER.strip().split(";"):
        stmt = stmt.strip()
        if stmt:
            session.run(stmt)
    try:
        session.run(_FULLTEXT_CYPHER)
    except Exception as e:
        _log(f"  WARNING: fulltext index creation: {e}")
    _log("  Schema: constraints, indexes, fulltext index created")


# ── Stage B1: Concepts ─────────────────────────────────────────────────────────

def load_concepts(driver: Any, database: str, concepts_df: pd.DataFrame) -> int:
    records = []
    for row in concepts_df.itertuples(index=False):
        synonyms = row.synonyms if isinstance(row.synonyms, list) else []
        records.append({
            "sctid":               str(row.sctid),
            "fsn":                 str(row.term) if pd.notna(row.term) else "",
            "preferred_term":      str(row.preferred_term) if pd.notna(row.preferred_term) else "",
            "semantic_tag":        str(row.semantic_tag) if pd.notna(row.semantic_tag) else "",
            "top_level_hierarchy": str(row.top_level_hierarchy) if pd.notna(row.top_level_hierarchy) else "",
            "active":              True,
            "synonyms":            [str(s) for s in synonyms],
        })
    total = 0
    with driver.session(database=database) as session:
        for i in range(0, len(records), BATCH_CONCEPTS):
            batch = records[i : i + BATCH_CONCEPTS]
            session.run(
                """
                UNWIND $batch AS row
                MERGE (c:Concept {sctid: row.sctid})
                SET c.fsn               = row.fsn,
                    c.preferred_term    = row.preferred_term,
                    c.semantic_tag      = row.semantic_tag,
                    c.top_level_hierarchy = row.top_level_hierarchy,
                    c.active            = row.active,
                    c.synonyms          = row.synonyms
                """,
                batch=batch,
            )
            total += len(batch)
            print(f"  Concepts: {total}/{len(records)}", end="\r", flush=True)
    print()
    return total


# ── Stage B2: IS_A edges ───────────────────────────────────────────────────────

def load_isa_edges(driver: Any, database: str, isa_df: pd.DataFrame, loaded_ids: Set[int]) -> int:
    mask = isa_df["sourceId"].isin(loaded_ids) & isa_df["destinationId"].isin(loaded_ids)
    filtered = isa_df[mask]
    records = [
        {"child": str(int(s)), "parent": str(int(d))}
        for s, d in zip(filtered["sourceId"].tolist(), filtered["destinationId"].tolist())
    ]
    total = 0
    with driver.session(database=database) as session:
        for i in range(0, len(records), BATCH_EDGES):
            batch = records[i : i + BATCH_EDGES]
            session.run(
                """
                UNWIND $batch AS row
                MATCH (child:Concept  {sctid: row.child})
                MATCH (parent:Concept {sctid: row.parent})
                MERGE (child)-[:IS_A]->(parent)
                """,
                batch=batch,
            )
            total += len(batch)
            print(f"  IS_A edges: {total}/{len(records)}", end="\r", flush=True)
    print()
    return total


# ── Stage B3: HAS_ROLE edges (logical definitions) ────────────────────────────

def load_role_edges(
    driver: Any,
    database: str,
    role_df: pd.DataFrame,
    loaded_ids: Set[int],
    id_to_preferred: Dict[int, str],
) -> int:
    mask = role_df["sourceId"].isin(loaded_ids) & role_df["destinationId"].isin(loaded_ids)
    filtered = role_df[mask]
    records = [
        {
            "src":      str(int(r.sourceId)),
            "dst":      str(int(r.destinationId)),
            "type_id":  str(int(r.typeId)),
            "type_fsn": id_to_preferred.get(int(r.typeId), str(int(r.typeId))),
            "group":    int(r.relationshipGroup),
        }
        for r in filtered.itertuples(index=False)
    ]
    total = 0
    with driver.session(database=database) as session:
        for i in range(0, len(records), BATCH_EDGES):
            batch = records[i : i + BATCH_EDGES]
            session.run(
                """
                UNWIND $batch AS row
                MATCH (src:Concept {sctid: row.src})
                MATCH (dst:Concept {sctid: row.dst})
                MERGE (src)-[r:HAS_ROLE {type_sctid: row.type_id, group: row.group}]->(dst)
                SET r.type_fsn = row.type_fsn
                """,
                batch=batch,
            )
            total += len(batch)
            print(f"  HAS_ROLE edges: {total}/{len(records)}", end="\r", flush=True)
    print()
    return total


# ── Stage B4: MRCM nodes ──────────────────────────────────────────────────────

def load_mrcm_range(driver: Any, database: str, attr_range: pd.DataFrame) -> int:
    records = []
    for i, row in enumerate(attr_range.itertuples(index=False)):
        records.append({
            "uid":            str(row.id),
            "attribute_sctid":str(int(row.referencedComponentId)),
            "range_constraint": str(row.rangeConstraint) if pd.notna(row.rangeConstraint) else "",
            "attribute_rule":  str(row.attributeRule)    if pd.notna(row.attributeRule)   else "",
            "rule_strength_id":str(int(row.ruleStrengthId)),
            "content_type_id": str(int(row.contentTypeId)),
        })
    with driver.session(database=database) as session:
        for i in range(0, len(records), BATCH_MRCM):
            batch = records[i : i + BATCH_MRCM]
            session.run(
                """
                UNWIND $batch AS row
                MERGE (r:MRCMRange {uid: row.uid})
                SET r.attribute_sctid  = row.attribute_sctid,
                    r.range_constraint = row.range_constraint,
                    r.attribute_rule   = row.attribute_rule,
                    r.rule_strength_id = row.rule_strength_id,
                    r.content_type_id  = row.content_type_id
                WITH r, row
                OPTIONAL MATCH (attr:Concept {sctid: row.attribute_sctid})
                FOREACH (_ IN CASE WHEN attr IS NOT NULL THEN [1] ELSE [] END |
                    MERGE (attr)-[:HAS_ATTRIBUTE_RANGE]->(r)
                )
                """,
                batch=batch,
            )
    return len(records)


def load_mrcm_attr_domain(driver: Any, database: str, attr_domain: pd.DataFrame) -> int:
    records = []
    for row in attr_domain.itertuples(index=False):
        records.append({
            "uid":             str(row.id),
            "attribute_sctid": str(int(row.referencedComponentId)),
            "domain_id":       str(int(row.domainId)),
            "grouped":         bool(int(row.grouped)),
            "attr_cardinality":str(row.attributeCardinality)        if pd.notna(row.attributeCardinality)        else "",
            "attr_in_group":   str(row.attributeInGroupCardinality) if pd.notna(row.attributeInGroupCardinality) else "",
            "rule_strength_id":str(int(row.ruleStrengthId)),
            "content_type_id": str(int(row.contentTypeId)),
        })
    with driver.session(database=database) as session:
        for i in range(0, len(records), BATCH_MRCM):
            batch = records[i : i + BATCH_MRCM]
            session.run(
                """
                UNWIND $batch AS row
                MERGE (d:MRCMAttributeDomain {uid: row.uid})
                SET d.attribute_sctid  = row.attribute_sctid,
                    d.domain_id        = row.domain_id,
                    d.grouped          = row.grouped,
                    d.attr_cardinality = row.attr_cardinality,
                    d.attr_in_group    = row.attr_in_group,
                    d.rule_strength_id = row.rule_strength_id,
                    d.content_type_id  = row.content_type_id
                WITH d, row
                OPTIONAL MATCH (attr:Concept {sctid: row.attribute_sctid})
                FOREACH (_ IN CASE WHEN attr IS NOT NULL THEN [1] ELSE [] END |
                    MERGE (attr)-[:HAS_ATTRIBUTE_DOMAIN_RULE]->(d)
                )
                """,
                batch=batch,
            )
    return len(records)


def load_mrcm_domain(driver: Any, database: str, mrcm_domain: pd.DataFrame) -> int:
    records = []
    for row in mrcm_domain.itertuples(index=False):
        records.append({
            "uid":             str(row.id),
            "concept_sctid":   str(int(row.referencedComponentId)),
            "domain_constraint":          str(row.domainConstraint)                    if pd.notna(row.domainConstraint)                    else "",
            "proximal_primitive_constraint": str(row.proximalPrimitiveConstraint)      if pd.notna(row.proximalPrimitiveConstraint)          else "",
            "domain_template_precoord":   str(row.domainTemplateForPrecoordination)    if pd.notna(row.domainTemplateForPrecoordination)     else "",
            "domain_template_postcoord":  str(row.domainTemplateForPostcoordination)   if pd.notna(row.domainTemplateForPostcoordination)    else "",
        })
    with driver.session(database=database) as session:
        for i in range(0, len(records), BATCH_MRCM):
            batch = records[i : i + BATCH_MRCM]
            session.run(
                """
                UNWIND $batch AS row
                MERGE (d:MRCMDomain {uid: row.uid})
                SET d.concept_sctid                  = row.concept_sctid,
                    d.domain_constraint               = row.domain_constraint,
                    d.proximal_primitive_constraint   = row.proximal_primitive_constraint,
                    d.domain_template_precoord        = row.domain_template_precoord,
                    d.domain_template_postcoord       = row.domain_template_postcoord
                WITH d, row
                OPTIONAL MATCH (c:Concept {sctid: row.concept_sctid})
                FOREACH (_ IN CASE WHEN c IS NOT NULL THEN [1] ELSE [] END |
                    MERGE (c)-[:HAS_DOMAIN_DEFINITION]->(d)
                )
                """,
                batch=batch,
            )
    return len(records)


# ── Stage C: snomed_conventions.md ────────────────────────────────────────────

HIERARCHY_LINGUISTIC_CUES: Dict[str, List[str]] = {
    "Clinical Finding": ["disorder","disease","syndrome","impairment","failure","deficiency",
                         "hypersensitivity","allergy","infection","history of","anaphylaxis",
                         "urticaria","anemia","hypertension","insufficiency","condition"],
    "Procedure":        ["administration","use of","treatment","therapy","coadministration",
                         "concurrent use","application"],
    "Substance":        ["nitrate","NSAID","aspirin","steroid","anticoagulant","inhibitor",
                         "blocker","donor","drug","compound","agent","medication","salt"],
    "Body Structure":   ["liver","renal","hepatic","cornea","cardiac","kidney","lung",
                         "heart","brain","structure of"],
    "Qualifier Value":  ["severe","mild","moderate","acute","chronic","recurrent",
                         "subacute","complete","partial","advanced"],
    "Observable Entity":["level","count","ratio","measurement","function"],
    "Organism":         ["virus","bacteria","fungus","parasite","pathogen"],
}

HIERARCHY_CONTRAINDICATION_ROLE: Dict[str, str] = {
    "Clinical Finding":  "focus — the disorder/finding that is contraindicated",
    "Procedure":         "focus — procedures contraindicated or contraindicated-by",
    "Substance":         "slot_filler — causative_agent (246075003)",
    "Body Structure":    "slot_filler — finding_site (363698007)",
    "Qualifier Value":   "slot_filler — severity (246112005), clinical_course (263502005)",
    "Observable Entity": "rarely_used — lab-threshold contraindications",
    "Organism":          "slot_filler — causative_agent (246075003) for biological organisms",
}

HIERARCHY_VALID_SLOTS: Dict[str, str] = {
    "Clinical Finding":  "none (focus concept)",
    "Procedure":         "none (focus concept)",
    "Substance":         "causative_agent (246075003)",
    "Body Structure":    "finding_site (363698007)",
    "Qualifier Value":   "severity (246112005), clinical_course (263502005)",
    "Observable Entity": "none",
    "Organism":          "causative_agent (246075003)",
}


def generate_conventions_md(driver: Any, database: str) -> None:
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    lines: List[str] = [
        "# SNOMED CT Conventions — Semantic Tags Reference",
        f"*Generated from SNOMED CT US Edition 20250901 on {now_str}*",
        "",
        "## Purpose",
        ("Reference for Agent 1 (SNOMED Concept Navigator) semantic categorization of "
         "contraindication components. Generated from SNOMED CT RF2 US Edition 20250901."),
        "",
        "## How to Use This File",
        ("Match linguistic cues to hierarchy descriptions below. Use `graph_client` tool "
         "only for ambiguous cases."),
        "",
        "## Top-Level Hierarchy Reference",
        "",
    ]
    with driver.session(database=database) as session:
        for hier_id in HIERARCHY_PRIORITY:
            if hier_id not in CONTRAINDICATION_RELEVANT:
                continue
            name = TOP_LEVEL_HIERARCHIES[hier_id]
            # Semantic tags
            tags = session.run(
                "MATCH (c:Concept {top_level_hierarchy: $h}) "
                "RETURN c.semantic_tag AS tag, count(c) AS n ORDER BY n DESC",
                h=name,
            ).data()
            # Examples
            examples = session.run(
                "MATCH (c:Concept {top_level_hierarchy: $h}) "
                "RETURN c.sctid AS sctid, c.preferred_term AS pt, c.semantic_tag AS st "
                "ORDER BY c.preferred_term LIMIT 50",
                h=name,
            ).data()

            lines += [f"### {name} ({hier_id})", ""]
            lines += [f"**Contraindication role:** {HIERARCHY_CONTRAINDICATION_ROLE.get(name,'')}", ""]
            lines += [f"**Valid as slot:** {HIERARCHY_VALID_SLOTS.get(name,'none')}", ""]
            lines += ["**Linguistic cues:**"]
            for cue in HIERARCHY_LINGUISTIC_CUES.get(name, []):
                lines.append(f"- {cue}")
            lines.append("")
            lines += ["**Semantic tags:**", "| Semantic Tag | Concept Count |", "|---|---|"]
            for r in tags[:20]:
                lines.append(f"| {r['tag'] or ''} | {r['n']} |")
            lines.append("")
            lines += ["**Example concepts (up to 50, A–Z):**",
                      "| SCTID | Preferred Term | Semantic Tag |", "|---|---|---|"]
            for r in examples:
                lines.append(f"| {r['sctid']} | {r['pt'] or ''} | {r['st'] or ''} |")
            lines += ["", "---", ""]

    CONVENTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONVENTIONS_PATH.write_text("\n".join(lines), encoding="utf-8")
    _log(f"  Written: {CONVENTIONS_PATH} ({len(lines)} lines)")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    load_dotenv(override=True)
    LOG_PATH.write_text(
        f"# SNOMED Neo4j Build Log\n*Started: {datetime.now(timezone.utc).isoformat()}*\n\n",
        encoding="utf-8",
    )

    uri      = os.environ.get("NEO4J_URI", "").strip()
    user     = os.environ.get("NEO4J_USER", "").strip()
    password = os.environ.get("NEO4J_PASSWORD", "").strip()
    database = os.environ.get("NEO4J_DATABASE", "neo4j").strip()
    missing  = [k for k, v in {"NEO4J_URI": uri, "NEO4J_USER": user, "NEO4J_PASSWORD": password}.items() if not v]
    if missing:
        print(f"STOP: Missing env vars: {missing}")
        sys.exit(1)

    _log("## Step 0: Connecting to Neo4j")
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        driver.verify_connectivity()
        _log(f"  Neo4j connection: VERIFIED ({uri})")
    except Exception as exc:
        _log(f"  Neo4j connection: FAILED — {exc}")
        sys.exit(1)

    snomed_dir = os.environ.get("SNOMED_SOURCE_DIR", "snomed_us_source")

    # Stage A
    _log("\n## Stage A: Load dataframes")
    concepts_df, isa_df, role_df, id_to_preferred, attr_range, attr_domain, mrcm_domain = (
        load_and_enrich(snomed_dir)
    )
    loaded_ids_int: Set[int] = set(concepts_df["conceptId"].astype(int).tolist())

    # Stage B — schema
    _log("\n## Stage B: Build Neo4j graph")
    with driver.session(database=database) as session:
        setup_schema(session)

    t0 = time.perf_counter()
    n_concepts = load_concepts(driver, database, concepts_df)
    _log(f"  Concepts loaded: {n_concepts}  ({time.perf_counter()-t0:.1f}s)")

    t0 = time.perf_counter()
    n_isa = load_isa_edges(driver, database, isa_df, loaded_ids_int)
    _log(f"  IS_A edges: {n_isa}  ({time.perf_counter()-t0:.1f}s)")

    t0 = time.perf_counter()
    n_roles = load_role_edges(driver, database, role_df, loaded_ids_int, id_to_preferred)
    _log(f"  HAS_ROLE edges: {n_roles}  ({time.perf_counter()-t0:.1f}s)")

    t0 = time.perf_counter()
    n_range = load_mrcm_range(driver, database, attr_range)
    _log(f"  MRCMRange nodes: {n_range}  ({time.perf_counter()-t0:.1f}s)")

    t0 = time.perf_counter()
    n_attr_dom = load_mrcm_attr_domain(driver, database, attr_domain)
    _log(f"  MRCMAttributeDomain nodes: {n_attr_dom}  ({time.perf_counter()-t0:.1f}s)")

    t0 = time.perf_counter()
    n_dom = load_mrcm_domain(driver, database, mrcm_domain)
    _log(f"  MRCMDomain nodes: {n_dom}  ({time.perf_counter()-t0:.1f}s)")

    _log(f"\n## Summary")
    _log(f"  Concepts:              {n_concepts}")
    _log(f"  IS_A edges:            {n_isa}")
    _log(f"  HAS_ROLE edges:        {n_roles}")
    _log(f"  MRCMRange nodes:       {n_range}")
    _log(f"  MRCMAttributeDomain:   {n_attr_dom}")
    _log(f"  MRCMDomain:            {n_dom}")

    # Stage C
    _log("\n## Stage C: Generating snomed_conventions.md")
    generate_conventions_md(driver, database)

    driver.close()
    _log(f"\n## Build complete: {datetime.now(timezone.utc).isoformat()}")
    print("\nDone. See SNOMED_BUILD_LOG.md and agents/snomed_conventions.md")


if __name__ == "__main__":
    main()
