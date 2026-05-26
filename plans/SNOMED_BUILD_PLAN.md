# SNOMED CT Knowledge Infrastructure — Build Plan

## Step 0 Findings

### `load_snomed_dataframes()` — Keys and Columns

| Key | Columns | Notes |
|---|---|---|
| `concept_df` | conceptId, term (FSN), semantic_tag | active concepts only; semantic_tag already computed |
| `synonym_df` | conceptId, term | active synonyms only |
| `terms_df` | conceptId, term_text, term_type ('preferred'\|'synonym') | union of FSN + synonyms |
| `snomed_complete_df` | conceptId, term (FSN), semantic_tag, synonyms (list) | concept-level with aggregated synonym list |
| `enriched_terms_df` | conceptId, term_text (pipe-delimited) | FSN \| syn1 \| syn2 \| semantic_tag \| attr=val … |
| `domain` | full MRCM domain refset columns | active rows only |
| `attr_domain` | full MRCM attribute-domain refset columns | active rows only |
| `attr_range` | full MRCM attribute-range refset columns | active rows only |
| `rel_df` | sourceId, destinationId, typeId, relationshipGroup | active only; ALL relationship types (IS-A + role) |

**RF2 release:** US Edition 20250901

### What is already computed

| Item | Status | How |
|---|---|---|
| `semantic_tag` | ✅ Already in `concept_df` | `extract_semantic_tag(fsn)` — regex `\(([^)]+)\)$`, returns lowercase |
| Active-only filter | ✅ Done inside loader | concept_rf2 active==1, desc active==1, rel active==1 |
| IS-A edges in rel_df | ✅ Present (not pre-filtered) | rel_df contains typeId==116680003 rows; filter manually |
| BFS ancestor traversal | ✅ `get_ancestors_with_depth(concept_id, rel_df)` | BFS upward, returns `{ancestor_id: hop_distance}`; requires df indexed by sourceId |

### What must be derived

| Item | Action |
|---|---|
| `preferred_term` | `_strip_semantic_tag(fsn)` — already in snomed_utils.py |
| `top_level_hierarchy` | BFS downward from each CONTRAINDICATION_RELEVANT root; `defaultdict(set)` parent→children; first-match-wins by priority |

### RF2 Files Present

```
snomed_us_source/
  sct2_Concept_Snapshot_US1000124_20250901.txt
  sct2_Description_Snapshot-en_US1000124_20250901.txt
  sct2_Description_Full-en_US1000124_20250901.txt
  sct2_Relationship_Snapshot_US1000124_20250901.txt
  der2_sssssssRefset_MRCMDomainSnapshot_US1000124_20250901.txt
  der2_cissccRefset_MRCMAttributeDomainSnapshot_US1000124_20250901.txt
  der2_ssccRefset_MRCMAttributeRangeSnapshot_US1000124_20250901.txt
```

### Record Counts

| Item | Count |
|---|---|
| Total concepts | 532,287 |
| Active concepts | 382,170 |
| Active IS-A rels | 631,617 |
| Total active rels | 1,336,381 |
| Direct children of root (138875005) | 19 |

### Backend: Neo4j 5.26.25 — VERIFIED

- URI: `bolt://139.52.39.81:7687`
- Build complete: 2026-04-30
- `neo4j` Python driver — OK

---

## Build Artifacts

| File | Purpose |
|---|---|
| `scripts/build_snomed_graph.py` | Loads full graph into Neo4j (concepts, IS_A, HAS_ROLE, MRCM) |
| `src/snomed/graph_client.py` | Runtime client — 7 methods covering all 5 capabilities |
| `agents/snomed_conventions.md` | Generated hierarchy reference for Agent 1 |
| `SNOMED_BUILD_LOG.md` | Run log with record counts |

## Final Graph Counts

| Element | Count |
|---|---|
| Concept nodes | 320,419 |
| IS_A edges | 550,810 |
| HAS_ROLE edges | 559,762 |
| MRCMRange nodes | 131 |
| MRCMAttributeDomain nodes | 144 |
| MRCMDomain nodes | 19 |

## Neo4j Schema

```cypher
// Nodes
(:Concept {sctid, fsn, preferred_term, semantic_tag, top_level_hierarchy, active, synonyms[]})
(:MRCMRange {uid, attribute_sctid, range_constraint, attribute_rule, rule_strength_id, content_type_id})
(:MRCMAttributeDomain {uid, attribute_sctid, domain_id, grouped, attr_cardinality, rule_strength_id})
(:MRCMDomain {uid, concept_sctid, domain_constraint, proximal_primitive_constraint, templates})

// Relationships
(:Concept)-[:IS_A]->(:Concept)                          // hierarchy traversal
(:Concept)-[:HAS_ROLE {type_sctid, type_fsn, group}]->(:Concept)  // logical definitions
(:Concept)-[:HAS_ATTRIBUTE_RANGE]->(:MRCMRange)          // attribute-aware explanations
(:Concept)-[:HAS_ATTRIBUTE_DOMAIN_RULE]->(:MRCMAttributeDomain)
(:Concept)-[:HAS_DOMAIN_DEFINITION]->(:MRCMDomain)
```

## graph_client.py Capabilities

| Method | Capability |
|---|---|
| `lookup_concept(text, hierarchy_filter)` | Exact → synonym → fulltext → substring |
| `get_top_level_hierarchy(sctid)` | Hierarchy label lookup |
| `get_ancestors(sctid, max_depth)` | IS_A* traversal |
| `get_logical_definition(sctid)` | All HAS_ROLE edges (SNOMED definition) |
| `get_attribute_range(attribute_sctid)` | MRCM ECL range constraints |
| `get_attribute_domain_rules(attribute_sctid)` | MRCM domain rules |
| `check_concept_exists(sctid)` | Boolean existence check |
