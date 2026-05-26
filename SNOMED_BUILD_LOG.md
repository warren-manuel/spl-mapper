# SNOMED Neo4j Build Log
*Started: 2026-05-19T19:59:13.572889+00:00*

## Step 0: Connecting to Neo4j
  Neo4j connection: VERIFIED (bolt://139.52.39.81:7687)

## Stage A: Load dataframes
  Calling load_snomed_dataframes()...
  Completed in 39.2s
  concept_df: 382151 rows
  rel_df: 1336381 rows
  IS-A rels: 631617   Role rels: 704764
  Building parent→children index for hierarchy BFS...
  Hierarchy map: 321631 concepts assigned
  Concepts in CONTRAINDICATION_RELEVANT: 321612
    Clinical Finding: 130225
    Procedure: 60999
    Substance: 27991
    Body Structure: 42752
    Qualifier Value: 12492
    Observable Entity: 11004
    Organism: 34956
    Attribute: 1193

## Stage B: Build Neo4j graph
  Schema: constraints, indexes, fulltext index created
  Concepts loaded: 321612  (19.5s)
  IS_A edges: 552002  (20.8s)
  HAS_ROLE edges: 559762  (24.7s)
  MRCMRange nodes: 131  (0.2s)
  MRCMAttributeDomain nodes: 144  (0.0s)
  MRCMDomain nodes: 19  (0.0s)

## Summary
  Concepts:              321612
  IS_A edges:            552002
  HAS_ROLE edges:        559762
  MRCMRange nodes:       131
  MRCMAttributeDomain:   144
  MRCMDomain:            19

## Stage C: Generating snomed_conventions.md
  Written: agents/snomed_conventions.md (647 lines)

## Build complete: 2026-05-19T20:01:01.718266+00:00
