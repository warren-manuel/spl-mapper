# SNOMED Neo4j Build Log
*Started: 2026-05-26T19:45:19.946594+00:00*

## Step 0: Connecting to Neo4j
  Neo4j connection: VERIFIED (bolt://139.52.39.81:7687)

## Stage A: Load dataframes
  Calling load_snomed_dataframes()...
  Completed in 38.7s
  concept_df: 382151 rows
  rel_df: 1336381 rows
  IS-A rels: 631617   Role rels: 704764
  Building parent→children index for hierarchy BFS...
  Hierarchy map: 347350 concepts assigned
  Concepts in CONTRAINDICATION_RELEVANT: 347331
    Clinical Finding: 130225
    Procedure: 60999
    Substance: 27991
    Pharmaceutical/Biological Product: 25719
    Body Structure: 42752
    Qualifier Value: 12492
    Observable Entity: 11004
    Organism: 34956
    Attribute: 1193

## Stage B: Build Neo4j graph
  Schema: constraints, indexes, fulltext index created
  Concepts loaded: 347331  (24.0s)
  IS_A edges: 589425  (23.7s)
  HAS_ROLE edges: 642442  (36.6s)
  MRCMRange nodes: 131  (0.1s)
  MRCMAttributeDomain nodes: 144  (0.0s)
  MRCMDomain nodes: 19  (0.0s)

## Summary
  Concepts:              347331
  IS_A edges:            589425
  HAS_ROLE edges:        642442
  MRCMRange nodes:       131
  MRCMAttributeDomain:   144
  MRCMDomain:            19

## Stage C: Generating snomed_conventions.md
  Written: agents/snomed_conventions.md (720 lines)

## Build complete: 2026-05-26T19:47:26.595356+00:00
