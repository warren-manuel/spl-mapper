# Evaluation Strategy: `evaluate_agg_results_3.py`

---

## Overview

The evaluation script compares system-predicted contraindication extractions against a human-annotated gold standard (`contra_gold_100_2.csv`). It operates at three levels of granularity, produces precision/recall/F1 at each level, and optionally applies SNOMED hierarchy-aware partial credit at the concept level.

---

## 1. Gold/Prediction Alignment (Row Matching)

Before any metric is computed, each predicted extraction must be aligned to a gold-standard annotation row **within the same SPL document** (`SPL_SET_ID`). This is a one-to-one assignment problem.

### Similarity Score

Each (gold row, predicted row) pair receives a combined score:

```
score = α · semantic_cosine(gold.annotation, pred.query_text)
      + β · token_jaccard(gold.annotation, pred.query_text)
```

Default weights: **α = 0.85**, **β = 0.15**.

- **Semantic cosine**: embeddings from a PubMed sentence-transformer (`all-MiniLM-L6-v2-pubmed-full`) compared via dot product of unit-normalised vectors.
- **Token Jaccard**: lowercased alphanumeric token sets, `|A∩B| / |A∪B|`. Used as fallback when the embedding model is unavailable.

### Assignment Algorithms

Two algorithms are supported (`--assignment`):

| Algorithm | Method | Notes |
|---|---|---|
| **Hungarian** (default) | `scipy.optimize.linear_sum_assignment` on cost = max_score − score | Globally optimal; guaranteed maximum total score |
| **Greedy** | Sort all pairs by score descending; assign greedily | Faster; can miss optimal global assignment |

A `--min-pair-score` threshold (default 0.0) discards assignments below the floor.

### Ignored Gold Rows

Gold rows with a blank/NA `Minimum Concept/s` field can be flagged as **ignored** (`--discard-na-gold-expression`). Ignored rows still participate in matching (so they can absorb a spurious prediction), but any matched pair involving an ignored gold row is excluded from all metrics.

---

## 2. Evaluation Levels

### Level 1 — Extraction

Measures whether the pipeline found the right set of contraindication items, independent of concept accuracy.

| Outcome | Condition |
|---|---|
| **TP** | A non-ignored gold row was matched to a prediction |
| **FN** | A non-ignored gold row had no match |
| **FP** | A prediction was not matched to any gold row |

### Level 2 — Contraindication

Measures whether the full concept set for a matched pair is exactly correct.

For each matched (gold, pred) pair:
- **TP** if `gold_concept_ids == pred_concept_ids` (exact set equality)
- **FP + FN** otherwise (one count each — the contraindication is wrong)

The concept union covers four slots: `problem_concept`, `causative_concept`, `severity_concept`, `course_concept`.

### Level 3 — Concept

Relaxed version of contraindication-level. Partial credit is given per individual concept rather than requiring a fully correct set.

**Without SNOMED hierarchy** (binary):
```
TP = |gold_ids ∩ pred_ids|
FP = |pred_ids − gold_ids|
FN = |gold_ids − pred_ids|
```

**With SNOMED hierarchy** (tiered, see §3 below): the TP/FP/FN are fractional values computed via a second Hungarian assignment over concept-level similarity scores.

---

## 3. SNOMED Hierarchy Partial Scoring

When a SNOMED RF2 relationship snapshot is available (`sct2_Relationship_Snapshot_*.txt`), concept comparisons move from binary (exact match only) to **tiered similarity**.

### Data Loading

Only active IS-A relationships (`typeId = 116680003`, `active = 1`) are loaded. The dataframe is indexed on `sourceId` for fast ancestor lookup.

### Ancestor Traversal

`get_ancestors_with_depth(concept_id, rel_df_indexed)` performs a BFS upward from a concept, returning a `{ancestor_id: hop_distance}` dict. Results are memoised in `_anc_depth_cache`.

Concept depth from the SNOMED root (`138875005`) is derived from the same traversal and cached in `_depth_cache`.

### Similarity Tiers (`concept_similarity_score`)

Given a predicted concept `p` and a gold concept `g`:

```
1. Exact match:              score = 1.0

2. p is ancestor of g        score = max(partial_floor,
   (prediction too general):           1.0 − hop_penalty × hops(g→p))

3. p is descendant of g      score = max(partial_floor,
   (prediction too specific):          1.0 − hop_penalty × hops(p→g))

4. Common ancestor exists    lca = deepest shared ancestor
   (sibling / cousin):       total_hops = depth(g)−depth(lca) + depth(p)−depth(lca)
                             score = max(0.0, sibling_base − sibling_decay × total_hops)

5. No relationship:          score = 0.0
```

**Default parameters:**

| Parameter | Default | Role |
|---|---|---|
| `--hierarchy-partial-score` | 0.0 | Floor for ancestor/descendant matches |
| `--hop-penalty` | 0.25 | Score lost per hop in ancestor/descendant tiers |
| `--sibling-base` | 0.4 | Starting score for sibling/cousin matches |
| `--sibling-decay` | 0.1 | Score lost per total hop from LCA |

> **Note:** The `--hierarchy-partial-score` default of 0.0 means an ancestor/descendant match only earns credit if the hop distance is small enough that `1.0 − hop_penalty × hops > 0`. Raising this floor (e.g. to 0.5) guarantees partial credit for any hierarchical relative.

### Concept-Level Hungarian Assignment

`compute_tiered_concept_metrics` builds an `|pred| × |gold|` score matrix and runs `linear_sum_assignment` to find the maximum-weight one-to-one pairing. The summed scores become fractional TP; residuals become fractional FP and FN:

```
tp = Σ score[assigned pairs]
fp = max(0, |pred| − tp)
fn = max(0, |gold| − tp)
```

This means a prediction that is a near-ancestor of the gold concept can contribute, say, 0.75 TP rather than a binary 0 or 1.

---

## 4. Decoupled vs. Coupled Metrics

The `--no-decoupled` flag controls whether unmatched rows propagate into contraindication/concept metrics.

| Mode | Unmatched gold row | Unmatched pred row |
|---|---|---|
| **Decoupled** (default) | FN only at extraction level | FP only at extraction level |
| **Coupled** | FN at all three levels | FP at all three levels |

Decoupled mode isolates concept-level performance to matched pairs, making it independent of extraction recall.

---

## 5. Outputs

| File | Content |
|---|---|
| `eval_metrics.json` | Aggregate precision/recall/F1 at all three levels, plus run configuration |
| `eval_details.csv` | Per-row breakdown: matched text, scores, concept IDs, per-row TP/FP/FN contribution |

---

## 6. Metric Summary Formula

For each level:

```
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 · Precision · Recall / (Precision + Recall)
```

At concept level with tiered scoring, TP/FP/FN are floats, so these become soft metrics.
