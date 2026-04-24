#!/usr/bin/env python3
"""Evaluate aggregated contraindication predictions against contra_gold_100_2.csv.

Scoring is performed per SPL_SET_ID using a global one-to-one assignment between
gold rows and prediction rows. Matching uses:

  score = alpha * semantic_cosine(annotation, query_text)
          + beta * token_jaccard(annotation, query_text)

Metric levels:
1) extraction_level
   - TP: matched non-ignored gold/pred pair
   - FN: unmatched non-ignored gold annotation
   - FP: unmatched prediction query_text not absorbed by an ignored gold row
2) contraindication_level
   - Exact-set metric on the union of problem/causative/severity/course concepts.
3) concept_level
   - Relaxed set-overlap metric on the same concept unions.

Ignored gold rows (blank/NA Minimum Concept/s) remain eligible for assignment so
they can absorb a predicted extraction, but any matched ignored-gold pair is
excluded from all metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

import numpy as np
import pandas as pd
from collections import deque
from scipy.optimize import linear_sum_assignment

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args: Any, **kwargs: Any) -> bool:
        return False

try:
    from VaxMapper.src.utils.embedding_utils import load_ST_model
except Exception:
    load_ST_model = None

try:
    from VaxMapper.src.utils.snomed_utils import _resolve_rf2_file, get_ancestors_with_depth
    _snomed_utils_available = True
except Exception:
    _snomed_utils_available = False


ID_PATTERN = re.compile(r"\d+")
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
GOLD_CONCEPT_COLUMNS = (
    "problem_concept",
    "causative_concept",
    "severity_concept",
    "course_concept",
)


def parse_args() -> argparse.Namespace:
    load_dotenv(override=False)
    default_st_model = os.environ.get("MAPPER_MODEL_NAME", "tavakolih/all-MiniLM-L6-v2-pubmed-full")

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred-csv", default="results/20260303/agg_results.csv")
    ap.add_argument("--gold-csv", default="results/contra_gold_100_2.csv")
    ap.add_argument("--out-json", default="results/20260303/eval_metrics.json")
    ap.add_argument("--out-details-csv", default="results/20260303/eval_details.csv")
    ap.add_argument(
        "--discard-na-gold-expression",
        action="store_true",
        help="Treat rows with blank/NA 'Minimum Concept/s' as ignored for metrics, but keep them in matching.",
    )
    ap.add_argument("--st-model-id", default=default_st_model)
    ap.add_argument("--st-device", default="cuda")
    ap.add_argument("--st-batch-size", type=int, default=256)
    ap.add_argument("--alpha", type=float, default=0.85, help="Weight for semantic cosine similarity")
    ap.add_argument("--beta", type=float, default=0.15, help="Weight for text-token Jaccard overlap")
    ap.add_argument(
        "--min-pair-score",
        type=float,
        default=0.0,
        help="Minimum combined score required to keep an assigned gold/pred pair.",
    )
    ap.add_argument(
        "--require-semantic",
        action="store_true",
        help="Fail if semantic matching dependencies are unavailable.",
    )
    ap.add_argument(
        "--no-decoupled",
        action="store_true",
        help="When set, unmatched non-ignored gold/pred rows also contribute to contraindication-level and concept-level metrics.",
    )
    ap.add_argument(
        "--assignment",
        choices=("hungarian", "greedy"),
        default="hungarian",
        help="One-to-one assignment algorithm for gold/pred row matching.",
    )
    ap.add_argument(
        "--snomed-rel",
        default="",
        help="Path to sct2_Relationship_Snapshot_*.txt for tiered concept scoring.",
    )
    ap.add_argument(
        "--snomed-source-dir",
        default="snomed_us_source",
        help="Auto-resolve RF2 relationship snapshot from this directory (used when --snomed-rel is not set).",
    )
    ap.add_argument(
        "--hierarchy-partial-score",
        type=float,
        default=0.0,
        help="Floor score for direct ancestor/descendant concept matches (default 0.5).",
    )
    ap.add_argument(
        "--sibling-base",
        type=float,
        default=0.4,
        help="Base score for sibling/cousin concept matches via LCA (default 0.4).",
    )
    ap.add_argument(
        "--sibling-decay",
        type=float,
        default=0.1,
        help="Score decay per hop from LCA for sibling/cousin matches (default 0.05).",
    )
    ap.add_argument(
        "--hop-penalty",
        type=float,
        default=0.25,
        help="Score penalty per hop for ancestor/descendant matches (default 0.25).",
    )
    return ap.parse_args()


def read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def norm_na(text: str) -> str:
    value = (text or "").strip()
    if not value or value.upper() in {"NA", "N/A"}:
        return ""
    return value


def safe_int(text: Any) -> int:
    try:
        return int(text)
    except Exception:
        return 0


def extract_ids(text: str) -> Set[str]:
    return set(ID_PATTERN.findall(text or ""))


def normalize_text_for_tokens(text: str) -> str:
    return " ".join(TOKEN_PATTERN.findall((text or "").lower()))


def token_set(text: str) -> Set[str]:
    return set(TOKEN_PATTERN.findall((text or "").lower()))


def parse_gold_union_ids(row: Dict[str, str]) -> Set[str]:
    ids: Set[str] = set()
    for col in GOLD_CONCEPT_COLUMNS:
        value = row.get(col, "")
        if not value:
            continue
        for part in str(value).splitlines():
            ids.update(extract_ids(part))
    return ids


def pred_ids_from_row(row: Dict[str, str]) -> Set[str]:
    out: Set[str] = set()
    for key in ("final_concept_id", "causative_agent_id", "severity_id", "clinical_course_id"):
        value = norm_na(row.get(key, ""))
        if value:
            out.add(value)
    return out


def gold_row_ignored(row: Dict[str, str], discard_na_gold_expression: bool) -> bool:
    if not discard_na_gold_expression:
        return False
    return not bool(norm_na(row.get("Minimum Concept/s", "")))


def semantic_unavailable_reasons() -> List[str]:
    reasons: List[str] = []
    if np is None:
        reasons.append("numpy import failed")
    if load_ST_model is None:
        reasons.append("embedding model loader import failed")
    return reasons


def compute_metrics(tp: float, fp: float, fn: float) -> Dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


# ---------------------------------------------------------------------------
# Tiered SNOMED hierarchy concept scoring
# ---------------------------------------------------------------------------

SNOMED_ROOT = 138875005  # "SNOMED CT Concept" — root of every active concept

_anc_depth_cache: Dict[int, Dict[int, int]] = {}
_depth_cache: Dict[int, int] = {}


def _cached_ancestors_with_depth(
    concept_id: int,
    rel_df_indexed: "pd.DataFrame",
) -> Dict[int, int]:
    """Memoised wrapper around get_ancestors_with_depth."""
    if concept_id not in _anc_depth_cache:
        _anc_depth_cache[concept_id] = get_ancestors_with_depth(concept_id, rel_df_indexed)
    return _anc_depth_cache[concept_id]


def get_concept_depth(concept_id: int, rel_df_indexed: "pd.DataFrame") -> int:
    """
    Depth of concept in the SNOMED hierarchy measured as hop distance to the root
    (SNOMED_ROOT = 138875005).  Root itself has depth 0; its direct children depth 1, etc.
    """
    if concept_id in _depth_cache:
        return _depth_cache[concept_id]
    anc = _cached_ancestors_with_depth(concept_id, rel_df_indexed)
    depth = anc.get(SNOMED_ROOT, 0)
    _depth_cache[concept_id] = depth
    return depth


def concept_similarity_score(
    predicted_id: str,
    gold_id: str,
    rel_df_indexed: "pd.DataFrame",
    partial_floor: float = 0.5,
    sibling_base: float = 0.4,
    sibling_decay: float = 0.05,
    hop_penalty: float = 0.15,
) -> float:
    """
    Hierarchical similarity score in [0.0, 1.0] between two SNOMED concept IDs.

    Scoring tiers:
      - Exact match                          → 1.0
      - pred is ancestor of gold (too gen.)  → max(partial_floor, 1.0 - hop_penalty * depth_diff)
      - pred is descendant of gold (too sp.) → max(partial_floor, 1.0 - hop_penalty * depth_diff)
      - Common ancestor (sibling/cousin)     → max(0.1, sibling_base - sibling_decay
                                                        * (gold_hops_to_lca + pred_hops_to_lca))
      - No relationship                      → 0.0
    """
    if predicted_id == gold_id:
        return 1.0
    try:
        p, g = int(predicted_id), int(gold_id)
    except ValueError:
        return 0.0

    gold_anc = _cached_ancestors_with_depth(g, rel_df_indexed)
    pred_anc = _cached_ancestors_with_depth(p, rel_df_indexed)

    if p in gold_anc:
        # Predicted is an ancestor of gold — too general
        depth_diff = gold_anc[p]
        return max(partial_floor, 1.0 - (hop_penalty * depth_diff))

    if g in pred_anc:
        # Predicted is a descendant of gold — too specific
        depth_diff = pred_anc[g]
        return max(partial_floor, 1.0 - (hop_penalty * depth_diff))

    # Check for common ancestors (siblings / cousins)
    common = set(gold_anc) & set(pred_anc)
    if common:
        lca = max(common, key=lambda a: get_concept_depth(a, rel_df_indexed))
        lca_depth = get_concept_depth(lca, rel_df_indexed)
        gold_depth = get_concept_depth(g, rel_df_indexed)
        pred_depth = get_concept_depth(p, rel_df_indexed)
        total_hops = (gold_depth - lca_depth) + (pred_depth - lca_depth)
        return max(0.0, sibling_base - (sibling_decay * total_hops))

    return 0.0


def compute_tiered_concept_metrics(
    gold_ids: Set[str],
    pred_ids: Set[str],
    rel_df_indexed: "pd.DataFrame",
    partial_floor: float = 0.5,
    sibling_base: float = 0.4,
    sibling_decay: float = 0.05,
    hop_penalty: float = 0.15,
) -> Tuple[float, float, float]:
    """
    Compute (tp, fp, fn) as floats using hierarchy-aware scoring and optimal
    Hungarian assignment between gold and pred concept sets.
    """
    gold_list = sorted(gold_ids)
    pred_list = sorted(pred_ids)
    if not gold_list or not pred_list:
        return 0.0, float(len(pred_list)), float(len(gold_list))

    score_mat = np.array(
        [
            [
                concept_similarity_score(
                    p, g, rel_df_indexed, partial_floor,
                    sibling_base, sibling_decay, hop_penalty,
                )
                for g in gold_list
            ]
            for p in pred_list
        ],
        dtype=np.float64,
    )

    row_ind, col_ind = linear_sum_assignment(-score_mat)
    tp = float(score_mat[row_ind, col_ind].sum())
    fp = max(0.0, float(len(pred_list)) - tp)
    fn = max(0.0, float(len(gold_list)) - tp)
    return tp, fp, fn


def write_csv(path: str, rows: Iterable[Dict[str, Any]], fieldnames: List[str]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def greedy_global_assignment(scores: "np.ndarray", min_score: float = 0.0) -> Dict[int, int]:
    """Greedy maximum-weight one-to-one matching. Returns gold_idx -> pred_idx."""
    gold_count, pred_count = scores.shape
    pairs: List[Tuple[float, int, int]] = []
    for gold_idx in range(gold_count):
        for pred_idx in range(pred_count):
            pairs.append((float(scores[gold_idx, pred_idx]), gold_idx, pred_idx))
    pairs.sort(key=lambda item: item[0], reverse=True)

    used_gold: Set[int] = set()
    used_pred: Set[int] = set()
    mapping: Dict[int, int] = {}

    for score, gold_idx, pred_idx in pairs:
        if score < min_score:
            break
        if gold_idx in used_gold or pred_idx in used_pred:
            continue
        mapping[gold_idx] = pred_idx
        used_gold.add(gold_idx)
        used_pred.add(pred_idx)

    return mapping


def hungarian_assignment(scores: "np.ndarray", min_score: float = 0.0) -> Dict[int, int]:
    """Maximum-weight one-to-one assignment using Hungarian optimization."""
    gold_count, pred_count = scores.shape
    if gold_count == 0 or pred_count == 0:
        return {}

    max_score = float(np.max(scores)) if scores.size else 0.0
    cost = max_score - scores
    gold_idx_arr, pred_idx_arr = linear_sum_assignment(cost)

    mapping: Dict[int, int] = {}
    for gold_idx, pred_idx in zip(gold_idx_arr.tolist(), pred_idx_arr.tolist()):
        if float(scores[gold_idx, pred_idx]) < min_score:
            continue
        mapping[gold_idx] = pred_idx
    return mapping


def assign_pairs(scores: "np.ndarray", method: str, min_score: float) -> Dict[int, int]:
    if method == "greedy":
        return greedy_global_assignment(scores, min_score=min_score)
    return hungarian_assignment(scores, min_score=min_score)


def build_pair_scores(
    gold_rows: List[Dict[str, str]],
    pred_rows: List[Dict[str, str]],
    st_model: Any,
    alpha: float,
    beta: float,
    st_batch_size: int,
) -> Tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
    """Return (combined_scores, semantic_matrix, token_jaccard_matrix). Shapes: (G,P)."""
    gold_count = len(gold_rows)
    pred_count = len(pred_rows)
    scores = np.zeros((gold_count, pred_count), dtype=np.float32)
    sem = np.zeros((gold_count, pred_count), dtype=np.float32)
    jac = np.zeros((gold_count, pred_count), dtype=np.float32)

    gold_texts = [(row.get("annotation") or "").strip() or "[EMPTY_ANNOTATION]" for row in gold_rows]
    pred_texts = [(row.get("query_text") or "").strip() or "[EMPTY_QUERY_TEXT]" for row in pred_rows]

    if st_model is not None:
        gold_vectors = st_model.encode(
            gold_texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=st_batch_size,
        )
        pred_vectors = st_model.encode(
            pred_texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=st_batch_size,
        )
        sem = (pred_vectors @ gold_vectors.T).T.astype(np.float32)

    gold_token_sets = [token_set(text) for text in gold_texts]
    pred_token_sets = [token_set(text) for text in pred_texts]

    for gold_idx in range(gold_count):
        for pred_idx in range(pred_count):
            jac[gold_idx, pred_idx] = float(jaccard(gold_token_sets[gold_idx], pred_token_sets[pred_idx]))
            scores[gold_idx, pred_idx] = alpha * float(sem[gold_idx, pred_idx]) + beta * float(jac[gold_idx, pred_idx])

    return scores, sem, jac


def append_detail(
    details: List[Dict[str, Any]],
    *,
    spl: str,
    contra_id: str = "",
    annotation: str = "",
    query_text: str = "",
    gold_ids: Set[str] | None = None,
    pred_ids: Set[str] | None = None,
    selection_method: str,
    assignment_method: str,
    semantic_score: Any = "",
    jaccard_score: Any = "",
    combined_score: Any = "",
    extraction_tp: int = 0,
    extraction_fp: int = 0,
    extraction_fn: int = 0,
    contraindication_tp: int = 0,
    contraindication_fp: int = 0,
    contraindication_fn: int = 0,
    concept_tp: int = 0,
    concept_fp: int = 0,
    concept_fn: int = 0,
    decoupled: bool,
    downstream_counted: int,
    item_index: str = "",
) -> None:
    details.append(
        {
            "SPL_SET_ID": spl,
            "contra_id": contra_id,
            "annotation": annotation,
            "query_text": query_text,
            "gold_concept_ids": "|".join(sorted(gold_ids or set())),
            "pred_concept_ids": "|".join(sorted(pred_ids or set())),
            "selection_method": selection_method,
            "assignment_method": assignment_method,
            "semantic_score": semantic_score,
            "jaccard": jaccard_score,
            "combined_score": combined_score,
            "selected_pred_item_index": item_index,
            "decoupled": int(decoupled),
            "downstream_counted": downstream_counted,
            "extraction_tp": extraction_tp,
            "extraction_fp": extraction_fp,
            "extraction_fn": extraction_fn,
            "contraindication_tp": contraindication_tp,
            "contraindication_fp": contraindication_fp,
            "contraindication_fn": contraindication_fn,
            "concept_tp": concept_tp,
            "concept_fp": concept_fp,
            "concept_fn": concept_fn,
        }
    )


def main() -> None:
    args = parse_args()
    decoupled = not args.no_decoupled

    pred_rows = read_csv(args.pred_csv)
    gold_rows = read_csv(args.gold_csv)
    gold_rows_total = len(gold_rows)
    dropped_gold_rows = sum(
        1 for row in gold_rows if gold_row_ignored(row, args.discard_na_gold_expression)
    )

    semantic_reasons = semantic_unavailable_reasons()
    semantic_enabled = not semantic_reasons
    if not semantic_enabled:
        msg = "Semantic matching disabled: " + "; ".join(semantic_reasons)
        if args.require_semantic:
            raise RuntimeError(msg)
        print(msg)
    st_model = load_ST_model(args.st_model_id, device=args.st_device) if semantic_enabled else None

    # ------------------------------------------------------------------
    # SNOMED hierarchy loading for tiered concept scoring
    # ------------------------------------------------------------------
    rel_df_indexed = None
    if _snomed_utils_available:
        snomed_rel_path: str | None = args.snomed_rel or None
        if not snomed_rel_path:
            try:
                snomed_rel_path = str(_resolve_rf2_file(args.snomed_source_dir, "sct2_Relationship_Snapshot_"))
            except FileNotFoundError:
                snomed_rel_path = None
        if snomed_rel_path:
            _rel = pd.read_csv(
                snomed_rel_path,
                sep="\t",
                usecols=["sourceId", "destinationId", "typeId", "active"],
                dtype={"sourceId": int, "destinationId": int, "typeId": int, "active": int},
            )
            _rel = _rel[(_rel["active"] == 1) & (_rel["typeId"] == 116680003)].copy()
            rel_df_indexed = _rel.set_index("sourceId", drop=True)
            print(f"Tiered concept scoring enabled ({len(rel_df_indexed):,} IS-A rows loaded).")
        else:
            print("Tiered concept scoring disabled (no SNOMED relationship file found).")
    else:
        print("Tiered concept scoring disabled (snomed_utils import failed).")

    pred_by_spl: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for pred_uid, row in enumerate(pred_rows):
        spl = (row.get("SPL_SET_ID") or "").strip()
        if not spl:
            continue
        row["__pred_uid"] = pred_uid
        pred_by_spl[spl].append(row)

    for spl in pred_by_spl:
        pred_by_spl[spl].sort(key=lambda row: safe_int(row.get("item_index")))

    gold_by_spl: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    gold_rows_used = 0
    for row in gold_rows:
        spl = (row.get("SPL_SET_ID") or "").strip()
        if not spl:
            continue
        gold_by_spl[spl].append(row)
        if not gold_row_ignored(row, args.discard_na_gold_expression):
            gold_rows_used += 1

    extraction_tp = extraction_fp = extraction_fn = 0
    contraindication_tp = contraindication_fp = contraindication_fn = 0
    concept_tp = concept_fp = concept_fn = 0
    details: List[Dict[str, Any]] = []

    for spl, gold_rows_for_matching in gold_by_spl.items():
        pred_rows_for_matching = pred_by_spl.get(spl, [])

        mapping: Dict[int, int] = {}
        score_mat = sem_mat = jac_mat = None

        if gold_rows_for_matching and pred_rows_for_matching:
            if semantic_enabled:
                score_mat, sem_mat, jac_mat = build_pair_scores(
                    gold_rows_for_matching,
                    pred_rows_for_matching,
                    st_model,
                    args.alpha,
                    args.beta,
                    args.st_batch_size,
                )
            else:
                score_mat = np.zeros((len(gold_rows_for_matching), len(pred_rows_for_matching)), dtype=np.float32)
                sem_mat = np.zeros_like(score_mat)
                jac_mat = np.zeros_like(score_mat)
                for gold_idx, gold_row in enumerate(gold_rows_for_matching):
                    gold_text = normalize_text_for_tokens(gold_row.get("annotation", ""))
                    gold_tokens = token_set(gold_text)
                    for pred_idx, pred_row in enumerate(pred_rows_for_matching):
                        pred_text = normalize_text_for_tokens(pred_row.get("query_text", ""))
                        pred_tokens = token_set(pred_text)
                        jac = float(jaccard(gold_tokens, pred_tokens))
                        jac_mat[gold_idx, pred_idx] = jac
                        score_mat[gold_idx, pred_idx] = jac

            mapping = assign_pairs(score_mat, method=args.assignment, min_score=args.min_pair_score)

        assigned_pred_local = set(mapping.values())

        for gold_idx, gold_row in enumerate(gold_rows_for_matching):
            gold_ids = parse_gold_union_ids(gold_row)
            annotation = gold_row.get("annotation", "")
            contra_id = gold_row.get("contra_id", "")
            ignored_gold = gold_row_ignored(gold_row, args.discard_na_gold_expression)

            if gold_idx in mapping:
                pred_idx = mapping[gold_idx]
                pred_row = pred_rows_for_matching[pred_idx]
                pred_ids = pred_ids_from_row(pred_row)
                sem_score = float(sem_mat[gold_idx, pred_idx]) if sem_mat is not None else ""
                jac_score = float(jac_mat[gold_idx, pred_idx]) if jac_mat is not None else ""
                combined_score = float(score_mat[gold_idx, pred_idx]) if score_mat is not None else ""

                if ignored_gold:
                    append_detail(
                        details,
                        spl=spl,
                        contra_id=contra_id,
                        annotation=annotation,
                        query_text=pred_row.get("query_text", ""),
                        gold_ids=gold_ids,
                        pred_ids=pred_ids,
                        selection_method="matched_ignored_gold",
                        assignment_method=args.assignment,
                        semantic_score=sem_score,
                        jaccard_score=jac_score,
                        combined_score=combined_score,
                        decoupled=decoupled,
                        downstream_counted=0,
                        item_index=str(pred_row.get("item_index", "")),
                    )
                    continue

                extraction_tp += 1

                matched_contra_tp = matched_contra_fp = matched_contra_fn = 0
                if rel_df_indexed is not None:
                    matched_concept_tp, matched_concept_fp, matched_concept_fn = (
                        compute_tiered_concept_metrics(
                            gold_ids, pred_ids, rel_df_indexed, args.hierarchy_partial_score,
                            args.sibling_base, args.sibling_decay, args.hop_penalty,
                        )
                    )
                else:
                    matched_concept_tp = len(gold_ids & pred_ids)
                    matched_concept_fp = len(pred_ids - gold_ids)
                    matched_concept_fn = len(gold_ids - pred_ids)

                if gold_ids == pred_ids:
                    matched_contra_tp = 1
                else:
                    matched_contra_fp = 1
                    matched_contra_fn = 1

                contraindication_tp += matched_contra_tp
                contraindication_fp += matched_contra_fp
                contraindication_fn += matched_contra_fn
                concept_tp += matched_concept_tp
                concept_fp += matched_concept_fp
                concept_fn += matched_concept_fn

                append_detail(
                    details,
                    spl=spl,
                    contra_id=contra_id,
                    annotation=annotation,
                    query_text=pred_row.get("query_text", ""),
                    gold_ids=gold_ids,
                    pred_ids=pred_ids,
                    selection_method=f"matched_{args.assignment}",
                    assignment_method=args.assignment,
                    semantic_score=sem_score,
                    jaccard_score=jac_score,
                    combined_score=combined_score,
                    extraction_tp=1,
                    contraindication_tp=matched_contra_tp,
                    contraindication_fp=matched_contra_fp,
                    contraindication_fn=matched_contra_fn,
                    concept_tp=matched_concept_tp,
                    concept_fp=matched_concept_fp,
                    concept_fn=matched_concept_fn,
                    decoupled=decoupled,
                    downstream_counted=1,
                    item_index=str(pred_row.get("item_index", "")),
                )
            else:
                if ignored_gold:
                    append_detail(
                        details,
                        spl=spl,
                        contra_id=contra_id,
                        annotation=annotation,
                        gold_ids=gold_ids,
                        selection_method="ignored_gold_unmatched",
                        assignment_method=args.assignment,
                        decoupled=decoupled,
                        downstream_counted=0,
                    )
                    continue

                extraction_fn += 1
                unmatched_contra_fn = 0 if decoupled else 1
                unmatched_concept_fn = 0 if decoupled else len(gold_ids)

                if not decoupled:
                    contraindication_fn += 1
                    concept_fn += len(gold_ids)

                append_detail(
                    details,
                    spl=spl,
                    contra_id=contra_id,
                    annotation=annotation,
                    gold_ids=gold_ids,
                    selection_method="unmatched_gold",
                    assignment_method=args.assignment,
                    extraction_fn=1,
                    contraindication_fn=unmatched_contra_fn,
                    concept_fn=unmatched_concept_fn,
                    decoupled=decoupled,
                    downstream_counted=int(not decoupled),
                )

        for pred_idx, pred_row in enumerate(pred_rows_for_matching):
            if pred_idx in assigned_pred_local:
                continue
            pred_ids = pred_ids_from_row(pred_row)
            extraction_fp += 1
            unmatched_contra_fp = 0 if decoupled else 1
            unmatched_concept_fp = 0 if decoupled else len(pred_ids)

            if not decoupled:
                contraindication_fp += 1
                concept_fp += len(pred_ids)

            append_detail(
                details,
                spl=spl,
                query_text=pred_row.get("query_text", ""),
                pred_ids=pred_ids,
                selection_method="unmatched_pred",
                assignment_method=args.assignment,
                extraction_fp=1,
                contraindication_fp=unmatched_contra_fp,
                concept_fp=unmatched_concept_fp,
                decoupled=decoupled,
                downstream_counted=int(not decoupled),
                item_index=str(pred_row.get("item_index", "")),
            )

    pred_only_spls = set(pred_by_spl) - set(gold_by_spl)
    for spl in sorted(pred_only_spls):
        for pred_row in pred_by_spl[spl]:
            pred_ids = pred_ids_from_row(pred_row)
            extraction_fp += 1
            unmatched_contra_fp = 0 if decoupled else 1
            unmatched_concept_fp = 0 if decoupled else len(pred_ids)
            if not decoupled:
                contraindication_fp += 1
                concept_fp += len(pred_ids)
            append_detail(
                details,
                spl=spl,
                query_text=pred_row.get("query_text", ""),
                pred_ids=pred_ids,
                selection_method="unmatched_pred_no_gold",
                assignment_method=args.assignment,
                extraction_fp=1,
                contraindication_fp=unmatched_contra_fp,
                concept_fp=unmatched_concept_fp,
                decoupled=decoupled,
                downstream_counted=int(not decoupled),
                item_index=str(pred_row.get("item_index", "")),
            )

    out_payload = {
        "inputs": {
            "pred_csv": args.pred_csv,
            "gold_csv": args.gold_csv,
            "gold_rows_total": gold_rows_total,
            "gold_rows_used": gold_rows_used,
            "gold_rows_dropped_na_expression": dropped_gold_rows,
            "discard_na_gold_expression": args.discard_na_gold_expression,
            "pred_rows": len(pred_rows),
            "gold_unique_spl": len(gold_by_spl),
            "semantic_matching_enabled": semantic_enabled,
            "semantic_matching_unavailable_reasons": semantic_reasons,
            "st_model_id": args.st_model_id,
            "alpha": args.alpha,
            "beta": args.beta,
            "min_pair_score": args.min_pair_score,
            "decoupled": decoupled,
            "assignment_method": args.assignment,
            "tiered_concept_scoring": rel_df_indexed is not None,
            "hierarchy_partial_score": args.hierarchy_partial_score if rel_df_indexed is not None else None,
        },
        "extraction_level": compute_metrics(extraction_tp, extraction_fp, extraction_fn),
        "contraindication_level": compute_metrics(
            contraindication_tp, contraindication_fp, contraindication_fn
        ),
        "concept_level": compute_metrics(concept_tp, concept_fp, concept_fn),
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")

    detail_fields = [
        "SPL_SET_ID",
        "contra_id",
        "annotation",
        "query_text",
        "gold_concept_ids",
        "pred_concept_ids",
        "selection_method",
        "assignment_method",
        "semantic_score",
        "jaccard",
        "combined_score",
        "selected_pred_item_index",
        "decoupled",
        "downstream_counted",
        "extraction_tp",
        "extraction_fp",
        "extraction_fn",
        "contraindication_tp",
        "contraindication_fp",
        "contraindication_fn",
        "concept_tp",
        "concept_fp",
        "concept_fn",
    ]
    write_csv(args.out_details_csv, details, detail_fields)

    print(json.dumps(out_payload, indent=2))
    print(f"Wrote details: {args.out_details_csv}")


if __name__ == "__main__":
    main()
