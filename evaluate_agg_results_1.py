#!/usr/bin/env python3
"""Evaluate agg_results.csv against contra_gold_100.csv.

Key change vs previous version:
- Attribute-level evaluation uses per-SPL *global* one-to-one assignment between
  gold rows and prediction rows to avoid greedy, order-dependent mismatches.

Evaluation rules:
1) Match within same SPL_SET_ID; item indexes are ignored.
2) Concept-level: exact normalized expression match between gold and prediction.
   - Normalization removes whitespace and braces from both expressions.
3) Attribute-level targets per gold row:
   - If concept-level matched: target IDs = all concept IDs from gold SNOMED_ID / Expression.
     Pred IDs = all concept IDs found in prediction postcoord_expression.
   - Else: target IDs = concept IDs from gold Minimum Concept/s.
     Pred IDs = {final_concept_id, causative_agent_id, severity_id, clinical_course_id}.

Attribute-level matching per SPL_SET_ID:
  A) First, reserve exact concept-expression matches (one-to-one).
  B) Then, globally assign remaining gold rows to remaining prediction rows using a
     combined score:
       score = alpha * semantic_cosine(annotation, pred.query_text)
               + beta * jaccard(target_ids, pred_ids)
     via greedy maximum-weight matching.
  C) Unmatched gold rows contribute FN; unmatched prediction rows contribute FP.

This prevents “partially correct” predictions from being matched to the wrong gold rows
when atomic splitting is imperfect.
"""

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
from VaxMapper.src.utils.embedding_utils import load_ST_model


ID_PATTERN = re.compile(r"\d+")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred-csv", default="results/20260303/agg_results.csv")
    ap.add_argument("--gold-csv", default="results/contra_gold_100_1.csv")
    ap.add_argument("--out-json", default="results/20260303/eval_metrics.json")
    ap.add_argument("--out-details-csv", default="results/20260303/eval_details.csv")
    ap.add_argument(
        "--discard-na-gold-expression",
        action="store_true",
        help="Discard rows where BOTH 'Minimum Concept/s' and 'SNOMED_ID / Expression' are blank/NA before evaluation.",
    )
    ap.add_argument("--st-model-id", default="tavakolih/all-MiniLM-L6-v2-pubmed-full")
    ap.add_argument("--st-device", default="cuda")
    ap.add_argument("--st-batch-size", type=int, default=256)
    ap.add_argument("--alpha", type=float, default=0.85, help="Weight for semantic cosine similarity")
    ap.add_argument("--beta", type=float, default=0.15, help="Weight for ID-overlap (Jaccard)")
    ap.add_argument(
        "--min-pair-score",
        type=float,
        default=0.0,
        help="Minimum combined score required to assign a gold row to a pred row in global matching.",
    )
    ap.add_argument(
        "--require-semantic",
        action="store_true",
        help="Fail if semantic matching dependencies are unavailable.",
    )
    return ap.parse_args()


def read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def norm_expression(text: str) -> str:
    text = (text or "").strip()
    if not text or text.upper() in {"NA", "N/A"}:
        return ""
    text = text.replace("{", "").replace("}", "")
    text = re.sub(r"\s+", "", text)
    return text


def extract_ids(text: str) -> Set[str]:
    return set(ID_PATTERN.findall(text or ""))


def safe_int(text: Any) -> int:
    try:
        return int(text)
    except Exception:
        return 0


def pred_ids_from_row(row: Dict[str, str]) -> Set[str]:
    out: Set[str] = set()
    for k in ("final_concept_id", "causative_agent_id", "severity_id", "clinical_course_id"):
        val = (row.get(k) or "").strip()
        if val and val.upper() not in {"NA", "N/A"}:
            out.add(val)
    return out


def concept_eval_eligible(row: Dict[str, str]) -> bool:
    status = (row.get("exp_status") or "").strip().upper()
    if not status:
        return False
    return status in {"S", "PO"}


def gold_row_ignored(row: Dict[str, str], discard_na_gold_expression: bool) -> bool:
    if not discard_na_gold_expression:
        return False
    has_min = bool(norm_expression(row.get("Minimum Concept/s", "")))
    has_expr = bool(norm_expression(row.get("SNOMED_ID / Expression", "")))
    return not (has_min or has_expr)


def semantic_unavailable_reasons() -> List[str]:
    reasons: List[str] = []
    if np is None:
        reasons.append("numpy import failed")
    if pd is None:
        reasons.append("pandas import failed")
    if load_ST_model is None:
        reasons.append("embedding model loader import failed")
    return reasons


def compute_metrics(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def write_csv(path: str, rows: Iterable[Dict[str, Any]], fieldnames: List[str]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def greedy_global_assignment(scores: "np.ndarray", min_score: float = 0.0) -> Dict[int, int]:
    """Greedy maximum-weight one-to-one matching. Returns gold_idx -> pred_idx."""
    G, P = scores.shape
    pairs: List[Tuple[float, int, int]] = []
    for i in range(G):
        for j in range(P):
            pairs.append((float(scores[i, j]), i, j))
    pairs.sort(key=lambda x: x[0], reverse=True)

    g_used: Set[int] = set()
    p_used: Set[int] = set()
    mapping: Dict[int, int] = {}

    for score, gi, pj in pairs:
        if score < min_score:
            break
        if gi in g_used or pj in p_used:
            continue
        mapping[gi] = pj
        g_used.add(gi)
        p_used.add(pj)
    return mapping


def build_pair_scores(
    gold_rows: List[Dict[str, str]],
    pred_rows: List[Dict[str, str]],
    st_model: Any,
    alpha: float,
    beta: float,
) -> Tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
    """Return (combined_scores, semantic_matrix, jaccard_matrix). Shapes: (G,P)."""
    G = len(gold_rows)
    P = len(pred_rows)
    scores = np.zeros((G, P), dtype=np.float32)
    sem = np.zeros((G, P), dtype=np.float32)
    jac = np.zeros((G, P), dtype=np.float32)

    gold_texts = [(g.get("annotation") or "").strip() or "[EMPTY_ANNOTATION]" for g in gold_rows]
    pred_texts = [(p.get("query_text") or "").strip() or "[EMPTY_QUERY_TEXT]" for p in pred_rows]

    g_vec = st_model.encode(gold_texts, normalize_embeddings=True, show_progress_bar=False)
    p_vec = st_model.encode(pred_texts, normalize_embeddings=True, show_progress_bar=False)
    sem = (p_vec @ g_vec.T).T.astype(np.float32)

    gold_ids = [extract_ids(g.get("Minimum Concept/s", "")) for g in gold_rows]
    pred_ids = [pred_ids_from_row(p) for p in pred_rows]

    for i in range(G):
        for j in range(P):
            jac[i, j] = float(jaccard(gold_ids[i], pred_ids[j]))
            scores[i, j] = alpha * float(sem[i, j]) + beta * float(jac[i, j])

    return scores, sem, jac


def main() -> None:
    args = parse_args()

    pred_rows = read_csv(args.pred_csv)
    gold_rows = read_csv(args.gold_csv)
    gold_rows_total = len(gold_rows)
    dropped_gold_rows = 0
    for row in gold_rows:
        if gold_row_ignored(row, args.discard_na_gold_expression):
            dropped_gold_rows += 1

    semantic_reasons = semantic_unavailable_reasons()
    semantic_enabled = not semantic_reasons
    if not semantic_enabled:
        msg = "Semantic matching disabled: " + "; ".join(semantic_reasons)
        if args.require_semantic:
            raise RuntimeError(msg)
        print(msg)
    st_model = load_ST_model(args.st_model_id, device=args.st_device) if semantic_enabled else None

    pred_by_spl: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    pred_expr_map: Dict[str, Dict[str, List[Dict[str, str]]]] = defaultdict(lambda: defaultdict(list))

    for pred_uid, r in enumerate(pred_rows):
        spl = (r.get("SPL_SET_ID") or "").strip()
        if not spl:
            continue
        r["__pred_uid"] = pred_uid
        pred_by_spl[spl].append(r)
        e = norm_expression(r.get("postcoord_expression", ""))
        if e:
            pred_expr_map[spl][e].append(r)

    for spl in pred_by_spl:
        pred_by_spl[spl].sort(key=lambda x: safe_int(x.get("item_index")))
    for spl in pred_expr_map:
        for e in pred_expr_map[spl]:
            pred_expr_map[spl][e].sort(key=lambda x: safe_int(x.get("item_index")))

    gold_by_spl: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for g in gold_rows:
        spl = (g.get("SPL_SET_ID") or "").strip()
        if spl:
            gold_by_spl[spl].append(g)

    # Metrics
    concept_tp = concept_fp = concept_fn = 0
    attr_tp = attr_fp = attr_fn = 0
    details: List[Dict[str, Any]] = []

    concept_gold_rows = [
        g for g in gold_rows
        if concept_eval_eligible(g) and not gold_row_ignored(g, args.discard_na_gold_expression)
    ]
    concept_exact_used_pred_uids_by_spl: Dict[str, Set[int]] = defaultdict(set)
    ignored_matched_pred_uids_by_spl: Dict[str, Set[int]] = defaultdict(set)

    # --- concept-level ---
    gold_scope_spls = {
        (g.get("SPL_SET_ID") or "").strip()
        for g in concept_gold_rows
        if (g.get("SPL_SET_ID") or "").strip()
    }
    for spl, g_rows in gold_by_spl.items():
        used_pred_uids: Set[int] = set()
        for g in g_rows:
            gold_expr_norm = norm_expression(g.get("SNOMED_ID / Expression", ""))
            if not gold_expr_norm:
                continue

            pick = None
            for r in pred_expr_map.get(spl, {}).get(gold_expr_norm, []):
                puid = int(r.get("__pred_uid", -1))
                if puid not in used_pred_uids:
                    pick = r
                    break

            if pick is not None:
                puid = int(pick.get("__pred_uid", -1))
                used_pred_uids.add(puid)
                concept_exact_used_pred_uids_by_spl[spl].add(puid)

            if concept_eval_eligible(g) and not gold_row_ignored(g, args.discard_na_gold_expression):
                if pick is not None:
                    concept_tp += 1
                else:
                    concept_fn += 1

    # --- attribute-level: per-SPL global assignment ---
    for spl, g_rows in gold_by_spl.items():
        p_rows = pred_by_spl.get(spl, [])

        if not p_rows:
            for g in g_rows:
                if gold_row_ignored(g, args.discard_na_gold_expression):
                    details.append({
                        "SPL_SET_ID": spl,
                        "contra_id": g.get("contra_id", ""),
                        "annotation": g.get("annotation", ""),
                        "exp_status": g.get("exp_status", ""),
                        "concept_eval_eligible": int(concept_eval_eligible(g)),
                        "concept_metric_counted": 0,
                        "concept_match": 0,
                        "gold_expression_norm": norm_expression(g.get("SNOMED_ID / Expression", "")),
                        "pred_expression_norm": "",
                        "attr_target_source": "ignored",
                        "gold_expression_ids": "|".join(sorted(extract_ids(g.get("SNOMED_ID / Expression", "")))),
                        "gold_minimum_ids": "|".join(sorted(extract_ids(g.get("Minimum Concept/s", "")))),
                        "target_ids": "",
                        "pred_ids": "",
                        "attr_tp": 0,
                        "attr_fp": 0,
                        "attr_fn": 0,
                        "selected_pred_item_index": "",
                        "selected_pred_query": "",
                        "selection_method": "ignored_gold_no_pred",
                        "semantic_score": "",
                        "combined_score": "",
                        "jaccard": "",
                    })
                    continue
                gold_expr_raw = g.get("SNOMED_ID / Expression", "")
                gold_expr_norm = norm_expression(gold_expr_raw)
                gold_expr_ids = extract_ids(gold_expr_raw)
                gold_min_ids = extract_ids(g.get("Minimum Concept/s", ""))
                target_ids = gold_min_ids

                attr_fn += len(target_ids)
                details.append({
                    "SPL_SET_ID": spl,
                    "contra_id": g.get("contra_id", ""),
                    "annotation": g.get("annotation", ""),
                    "exp_status": g.get("exp_status", ""),
                    "concept_eval_eligible": int(concept_eval_eligible(g)),
                    "concept_metric_counted": 0,
                    "concept_match": 0,
                    "gold_expression_norm": gold_expr_norm,
                    "pred_expression_norm": "",
                    "attr_target_source": "minimum_concepts",
                    "gold_expression_ids": "|".join(sorted(gold_expr_ids)),
                    "gold_minimum_ids": "|".join(sorted(gold_min_ids)),
                    "target_ids": "|".join(sorted(target_ids)),
                    "pred_ids": "",
                    "attr_tp": 0,
                    "attr_fp": 0,
                    "attr_fn": len(target_ids),
                    "selected_pred_item_index": "",
                    "selected_pred_query": "",
                    "selection_method": "no_preds",
                    "semantic_score": "",
                    "combined_score": "",
                    "jaccard": "",
                })
            continue

        # A) reserve exact expression matches first
        used_pred_uids: Set[int] = set()
        matched_gold_idx: Set[int] = set()

        for gi, g in enumerate(g_rows):
            gold_expr_raw = g.get("SNOMED_ID / Expression", "")
            gold_expr_norm = norm_expression(gold_expr_raw)
            if not gold_expr_norm:
                continue
            candidates = pred_expr_map.get(spl, {}).get(gold_expr_norm, [])
            pick = None
            for r in candidates:
                puid = int(r.get("__pred_uid", -1))
                if puid not in used_pred_uids:
                    pick = r
                    break
            if pick is None:
                continue

            used_pred_uids.add(int(pick.get("__pred_uid", -1)))
            matched_gold_idx.add(gi)

            if gold_row_ignored(g, args.discard_na_gold_expression):
                ignored_matched_pred_uids_by_spl[spl].add(int(pick.get("__pred_uid", -1)))
                details.append({
                    "SPL_SET_ID": spl,
                    "contra_id": g.get("contra_id", ""),
                    "annotation": g.get("annotation", ""),
                    "exp_status": g.get("exp_status", ""),
                    "concept_eval_eligible": int(concept_eval_eligible(g)),
                    "concept_metric_counted": 0,
                    "concept_match": 0,
                    "gold_expression_norm": gold_expr_norm,
                    "pred_expression_norm": norm_expression(pick.get("postcoord_expression", "")),
                    "attr_target_source": "ignored",
                    "gold_expression_ids": "|".join(sorted(extract_ids(gold_expr_raw))),
                    "gold_minimum_ids": "|".join(sorted(extract_ids(g.get("Minimum Concept/s", "")))),
                    "target_ids": "",
                    "pred_ids": "",
                    "attr_tp": 0,
                    "attr_fp": 0,
                    "attr_fn": 0,
                    "selected_pred_item_index": pick.get("item_index", ""),
                    "selected_pred_query": pick.get("query_text", ""),
                    "selection_method": "ignored_gold_exact_match",
                    "semantic_score": "",
                    "combined_score": "",
                    "jaccard": "",
                })
                continue

            target_ids = extract_ids(g.get("Minimum Concept/s", ""))
            pred_ids = pred_ids_from_row(pick)

            attr_tp += len(target_ids & pred_ids)
            attr_fp += len(pred_ids - target_ids)
            attr_fn += len(target_ids - pred_ids)

            details.append({
                "SPL_SET_ID": spl,
                "contra_id": g.get("contra_id", ""),
                "annotation": g.get("annotation", ""),
                "exp_status": g.get("exp_status", ""),
                "concept_eval_eligible": int(concept_eval_eligible(g)),
                "concept_metric_counted": int(concept_eval_eligible(g)),
                "concept_match": 1,
                "gold_expression_norm": gold_expr_norm,
                "pred_expression_norm": norm_expression(pick.get("postcoord_expression", "")),
                "attr_target_source": "minimum_concepts",
                    "gold_expression_ids": "|".join(sorted(extract_ids(gold_expr_raw))),
                    "gold_minimum_ids": "|".join(sorted(extract_ids(g.get("Minimum Concept/s", "")))),
                    "target_ids": "|".join(sorted(target_ids)),
                    "pred_ids": "|".join(sorted(pred_ids)),
                "attr_tp": len(target_ids & pred_ids),
                "attr_fp": len(pred_ids - target_ids),
                "attr_fn": len(target_ids - pred_ids),
                "selected_pred_item_index": pick.get("item_index", ""),
                "selected_pred_query": pick.get("query_text", ""),
                "selection_method": "concept_expression_exact",
                "semantic_score": "",
                "combined_score": "",
                "jaccard": "",
            })

        # B) global assignment for remaining (minimum concepts)
        rem_gold = [g for i, g in enumerate(g_rows) if i not in matched_gold_idx]
        if not semantic_enabled:
            ignored_without_semantic = [
                g for g in rem_gold if gold_row_ignored(g, args.discard_na_gold_expression)
            ]
            rem_gold = [
                g for g in rem_gold if not gold_row_ignored(g, args.discard_na_gold_expression)
            ]
            for g in ignored_without_semantic:
                details.append({
                    "SPL_SET_ID": spl,
                    "contra_id": g.get("contra_id", ""),
                    "annotation": g.get("annotation", ""),
                    "exp_status": g.get("exp_status", ""),
                    "concept_eval_eligible": int(concept_eval_eligible(g)),
                    "concept_metric_counted": 0,
                    "concept_match": 0,
                    "gold_expression_norm": norm_expression(g.get("SNOMED_ID / Expression", "")),
                    "pred_expression_norm": "",
                    "attr_target_source": "ignored",
                    "gold_expression_ids": "|".join(sorted(extract_ids(g.get("SNOMED_ID / Expression", "")))),
                    "gold_minimum_ids": "|".join(sorted(extract_ids(g.get("Minimum Concept/s", "")))),
                    "target_ids": "",
                    "pred_ids": "",
                    "attr_tp": 0,
                    "attr_fp": 0,
                    "attr_fn": 0,
                    "selected_pred_item_index": "",
                    "selected_pred_query": "",
                    "selection_method": "ignored_gold_unmatched_no_semantic",
                    "semantic_score": "",
                    "combined_score": "",
                    "jaccard": "",
                })
        rem_pred = [r for r in p_rows if int(r.get("__pred_uid", -1)) not in used_pred_uids]

        mapping: Dict[int, int] = {}
        score_mat = sem_mat = jac_mat = None

        if rem_gold and rem_pred:
            if semantic_enabled:
                score_mat, sem_mat, jac_mat = build_pair_scores(rem_gold, rem_pred, st_model, args.alpha, args.beta)
            else:
                score_mat = np.zeros((len(rem_gold), len(rem_pred)), dtype=np.float32)
                sem_mat = np.zeros_like(score_mat)
                jac_mat = np.zeros_like(score_mat)
                for i, g in enumerate(rem_gold):
                    g_ids = extract_ids(g.get("Minimum Concept/s", ""))
                    for j, p in enumerate(rem_pred):
                        jac = float(jaccard(g_ids, pred_ids_from_row(p)))
                        jac_mat[i, j] = jac
                        score_mat[i, j] = jac

            mapping = greedy_global_assignment(score_mat, min_score=args.min_pair_score)

        assigned_pred_local = set(mapping.values())

        # Apply mapping
        for i, g in enumerate(rem_gold):
            target_ids = extract_ids(g.get("Minimum Concept/s", ""))

            if i in mapping:
                j = mapping[i]
                p = rem_pred[j]
                used_pred_uids.add(int(p.get("__pred_uid", -1)))

                if gold_row_ignored(g, args.discard_na_gold_expression):
                    ignored_matched_pred_uids_by_spl[spl].add(int(p.get("__pred_uid", -1)))
                    details.append({
                        "SPL_SET_ID": spl,
                        "contra_id": g.get("contra_id", ""),
                        "annotation": g.get("annotation", ""),
                        "exp_status": g.get("exp_status", ""),
                        "concept_eval_eligible": int(concept_eval_eligible(g)),
                        "concept_metric_counted": 0,
                        "concept_match": 0,
                        "gold_expression_norm": norm_expression(g.get("SNOMED_ID / Expression", "")),
                        "pred_expression_norm": norm_expression(p.get("postcoord_expression", "")),
                        "attr_target_source": "ignored",
                        "gold_expression_ids": "|".join(sorted(extract_ids(g.get("SNOMED_ID / Expression", "")))),
                        "gold_minimum_ids": "|".join(sorted(extract_ids(g.get("Minimum Concept/s", "")))),
                        "target_ids": "",
                        "pred_ids": "",
                        "attr_tp": 0,
                        "attr_fp": 0,
                        "attr_fn": 0,
                        "selected_pred_item_index": p.get("item_index", ""),
                        "selected_pred_query": p.get("query_text", ""),
                        "selection_method": "ignored_gold_matched",
                        "semantic_score": float(sem_mat[i, j]) if sem_mat is not None else "",
                        "combined_score": float(score_mat[i, j]) if score_mat is not None else "",
                        "jaccard": float(jac_mat[i, j]) if jac_mat is not None else "",
                    })
                    continue

                pred_ids = pred_ids_from_row(p)
                attr_tp += len(target_ids & pred_ids)
                attr_fp += len(pred_ids - target_ids)
                attr_fn += len(target_ids - pred_ids)

                sem_score = float(sem_mat[i, j]) if sem_mat is not None else ""
                comb_score = float(score_mat[i, j]) if score_mat is not None else ""
                jac_score = float(jac_mat[i, j]) if jac_mat is not None else ""

                details.append({
                    "SPL_SET_ID": spl,
                    "contra_id": g.get("contra_id", ""),
                    "annotation": g.get("annotation", ""),
                    "exp_status": g.get("exp_status", ""),
                    "concept_eval_eligible": int(concept_eval_eligible(g)),
                    "concept_metric_counted": int(concept_eval_eligible(g)),
                    "concept_match": 0,
                    "gold_expression_norm": norm_expression(g.get("SNOMED_ID / Expression", "")),
                    "pred_expression_norm": norm_expression(p.get("postcoord_expression", "")),
                    "attr_target_source": "minimum_concepts",
                    "gold_expression_ids": "|".join(sorted(extract_ids(g.get("SNOMED_ID / Expression", "")))),
                    "gold_minimum_ids": "|".join(sorted(extract_ids(g.get("Minimum Concept/s", "")))),
                    "target_ids": "|".join(sorted(target_ids)),
                    "pred_ids": "|".join(sorted(pred_ids)),
                    "attr_tp": len(target_ids & pred_ids),
                    "attr_fp": len(pred_ids - target_ids),
                    "attr_fn": len(target_ids - pred_ids),
                    "selected_pred_item_index": p.get("item_index", ""),
                    "selected_pred_query": p.get("query_text", ""),
                    "selection_method": "global_greedy",
                    "semantic_score": sem_score,
                    "combined_score": comb_score,
                    "jaccard": jac_score,
                })
            else:
                if gold_row_ignored(g, args.discard_na_gold_expression):
                    details.append({
                        "SPL_SET_ID": spl,
                        "contra_id": g.get("contra_id", ""),
                        "annotation": g.get("annotation", ""),
                        "exp_status": g.get("exp_status", ""),
                        "concept_eval_eligible": int(concept_eval_eligible(g)),
                        "concept_metric_counted": 0,
                        "concept_match": 0,
                        "gold_expression_norm": norm_expression(g.get("SNOMED_ID / Expression", "")),
                        "pred_expression_norm": "",
                        "attr_target_source": "ignored",
                        "gold_expression_ids": "|".join(sorted(extract_ids(g.get("SNOMED_ID / Expression", "")))),
                        "gold_minimum_ids": "|".join(sorted(extract_ids(g.get("Minimum Concept/s", "")))),
                        "target_ids": "",
                        "pred_ids": "",
                        "attr_tp": 0,
                        "attr_fp": 0,
                        "attr_fn": 0,
                        "selected_pred_item_index": "",
                        "selected_pred_query": "",
                        "selection_method": "ignored_gold_unmatched",
                        "semantic_score": "",
                        "combined_score": "",
                        "jaccard": "",
                    })
                    continue
                # unmatched gold
                attr_fn += len(target_ids)
                details.append({
                    "SPL_SET_ID": spl,
                    "contra_id": g.get("contra_id", ""),
                    "annotation": g.get("annotation", ""),
                    "exp_status": g.get("exp_status", ""),
                    "concept_eval_eligible": int(concept_eval_eligible(g)),
                    "concept_metric_counted": int(concept_eval_eligible(g)),
                    "concept_match": 0,
                    "gold_expression_norm": norm_expression(g.get("SNOMED_ID / Expression", "")),
                    "pred_expression_norm": "",
                    "attr_target_source": "minimum_concepts",
                    "gold_expression_ids": "|".join(sorted(extract_ids(g.get("SNOMED_ID / Expression", "")))),
                    "gold_minimum_ids": "|".join(sorted(extract_ids(g.get("Minimum Concept/s", "")))),
                    "target_ids": "|".join(sorted(target_ids)),
                    "pred_ids": "",
                    "attr_tp": 0,
                    "attr_fp": 0,
                    "attr_fn": len(target_ids),
                    "selected_pred_item_index": "",
                    "selected_pred_query": "",
                    "selection_method": "unmatched_gold",
                    "semantic_score": "",
                    "combined_score": "",
                    "jaccard": "",
                })

        # C) unmatched preds -> FP (penalize over-extraction)
        for j, p in enumerate(rem_pred):
            if j in assigned_pred_local:
                continue
            pred_ids = pred_ids_from_row(p)
            if not pred_ids:
                continue
            attr_fp += len(pred_ids)
            details.append({
                "SPL_SET_ID": spl,
                "contra_id": "",
                "annotation": "",
                "exp_status": "",
                "concept_eval_eligible": "",
                "concept_metric_counted": "",
                "concept_match": 0,
                "gold_expression_norm": "",
                "pred_expression_norm": norm_expression(p.get("postcoord_expression", "")),
                "attr_target_source": "none",
                "gold_expression_ids": "",
                "gold_minimum_ids": "",
                "target_ids": "",
                "pred_ids": "|".join(sorted(pred_ids)),
                "attr_tp": 0,
                "attr_fp": len(pred_ids),
                "attr_fn": 0,
                "selected_pred_item_index": p.get("item_index", ""),
                "selected_pred_query": p.get("query_text", ""),
                "selection_method": "unmatched_pred",
                "semantic_score": "",
                "combined_score": "",
                "jaccard": "",
            })

    for spl in gold_scope_spls:
        ignored_used = ignored_matched_pred_uids_by_spl.get(spl, set())
        concept_exact_used = concept_exact_used_pred_uids_by_spl.get(spl, set())
        for p in pred_by_spl.get(spl, []):
            puid = int(p.get("__pred_uid", -1))
            pred_expr = norm_expression(p.get("postcoord_expression", ""))
            if not pred_expr:
                continue
            if puid in concept_exact_used or puid in ignored_used:
                continue
            concept_fp += 1

    concept_metrics = compute_metrics(concept_tp, concept_fp, concept_fn)
    attr_metrics = compute_metrics(attr_tp, attr_fp, attr_fn)

    out_payload = {
        "inputs": {
            "pred_csv": args.pred_csv,
            "gold_csv": args.gold_csv,
            "gold_rows_total": gold_rows_total,
            "gold_rows_used": len([g for g in gold_rows if not gold_row_ignored(g, args.discard_na_gold_expression)]),
            "concept_gold_rows_used": len(concept_gold_rows),
            "gold_rows_dropped_na_expression": dropped_gold_rows,
            "discard_na_gold_expression": args.discard_na_gold_expression,
            "pred_rows": len(pred_rows),
            "gold_unique_spl": len(gold_by_spl),
            "semantic_matching_enabled": semantic_enabled,
            "semantic_matching_unavailable_reasons": semantic_reasons,
            "alpha": args.alpha,
            "beta": args.beta,
            "min_pair_score": args.min_pair_score,
        },
        "concept_level": concept_metrics,
        "attribute_level": attr_metrics,
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")

    detail_fields = [
        "SPL_SET_ID",
        "contra_id",
        "annotation",
        "exp_status",
        "concept_eval_eligible",
        "concept_metric_counted",
        "concept_match",
        "gold_expression_norm",
        "pred_expression_norm",
        "attr_target_source",
        "gold_expression_ids",
        "gold_minimum_ids",
        "target_ids",
        "pred_ids",
        "attr_tp",
        "attr_fp",
        "attr_fn",
        "selected_pred_item_index",
        "selected_pred_query",
        "selection_method",
        "semantic_score",
        "combined_score",
        "jaccard",
    ]
    write_csv(args.out_details_csv, details, detail_fields)

    print(json.dumps(out_payload, indent=2))
    print(f"Wrote details: {args.out_details_csv}")


if __name__ == "__main__":
    main()
