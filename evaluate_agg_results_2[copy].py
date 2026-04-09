#!/usr/bin/env python3
"""Evaluate aggregated contraindication predictions against contra_gold_100_2.csv.

Scoring is performed per SPL_SET_ID using a global one-to-one assignment between
gold rows and prediction rows. Matching uses:

  score = alpha * semantic_cosine(annotation, query_text)
          + beta * jaccard(gold_union_ids, pred_union_ids)

Metric levels:
1) extraction_level
   - TP: matched gold/pred pair
   - FN: unmatched gold annotation
   - FP: unmatched predicted query_text
2) contraindication_level
   - Exact-set metric on the union of problem/causative/severity/course concepts.
   - When decoupled, count matched pairs only.
3) concept_level
   - Relaxed set-overlap metric on the same concept unions.
   - When decoupled, count matched pairs only.
"""

#ToDo: Jaccard scoring needs to use text signals not concept ID. Since the information extraction is text based. 

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

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args: Any, **kwargs: Any) -> bool:
        return False

try:
    from VaxMapper.src.utils.embedding_utils import load_ST_model
except Exception:
    load_ST_model = None


ID_PATTERN = re.compile(r"\d+")
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
        help="Discard rows where 'Minimum Concept/s' is blank/NA before evaluation.",
    )
    ap.add_argument("--st-model-id", default=default_st_model)
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
    ap.add_argument(
        "--no-decoupled",
        action="store_true",
        help="When set, unmatched gold/pred rows also contribute to contraindication-level and concept-level metrics.",
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


def compute_metrics(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


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


def build_pair_scores(
    gold_rows: List[Dict[str, str]],
    pred_rows: List[Dict[str, str]],
    st_model: Any,
    alpha: float,
    beta: float,
    st_batch_size: int,
) -> Tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
    """Return (combined_scores, semantic_matrix, jaccard_matrix). Shapes: (G,P)."""
    gold_count = len(gold_rows)
    pred_count = len(pred_rows)
    scores = np.zeros((gold_count, pred_count), dtype=np.float32)
    sem = np.zeros((gold_count, pred_count), dtype=np.float32)
    jac = np.zeros((gold_count, pred_count), dtype=np.float32)

    gold_texts = [(row.get("annotation") or "").strip() or "[EMPTY_ANNOTATION]" for row in gold_rows]
    pred_texts = [(row.get("query_text") or "").strip() or "[EMPTY_QUERY_TEXT]" for row in pred_rows]

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

    gold_id_sets = [parse_gold_union_ids(row) for row in gold_rows]
    pred_id_sets = [pred_ids_from_row(row) for row in pred_rows]

    for gold_idx in range(gold_count):
        for pred_idx in range(pred_count):
            jac[gold_idx, pred_idx] = float(jaccard(gold_id_sets[gold_idx], pred_id_sets[pred_idx]))
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

    for spl, all_gold_rows in gold_by_spl.items():
        gold_rows_for_matching = [
            row for row in all_gold_rows if not gold_row_ignored(row, args.discard_na_gold_expression)
        ]
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
                    gold_ids = parse_gold_union_ids(gold_row)
                    for pred_idx, pred_row in enumerate(pred_rows_for_matching):
                        jac = float(jaccard(gold_ids, pred_ids_from_row(pred_row)))
                        jac_mat[gold_idx, pred_idx] = jac
                        score_mat[gold_idx, pred_idx] = jac

            mapping = greedy_global_assignment(score_mat, min_score=args.min_pair_score)

        assigned_pred_local = set(mapping.values())

        for gold_idx, gold_row in enumerate(gold_rows_for_matching):
            gold_ids = parse_gold_union_ids(gold_row)
            annotation = gold_row.get("annotation", "")
            contra_id = gold_row.get("contra_id", "")

            if gold_idx in mapping:
                pred_idx = mapping[gold_idx]
                pred_row = pred_rows_for_matching[pred_idx]
                pred_ids = pred_ids_from_row(pred_row)
                extraction_tp += 1

                matched_contra_tp = matched_contra_fp = matched_contra_fn = 0
                matched_concept_tp = len(gold_ids & pred_ids)
                matched_concept_fp = len(pred_ids - gold_ids)
                matched_concept_fn = len(gold_ids - pred_ids)

                if gold_ids == pred_ids:
                    matched_contra_tp = 1
                else:
                    matched_contra_fp = 1
                    matched_contra_fn = 1

                if decoupled:
                    contraindication_tp += matched_contra_tp
                    contraindication_fp += matched_contra_fp
                    contraindication_fn += matched_contra_fn
                    concept_tp += matched_concept_tp
                    concept_fp += matched_concept_fp
                    concept_fn += matched_concept_fn

                sem_score = float(sem_mat[gold_idx, pred_idx]) if sem_mat is not None else ""
                jac_score = float(jac_mat[gold_idx, pred_idx]) if jac_mat is not None else ""
                combined_score = float(score_mat[gold_idx, pred_idx]) if score_mat is not None else ""

                append_detail(
                    details,
                    spl=spl,
                    contra_id=contra_id,
                    annotation=annotation,
                    query_text=pred_row.get("query_text", ""),
                    gold_ids=gold_ids,
                    pred_ids=pred_ids,
                    selection_method="global_greedy",
                    semantic_score=sem_score,
                    jaccard_score=jac_score,
                    combined_score=combined_score,
                    extraction_tp=1,
                    contraindication_tp=matched_contra_tp if decoupled else 0,
                    contraindication_fp=matched_contra_fp if decoupled else 0,
                    contraindication_fn=matched_contra_fn if decoupled else 0,
                    concept_tp=matched_concept_tp if decoupled else 0,
                    concept_fp=matched_concept_fp if decoupled else 0,
                    concept_fn=matched_concept_fn if decoupled else 0,
                    decoupled=decoupled,
                    downstream_counted=int(decoupled),
                    item_index=str(pred_row.get("item_index", "")),
                )
            else:
                extraction_fn += 1
                gold_ids = parse_gold_union_ids(gold_row)
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
                    extraction_fn=1,
                    contraindication_fn=0 if decoupled else 1,
                    concept_fn=0 if decoupled else len(gold_ids),
                    decoupled=decoupled,
                    downstream_counted=int(not decoupled),
                )

        for pred_idx, pred_row in enumerate(pred_rows_for_matching):
            if pred_idx in assigned_pred_local:
                continue
            pred_ids = pred_ids_from_row(pred_row)
            extraction_fp += 1
            if not decoupled:
                contraindication_fp += 1
                concept_fp += len(pred_ids)

            append_detail(
                details,
                spl=spl,
                query_text=pred_row.get("query_text", ""),
                pred_ids=pred_ids,
                selection_method="unmatched_pred",
                extraction_fp=1,
                contraindication_fp=0 if decoupled else 1,
                concept_fp=0 if decoupled else len(pred_ids),
                decoupled=decoupled,
                downstream_counted=int(not decoupled),
                item_index=str(pred_row.get("item_index", "")),
            )

    pred_only_spls = set(pred_by_spl) - set(gold_by_spl)
    for spl in sorted(pred_only_spls):
        for pred_row in pred_by_spl[spl]:
            pred_ids = pred_ids_from_row(pred_row)
            extraction_fp += 1
            if not decoupled:
                contraindication_fp += 1
                concept_fp += len(pred_ids)
            append_detail(
                details,
                spl=spl,
                query_text=pred_row.get("query_text", ""),
                pred_ids=pred_ids,
                selection_method="unmatched_pred_no_gold",
                extraction_fp=1,
                contraindication_fp=0 if decoupled else 1,
                concept_fp=0 if decoupled else len(pred_ids),
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
