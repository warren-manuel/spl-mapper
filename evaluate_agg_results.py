#!/usr/bin/env python3
"""Evaluate agg_results.csv against contra_gold_100.csv.

Evaluation rules implemented:
1) Match within same SPL_SET_ID; item indexes are ignored.
2) Concept-level: exact normalized expression match between gold and prediction.
   - Normalization removes whitespace and braces from both expressions.
3) Attribute-level target per gold row:
   - If concept-level matched: target IDs = all concept IDs from gold SNOMED_ID / Expression.
   - Else: target IDs = concept IDs from gold Minimum Concept/s.
   - Predicted IDs are from {final_concept_id, causative_agent_id, severity_id, clinical_course_id}
     of the selected prediction row in the same SPL.
"""

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

try:
    import pandas as pd
    from VaxMapper.src.utils.embedding_utils import build_and_save_dense_index, load_ST_model
except Exception:
    pd = None
    build_and_save_dense_index = None
    load_ST_model = None

ID_PATTERN = re.compile(r"\d+")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred-csv", default="results/20260303/agg_results.csv")
    ap.add_argument("--gold-csv", default="results/contra_gold_100.csv")
    ap.add_argument("--out-json", default="results/20260303/eval_metrics.json")
    ap.add_argument("--out-details-csv", default="results/20260303/eval_details.csv")
    ap.add_argument(
        "--discard-na-gold-expression",
        action="store_true",
        help="Discard rows where BOTH 'Minimum Concept/s' and 'SNOMED_ID / Expression' are blank/NA before evaluation.",
    )
    ap.add_argument("--st-model-id", default="tavakolih/all-MiniLM-L6-v2-pubmed-full")
    ap.add_argument("--st-device", default="cuda")
    ap.add_argument("--st-batch-size", type=int, default=128)
    return ap.parse_args()


def read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def norm_expression(text: str) -> str:
    text = (text or "").strip()
    if not text or text.upper() in {"NA", "N/A"}:
        return ""
    # Remove braces and all spaces prior to concept-level comparison.
    text = text.replace("{", "").replace("}", "")
    text = re.sub(r"\s+", "", text)
    return text


def extract_ids(text: str) -> Set[str]:
    text = text or ""
    ids = set(ID_PATTERN.findall(text))
    return ids


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


def pick_best_overlap_row(
    rows: List[Dict[str, str]],
    target_ids: Set[str],
    used_pred_uids: Optional[Set[int]] = None,
) -> Optional[Dict[str, str]]:
    if not rows:
        return None

    used_pred_uids = used_pred_uids or set()
    best = None
    best_key: Tuple[int, int] = (-1, 10**9)
    for r in rows:
        pred_uid = int(r.get("__pred_uid", -1))
        if pred_uid in used_pred_uids:
            continue
        pids = pred_ids_from_row(r)
        overlap = len(target_ids & pids)
        idx = safe_int(r.get("item_index"))
        key = (overlap, -idx)
        if key > best_key:
            best = r
            best_key = key
    return best


def compute_metrics(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def write_csv(path: str, rows: Iterable[Dict[str, Any]], fieldnames: List[str]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def build_spl_semantic_index(
    rows: List[Dict[str, str]],
    model: Any,
    batch_size: int = 128,
) -> Dict[str, Any]:
    if pd is None or build_and_save_dense_index is None:
        return {"index": None, "id_to_row": {}, "k_search": 0}
    if not rows:
        return {"index": None, "id_to_row": {}, "k_search": 0}

    df = pd.DataFrame(
        {
            "pred_uid": [int(r["__pred_uid"]) for r in rows],
            "query_text": [(r.get("query_text") or "").strip() for r in rows],
        }
    )
    # Ensure no empty text rows break dense retrieval.
    df["query_text"] = df["query_text"].replace("", "[EMPTY_QUERY_TEXT]")

    index = build_and_save_dense_index(
        df=df,
        model=model,
        text_column="query_text",
        id_column="pred_uid",
        batch_size=batch_size,
        normalize=True,
        use_gpu_for_queries=False,
        save_index=False,
    )
    id_to_row = {int(r["__pred_uid"]): r for r in rows}
    return {"index": index, "id_to_row": id_to_row, "k_search": len(rows)}


def pick_semantic_best_unused_row(
    annotation_text: str,
    rows: List[Dict[str, str]],
    used_pred_uids: Set[int],
    model: Any,
    spl_index_cache: Dict[str, Dict[str, Any]],
    spl: str,
    batch_size: int = 128,
) -> Tuple[Optional[Dict[str, str]], Optional[float]]:
    if not rows or model is None:
        return None, None

    if spl not in spl_index_cache:
        spl_index_cache[spl] = build_spl_semantic_index(rows, model=model, batch_size=batch_size)

    cache = spl_index_cache[spl]
    if cache["index"] is None or cache["k_search"] == 0:
        return None, None

    query = (annotation_text or "").strip() or "[EMPTY_ANNOTATION]"
    q_vec = model.encode(
        [query],
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    dists, ids = cache["index"].search(q_vec, cache["k_search"])
    for row_id, score in zip(ids[0], dists[0]):
        if int(row_id) < 0:
            continue
        pred_uid = int(row_id)
        if pred_uid in used_pred_uids:
            continue
        row = cache["id_to_row"].get(pred_uid)
        if row is not None:
            return row, float(score)

    return None, None


def main() -> None:
    args = parse_args()

    pred_rows = read_csv(args.pred_csv)
    gold_rows = read_csv(args.gold_csv)
    gold_rows_total = len(gold_rows)
    dropped_gold_rows = 0
    if args.discard_na_gold_expression:
        kept_rows: List[Dict[str, str]] = []
        for row in gold_rows:
            has_min = bool(norm_expression(row.get("Minimum Concept/s", "")))
            has_expr = bool(norm_expression(row.get("SNOMED_ID / Expression", "")))
            if has_min or has_expr:
                kept_rows.append(row)
            else:
                dropped_gold_rows += 1
        gold_rows = kept_rows

    semantic_enabled = (pd is not None) and (load_ST_model is not None) and (build_and_save_dense_index is not None)
    st_model = load_ST_model(args.st_model_id, device=args.st_device) if semantic_enabled else None

    pred_by_spl: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    pred_expr_map: Dict[str, Dict[str, List[Dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    spl_index_cache: Dict[str, Dict[str, Any]] = {}
    used_pred_uids_by_spl: Dict[str, Set[int]] = defaultdict(set)

    for pred_uid, r in enumerate(pred_rows):
        spl = (r.get("SPL_SET_ID") or "").strip()
        if not spl:
            continue
        r["__pred_uid"] = pred_uid
        pred_by_spl[spl].append(r)
        norm_pred_expr = norm_expression(r.get("postcoord_expression", ""))
        if norm_pred_expr:
            pred_expr_map[spl][norm_pred_expr].append(r)

    # Deterministic order for tie-breaks.
    for spl in pred_by_spl:
        pred_by_spl[spl].sort(key=lambda x: safe_int(x.get("item_index")))
    for spl in pred_expr_map:
        for expr in pred_expr_map[spl]:
            pred_expr_map[spl][expr].sort(key=lambda x: safe_int(x.get("item_index")))

    concept_tp = 0
    concept_fp = 0
    concept_fn = 0

    attr_tp = 0
    attr_fp = 0
    attr_fn = 0

    details: List[Dict[str, Any]] = []

    # Concept-level from gold perspective: exact expression match within SPL.
    for g in gold_rows:
        spl = (g.get("SPL_SET_ID") or "").strip()
        gold_expr_raw = g.get("SNOMED_ID / Expression", "")
        gold_expr_norm = norm_expression(gold_expr_raw)

        gold_expr_ids = extract_ids(gold_expr_raw)
        gold_min_ids = extract_ids(g.get("Minimum Concept/s", ""))

        candidate_rows = pred_by_spl.get(spl, [])

        concept_match_row: Optional[Dict[str, str]] = None
        concept_match = False
        if gold_expr_norm:
            matches = pred_expr_map.get(spl, {}).get(gold_expr_norm, [])
            if matches:
                concept_match = True
                concept_match_row = matches[0]

        # Concept-level counts (skip NA/empty gold expressions from TP/FN denominator).
        if gold_expr_norm:
            if concept_match:
                concept_tp += 1
            else:
                concept_fn += 1

        # Attribute-level target rule.
        if concept_match:
            target_ids = set(gold_expr_ids)
            selected_pred = concept_match_row
            attr_target_source = "gold_expression"
            selection_method = "concept_expression_exact"
            semantic_score = None
        else:
            target_ids = set(gold_min_ids)
            selected_pred = None
            semantic_score = None
            if semantic_enabled:
                selected_pred, semantic_score = pick_semantic_best_unused_row(
                    annotation_text=g.get("annotation", ""),
                    rows=candidate_rows,
                    used_pred_uids=used_pred_uids_by_spl[spl],
                    model=st_model,
                    spl_index_cache=spl_index_cache,
                    spl=spl,
                    batch_size=args.st_batch_size,
                )
            if selected_pred is None:
                selected_pred = pick_best_overlap_row(
                    candidate_rows, target_ids, used_pred_uids=used_pred_uids_by_spl[spl]
                )
                selection_method = "fallback_overlap"
            else:
                selection_method = "semantic_nn"
            attr_target_source = "minimum_concepts"
            if selected_pred is not None:
                used_pred_uids_by_spl[spl].add(int(selected_pred.get("__pred_uid", -1)))
        if concept_match:
            pred_ids = extract_ids((selected_pred or {}).get("postcoord_expression", ""))
        else:
            pred_ids = pred_ids_from_row(selected_pred) if selected_pred else set()

        attr_tp += len(target_ids & pred_ids)
        attr_fp += len(pred_ids - target_ids)
        attr_fn += len(target_ids - pred_ids)

        details.append(
            {
                "SPL_SET_ID": spl,
                "contra_id": g.get("contra_id", ""),
                "annotation": g.get("annotation", ""),
                "concept_match": int(concept_match),
                "gold_expression_norm": gold_expr_norm,
                "pred_expression_norm": norm_expression((selected_pred or {}).get("postcoord_expression", "")),
                "attr_target_source": attr_target_source,
                "gold_expression_ids": "|".join(sorted(gold_expr_ids)),
                "gold_minimum_ids": "|".join(sorted(gold_min_ids)),
                "target_ids": "|".join(sorted(target_ids)),
                "pred_ids": "|".join(sorted(pred_ids)),
                "attr_tp": len(target_ids & pred_ids),
                "attr_fp": len(pred_ids - target_ids),
                "attr_fn": len(target_ids - pred_ids),
                "selected_pred_item_index": "" if not selected_pred else selected_pred.get("item_index", ""),
                "selected_pred_query": "" if not selected_pred else selected_pred.get("query_text", ""),
                "selection_method": selection_method,
                "semantic_score": "" if semantic_score is None else semantic_score,
            }
        )

    # Optional concept-level FP from prediction perspective for SPLs/expressions present in gold scope.
    gold_scope_spls = {(g.get("SPL_SET_ID") or "").strip() for g in gold_rows}
    gold_scope_spls.discard("")

    gold_expr_counter: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for g in gold_rows:
        spl = (g.get("SPL_SET_ID") or "").strip()
        expr = norm_expression(g.get("SNOMED_ID / Expression", ""))
        if spl and expr:
            gold_expr_counter[spl][expr] += 1

    pred_expr_counter: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for spl in gold_scope_spls:
        for r in pred_by_spl.get(spl, []):
            expr = norm_expression(r.get("postcoord_expression", ""))
            if expr:
                pred_expr_counter[spl][expr] += 1

    for spl in gold_scope_spls:
        gmap = gold_expr_counter.get(spl, {})
        pmap = pred_expr_counter.get(spl, {})
        for expr, pcount in pmap.items():
            extra = pcount - gmap.get(expr, 0)
            if extra > 0:
                concept_fp += extra

    concept_metrics = compute_metrics(concept_tp, concept_fp, concept_fn)
    attr_metrics = compute_metrics(attr_tp, attr_fp, attr_fn)

    out_payload = {
        "inputs": {
            "pred_csv": args.pred_csv,
            "gold_csv": args.gold_csv,
            "gold_rows_total": gold_rows_total,
            "gold_rows_used": len(gold_rows),
            "gold_rows_dropped_na_expression": dropped_gold_rows,
            "discard_na_gold_expression": args.discard_na_gold_expression,
            "pred_rows": len(pred_rows),
            "gold_unique_spl": len({(g.get('SPL_SET_ID') or '').strip() for g in gold_rows if (g.get('SPL_SET_ID') or '').strip()}),
            "semantic_matching_enabled": semantic_enabled,
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
    ]
    write_csv(args.out_details_csv, details, detail_fields)

    print(json.dumps(out_payload, indent=2))
    print(f"Wrote details: {args.out_details_csv}")


if __name__ == "__main__":
    main()
