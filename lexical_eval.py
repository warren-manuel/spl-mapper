#!/usr/bin/env python3
"""
lexical_eval.py

Document-level evaluation (per SPL_SET_ID) of contraindication extraction.
No spans required: uses deterministic canonical mention matching.

Inputs:
- Gold CSV with SPL_SET_ID and a gold annotation column (one annotation per row)
- One or more prediction JSONL files (from multiple GPUs) that include:
  - spl_set_id (or SPL_SET_ID)
  - parsed_json.items[].condition_text (list of predicted mentions)

Outputs:
- Prints micro-averaged TP/FP/FN, Precision/Recall/F1
- Writes per-document metrics CSV (and optional FP/FN keys)
"""

import os
import re
import json
import glob
import argparse
from typing import List, Dict, Any, Tuple

import pandas as pd

from VaxMapper.src.llm import extract_json


# -----------------------------
# Canonicalization (Method 1)
# -----------------------------
_STOP_PREFIXES = [
    "known", "history of", "patients with", "patient with", "individuals with",
    "those with", "contraindicated in", "contraindicated for",
    "do not use in", "should not be used in", "is contraindicated in",
]

_HYPERSENS_PATTERNS = [
    re.compile(r"^(?:known\s+)?(?:hypersensitivity|allergy|allergic\s+reaction|anaphylaxis)\s+(?:to|against)\s+(?P<x>.+)$"),
    re.compile(r"^(?P<x>.+?)\s+(?:hypersensitivity|allergy)$"),
]

def _basic_normalize(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.replace("‚Ä¢", " ").replace("â€¢", " ").replace("•", " ")
    s = re.sub(r"[^\w\s\-/]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def canonicalize_contra_term(s: str) -> str:
    s = _basic_normalize(s)

    for p in _STOP_PREFIXES:
        if s.startswith(p + " "):
            s = s[len(p) + 1 :].strip()

    # stabilize ingredients/components/excipients phrasing
    s = re.sub(r"\b(any|all)\s+of\s+the\s+\b", "", s)
    s = re.sub(r"\b(other|additional)\s+\b", "", s)
    s = re.sub(r"\b(ingredients?|components?|excipients?)\s+of\s+(?P<drug>[\w\-/ ]+)", r"ingredients|\g<drug>", s)
    s = re.sub(r"\b(ingredients?|components?|excipients?)\s+in\s+(?P<drug>[\w\-/ ]+)", r"ingredients|\g<drug>", s)

    for pat in _HYPERSENS_PATTERNS:
        m = pat.match(s)
        if m:
            x = _basic_normalize(m.group("x"))
            x = re.sub(r"^(a|an|the)\s+", "", x)
            return f"hypersensitivity|{x}"

    return s

def eval_one_doc(gold_terms: List[str], pred_terms: List[str]) -> Dict[str, Any]:
    G = set(canonicalize_contra_term(x) for x in gold_terms if (x or "").strip() and str(x).lower() != "nan")
    P = set(canonicalize_contra_term(x) for x in pred_terms if (x or "").strip() and str(x).lower() != "nan")

    tp_keys = G & P
    fp_keys = P - G
    fn_keys = G - P

    tp, fp, fn = len(tp_keys), len(fp_keys), len(fn_keys)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": prec, "recall": rec, "f1": f1,
        "fp_keys": sorted(fp_keys),
        "fn_keys": sorted(fn_keys),
    }

def micro_from_counts(tp: int, fp: int, fn: int) -> Dict[str, float]:
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": f1}


# -----------------------------
# IO helpers
# -----------------------------
def read_gold(gold_csv: str, id_col: str, gold_term_col: str) -> pd.DataFrame:
    df = pd.read_csv(gold_csv)
    if id_col not in df.columns:
        raise ValueError(f"Gold CSV missing '{id_col}'. Columns: {list(df.columns)}")
    if gold_term_col not in df.columns:
        raise ValueError(f"Gold CSV missing '{gold_term_col}'. Columns: {list(df.columns)}")
    df[id_col] = df[id_col].astype(str)
    return df

def read_predictions_jsonl_files(pred_files: List[str], id_field_candidates=("spl_set_id", "SPL_SET_ID","item_index")) -> pd.DataFrame:
    """
    Returns DataFrame with columns:
      - SPL_SET_ID
      - pred_terms : list[str]
    """
    rows = []
    for path in pred_files:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)

                spl_id = None
                for k in id_field_candidates:
                    if k in rec and rec[k] is not None:
                        spl_id = str(rec[k])
                        break
                if spl_id is None:
                    raise ValueError(f"Prediction record missing SPL_SET_ID/spl_set_id in file {path}")

                raw = rec.get("raw_output") or {}
                parsed = extract_json(raw)
                # parsed = rec.get("parsed_json") or {}
                items = parsed.get("items") or []
                pred_terms = []
                for it in items:
                    if isinstance(it, dict):
                        ct = it.get("condition_text")
                        if ct:
                            pred_terms.append(str(ct))

                rows.append({"SPL_SET_ID": spl_id, "pred_terms": pred_terms})
    return pd.DataFrame(rows)

def expand_glob_inputs(inputs: List[str]) -> List[str]:
    files = []
    for x in inputs:
        if any(ch in x for ch in ["*", "?", "["]):
            files.extend(glob.glob(x))
        else:
            files.append(x)
    files = [f for f in files if os.path.isfile(f)]
    if not files:
        raise ValueError("No prediction JSONL files found from provided paths/globs.")
    return sorted(files)


# -----------------------------
# Main evaluation
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold_csv", required=True, help="Gold CSV path")
    ap.add_argument("--gold_id_col", default="SPL_SET_ID", help="Gold SPL id column name")
    ap.add_argument("--gold_term_col", required=True, help="Gold annotation column name (one contraindication per row)")
    ap.add_argument("--pred_jsonl", nargs="+", required=True,
                    help="One or more prediction JSONL files OR globs (e.g., outputs/out_gpu*.jsonl)")
    ap.add_argument("--out_per_doc_csv", default="per_doc_eval.csv", help="Where to write per-doc results")
    ap.add_argument("--include_fp_fn", action="store_true", help="Include fp_keys and fn_keys columns in output CSV")
    args = ap.parse_args()

    # Load gold
    gold_df = read_gold(args.gold_csv, args.gold_id_col, args.gold_term_col)

    gold_group = (
        gold_df
        .groupby(args.gold_id_col)[args.gold_term_col]
        .apply(lambda s: [x for x in s.astype(str).tolist() if x and str(x).strip() and str(x).lower() != "nan"])
        .reset_index()
        .rename(columns={args.gold_term_col: "gold_terms"})
    )

    # Load preds from multiple jsonl files
    pred_files = expand_glob_inputs(args.pred_jsonl)
    pred_df = read_predictions_jsonl_files(pred_files)

    pred_group = (
        pred_df
        .groupby("SPL_SET_ID")["pred_terms"]
        .apply(lambda lists: [t for sub in lists for t in (sub or [])])  # flatten
        .reset_index()
    )

    # Merge and evaluate
    merged = gold_group.merge(pred_group, left_on=args.gold_id_col, right_on="SPL_SET_ID", how="left")
    merged["pred_terms"] = merged["pred_terms"].apply(lambda x: x if isinstance(x, list) else [])

    per_rows = []
    tp_total = fp_total = fn_total = 0

    for _, r in merged.iterrows():
        spl_id = str(r[args.gold_id_col])
        gold_terms = r["gold_terms"]
        pred_terms = r["pred_terms"]

        s = eval_one_doc(gold_terms, pred_terms)
        tp_total += s["tp"]
        fp_total += s["fp"]
        fn_total += s["fn"]

        row = {
            "SPL_SET_ID": spl_id,
            "tp": s["tp"], "fp": s["fp"], "fn": s["fn"],
            "precision": s["precision"], "recall": s["recall"], "f1": s["f1"],
        }
        if args.include_fp_fn:
            row["fp_keys"] = s["fp_keys"]
            row["fn_keys"] = s["fn_keys"]
        per_rows.append(row)

    micro = micro_from_counts(tp_total, fp_total, fn_total)

    print("=== Method 1 (Canonical Mention Matching) Micro-Average ===")
    print(f"TP={micro['tp']} FP={micro['fp']} FN={micro['fn']}")
    print(f"Precision={micro['precision']:.4f} Recall={micro['recall']:.4f} F1={micro['f1']:.4f}")
    print(f"Gold docs evaluated: {len(gold_group)} | Pred docs found: {pred_group['SPL_SET_ID'].nunique()}")

    if args.out_per_doc_csv:
        per_doc_df = pd.DataFrame(per_rows)
        per_doc_df.to_csv(args.out_per_doc_csv, index=False)
        print(f"Wrote per-doc results to: {args.out_per_doc_csv}")


if __name__ == "__main__":
    main()
