#!/usr/bin/env python3

import json
import glob
import argparse
from typing import List, Dict, Tuple

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from src.llm.backends import extract_json
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# -----------------------------
# Basic normalization
# -----------------------------
def normalize_text(s: str) -> str:
    return (s or "").strip().lower()


# -----------------------------
# Load gold
# -----------------------------
def load_gold(gold_csv: str, id_col: str, term_col: str) -> pd.DataFrame:
    df = pd.read_csv(gold_csv)
    if id_col not in df.columns or term_col not in df.columns:
        raise ValueError("Gold CSV missing required columns.")
    df[id_col] = df[id_col].astype(str)
    return (
        df.groupby(id_col)[term_col]
        .apply(lambda x: [normalize_text(v) for v in x if str(v).strip()])
        .reset_index()
        .rename(columns={term_col: "gold_terms"})
    )


# -----------------------------
# Load predictions (multiple JSONL files)
# -----------------------------
def load_predictions(jsonl_paths: List[str]) -> pd.DataFrame:
    rows = []
    for path in jsonl_paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                spl_id = rec.get("item_index") or rec.get("SPL_SET_ID")
                if not spl_id:
                    raise ValueError(f"Missing SPL_SET_ID in {path}")

                raw = rec.get("raw_output") or {}
                parsed = extract_json(raw)
                # items = (rec.get("parsed_json") or {}).get("items") or []
                items = parsed.get("items") or []
                preds = []
                for it in items:
                    ct = it.get("condition_text")
                    if ct:
                        preds.append(normalize_text(ct))

                rows.append({
                    "SPL_SET_ID": str(spl_id),
                    "pred_terms": preds
                })

    df = pd.DataFrame(rows)
    return (
        df.groupby("SPL_SET_ID")["pred_terms"]
        .apply(lambda lists: [t for sub in lists for t in sub])
        .reset_index()
    )


# -----------------------------
# Greedy one-to-one semantic matching
# -----------------------------
# def semantic_match(
#     gold_terms: List[str],
#     pred_terms: List[str],
#     model: SentenceTransformer,
#     threshold: float = 0.90
# ) -> Tuple[int, int, int]:
#     if not gold_terms and not pred_terms:
#         return 0, 0, 0
#     if not gold_terms:
#         return 0, len(pred_terms), 0
#     if not pred_terms:
#         return 0, 0, len(gold_terms)

#     # Deduplicate
#     gold_unique = list(dict.fromkeys(gold_terms))
#     pred_unique = list(dict.fromkeys(pred_terms))

#     # Embed
#     gold_emb = model.encode(gold_unique, normalize_embeddings=True)
#     pred_emb = model.encode(pred_unique, normalize_embeddings=True)

#     sim = pred_emb @ gold_emb.T  # cosine similarity matrix

#     used_gold = set()
#     tp = 0

#     # Greedy: sort pred rows by max similarity
#     order = np.argsort(-sim.max(axis=1))

#     for i in order:
#         best_gold = np.argmax(sim[i])
#         score = sim[i, best_gold]

#         if score >= threshold and best_gold not in used_gold:
#             tp += 1
#             used_gold.add(best_gold)

#     fp = len(pred_unique) - tp
#     fn = len(gold_unique) - tp

#     return tp, fp, fn


def semantic_match_detailed(
    gold_terms: List[str],
    pred_terms: List[str],
    model,
    threshold: float = 0.90
) -> Dict:

    # Deduplicate (preserve order)
    gold_unique = list(dict.fromkeys(gold_terms))
    pred_unique = list(dict.fromkeys(pred_terms))

    if not gold_unique and not pred_unique:
        return {
            "tp": 0, "fp": 0, "fn": 0,
            "tp_pairs": [],
            "fp_terms": [],
            "fn_terms": []
        }

    if not gold_unique:
        return {
            "tp": 0,
            "fp": len(pred_unique),
            "fn": 0,
            "tp_pairs": [],
            "fp_terms": pred_unique,
            "fn_terms": []
        }

    if not pred_unique:
        return {
            "tp": 0,
            "fp": 0,
            "fn": len(gold_unique),
            "tp_pairs": [],
            "fp_terms": [],
            "fn_terms": gold_unique
        }

    # Embed
    gold_emb = model.encode(gold_unique, normalize_embeddings=True)
    pred_emb = model.encode(pred_unique, normalize_embeddings=True)

    sim = pred_emb @ gold_emb.T

    used_gold = set()
    tp_pairs = []

    # Greedy order by highest max similarity
    order = np.argsort(-sim.max(axis=1))

    for i in order:
        best_gold = np.argmax(sim[i])
        score = float(sim[i, best_gold])

        if score >= threshold and best_gold not in used_gold:
            tp_pairs.append({
                "pred": pred_unique[i],
                "gold": gold_unique[best_gold],
                "similarity": score
            })
            used_gold.add(best_gold)

    matched_pred = {pair["pred"] for pair in tp_pairs}
    matched_gold = {pair["gold"] for pair in tp_pairs}

    fp_terms = [p for p in pred_unique if p not in matched_pred]
    fn_terms = [g for g in gold_unique if g not in matched_gold]
    
    tp, fp, fn = len(tp_pairs),len(fp_terms),len(fn_terms)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0

    return {
        "tp": tp,"fp": fp,"fn": fn,
        "precision": prec, "recall": rec, "f1":f1,
        "tp_pairs": tp_pairs,
        "fp_terms": fp_terms,
        "fn_terms": fn_terms
    }

def micro_from_counts(tp: int, fp: int, fn: int) -> Dict[str, float]:
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": f1}


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold_csv", required=True)
    parser.add_argument("--gold_term_col", required=True)
    parser.add_argument("--gold_id_col", default="SPL_SET_ID", help="Gold SPL id column name")
    parser.add_argument("--pred_jsonl", nargs="+", required=True)
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--model_name", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--out_per_doc_csv", default="per_doc_eval.csv", help="Where to write per-doc results")
    parser.add_argument("--include_fp_fn", action="store_true", help="Include fp_keys and fn_keys columns in output CSV")
    args = parser.parse_args()

    pred_files = []
    for p in args.pred_jsonl:
        pred_files.extend(glob.glob(p))

    gold_df = load_gold(args.gold_csv, "SPL_SET_ID", args.gold_term_col)
    pred_df = load_predictions(pred_files)

    merged = gold_df.merge(pred_df, on="SPL_SET_ID", how="left")
    merged["pred_terms"] = merged["pred_terms"].apply(lambda x: x if isinstance(x, list) else [])

    model = SentenceTransformer(args.model_name)

    per_rows = []
    tp_total = fp_total = fn_total = 0

    for _, row in merged.iterrows():
        spl_id = str(row[args.gold_id_col])
        s = semantic_match_detailed(
            row["gold_terms"],
            row["pred_terms"],
            model,
            threshold=args.threshold
        )
        tp_total += s["tp"]
        fp_total += s["fp"]
        fn_total += s["fn"]

        row = {
            "SPL_SET_ID": spl_id,
            "tp": s["tp"], "fp": s["fp"], "fn": s["fn"],
            "precision": s["precision"], "recall": s["recall"], "f1": s["f1"],
        }
        if args.include_fp_fn:
            row["tp_pairs"] = s["tp_pairs"]
            row["fp_terms"] = s["fp_terms"]
            row["fn_terms"] = s["fn_terms"]
        per_rows.append(row)

    micro = micro_from_counts(tp_total, fp_total, fn_total)


    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) else 0
    recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0

    print("=== Embedding-Based Extraction Evaluation ===")
    print(f"Threshold: {args.threshold}")
    print(f"TP={tp_total} FP={fp_total} FN={fn_total}")
    print(f"Precision={precision:.4f}")
    print(f"Recall={recall:.4f}")
    print(f"F1={f1:.4f}")

    print("=== Method 1 (Canonical Mention Matching) Micro-Average ===")
    print(f"TP={micro['tp']} FP={micro['fp']} FN={micro['fn']}")
    print(f"Precision={micro['precision']:.4f} Recall={micro['recall']:.4f} F1={micro['f1']:.4f}")

    per_doc_df = pd.DataFrame(per_rows)
    per_doc_df.to_csv(args.out_per_doc_csv, index=False)
    print(f"Wrote per-doc results to: {args.out_per_doc_csv}")



if __name__ == "__main__":
    main()
