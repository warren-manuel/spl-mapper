# diff from 20260317 2:34 am 
#!/usr/bin/env python3
"""Build SNOMED retrieval indexes and map extracted contraindication terms to concepts."""

import argparse
import glob
import json
import os
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Sequence, Tuple, Union

import faiss
import pandas as pd

from src.llm.backends import extract_json
from src.retrieval.es_utils import (
    bulk_index,
    create_index,
    get_es_client,
    run_elasticsearch,
    stop_elasticsearch,
)
from src.retrieval.embedding_utils import (
    build_and_save_dense_index,
    load_ST_model,
    maybe_move_index_to_gpu,
)
from src.retrieval.search_utils import search_query

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


SNOMED_CT_SETTINGS = {
    "settings": {
        "index": {
            "number_of_shards": 1,
            "number_of_replicas": 1,
            "similarity": {
                "default": {
                    "type": "BM25",
                    "k1": 1.2,
                    "b": 0.75,
                }
            },
        },
        "analysis": {
            "analyzer": {
                "snomed_text": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": ["lowercase", "asciifolding"],
                }
            }
        },
    },
    "mappings": {
        "properties": {
            "conceptId": {"type": "keyword"},
            "preferredTerm": {
                "type": "text",
                "analyzer": "snomed_text",
                "search_analyzer": "snomed_text",
                "copy_to": ["all_terms"],
            },
            "synonyms": {
                "type": "text",
                "analyzer": "snomed_text",
                "search_analyzer": "snomed_text",
                "copy_to": ["all_terms"],
            },
            "all_terms": {
                "type": "text",
                "analyzer": "snomed_text",
                "search_analyzer": "snomed_text",
            },
            "semantic_tag":{
                "type": "keyword"
            }
        }
    },
}

def extract_semantic_tag(term: str) -> str:
    match = re.search(r"\(([^)]+)\)\s*$", term)
    return match.group(1).lower() if match else ""

def make_terms_df(concept_df: pd.DataFrame, synonym_df: pd.DataFrame) -> pd.DataFrame:
    pref = concept_df[["conceptId", "term"]].rename(columns={"term": "term_text"})
    pref["term_type"] = "preferred"

    syn = synonym_df[["conceptId", "term"]].rename(columns={"term": "term_text"})
    syn["term_type"] = "synonym"

    terms_df = pd.concat([pref, syn], ignore_index=True)
    terms_df["term_text"] = terms_df["term_text"].astype(str)
    terms_df = terms_df[terms_df["term_text"].str.strip() != ""]
    terms_df = terms_df.drop_duplicates(subset=["conceptId", "term_text"])
    return terms_df


def load_snomed_frames(concept_path: str, description_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    snomed_con_df = pd.read_csv(concept_path, sep="\t")
    snomed_des_df = pd.read_csv(description_path, sep="\t")

    snomed_active_con = snomed_con_df[snomed_con_df["active"] == 1]
    snomed_des_df = snomed_des_df[
        (snomed_des_df["conceptId"].isin(snomed_active_con["id"]))
        & (snomed_des_df["active"] == 1)
    ]
    snomed_des_df = snomed_des_df[["conceptId", "term", "typeId"]]

    concept_df = snomed_des_df[snomed_des_df["typeId"] == 900000000000003001][["conceptId", "term"]]
    concept_df["semantic_tag"] = concept_df["term"].apply(extract_semantic_tag)
    synonym_df = snomed_des_df[snomed_des_df["typeId"] == 900000000000013009][["conceptId", "term"]]
    syn_agg_df = synonym_df.groupby("conceptId")["term"].apply(list).reset_index()

    snomed_complete_df = pd.merge(concept_df, syn_agg_df, on="conceptId", how="outer")
    snomed_complete_df = snomed_complete_df.rename(
        columns={"term_x": "preferredTerm", "term_y": "synonymTerms"}
    )

    terms_df = make_terms_df(concept_df, synonym_df)
    return concept_df, terms_df, snomed_complete_df


def build_es_index(es, index_name: str, snomed_complete_df: pd.DataFrame) -> None:
    create_index(es, index_name, SNOMED_CT_SETTINGS, delete_if_exists=True)

    field_map = {
        "conceptId": "conceptId",
        "preferredTerm": "preferredTerm",
        "synonymTerms": "synonyms",
        "semantic_tag": "semantic_tag"
    }
    bulk_index(
        es=es,
        df=snomed_complete_df,
        id_col="conceptId",
        index_name=index_name,
        field_map=field_map,
    )


def build_or_load_faiss_index(
    terms_df: pd.DataFrame,
    model_name: str,
    device: str,
    index_path: str,
    rebuild_index: bool,
):
    st_model = load_ST_model(model_name, device=device)

    if rebuild_index or not os.path.exists(index_path):
        _ = build_and_save_dense_index(
            df=terms_df,
            model=st_model,
            text_column="term_text",
            id_column="conceptId",
            batch_size=256,
            normalize=True,
            use_gpu_for_queries=True,
            save_index=True,
            index_filename=index_path,
        )

    cpu_index = faiss.read_index(index_path)
    faiss_index = maybe_move_index_to_gpu(cpu_index)
    return st_model, faiss_index


def read_predictions_jsonl_files(
    pred_files: Union[str, Sequence[str]],
    id_field_candidates: Iterable[str] = ("spl_set_id", "SPL_SET_ID", "item_index"),
    item_term_keys: Sequence[str] = ("ci_text", "condition_text", "substance_text","severity_span",),
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    normalized_term_keys = [k for k in item_term_keys if isinstance(k, str) and k.strip()]
    resolved_pred_files: List[str] = []

    if isinstance(pred_files, str):
        resolved_pred_files = sorted(glob.glob(pred_files))
        if not resolved_pred_files:
            raise FileNotFoundError(f"No files matched pattern: {pred_files}")
    else:
        for p in pred_files:
            if isinstance(p, str):
                matches = sorted(glob.glob(p))
                if matches:
                    resolved_pred_files.extend(matches)
                else:
                    resolved_pred_files.append(p)
        if not resolved_pred_files:
            raise ValueError("pred_files is empty after resolving input paths/patterns")

    for path in resolved_pred_files:
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
                    raise ValueError(
                        f"Prediction record missing SPL_SET_ID/spl_set_id in file {path}"
                    )

                raw = rec.get("raw_output") or {}
                parsed = extract_json(raw)
                if not isinstance(parsed, dict):
                    parsed = {}

                items = parsed.get("items") or []
                for idx,it in enumerate(items):
                    if not isinstance(it, dict):
                        continue
                    row: Dict[str, Any] = {"SPL_SET_ID": spl_id, "item_index": idx}
                    for key in normalized_term_keys:
                        value = it.get(key)
                        row[key] = str(value) if value is not None and str(value).strip() != "" else None
                    rows.append(row)

    return pd.DataFrame(rows)


def map_terms(
    pred_df: pd.DataFrame,
    item_term_keys: Sequence[str],
    st_model,
    faiss_index,
    concept_meta_df: pd.DataFrame,
    es,
    bm25_index: str,
    k_dense: int,
    k_bm25: int,
    k_final: int,
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    normalized_term_keys = [k for k in item_term_keys if isinstance(k, str) and k.strip()]

    for _, row in pred_df.iterrows():
        spl_id = str(row["SPL_SET_ID"])
        item_index = row.get("item_index", None)
        ci_text = row.get("ci_text", "") # only used to keep track of query text later on
        mapped_row: Dict[str, Any] = {"SPL_SET_ID": spl_id, "item_index": item_index, "ci_text": ci_text}
        for key in normalized_term_keys:
            raw_query = row.get(key, None)
            query = str(raw_query).strip() if raw_query is not None else ""
            out_key = f"{key}_terms"
            if not query:
                mapped_row[out_key] = []
                continue
            hits = search_query(
                query_text=query,
                model=st_model,
                faiss_index=faiss_index,
                concept_meta_df=concept_meta_df,
                es=es,
                bm25_index=bm25_index,
                label_column="term",
                bm25_text_field="all_terms",
                bm25_id_field="conceptId",
                bm25_label_field="preferredTerm",
                k_dense=k_dense,
                k_bm25=k_bm25,
                k_final=k_final,
                normalize_query=True,
            )
            mapped_row[out_key] = hits

        output.append(mapped_row)

    return output


def write_jsonl(records: Sequence[Dict[str, Any]], out_path: str) -> None:
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with out_file.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--concept-path",
        default="snomed_us_source/sct2_Concept_Snapshot_US1000124_20250901.txt",
        help="Path to SNOMED concept snapshot TXT",
    )
    ap.add_argument(
        "--description-path",
        default="snomed_us_source/sct2_Description_Snapshot-en_US1000124_20250901.txt",
        help="Path to SNOMED description snapshot TXT",
    )
    ap.add_argument(
        "--pred-jsonl",
        nargs="+",
        default=["results/contra_ie1/out_gpu0.jsonl", "results/contra_ie1/out_gpu1.jsonl"],
        help="One or more prediction JSONL files",
    )
    ap.add_argument(
        "--item-term-keys",
        nargs="+",
        default=["condition_text"],
        help="Keys under parsed items[] to extract/map (e.g., condition_text substance_text)",
    )
    ap.add_argument(
        "--out-jsonl",
        default="results/contra_ie1/mapped_snomed_hits.jsonl",
        help="Output JSONL path",
    )
    ap.add_argument(
        "--es-index",
        default="snomed_ct_es_index",
        help="Elasticsearch index name",
    )
    ap.add_argument(
        "--dense-index-path",
        default="results/snomed_terms_dense_test.bin",
        help="Path to save/load FAISS dense index",
    )
    ap.add_argument(
        "--model-name",
        default="tavakolih/all-MiniLM-L6-v2-pubmed-full",
        help="SentenceTransformer model",
    )
    ap.add_argument(
        "--device",
        default="cuda",
        help="Embedding model device (e.g., cuda or cpu)",
    )
    ap.add_argument("--k-dense", type=int, default=50)
    ap.add_argument("--k-bm25", type=int, default=50)
    ap.add_argument("--k-final", type=int, default=20)
    ap.add_argument(
        "--rebuild-dense-index",
        action="store_true",
        help="Force re-build of dense index even if file exists",
    )
    ap.add_argument(
        "--run-es",
        action="store_true",
        help="Start local Elasticsearch via helper before processing",
    )
    ap.add_argument(
        "--stop-es-on-exit",
        action="store_true",
        help="Stop local Elasticsearch via helper on exit",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    if args.run_es:
        run_elasticsearch()

    es = get_es_client()

    try:
        concept_df, terms_df, snomed_complete_df = load_snomed_frames(
            args.concept_path,
            args.description_path,
        )
        build_es_index(es, args.es_index, snomed_complete_df)

        st_model, faiss_index = build_or_load_faiss_index(
            terms_df=terms_df,
            model_name=args.model_name,
            device=args.device,
            index_path=args.dense_index_path,
            rebuild_index=args.rebuild_dense_index,
        )

        concept_meta_df = concept_df.set_index("conceptId")

        pred_df = read_predictions_jsonl_files(
            args.pred_jsonl,
            item_term_keys=args.item_term_keys,
        )
        mapped = map_terms(
            pred_df=pred_df,
            item_term_keys=args.item_term_keys,
            st_model=st_model,
            faiss_index=faiss_index,
            concept_meta_df=concept_meta_df,
            es=es,
            bm25_index=args.es_index,
            k_dense=args.k_dense,
            k_bm25=args.k_bm25,
            k_final=args.k_final,
        )

        write_jsonl(mapped, args.out_jsonl)
        print(f"Wrote {len(mapped)} records to {args.out_jsonl}")
    finally:
        if args.stop_es_on_exit:
            stop_elasticsearch()


if __name__ == "__main__":
    main()
