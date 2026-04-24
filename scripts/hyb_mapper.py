#!/usr/bin/env python3
"""Build SNOMED retrieval indexes and map extracted contraindication terms to concepts."""

import argparse

from src.retrieval.es_utils import run_elasticsearch, stop_elasticsearch
from src.retrieval.hybrid_mapper import (
    DEFAULT_ITEM_TERM_KEYS,
    build_es_index,
    build_or_load_faiss_index,
    load_snomed_dataframes,
    map_terms,
    read_predictions_jsonl_files,
    write_jsonl,
)


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
        default=list(DEFAULT_ITEM_TERM_KEYS),
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

    es_process = None
    if args.run_es:
        es_process = run_elasticsearch()

    try:
        snomed_frames = load_snomed_dataframes(
            concept_snapshot_path=args.concept_path,
            description_snapshot_path=args.description_path,
            snomed_source_dir="snomed_us_source",
        )

        from src.retrieval.es_utils import get_es_client

        es = get_es_client()
        build_es_index(es, args.es_index, snomed_frames["snomed_complete_df"])

        st_model, faiss_index = build_or_load_faiss_index(
            terms_df=snomed_frames["terms_df"],
            model_name=args.model_name,
            device=args.device,
            index_path=args.dense_index_path,
            rebuild_index=args.rebuild_dense_index,
        )

        concept_meta_df = snomed_frames["concept_df"].set_index("conceptId")
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
        if args.stop_es_on_exit and es_process is not None:
            stop_elasticsearch(es_process)


if __name__ == "__main__":
    main()
