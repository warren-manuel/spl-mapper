#!/usr/bin/env python3
"""
Build or reuse retrieval assets for contraindication mapping.

This script prepares:
- Elasticsearch BM25 index
- FAISS dense index

It does not run the LangGraph pipeline.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

from VaxMapper.src.utils.elastisearch_utils import (
    get_es_client,
    run_elasticsearch,
    stop_elasticsearch,
)
from VaxMapper.src.utils.hyb_mapper import load_mapper_resources


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _check_es_index_exists(index_name: str) -> bool:
    es = get_es_client()
    return bool(es.indices.exists(index=index_name))


def _check_es_ready() -> bool:
    try:
        es = get_es_client()
        return bool(es.ping())
    except Exception:
        return False


def build_assets(args: argparse.Namespace) -> Dict[str, Any]:
    dense_index_path = Path(args.dense_index_path)
    dense_index_path.parent.mkdir(parents=True, exist_ok=True)

    faiss_preexisting = dense_index_path.exists()
    es_preexisting = _check_es_index_exists(args.es_index)

    _ = load_mapper_resources(
        snomed_source_dir=args.snomed_source_dir,
        concept_path=args.concept_path,
        description_path=args.description_path,
        es_index=args.es_index,
        dense_index_path=str(dense_index_path),
        model_name=args.model_name,
        device=args.device,
        k_dense=args.k_dense,
        k_bm25=args.k_bm25,
        k_final=args.k_final,
        rebuild_dense_index=args.rebuild_dense_index,
        rebuild_es_index=args.rebuild_es_index,
    )

    faiss_exists = dense_index_path.exists()
    es_exists = _check_es_index_exists(args.es_index)

    return {
        "faiss_preexisting": faiss_preexisting,
        "faiss_exists": faiss_exists,
        "es_preexisting": es_preexisting,
        "es_exists": es_exists,
        "dense_index_path": str(dense_index_path),
        "es_index": args.es_index,
        "faiss_action": "rebuilt" if args.rebuild_dense_index else ("reused" if faiss_preexisting else "created"),
        "es_action": "rebuilt" if args.rebuild_es_index else ("reused" if es_preexisting else "created"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snomed-source-dir",
        default=os.environ.get("SNOMED_SOURCE_DIR", "snomed_us_source"),
        help="Directory containing SNOMED source snapshots.",
    )
    parser.add_argument(
        "--concept-path",
        default=os.environ.get("SNOMED_CONCEPT_PATH") or None,
        help="Optional explicit SNOMED concept snapshot path.",
    )
    parser.add_argument(
        "--description-path",
        default=os.environ.get("SNOMED_DESCRIPTION_PATH") or None,
        help="Optional explicit SNOMED description snapshot path.",
    )
    parser.add_argument(
        "--es-index",
        default=os.environ.get("MAPPER_ES_INDEX", "snomed_ct_es_index"),
        help="Elasticsearch index name for BM25 retrieval.",
    )
    parser.add_argument(
        "--dense-index-path",
        default=os.environ.get("MAPPER_DENSE_INDEX_PATH", "results/snomed_terms_dense_test.bin"),
        help="Path to the persisted FAISS index file.",
    )
    parser.add_argument(
        "--model-name",
        default=os.environ.get("MAPPER_MODEL_NAME", "tavakolih/all-MiniLM-L6-v2-pubmed-full"),
        help="Sentence-transformer model used for dense retrieval.",
    )
    parser.add_argument(
        "--device",
        default=os.environ.get("MAPPER_DEVICE", "cuda"),
        help="Device for dense embedding model and FAISS query path.",
    )
    parser.add_argument(
        "--k-dense",
        type=int,
        default=int(os.environ.get("MAPPER_K_DENSE", "50")),
        help="Dense retrieval candidate count.",
    )
    parser.add_argument(
        "--k-bm25",
        type=int,
        default=int(os.environ.get("MAPPER_K_BM25", "50")),
        help="BM25 retrieval candidate count.",
    )
    parser.add_argument(
        "--k-final",
        type=int,
        default=int(os.environ.get("MAPPER_K_FINAL", "20")),
        help="Final fused retrieval candidate count.",
    )
    parser.add_argument(
        "--rebuild-dense-index",
        action="store_true",
        default=_env_bool("MAPPER_REBUILD_DENSE_INDEX", False),
        help="Force rebuilding the FAISS index even if the file already exists.",
    )
    parser.add_argument(
        "--rebuild-es-index",
        action="store_true",
        default=_env_bool("MAPPER_REBUILD_ES_INDEX", False),
        help="Force deleting and rebuilding the Elasticsearch index.",
    )
    parser.add_argument(
        "--start-es",
        action="store_true",
        help="Start the bundled Elasticsearch service if it is not already running.",
    )
    parser.add_argument(
        "--stop-es-on-exit",
        action="store_true",
        help="Stop Elasticsearch on exit if this script started it.",
    )
    return parser


def main() -> None:
    if load_dotenv is not None:
        load_dotenv()

    parser = build_parser()
    args = parser.parse_args()

    es_process: Optional[Any] = None
    started_es = False

    if not _check_es_ready():
        if not args.start_es:
            raise RuntimeError(
                "Elasticsearch is not reachable at the configured host/port. "
                "Start it first or rerun with --start-es."
            )
        es_process = run_elasticsearch()
        started_es = True
        if not _check_es_ready():
            raise RuntimeError("Elasticsearch did not become ready after startup.")

    try:
        summary = build_assets(args)
    finally:
        if started_es and args.stop_es_on_exit:
            stop_elasticsearch(es_process)

    if not summary["faiss_exists"]:
        raise RuntimeError(f"FAISS index was not created: {summary['dense_index_path']}")
    if not summary["es_exists"]:
        raise RuntimeError(f"Elasticsearch index was not created: {summary['es_index']}")

    print(f"BM25 index: {summary['es_index']} ({summary['es_action']})")
    print(f"FAISS index: {summary['dense_index_path']} ({summary['faiss_action']})")
    print(f"Elasticsearch started by script: {'yes' if started_es else 'no'}")
    print(f"Elasticsearch stopped on exit: {'yes' if started_es and args.stop_es_on_exit else 'no'}")


if __name__ == "__main__":
    main()
