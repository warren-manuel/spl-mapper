#!/usr/bin/env python3
"""
Build (or rebuild) a SNOMED Elasticsearch index without running the full pipeline.

Usage examples:

  # Build the tuned index (b=0.5, default)
  python build_es_index.py

  # Build the original index (b=0.75) used when USE_BM25_TUNING=0
  python build_es_index.py --index-name snomed_ct_es_index_original --bm25-b 0.75

  # Force rebuild of an existing index
  python build_es_index.py --rebuild

  # Full example
  python build_es_index.py \
      --index-name snomed_ct_es_index_original \
      --bm25-b 0.75 \
      --snomed-source snomed_us_source \
      --rebuild
"""

import argparse
import os
import sys

from dotenv import load_dotenv

load_dotenv(override=True)

from VaxMapper.src.utils.hyb_mapper import (
    SNOMED_CT_SETTINGS,
    SNOMED_CT_SETTINGS_ORIGINAL,
    build_es_index,
)
from VaxMapper.src.utils.elastisearch_utils import get_es_client
from VaxMapper.src.utils.snomed_utils import load_snomed_dataframes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--index-name",
        default=os.environ.get("MAPPER_ES_INDEX", "snomed_ct_es_index"),
        help="Elasticsearch index name to create (default: snomed_ct_es_index)",
    )
    parser.add_argument(
        "--bm25-b",
        type=float,
        default=0.5,
        choices=[0.5, 0.75],
        help="BM25 b parameter: 0.5 = tuned, 0.75 = original (default: 0.5)",
    )
    parser.add_argument(
        "--snomed-source",
        default=os.environ.get("SNOMED_SOURCE_DIR", "snomed_us_source"),
        help="Path to SNOMED snapshot directory (default: snomed_us_source)",
    )
    parser.add_argument(
        "--concept-path",
        default=os.environ.get("SNOMED_CONCEPT_PATH"),
        help="Override path to SNOMED concept snapshot file",
    )
    parser.add_argument(
        "--description-path",
        default=os.environ.get("SNOMED_DESCRIPTION_PATH"),
        help="Override path to SNOMED description snapshot file",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete and recreate the index if it already exists",
    )
    args = parser.parse_args()

    index_settings = SNOMED_CT_SETTINGS_ORIGINAL if args.bm25_b == 0.75 else SNOMED_CT_SETTINGS

    print(f"Loading SNOMED dataframes from '{args.snomed_source}'...")
    snomed_frames = load_snomed_dataframes(
        concept_snapshot_path=args.concept_path,
        description_snapshot_path=args.description_path,
        snomed_source_dir=args.snomed_source,
    )

    print(f"Connecting to Elasticsearch...")
    es = get_es_client()

    action = "Rebuilding" if args.rebuild else "Building"
    print(f"{action} index '{args.index_name}' with BM25 b={args.bm25_b}...")
    build_es_index(
        es,
        args.index_name,
        snomed_frames["snomed_complete_df"],
        rebuild_index=args.rebuild,
        index_settings=index_settings,
    )
    print(f"Done. Index '{args.index_name}' is ready.")


if __name__ == "__main__":
    main()
