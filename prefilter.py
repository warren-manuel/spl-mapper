#!/usr/bin/env python3
"""
Precompute SNOMED ECL range-membership cache for postcoord candidate filtering.

Output JSON is consumable by postcord.py via --prefilter-cache.
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

from VaxMapper.src.utils.snomed_utils import (
    ATTRIBUTE_TABLE,
    DEFAULT_PREFILTER_CONTENT_TYPE,
    base as SNOMED_BASE_URL,
    check_snomed_connection,
    concept_matches_any_ecl,
    get_attribute_range_constraints,
    load_snomed_dataframes,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mapped-jsonl", default="results/20260225/mapped_hits.jsonl")
    ap.add_argument("--out-json", default="results/20260225/prefilter_cache.json")
    ap.add_argument("--snomed-source-dir", default="snomed_us_source")
    ap.add_argument("--max-workers", type=int, default=16)
    ap.add_argument("--timeout", type=int, default=10)
    ap.add_argument("--retries", type=int, default=1)
    ap.add_argument(
        "--skip-connection-check",
        action="store_true",
        help="Skip initial base URL connectivity check.",
    )
    return ap.parse_args()


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def normalize_candidate_ids(terms: Any) -> Set[int]:
    ids: Set[int] = set()
    for term in terms or []:
        if not isinstance(term, dict):
            continue
        cid = term.get("id")
        try:
            ids.add(int(cid))
        except Exception:
            continue
    return ids


def collect_unique_candidate_ids(mapped_rows: Iterable[Dict[str, Any]]) -> Dict[str, Set[int]]:
    pools = {
        "causative_agent": set(),
        "severity": set(),
        "clinical_course": set(),
    }
    for row in mapped_rows:
        pools["causative_agent"].update(normalize_candidate_ids(row.get("substance_text_terms")))
        pools["severity"].update(normalize_candidate_ids(row.get("severity_span_terms")))
        pools["clinical_course"].update(normalize_candidate_ids(row.get("course_span_terms")))
    return pools


def build_membership_for_attribute(
    attribute_key: str,
    concept_ids: Set[int],
    ecls: List[str],
    max_workers: int,
    timeout: int,
    retries: int,
) -> Tuple[Dict[int, bool], Dict[str, int]]:
    if not concept_ids:
        return {}, {"concept_count": 0, "allowed_count": 0, "http_error_count": 0}

    membership: Dict[int, bool] = {}
    stats = {
        "concept_count": len(concept_ids),
        "allowed_count": 0,
        "http_error_count": 0,
    }

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(
                concept_matches_any_ecl,
                cid,
                ecls,
                SNOMED_BASE_URL,
                timeout,
                retries,
                True,
            ): cid
            for cid in sorted(concept_ids)
        }
        for fut in as_completed(futures):
            cid = futures[fut]
            allowed, errors = fut.result()
            membership[cid] = bool(allowed)
            if allowed:
                stats["allowed_count"] += 1
            stats["http_error_count"] += int(errors)

    print(
        f"[{attribute_key}] concepts={stats['concept_count']} "
        f"allowed={stats['allowed_count']} http_errors={stats['http_error_count']}"
    )
    return membership, stats


def main() -> None:
    args = parse_args()

    if not args.skip_connection_check:
        check_snomed_connection(timeout=args.timeout, base_url=SNOMED_BASE_URL)

    mapped_rows = list(read_jsonl(args.mapped_jsonl))
    pools = collect_unique_candidate_ids(mapped_rows)

    attr_range_df = load_snomed_dataframes(snomed_source_dir=args.snomed_source_dir)["attr_range"]

    constraints = get_attribute_range_constraints(
        attr_range_df,
        content_type_by_attribute=DEFAULT_PREFILTER_CONTENT_TYPE,
    )

    memberships: Dict[str, Dict[int, bool]] = {}
    stats_by_attr: Dict[str, Dict[str, int]] = {}
    for key in ("causative_agent", "severity", "clinical_course"):
        membership, stats = build_membership_for_attribute(
            attribute_key=key,
            concept_ids=pools.get(key, set()),
            ecls=constraints.get(key, []),
            max_workers=max(1, args.max_workers),
            timeout=args.timeout,
            retries=max(1, args.retries),
        )
        memberships[key] = membership
        stats_by_attr[key] = stats

    out_obj = {
        "meta": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "mapped_jsonl": args.mapped_jsonl,
            "snomed_source_dir": args.snomed_source_dir,
            "snomed_base_url": SNOMED_BASE_URL,
            "attribute_table": {k: int(v) for k, v in ATTRIBUTE_TABLE.items()},
            "content_type_ids": DEFAULT_PREFILTER_CONTENT_TYPE,
            "constraints": constraints,
            "stats_by_attribute": stats_by_attr,
        },
        "memberships": {
            key: {str(cid): allowed for cid, allowed in memberships[key].items()}
            for key in memberships
        },
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)

    print(f"Wrote prefilter cache: {out_path}")


if __name__ == "__main__":
    main()
