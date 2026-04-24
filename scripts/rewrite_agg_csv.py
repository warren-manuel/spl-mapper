#!/usr/bin/env python3
"""
One-time script: regenerate aggregated_hits.csv from an aggregated_results.jsonl.

The original CSV had final_concept_id = N/A for postcoord/minimal rows because
aggregate_agent_results() was looking for selected_focus_id while the runner was
already emitting selected_problem_id.  The aggregated JSONL carries
postcoord_expression (e.g. "18629005:{246075003=89119000}"), so we can recover
the focus concept ID from the expression prefix.

Usage:
    python rewrite_agg_csv.py                          # uses defaults below
    python rewrite_agg_csv.py --in  path/to/aggregated_results.jsonl \
                               --out path/to/aggregated_hits.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluation.evaluator import AGG_CSV_COLUMNS, aggregated_item_to_csv_row, write_csv_rows


def fix_final_concept_id(item: dict) -> dict:
    """
    If final_concept_id is N/A for a postcoord/minimal row, derive it from
    postcoord_expression by taking the token before the first ':'.
    e.g. "18629005:{246075003=89119000}" → "18629005"
    """
    if item.get("final_concept_id", "N/A") != "N/A":
        return item
    if item.get("mapping_source") not in ("postcoord", "minimal"):
        return item
    expr = item.get("postcoord_expression", "N/A")
    if expr and expr != "N/A":
        item = dict(item)
        item["final_concept_id"] = expr.split(":")[0]
    return item


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--in",
        dest="in_jsonl",
        default="results/20260414_isolation/000_baseline/aggregated_results.jsonl",
        help="Path to aggregated_results.jsonl",
    )
    ap.add_argument(
        "--out",
        dest="out_csv",
        default=None,
        help="Path for output CSV (default: same dir as --in, filename aggregated_hits.csv)",
    )
    args = ap.parse_args()

    in_path = Path(args.in_jsonl)
    out_path = Path(args.out_csv) if args.out_csv else in_path.parent / "aggregated_hits.csv"

    csv_rows: list[dict] = []
    n_spls = 0
    with in_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            n_spls += 1
            for item in sorted(row.get("items") or [], key=lambda x: int(x.get("item_index", 0))):
                fixed = fix_final_concept_id(item)
                csv_rows.append(aggregated_item_to_csv_row(fixed))

    write_csv_rows(str(out_path), csv_rows, AGG_CSV_COLUMNS)
    print(f"Read {n_spls} SPL groups ({len(csv_rows)} items) from: {in_path}")
    print(f"Wrote CSV ({len(csv_rows)} rows) to: {out_path}")


if __name__ == "__main__":
    main()
