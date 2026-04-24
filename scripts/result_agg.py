#!/usr/bin/env python3
"""Aggregate verified + postcoord mapping outputs by SPL_SET_ID.

Rules:
- Use verified selected SNOMED when present.
- If verified selected_snomed_id == "N/A", use postcoord selected_focus + selected fills.
- Build postcoord expression using ATTRIBUTE_TABLE ids and omit missing/N/A attributes.
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

try:
    from src.snomed.snomed_utils import ATTRIBUTE_TABLE
except Exception:
    # Fallback values matching VaxMapper/src/utils/snomed_utils.py
    ATTRIBUTE_TABLE = {
        "causative_agent": 246075003,
        "severity": 246112005,
        "clinical_course": 263502005,
    }

ATTRIBUTE_KEYS = ("causative_agent", "severity", "clinical_course")
CSV_COLUMNS = [
    "SPL_SET_ID",
    "item_index",
    "query_text",
    "mapping_source",
    "final_concept_id",
    "final_concept_term",
    "postcoord_expression",
    "causative_agent_id",
    "causative_agent_term",
    "severity_id",
    "severity_term",
    "clinical_course_id",
    "clinical_course_term",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verified-jsonl", default="results/20260225/verified_hits.jsonl")
    ap.add_argument("--postcoord-jsonl", default="results/20260225/postcoord_hits.jsonl")
    ap.add_argument("--out-jsonl", default="results/20260225/aggregated_hits.jsonl")
    ap.add_argument("--out-csv", default="results/20260225/aggregated_hits.csv")
    return ap.parse_args()


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def row_key(row: Dict[str, Any]) -> Tuple[str, int]:
    return str(row["SPL_SET_ID"]), int(row["item_index"])


def normalize_fill(fills: Dict[str, Any], key: str) -> Dict[str, str]:
    raw = (fills or {}).get(key) or {}
    return {
        "id": str(raw.get("id", "N/A")),
        "term": str(raw.get("term", "N/A")),
    }


def build_postcoord_expression(focus_id: str, attr_values: Dict[str, str]) -> str:
    if not focus_id or focus_id == "N/A":
        return "N/A"

    parts: List[str] = []
    for key in ATTRIBUTE_KEYS:
        val = attr_values.get(key, "N/A")
        if val == "N/A":
            continue
        attr_id = ATTRIBUTE_TABLE[key]
        parts.append(f"{attr_id}={val}")

    if not parts:
        return focus_id
    return f"{focus_id}:{{{','.join(parts)}}}"


def build_item_row(verified_row: Dict[str, Any], postcoord_row: Dict[str, Any]) -> Dict[str, Any]:
    selected_snomed_id = str(verified_row.get("selected_snomed_id", "N/A"))
    selected_snomed_term = str(verified_row.get("selected_snomed_term", "N/A"))

    if selected_snomed_id != "N/A":
        return {
            "SPL_SET_ID": str(verified_row["SPL_SET_ID"]),
            "item_index": int(verified_row["item_index"]),
            "query_text": str(verified_row.get("query_text", "")),
            "mapping_source": "verified",
            "final_concept_id": selected_snomed_id,
            "final_concept_term": selected_snomed_term,
            "postcoord_expression": selected_snomed_id,
            "postcoord_schema": {selected_snomed_id: {}},
            "attributes": {},
        }

    if postcoord_row is None:
        raise ValueError(
            "Missing postcoord row for verified N/A item: "
            f"{verified_row.get('SPL_SET_ID')}#{verified_row.get('item_index')}"
        )

    focus_id = str(postcoord_row.get("selected_focus_id", "N/A"))
    focus_term = str(postcoord_row.get("selected_focus_term", "N/A"))
    fills = postcoord_row.get("fills") or {}

    fill_objects = {
        key: normalize_fill(fills, key)
        for key in ATTRIBUTE_KEYS
    }

    attr_values = {key: fill_objects[key]["id"] for key in ATTRIBUTE_KEYS}
    included_attrs = {
        key: fill_objects[key]["id"]
        for key in ATTRIBUTE_KEYS
        if fill_objects[key]["id"] != "N/A"
    }
    included_attrs_with_alias = dict(included_attrs)
    if "causative_agent" in included_attrs_with_alias:
        included_attrs_with_alias["causative"] = included_attrs_with_alias["causative_agent"]
    included_attrs_by_attr_id = {
        str(ATTRIBUTE_TABLE[key]): fill_objects[key]["id"]
        for key in ATTRIBUTE_KEYS
        if fill_objects[key]["id"] != "N/A"
    }

    return {
        "SPL_SET_ID": str(verified_row["SPL_SET_ID"]),
        "item_index": int(verified_row["item_index"]),
        "query_text": str(verified_row.get("query_text", "")),
        "mapping_source": "postcoord",
        "final_concept_id": focus_id,
        "final_concept_term": focus_term,
        "postcoord_expression": build_postcoord_expression(focus_id, attr_values),
        "postcoord_schema": {focus_id: included_attrs_with_alias} if focus_id != "N/A" else {},
        "postcoord_schema_by_attribute_id": (
            {focus_id: included_attrs_by_attr_id} if focus_id != "N/A" else {}
        ),
        "attributes": {
            key: {
                "attribute_id": str(ATTRIBUTE_TABLE[key]),
                "value_id": fill_objects[key]["id"],
                "value_term": fill_objects[key]["term"],
            }
            for key in ATTRIBUTE_KEYS
            if fill_objects[key]["id"] != "N/A"
        },
    }


def to_csv_row(item: Dict[str, Any]) -> Dict[str, Any]:
    attributes = item.get("attributes") or {}

    def val(attr_key: str, field: str) -> str:
        return str(((attributes.get(attr_key) or {}).get(field)) or "")

    return {
        "SPL_SET_ID": item["SPL_SET_ID"],
        "item_index": item["item_index"],
        "query_text": item.get("query_text", ""),
        "mapping_source": item.get("mapping_source", ""),
        "final_concept_id": item.get("final_concept_id", ""),
        "final_concept_term": item.get("final_concept_term", ""),
        "postcoord_expression": item.get("postcoord_expression", ""),
        "causative_agent_id": val("causative_agent", "value_id"),
        "causative_agent_term": val("causative_agent", "value_term"),
        "severity_id": val("severity", "value_id"),
        "severity_term": val("severity", "value_term"),
        "clinical_course_id": val("clinical_course", "value_id"),
        "clinical_course_term": val("clinical_course", "value_term"),
    }


def write_csv(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()

    verified_rows = read_jsonl(args.verified_jsonl)
    postcoord_rows = read_jsonl(args.postcoord_jsonl)
    postcoord_index = {row_key(r): r for r in postcoord_rows}

    items: List[Dict[str, Any]] = []
    for v in verified_rows:
        key = row_key(v)
        item = build_item_row(v, postcoord_index.get(key))
        items.append(item)

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in items:
        grouped[item["SPL_SET_ID"]].append(item)

    aggregated_rows: List[Dict[str, Any]] = []
    csv_rows: List[Dict[str, Any]] = []

    for spl_set_id in sorted(grouped.keys()):
        ordered_items = sorted(grouped[spl_set_id], key=lambda r: int(r["item_index"]))
        aggregated_rows.append(
            {
                "SPL_SET_ID": spl_set_id,
                "item_count": len(ordered_items),
                "items": ordered_items,
            }
        )
        csv_rows.extend(to_csv_row(item) for item in ordered_items)

    write_jsonl(args.out_jsonl, aggregated_rows)
    write_csv(args.out_csv, csv_rows)

    print(f"Wrote {len(aggregated_rows)} SPL groups to {args.out_jsonl}")
    print(f"Wrote {len(csv_rows)} item rows to {args.out_csv}")


if __name__ == "__main__":
    main()
