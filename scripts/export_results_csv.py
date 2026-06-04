#!/usr/bin/env python3
"""Convert agent_results.jsonl to a flat CSV for analysis.

Each row in the output CSV represents one mapped contraindication item.
SPL-level fields (product_name, contraindication_text) are repeated for
every item that belongs to that SPL.

Concept FSNs are resolved from the SNOMED RF2 flatfiles so that
MINIMAL/REVIEW rows get clean FSNs rather than truncated LLM reasoning text.

Usage:
    python3 -m scripts.export_results_csv \\
        --input  results/20260527/agent_results.jsonl \\
        --output results/20260527/agent_results.csv

    # Custom SNOMED source directory (default: SNOMED_SOURCE_DIR env var or snomed_us_source)
    python3 -m scripts.export_results_csv \\
        --input results/20260527/agent_results.jsonl \\
        --snomed-source-dir /path/to/snomed_us_source

    # Skip FSN lookup (use raw stored terms; faster but MINIMAL/REVIEW FSNs may be truncated)
    python3 -m scripts.export_results_csv --input ... --no-snomed-lookup
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Allow running as 'python3 scripts/export_results_csv.py' without installing
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.evaluator import write_csv_rows  # noqa: E402

# ---------------------------------------------------------------------------
# XLSX writer (openpyxl)
# ---------------------------------------------------------------------------

# Columns whose cells are merged vertically across all rows sharing the same SPL
_MERGE_COLS = {"SPL_SET_ID", "product_name", "contraindication_text"}


def _spl_key(row: dict) -> tuple:
    return (row.get("SPL_SET_ID", ""), row.get("product_name", ""),
            row.get("contraindication_text", ""))


def _write_xlsx(output_path: str, rows: list[dict], fieldnames: list[str]) -> None:
    """Write rows to an Excel workbook with merged cells for SPL-level fields."""
    from openpyxl import Workbook
    from openpyxl.styles import Font, Alignment

    wb = Workbook()
    ws = wb.active

    # ── Header row (bold, frozen) ─────────────────────────────────────────
    for col_idx, name in enumerate(fieldnames, 1):
        cell = ws.cell(row=1, column=col_idx, value=name)
        cell.font = Font(bold=True)
    ws.freeze_panes = "A2"

    # ── Data rows ─────────────────────────────────────────────────────────
    # Write actual values for merge columns (not blank — merging handles display)
    for row_idx, row in enumerate(rows, 2):
        for col_idx, name in enumerate(fieldnames, 1):
            ws.cell(row=row_idx, column=col_idx, value=row.get(name, ""))

    # ── Merge cells for SPL-level header columns ──────────────────────────
    merge_col_indices = [
        i + 1 for i, name in enumerate(fieldnames) if name in _MERGE_COLS
    ]

    if rows:
        group_start = 2  # 1-indexed worksheet row of current group's first data row
        for i in range(1, len(rows) + 1):
            is_last = (i == len(rows))
            same_group = (not is_last) and (_spl_key(rows[i]) == _spl_key(rows[i - 1]))
            if not same_group:
                group_end = i + 1  # worksheet row index (data starts at row 2)
                if group_end > group_start:  # group spans multiple rows → merge
                    for col in merge_col_indices:
                        ws.merge_cells(
                            start_row=group_start, end_row=group_end,
                            start_column=col, end_column=col,
                        )
                        merged_cell = ws.cell(row=group_start, column=col)
                        merged_cell.alignment = Alignment(
                            vertical="center", wrap_text=True
                        )
                group_start = group_end + 1

    # ── Column widths ─────────────────────────────────────────────────────
    for col_idx, name in enumerate(fieldnames, 1):
        col_letter = ws.cell(row=1, column=col_idx).column_letter
        ws.column_dimensions[col_letter].width = (
            80 if name == "contraindication_text" else 30
        )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    wb.save(output_path)

# (SLOT_TO_ATTR removed — fills slot is always empty; refinements are read from
#  trace.postcoord_pattern.refinements instead)

FIELDNAMES = [
    "SPL_SET_ID",
    "product_name",
    "contraindication_text",
    "item_index",
    "extracted_contraindication",
    "status",
    "concept_id",
    "concept_fsn",
    "attribute_ids",
    "attribute_fsns",
    "value_ids",
    "value_fsns",
]


# ---------------------------------------------------------------------------
# SNOMED FSN lookup
# ---------------------------------------------------------------------------

def _build_fsn_lookup(snomed_source_dir: str) -> dict[int, str]:
    """Load the SNOMED RF2 concept descriptions and return a {conceptId: FSN} dict.

    Uses ``create_concept_df`` from snomed_utils which reads the snapshot files
    and keeps only active concepts with their Fully Specified Names.
    """
    from src.snomed.snomed_utils import create_concept_df  # noqa: E402 (lazy import)

    print(f"  Loading SNOMED concept FSNs from {snomed_source_dir!r} …", file=sys.stderr)
    concept_df = create_concept_df(snomed_source_dir=snomed_source_dir)
    # concept_df columns: conceptId (int64), term (FSN string), semantic_tag
    return dict(zip(concept_df["conceptId"].astype(int), concept_df["term"]))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_refinements(item: dict) -> tuple[str, str, str, str]:
    """Return four pipe-delimited strings for the attribute/value columns.

    Reads from ``trace.postcoord_pattern.refinements``, which is a list of dicts:
        {attribute_sctid, attribute_fsn, value_sctid, value_term}

    Returns empty strings when no refinements are present.
    """
    refinements: list[dict] = (
        ((item.get("trace") or {}).get("postcoord_pattern") or {}).get("refinements") or []
    )

    attr_ids:  list[str] = []
    attr_fsns: list[str] = []
    val_ids:   list[str] = []
    val_fsns:  list[str] = []

    for ref in refinements:
        attr_ids.append(str(ref.get("attribute_sctid", "")))
        attr_fsns.append(str(ref.get("attribute_fsn", "")))
        val_ids.append(str(ref.get("value_sctid", "")))
        val_fsns.append(str(ref.get("value_term", "")))

    sep = " | "
    return sep.join(attr_ids), sep.join(attr_fsns), sep.join(val_ids), sep.join(val_fsns)


def _resolve_fsn(
    sctid_str: str,
    fallback: str,
    fsn_lookup: dict[int, str] | None,
) -> str:
    """Return the FSN for *sctid_str* from the lookup table.

    Falls back to *fallback* when the lookup is unavailable or the SCTID is
    not found (e.g., extension concept not in the loaded snapshot).
    """
    if fsn_lookup is None:
        return fallback or "N/A"
    try:
        sctid = int(sctid_str)
    except (TypeError, ValueError):
        return fallback or "N/A"
    return fsn_lookup.get(sctid, fallback or "N/A")


def _item_to_row(
    item: dict,
    spl: dict,
    fsn_lookup: dict[int, str] | None,
) -> dict:
    """Build one CSV row from a result item + its parent SPL record."""
    status_raw = item.get("status", "")

    if status_raw == "DIRECT":
        concept_id  = str(item.get("selected_id", "N/A"))
        # DIRECT selected_term is already a clean FSN from SNOMED search results
        concept_fsn = str(item.get("selected_term", "N/A"))
        status_out  = "direct"

    elif status_raw in ("MINIMAL", "POSTCOORD"):
        concept_id  = str(item.get("selected_problem_id", "N/A"))
        # selected_focus_term is truncated reasoning text — resolve via SNOMED data
        concept_fsn = _resolve_fsn(
            concept_id,
            fallback=str(item.get("selected_focus_term", "N/A")),
            fsn_lookup=fsn_lookup,
        )
        status_out  = "postcoord"

    elif status_raw == "REVIEW":
        concept_id  = str(item.get("selected_problem_id", "N/A"))
        concept_fsn = _resolve_fsn(
            concept_id,
            fallback=str(item.get("selected_focus_term", "N/A")),
            fsn_lookup=fsn_lookup,
        )
        status_out  = "review"

    else:
        concept_id  = "N/A"
        concept_fsn = "N/A"
        status_out  = status_raw.lower() if status_raw else "unknown"

    attr_ids, attr_fsns, val_ids, val_fsns = _extract_refinements(item)

    return {
        "SPL_SET_ID":                 spl.get("SPL_SET_ID", ""),
        "product_name":               spl.get("product_name", ""),
        "contraindication_text":      spl.get("contra_section_text", ""),
        "item_index":                 item.get("item_index", ""),
        "extracted_contraindication": item.get("query_text", ""),
        "status":                     status_out,
        "concept_id":                 concept_id,
        "concept_fsn":                concept_fsn,
        "attribute_ids":              attr_ids,
        "attribute_fsns":             attr_fsns,
        "value_ids":                  val_ids,
        "value_fsns":                 val_fsns,
    }


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def convert(
    input_path: str,
    output_path: str,
    fsn_lookup: dict[int, str] | None,
) -> int:
    """Read JSONL, build rows, write CSV.  Returns the number of rows written."""
    rows: list[dict] = []

    with open(input_path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                spl = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"  [warn] line {lineno}: JSON parse error — {exc}", file=sys.stderr)
                continue

            for item in spl.get("results") or []:
                rows.append(_item_to_row(item, spl, fsn_lookup))

    if output_path.lower().endswith(".xlsx"):
        # XLSX: merge cells handle display — no blanking needed
        _write_xlsx(output_path, rows, FIELDNAMES)
    else:
        # CSV: blank repeated SPL-level fields so each group header appears once
        _BLANK_FIELDS = ("SPL_SET_ID", "product_name", "contraindication_text")
        last_key: tuple = ()
        for row in rows:
            key = (row["SPL_SET_ID"], row["product_name"], row["contraindication_text"])
            if key == last_key:
                for f in _BLANK_FIELDS:
                    row[f] = ""
            else:
                last_key = key
        write_csv_rows(output_path, rows, FIELDNAMES)

    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert agent_results.jsonl to a flat analysis CSV."
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to agent_results.jsonl",
    )
    parser.add_argument(
        "--output", "-o",
        help="Output CSV path (default: same dir/stem as input with .csv extension)",
    )
    parser.add_argument(
        "--snomed-source-dir",
        default=os.environ.get("SNOMED_SOURCE_DIR", "snomed_us_source"),
        help="Directory containing SNOMED RF2 snapshot files "
             "(default: SNOMED_SOURCE_DIR env var or 'snomed_us_source')",
    )
    parser.add_argument(
        "--no-snomed-lookup",
        action="store_true",
        help="Skip SNOMED FSN lookup — faster but MINIMAL/REVIEW concept_fsn "
             "will contain raw (possibly truncated) stored text",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    output_path = args.output or str(input_path.with_suffix(".xlsx"))

    # Build FSN lookup unless suppressed
    fsn_lookup: dict[int, str] | None = None
    if not args.no_snomed_lookup:
        snomed_dir = Path(args.snomed_source_dir)
        if not snomed_dir.exists():
            print(
                f"  [warn] SNOMED source dir not found: {snomed_dir}. "
                "Falling back to stored terms (use --no-snomed-lookup to silence).",
                file=sys.stderr,
            )
        else:
            fsn_lookup = _build_fsn_lookup(str(snomed_dir))
            print(f"  FSN lookup: {len(fsn_lookup):,} concepts loaded.", file=sys.stderr)

    print(f"Reading:  {input_path}")
    print(f"Writing:  {output_path}")
    n = convert(str(input_path), output_path, fsn_lookup)
    print(f"Done — {n} rows written.")


if __name__ == "__main__":
    main()
