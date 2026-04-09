#!/usr/bin/env python3
"""
Post-coordinate SNOMED mapping over unmatched verified rows.

Flow:
1) Read verified_hits and keep rows with selected_snomed_id == "N/A".
2) Join mapped_hits by (SPL_SET_ID, item_index).
3) Build prompts with candidates:
   - focus: condition_text_terms
   - causative: substance_text_terms (ECL-filtered by ATTRIBUTE_TABLE range constraints)
   - severity: severity_span_terms (ECL-filtered)
   - course: course_span_terms (ECL-filtered)
4) Run local LLM with map_verify-style multi-GPU workers.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Avoid torch.compile / dynamo instability for per-item generation workers.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import pandas as pd

from VaxMapper.src.llm import build_messages_from_iter, extract_json, load_model_local
from VaxMapper.src.utils.snomed_utils import (
    ATTRIBUTE_TABLE,
    base as SNOMED_BASE_URL,
    check_snomed_connection,
    concept_matches_ecl,
    get_range_constraints_for_attribute,
    load_snomed_dataframes,
)


SYSTEM_PROMPT = """
You are an expert Medical Informatician specializing in SNOMED CT post-coordination for Clinical Decision Support.

Task:
1) Analyze the QUERY, which represents a contraindication to which a patient should not receive a drug.
2) Choose the single best focus concept from FOCUS_CANDIDATES.
   - The focus represents the primary condition/diagnosis/physiological state that is the target of the post-coordination.
   - For conditions prioritize the 'Disease' or 'Disorder' type concepts.
   - For physiological states and other latent conditions, prioritize the 'Finding' type concept representing the 'State' or 'Dispositional' aspect.
3) For each attribute in ATTRIBUTE_TABLE, select the best VALUE from its candidate list.
   - If a candidate list is empty or no value is supported by the text, output "N/A".
   - You MUST ONLY use conceptIds provided in the candidate lists. Do NOT invent IDs.

Output ONLY minified JSON followed by <<END_JSON>>. 
No markdown fences.

Schema:
{{
  "selected_focus_id": "<conceptId from FOCUS_CANDIDATES or N/A>",
  "fills": {{
    "<attribute_key>": "<conceptId from that attribute's candidates or N/A>",
    ...
  }}
}}
<<END_JSON>>
"""

USER_PROMPT = """
QUERY:
{span_text}

ATTRIBUTE_TABLE (attribute_key -> SNOMED attributeId):
{attribute_table_json}

FOCUS_CANDIDATES (choose 1):
{focus_candidates_block}

CANDIDATES FOR causative_agent (choose 1 or N/A):
{causative_candidates_block}

CANDIDATES FOR severity (choose 1 or N/A):
{severity_candidates_block}

CANDIDATES FOR clinical_course (choose 1 or N/A):
{course_candidates_block}
"""

ATTRIBUTE_KEYS = ("causative_agent", "severity", "clinical_course")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mapped-jsonl", default="results/20260225/mapped_hits.jsonl")
    ap.add_argument("--verified-jsonl", default="results/20260225/verified_hits.jsonl")
    ap.add_argument("--out-jsonl", default="results/20260225/postcoord_hits.jsonl")
    ap.add_argument("--model-id", default="google/medgemma-27b-text-it")
    ap.add_argument("--max-new-tokens", type=int, default=320)
    ap.add_argument(
        "--filter-by-range",
        action="store_true",
        help="Apply ECL range-constraint filtering for causative/severity/course candidates (default: off).",
    )
    ap.add_argument(
        "--debug-payloads",
        action="store_true",
        help="Include payload debug diagnostics and raw/filtered candidates in output rows.",
    )
    ap.add_argument(
        "--snomed-source-dir",
        default="snomed_us_source",
        help="Directory containing RF2 MRCM files (domain/attribute range snapshots).",
    )
    ap.add_argument(
        "--prefilter-cache",
        default="",
        help="Optional JSON cache from prefilter.py for range membership lookups.",
    )
    ap.add_argument(
        "--allow-dynamo",
        action="store_true",
        help="Allow TorchDynamo/torch.compile (disabled by default for stability).",
    )
    ap.add_argument(
        "--gpu-ids",
        default="0",
        help="Comma-separated GPU IDs (e.g., 0 or 0,1,2,3).",
    )
    ap.add_argument(
        "--_worker",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    ap.add_argument(
        "--_worker-out",
        default="",
        help=argparse.SUPPRESS,
    )
    return ap.parse_args()


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_gpu_ids(gpu_ids_str: str) -> List[str]:
    ids = [x.strip() for x in gpu_ids_str.split(",") if x.strip()]
    if not ids:
        raise ValueError("--gpu-ids must contain at least one GPU id.")
    return ids


def configure_torch_runtime(allow_dynamo: bool) -> None:
    if allow_dynamo:
        return
    try:
        import torch._dynamo as dynamo  # type: ignore

        dynamo.config.disable = True
        dynamo.config.suppress_errors = True
        if hasattr(dynamo.config, "fail_on_recompile_limit_hit"):
            dynamo.config.fail_on_recompile_limit_hit = False
    except Exception:
        pass


def normalize_terms(terms: Optional[List[Dict[str, Any]]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for t in terms or []:
        cid = t.get("id")
        label = t.get("label")
        if cid is None or label is None:
            continue
        out.append({"id": str(cid), "label": str(label)})
    return out


def row_key(row: Dict[str, Any]) -> Tuple[str, int]:
    spl = str(row.get("SPL_SET_ID", ""))
    idx = int(row.get("item_index"))
    return spl, idx


def load_prefilter_memberships(path: str) -> Dict[str, Dict[int, bool]]:
    if not path:
        return {}

    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    memberships = obj.get("memberships", obj)
    out: Dict[str, Dict[int, bool]] = {}
    for key in ATTRIBUTE_KEYS:
        raw_map = memberships.get(key, {})
        cleaned: Dict[int, bool] = {}
        if isinstance(raw_map, dict):
            for cid, allowed in raw_map.items():
                try:
                    cleaned[int(cid)] = bool(allowed)
                except Exception:
                    continue
        out[key] = cleaned
    return out


def ecl_match_cached(
    concept_id: Any,
    ecl: str,
    cache: Dict[Tuple[int, str], bool],
    stats: Optional[Dict[str, int]] = None,
) -> bool:
    try:
        concept_id_int = int(concept_id)
    except Exception:
        if stats is not None:
            stats["invalid_id_count"] = stats.get("invalid_id_count", 0) + 1
        return False

    key = (concept_id_int, ecl)
    if key in cache:
        if stats is not None:
            stats["cache_hit_count"] = stats.get("cache_hit_count", 0) + 1
        return cache[key]

    if stats is not None:
        stats["cache_miss_count"] = stats.get("cache_miss_count", 0) + 1
        stats["http_check_count"] = stats.get("http_check_count", 0) + 1

    ok_with_meta = concept_matches_ecl(
        concept_id=concept_id_int,
        ecl=ecl,
        base=SNOMED_BASE_URL,
        timeout=10,
        retries=1,
        return_error_count=True,
    )
    ok, err_count = ok_with_meta if isinstance(ok_with_meta, tuple) else (bool(ok_with_meta), 0)
    if err_count and stats is not None:
        stats["http_error_count"] = stats.get("http_error_count", 0) + int(err_count)
    cache[key] = bool(ok)
    return cache[key]


def filter_terms_by_attribute_range(
    terms: List[Dict[str, str]],
    attribute_key: str,
    attr_range_df: pd.DataFrame,
    ecl_cache: Dict[Tuple[int, str], bool],
    membership_map: Optional[Dict[int, bool]] = None,
    live_fallback: bool = True,
    content_type_id: Optional[int] = None,
    diagnostics: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, str]]:
    attribute_id = int(ATTRIBUTE_TABLE[attribute_key])
    ecls = [
        str(e).strip()
        for e in get_range_constraints_for_attribute(attribute_id=attribute_id, attr_range_df=attr_range_df, content_type_id=content_type_id)
        if str(e).strip()
    ]
    stats: Dict[str, int] = {}
    if diagnostics is not None:
        diagnostics["attribute_id"] = attribute_id
        diagnostics["content_type_id"] = content_type_id
        diagnostics["ecl_constraints"] = ecls
        diagnostics["ecl_constraints_count"] = len(ecls)
        diagnostics["raw_candidate_count"] = len(terms)
        diagnostics["prefilter_cache_enabled"] = membership_map is not None
        diagnostics["prefilter_hits"] = 0
        diagnostics["prefilter_misses"] = 0

    if not ecls:
        if diagnostics is not None:
            diagnostics["kept_candidate_count"] = len(terms)
            diagnostics["dropped_candidate_count"] = 0
            diagnostics["no_ecl_constraints"] = True
            diagnostics["http_error_count"] = 0
            diagnostics["cache_hit_count"] = 0
            diagnostics["cache_miss_count"] = 0
            diagnostics["http_check_count"] = 0
            diagnostics["invalid_id_count"] = 0
        return terms

    filtered: List[Dict[str, str]] = []
    for term in terms:
        cid_raw = term.get("id")
        try:
            cid = int(cid_raw)
        except Exception:
            continue

        is_match: Optional[bool] = None
        if membership_map is not None:
            if cid in membership_map:
                is_match = bool(membership_map[cid])
                if diagnostics is not None:
                    diagnostics["prefilter_hits"] = diagnostics.get("prefilter_hits", 0) + 1
            else:
                if diagnostics is not None:
                    diagnostics["prefilter_misses"] = diagnostics.get("prefilter_misses", 0) + 1

        if is_match is None and live_fallback:
            is_match = any(ecl_match_cached(cid, ecl, ecl_cache, stats=stats) for ecl in ecls)
        elif is_match is None:
            is_match = False

        if is_match:
            kept = dict(term)
            kept["id"] = cid
            filtered.append(kept)
    if diagnostics is not None:
        diagnostics["kept_candidate_count"] = len(filtered)
        diagnostics["dropped_candidate_count"] = max(0, len(terms) - len(filtered))
        diagnostics["no_ecl_constraints"] = False
        diagnostics["http_error_count"] = stats.get("http_error_count", 0)
        diagnostics["cache_hit_count"] = stats.get("cache_hit_count", 0)
        diagnostics["cache_miss_count"] = stats.get("cache_miss_count", 0)
        diagnostics["http_check_count"] = stats.get("http_check_count", 0)
        diagnostics["invalid_id_count"] = stats.get("invalid_id_count", 0)
    return filtered


def format_candidates_block(candidates: List[Dict[str, str]]) -> str:
    if not candidates:
        return "[]"
    return json.dumps(candidates, ensure_ascii=False, indent=2)


def build_payloads(
    mapped_rows: Iterable[Dict[str, Any]],
    verified_rows: Iterable[Dict[str, Any]],
    attr_range_df: pd.DataFrame,
    debug_payloads: bool = False,
    filter_by_range: bool = False,
    prefilter_memberships: Optional[Dict[str, Dict[int, bool]]] = None,
) -> List[Dict[str, Any]]:
    targets: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for row in verified_rows:
        if str(row.get("selected_snomed_id", "")) != "N/A":
            continue
        targets[row_key(row)] = row

    ecl_cache: Dict[Tuple[int, str], bool] = {}
    payloads: List[Dict[str, Any]] = []
    for row in mapped_rows:
        key = row_key(row)
        verified_row = targets.get(key)
        if verified_row is None:
            continue

        query_text = str(
            verified_row.get("query_text")
            or row.get("ci_text")
            or ""
        ).strip()

        focus_candidates = normalize_terms(row.get("condition_text_terms"))
        causative_candidates_raw = normalize_terms(row.get("substance_text_terms"))
        severity_candidates_raw = normalize_terms(row.get("severity_span_terms"))
        course_candidates_raw = normalize_terms(row.get("course_span_terms"))

        causative_diag: Dict[str, Any] = {}
        severity_diag: Dict[str, Any] = {}
        course_diag: Dict[str, Any] = {}

        if filter_by_range:
            causative_candidates = filter_terms_by_attribute_range(
                causative_candidates_raw,
                "causative_agent",
                attr_range_df=attr_range_df,
                ecl_cache=ecl_cache,
                membership_map=(prefilter_memberships or {}).get("causative_agent"),
                content_type_id=723594008,  # precoordinated only
                diagnostics=causative_diag,
            )
            severity_candidates = filter_terms_by_attribute_range(
                severity_candidates_raw,
                "severity",
                attr_range_df=attr_range_df,
                ecl_cache=ecl_cache,
                membership_map=(prefilter_memberships or {}).get("severity"),
                diagnostics=severity_diag,
            )
            course_candidates = filter_terms_by_attribute_range(
                course_candidates_raw,
                "clinical_course",
                attr_range_df=attr_range_df,
                ecl_cache=ecl_cache,
                membership_map=(prefilter_memberships or {}).get("clinical_course"),
                diagnostics=course_diag,
            )
        else:
            causative_candidates = list(causative_candidates_raw)
            severity_candidates = list(severity_candidates_raw)
            course_candidates = list(course_candidates_raw)
            if debug_payloads:
                causative_diag.update(
                    {
                        "range_filter_applied": False,
                        "raw_candidate_count": len(causative_candidates_raw),
                        "kept_candidate_count": len(causative_candidates),
                        "dropped_candidate_count": 0,
                    }
                )
                severity_diag.update(
                    {
                        "range_filter_applied": False,
                        "raw_candidate_count": len(severity_candidates_raw),
                        "kept_candidate_count": len(severity_candidates),
                        "dropped_candidate_count": 0,
                    }
                )
                course_diag.update(
                    {
                        "range_filter_applied": False,
                        "raw_candidate_count": len(course_candidates_raw),
                        "kept_candidate_count": len(course_candidates),
                        "dropped_candidate_count": 0,
                    }
                )

        payload: Dict[str, Any] = {
            "SPL_SET_ID": key[0],
            "item_index": key[1],
            "query_text": query_text,
            "span_text": query_text,
            "attribute_table_json": json.dumps(ATTRIBUTE_TABLE, ensure_ascii=False, indent=2),
            "focus_candidates": focus_candidates,
            "causative_candidates": causative_candidates,
            "severity_candidates": severity_candidates,
            "course_candidates": course_candidates,
            "focus_candidates_block": format_candidates_block(focus_candidates),
            "causative_candidates_block": format_candidates_block(causative_candidates),
            "severity_candidates_block": format_candidates_block(severity_candidates),
            "course_candidates_block": format_candidates_block(course_candidates),
        }
        if debug_payloads:
            payload.update(
                {
                    "causative_candidates_raw": causative_candidates_raw,
                    "severity_candidates_raw": severity_candidates_raw,
                    "course_candidates_raw": course_candidates_raw,
                    "payload_debug": {
                        "range_filter_applied": filter_by_range,
                        "focus_raw_candidate_count": len(focus_candidates),
                        "causative_filter": causative_diag,
                        "severity_filter": severity_diag,
                        "course_filter": course_diag,
                    },
                }
            )
        payloads.append(payload)

    return payloads


def parse_and_validate(raw_text: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    text = (raw_text or "").replace("<<END_JSON>>", "").strip()
    parsed = extract_json(text)
    if not isinstance(parsed, dict):
        parsed = {}

    focus_lookup = {str(c["id"]): c["label"] for c in payload["focus_candidates"]}
    attr_lookup = {
        "causative_agent": {str(c["id"]): c["label"] for c in payload["causative_candidates"]},
        "severity": {str(c["id"]): c["label"] for c in payload["severity_candidates"]},
        "clinical_course": {str(c["id"]): c["label"] for c in payload["course_candidates"]},
    }

    selected_focus_id = str(parsed.get("selected_focus_id", "N/A"))
    if selected_focus_id not in focus_lookup:
        selected_focus_id = "N/A"
        selected_focus_term = "N/A"
    else:
        selected_focus_term = focus_lookup[selected_focus_id]

    raw_fills = parsed.get("fills", {})
    if not isinstance(raw_fills, dict):
        raw_fills = {}

    fills: Dict[str, Dict[str, str]] = {}
    for key in ATTRIBUTE_KEYS:
        picked = str(raw_fills.get(key, "N/A"))
        if picked in attr_lookup[key]:
            fills[key] = {"id": picked, "term": attr_lookup[key][picked]}
        else:
            fills[key] = {"id": "N/A", "term": "N/A"}

    return {
        "selected_focus_id": selected_focus_id,
        "selected_focus_term": selected_focus_term,
        "fills": fills,
    }


def run_single_worker(
    args: argparse.Namespace,
    gpu_id: str,
    worker_idx: int = 0,
    num_workers: int = 1,
) -> List[Dict[str, Any]]:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    configure_torch_runtime(args.allow_dynamo)
    print(
        f"[postcord] worker={worker_idx} physical_gpu={gpu_id} "
        f"visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', '')}"
    )

    mapped_rows = read_jsonl(args.mapped_jsonl)
    verified_rows = read_jsonl(args.verified_jsonl)
    attr_range_df = load_snomed_dataframes(snomed_source_dir=args.snomed_source_dir)["attr_range"]
    prefilter_memberships = load_prefilter_memberships(args.prefilter_cache)

    all_payloads = build_payloads(
        mapped_rows=mapped_rows,
        verified_rows=verified_rows,
        attr_range_df=attr_range_df,
        debug_payloads=args.debug_payloads,
        filter_by_range=args.filter_by_range,
        prefilter_memberships=prefilter_memberships,
    )
    payloads = [
        payload
        for idx, payload in enumerate(all_payloads)
        if (idx % num_workers) == worker_idx
    ]
    chats = build_messages_from_iter(SYSTEM_PROMPT, USER_PROMPT, payloads)
    total = len(payloads)

    model = load_model_local(args.model_id)

    t0 = time.perf_counter()
    rows: List[Dict[str, Any]] = []
    print(f"[Worker {worker_idx} | GPU {gpu_id}] Starting {total} items")
    for shard_idx, (payload, chat) in enumerate(zip(payloads, chats)):
        global_idx = worker_idx + shard_idx * num_workers
        raw_out = model.generate(
            chat,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
        parsed = parse_and_validate(raw_out, payload)
        out_row: Dict[str, Any] = {
            "SPL_SET_ID": payload["SPL_SET_ID"],
            "item_index": payload["item_index"],
            "query_text": payload["query_text"],
            "selected_focus_id": parsed["selected_focus_id"],
            "selected_focus_term": parsed["selected_focus_term"],
            "fills": parsed["fills"],
            "_payload_index": global_idx,
        }
        if args.debug_payloads:
            out_row.update(
                {
                    "payload_debug": payload.get("payload_debug", {}),
                    "focus_candidates_raw": payload.get("focus_candidates", []),
                    "causative_candidates_raw": payload.get("causative_candidates_raw", []),
                    "severity_candidates_raw": payload.get("severity_candidates_raw", []),
                    "course_candidates_raw": payload.get("course_candidates_raw", []),
                    "focus_candidates_filtered": payload.get("focus_candidates", []),
                    "causative_candidates_filtered": payload.get("causative_candidates", []),
                    "severity_candidates_filtered": payload.get("severity_candidates", []),
                    "course_candidates_filtered": payload.get("course_candidates", []),
                }
            )
        rows.append(out_row)
        done = shard_idx + 1
        if done == 1 or done % 25 == 0 or done == total:
            elapsed = time.perf_counter() - t0
            pct = (100.0 * done / total) if total else 100.0
            print(
                f"[Worker {worker_idx} | GPU {gpu_id}] "
                f"{done}/{total} ({pct:.1f}%) elapsed={elapsed:.1f}s"
            )
    t1 = time.perf_counter()
    print(f"[GPU {gpu_id}] Processed {len(rows)} items in {t1 - t0:.2f}s")
    return rows


def run_multi_gpu(args: argparse.Namespace, gpu_ids: List[str]) -> None:
    tmp_dir = Path(args.out_jsonl).parent / ".postcord_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    workers = []
    worker_files = []
    for i, gpu in enumerate(gpu_ids):
        worker_out = tmp_dir / f"worker_{i}.jsonl"
        worker_files.append(worker_out)
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--mapped-jsonl",
            args.mapped_jsonl,
            "--verified-jsonl",
            args.verified_jsonl,
            "--out-jsonl",
            args.out_jsonl,
            "--model-id",
            args.model_id,
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--filter-by-range" if args.filter_by_range else "",
            "--debug-payloads" if args.debug_payloads else "",
            "--snomed-source-dir",
            args.snomed_source_dir,
            "--allow-dynamo" if args.allow_dynamo else "",
            "--gpu-ids",
            gpu,
            "--_worker",
            "--_worker-out",
            str(worker_out),
        ]
        if args.prefilter_cache:
            cmd.extend(["--prefilter-cache", args.prefilter_cache])
        cmd = [c for c in cmd if c]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        if not args.allow_dynamo:
            env["TORCHDYNAMO_DISABLE"] = "1"
            env["TORCH_COMPILE_DISABLE"] = "1"
        env["POSTCORD_WORKER_INDEX"] = str(i)
        env["POSTCORD_NUM_WORKERS"] = str(len(gpu_ids))
        workers.append(subprocess.Popen(cmd, env=env))

    exit_codes = [p.wait() for p in workers]
    if any(code != 0 for code in exit_codes):
        raise RuntimeError(f"One or more worker processes failed: {exit_codes}")

    merged: List[Dict[str, Any]] = []
    for wf in worker_files:
        if wf.exists():
            merged.extend(read_jsonl(str(wf)))

    merged.sort(key=lambda r: int(r.get("_payload_index", 10**18)))
    for row in merged:
        row.pop("_payload_index", None)
    write_jsonl(args.out_jsonl, merged)
    print(f"Merged {len(merged)} rows to {args.out_jsonl}")


def main() -> None:
    args = parse_args()
    configure_torch_runtime(args.allow_dynamo)
    check_snomed_connection()
    gpu_ids = parse_gpu_ids(args.gpu_ids)

    if not args._worker and len(gpu_ids) > 1:
        run_multi_gpu(args, gpu_ids)
        return

    worker_idx = int(os.environ.get("POSTCORD_WORKER_INDEX", "0"))
    num_workers = int(os.environ.get("POSTCORD_NUM_WORKERS", "1"))
    gpu_id = gpu_ids[0]

    rows = run_single_worker(
        args=args,
        gpu_id=gpu_id,
        worker_idx=worker_idx,
        num_workers=num_workers,
    )

    if args._worker:
        if not args._worker_out:
            raise ValueError("Worker mode requires --_worker-out")
        write_jsonl(args._worker_out, rows)
        print(f"[Worker {worker_idx}] Wrote {len(rows)} rows to {args._worker_out}")
    else:
        for row in rows:
            row.pop("_payload_index", None)
        write_jsonl(args.out_jsonl, rows)
        print(f"Wrote: {args.out_jsonl}")


if __name__ == "__main__":
    main()
