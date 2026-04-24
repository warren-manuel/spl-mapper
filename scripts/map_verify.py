#!/usr/bin/env python3
"""
Verify SNOMED direct matches from mapped candidate hits.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

# Avoid torch.compile / dynamo instability for per-item generation workers.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

from src.llm.backends import build_messages_from_iter, extract_json, load_model_local


SYSTEM = """
You are a strict biomedical terminology validator. 
Your role is to determine if a pre-selected candidate is a LEXICAL and SEMANTIC identity match for a query.

Your ONLY task:
- Identify if a candidate is a DIRECT MATCH.
- If a match exists, select exactly ONE.
- Otherwise, return "N/A".

--------------------
DIRECT MATCH (STRICT)
--------------------
A candidate is a DIRECT MATCH only if it is an exact semantic equivalent. Do NOT 'bridge' concepts even if they are clinically related.

1) Lexical-Semantic Alignment:
- The candidate must match the specificity and naming of the query. Do NOT bridge concepts that use different primary terms. even if they are clinically related. 
- If the query text and candidate label belong to different levels of the hierarchy, return "N/A".
2) Temporal/Contextual Scope:
   - "Post-X" (after) is NOT a match for "X" (the event/procedure). 
   - "History of X" is NOT a match for "X".
   - If the query implies a state *after* an event, and the candidate is just the event, return "N/A".
3) Meaning Completeness:
   - Do NOT select a candidate that represents only a part of the query.
4) Semantic Type Guardrail:
   - If the query implies a 'Condition' (e.g., Post-surgical state), do NOT select a 'Procedure' concept (e.g., the surgery itself).
5) Hierarchical Granularity (Subsumption is NOT Identity):
   - A match must exist at the same level of specificity as the query.
   - If a candidate is a broader category (parent) or a more specific sub-type (child) of the query text, it is NOT a direct match.
   - Logic: If the query specifies a general mechanism and the candidate specifies a specific clinical manifestation, they are distinct. 
   - Rule: Lexical headers matter. Do not bridge "Condition X" to "Specific Type of X" even if they are clinically inseparable.

--------------------
CANDIDATE ELIGIBILITY FILTERS
--------------------
Unless the query explicitly describes a procedure or product, DO NOT select candidates whose label indicates:
- procedure (e.g., "administration", "vaccination", "(procedure)")
- overly broad parent concepts when query is specific

--------------------
OUTPUT FORMAT (STRICT)
--------------------
Return ONLY a single line of MINIFIED JSON followed immediately by the token <<END_JSON>>.
Do NOT use markdown fences (no ```).
Do NOT include explanations or extra fields.

Exact structure:
{{"query_text":"<original query>","selected_snomed_id":"<candidate id or N/A>","selected_snomed_term":"<candidate label or N/A>"}}<<END_JSON>>
"""

USER = """
QUERY:
"{query_text}"

CANDIDATES (each line is one SNOMED CT concept):
{hits_json}
"""


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mapped-jsonl", default="results/contra_ie2/mapped_hits.jsonl")
    ap.add_argument("--out-jsonl", default="results/contra_ie2/verified_hits.jsonl")
    ap.add_argument("--model-id", default="google/medgemma-27b-text-it")
    ap.add_argument("--max-new-tokens", type=int, default=256)
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


def clean_query_label(label: str) -> str:
    text = (label or "").strip()
    if text.endswith(")") and "(" in text:
        # Remove trailing semantic tag from SNOMED labels.
        text = text.rsplit("(", 1)[0].strip()
    return text


def iter_payloads(mapped_rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    New format only (mapped_hits_2 style):
    - one row per item
    - item_index from row
    - query_text from ci_text
    - hits from ci_text_terms
    """
    payloads: List[Dict[str, Any]] = []
    missing_ci_text = 0
    for row in mapped_rows:
        spl_set_id = str(row.get("SPL_SET_ID", ""))
        item_index = row.get("item_index")
        if item_index is None:
            raise ValueError(f"Missing item_index for SPL_SET_ID={spl_set_id}")

        hits = row.get("ci_text_terms") or [] # direct mapping only for the ci_text query field.
        filtered_hits = [{"id": h.get("id"), "label": h.get("label")} for h in hits]

        query_text = str(row.get("ci_text", "")).strip()
        if not query_text:
            # Keep execution possible when ci_text is absent in mapped_hits_2.
            query_text = clean_query_label(str(filtered_hits[0].get("label", ""))) if filtered_hits else ""
            missing_ci_text += 1

        payloads.append(
            {
                "SPL_SET_ID": spl_set_id,
                "item_index": item_index,
                "query_text": query_text,
                "hits": filtered_hits,
                "hits_json": json.dumps(filtered_hits, ensure_ascii=False, indent=2),
            }
        )

    if missing_ci_text:
        print(
            f"[iter_payloads] ci_text missing in {missing_ci_text} rows; "
            "used first ci_text_terms label as query_text fallback."
        )
    return payloads


def fallback_exact(query_text: str, hits: List[Dict[str, Any]]) -> Dict[str, str]:
    q = query_text.strip().lower()
    for h in hits:
        label = str(h.get("label", ""))
        label_core = label.rsplit("(", 1)[0].strip().lower() if "(" in label else label.lower()
        if q and q == label_core:
            return {
                "selected_snomed_id": str(h.get("id")),
                "selected_snomed_term": label,
            }
    return {"selected_snomed_id": "N/A", "selected_snomed_term": "N/A"}


def parse_and_validate(raw_text: str, payload: Dict[str, Any]) -> Dict[str, str]:
    text = (raw_text or "").replace("<<END_JSON>>", "").strip()
    parsed = extract_json(text)

    valid_ids = {str(h.get("id")): str(h.get("label", "")) for h in payload["hits"]}
    if not isinstance(parsed, dict):
        return fallback_exact(payload["query_text"], payload["hits"])

    selected_id = str(parsed.get("selected_snomed_id", "N/A"))
    selected_term = str(parsed.get("selected_snomed_term", "N/A"))

    if selected_id == "N/A":
        return {"selected_snomed_id": "N/A", "selected_snomed_term": "N/A"}

    if selected_id in valid_ids:
        expected_label = valid_ids[selected_id]
        return {
            "selected_snomed_id": selected_id,
            "selected_snomed_term": expected_label if expected_label else selected_term,
        }

    return fallback_exact(payload["query_text"], payload["hits"])


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
        import torch  # type: ignore
        import torch._dynamo as dynamo  # type: ignore

        dynamo.config.disable = True
        dynamo.config.suppress_errors = True
        if hasattr(dynamo.config, "fail_on_recompile_limit_hit"):
            dynamo.config.fail_on_recompile_limit_hit = False
    except Exception:
        pass


def run_single_worker(
    args: argparse.Namespace,
    gpu_id: str,
    worker_idx: int = 0,
    num_workers: int = 1,
) -> List[Dict[str, Any]]:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    configure_torch_runtime(args.allow_dynamo)
    print(
        f"[map_verify] worker={worker_idx} physical_gpu={gpu_id} "
        f"visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', '')}"
    )

    mapped_rows = read_jsonl(args.mapped_jsonl)
    all_payloads = iter_payloads(mapped_rows)
    payloads = [
        payload
        for idx, payload in enumerate(all_payloads)
        if (idx % num_workers) == worker_idx
    ]
    chats = build_messages_from_iter(SYSTEM, USER, payloads)
    total = len(payloads)

    model = load_model_local(args.model_id)

    t0 = time.perf_counter()
    verified_rows: List[Dict[str, Any]] = []
    print(f"[Worker {worker_idx} | GPU {gpu_id}] Starting {total} items")
    for shard_idx, (payload, chat) in enumerate(zip(payloads, chats)):
        global_idx = worker_idx + shard_idx * num_workers
        raw_out = model.generate(
            chat,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
        picked = parse_and_validate(raw_out, payload)
        verified_rows.append(
            {
                "SPL_SET_ID": payload["SPL_SET_ID"],
                "item_index": payload["item_index"],
                "query_text": payload["query_text"],
                "selected_snomed_id": picked["selected_snomed_id"],
                "selected_snomed_term": picked["selected_snomed_term"],
                "_payload_index": global_idx,
            }
        )
        done = shard_idx + 1
        if done == 1 or done % 25 == 0 or done == total:
            elapsed = time.perf_counter() - t0
            pct = (100.0 * done / total) if total else 100.0
            print(
                f"[Worker {worker_idx} | GPU {gpu_id}] "
                f"{done}/{total} ({pct:.1f}%) elapsed={elapsed:.1f}s"
            )
    t1 = time.perf_counter()

    print(f"[GPU {gpu_id}] Verified {len(verified_rows)} items in {t1 - t0:.2f}s")
    return verified_rows


def run_multi_gpu(args: argparse.Namespace, gpu_ids: List[str]) -> None:
    tmp_dir = Path(args.out_jsonl).parent / ".map_verify_tmp"
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
            "--out-jsonl",
            args.out_jsonl,
            "--model-id",
            args.model_id,
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--allow-dynamo" if args.allow_dynamo else "",
            "--gpu-ids",
            gpu,
            "--_worker",
            "--_worker-out",
            str(worker_out),
        ]
        cmd = [c for c in cmd if c]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        if not args.allow_dynamo:
            env["TORCHDYNAMO_DISABLE"] = "1"
            env["TORCH_COMPILE_DISABLE"] = "1"
        env["MAP_VERIFY_WORKER_INDEX"] = str(i)
        env["MAP_VERIFY_NUM_WORKERS"] = str(len(gpu_ids))
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
    print(f"Merged {len(merged)} verified rows to {args.out_jsonl}")


def main() -> None:
    args = parse_args()
    configure_torch_runtime(args.allow_dynamo)
    gpu_ids = parse_gpu_ids(args.gpu_ids)

    if not args._worker and len(gpu_ids) > 1:
        run_multi_gpu(args, gpu_ids)
        return

    worker_idx = int(os.environ.get("MAP_VERIFY_WORKER_INDEX", "0"))
    num_workers = int(os.environ.get("MAP_VERIFY_NUM_WORKERS", "1"))
    gpu_id = gpu_ids[0]

    verified_rows = run_single_worker(
        args,
        gpu_id=gpu_id,
        worker_idx=worker_idx,
        num_workers=num_workers,
    )

    if args._worker:
        if not args._worker_out:
            raise ValueError("Worker mode requires --_worker-out")
        write_jsonl(args._worker_out, verified_rows)
        print(f"[Worker {worker_idx}] Wrote {len(verified_rows)} rows to {args._worker_out}")
    else:
        for row in verified_rows:
            row.pop("_payload_index", None)
        write_jsonl(args.out_jsonl, verified_rows)
        print(f"Wrote: {args.out_jsonl}")


if __name__ == "__main__":
    main()
