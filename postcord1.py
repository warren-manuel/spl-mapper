#!/usr/bin/env python3
"""Run post-coordination pattern inference from mapped SNOMED candidate hits."""

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

# Avoid torch.compile / dynamo instability for per-item generation workers.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

from VaxMapper.src.llm import build_messages_from_iter, extract_json, load_model_local
from VaxMapper.src.utils.snomed_utils import (
    extract_snomed_relationships,
    generate_scg_from_table,
    select_structured_exemplars,
)

SYSTEM = """
You are a SNOMED CT modeling assistant.

You will be given:
- A query contraindication
- Similar exemplar SNOMED CT concepts with their INFERRED relationship axioms grouped by role group and their SNOMED CT compositional grammar (SCG) expression.

TASK:
1) Identify the common modeling pattern used by the examples that can be applied to the query text.
2) Output a JSON object with:
   - pattern_found: true/false
   - expression: proposed SNOMED CT compositional grammar template
3) If examples do not support a consistent pattern, return pattern_found=false and expression="".

--------------------
OUTPUT FORMAT (STRICT)
--------------------
Output ONLY minified JSON followed by <<END_JSON>>.
Do not include markdown, prose, or extra fields.

Exact structure:
{{"query_text":"<original query>","pattern_found":true,"expression":"<proposed expression template or empty>"}}<<END_JSON>>
"""

USER = """
QUERY:
"{query_text}"

EXEMPLARS (each line includes concept id, label, inferred relationships, and SCG):
{hits_json}
"""


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mapped-jsonl", default="results/contra_ie2/mapped_hits.jsonl")
    ap.add_argument("--out-jsonl", default="results/contra_ie2/postcoord1_hits.jsonl")
    ap.add_argument("--model-id", default="google/medgemma-27b-text-it")
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--max-candidates-per-item", type=int, default=40)
    ap.add_argument("--exemplar-k", type=int, default=8)
    ap.add_argument("--min-attrs", type=int, default=2)
    ap.add_argument(
        "--concept-path",
        default="snomed_source/sct2_Concept_Snapshot_US1000124_20250901.txt",
        help="Path to SNOMED concept snapshot TXT",
    )
    ap.add_argument(
        "--description-path",
        default="snomed_source/sct2_Description_Snapshot-en_US1000124_20250901.txt",
        help="Path to SNOMED description snapshot TXT",
    )
    ap.add_argument(
        "--relationship-path",
        default="snomed_source/sct2_Relationship_Snapshot_US1000124_20250901.txt",
        help="Path to SNOMED relationship snapshot TXT",
    )
    ap.add_argument(
        "--gpu-ids",
        default="0",
        help="Comma-separated GPU IDs (first one is used for this script).",
    )
    ap.add_argument(
        "--allow-dynamo",
        action="store_true",
        help="Allow TorchDynamo/torch.compile (disabled by default for stability).",
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


def load_snomed_frames(
    concept_path: str,
    description_path: str,
    relationship_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    snomed_con_df = pd.read_csv(concept_path, sep="\t")
    snomed_des_df = pd.read_csv(description_path, sep="\t")
    snomed_rel_df = pd.read_csv(relationship_path, sep="\t")

    active_concepts = set(snomed_con_df.loc[snomed_con_df["active"] == 1, "id"].astype(int))

    concept_df = snomed_des_df[
        (snomed_des_df["conceptId"].isin(active_concepts))
        & (snomed_des_df["active"] == 1)
        & (snomed_des_df["typeId"] == 900000000000003001)
    ][["conceptId", "term"]].copy()
    concept_df["conceptId"] = concept_df["conceptId"].astype(str)

    rel = snomed_rel_df[snomed_rel_df["active"] == 1].copy()
    if "characteristicTypeId" in rel.columns:
        rel = rel[rel["characteristicTypeId"] == 900000000000011006].copy()
    rel = rel[["sourceId", "typeId", "destinationId", "relationshipGroup"]]
    rel["sourceId"] = rel["sourceId"].astype(int)
    rel["typeId"] = rel["typeId"].astype(str)
    rel["destinationId"] = rel["destinationId"].astype(str)
    rel["relationshipGroup"] = rel["relationshipGroup"].astype(int)
    rel = rel.set_index("sourceId", drop=True)

    return concept_df, rel


def collect_id_frequency(mapped_rows: Iterable[Dict[str, Any]]) -> Counter:
    counts: Counter = Counter()
    for row in mapped_rows:
        for item in row.get("items") or []:
            for hit in item.get("hits") or []:
                cid = hit.get("id")
                if cid is not None:
                    counts[str(cid)] += 1
    return counts


def rank_candidates(
    hits: List[Dict[str, Any]],
    id_counts: Counter,
    max_candidates: int,
) -> List[Dict[str, Any]]:
    by_id: Dict[str, Dict[str, Any]] = {}
    for hit in hits:
        cid = hit.get("id")
        if cid is None:
            continue
        cid_str = str(cid)
        score = float(hit.get("fused", 0.0) or 0.0)
        prev = by_id.get(cid_str)
        if prev is None or score > float(prev.get("fused", 0.0) or 0.0):
            by_id[cid_str] = {
                "id": cid_str,
                "label": hit.get("label", ""),
                "fused": score,
                "global_count": int(id_counts.get(cid_str, 0)),
            }

    ranked = sorted(
        by_id.values(),
        key=lambda h: (-h["global_count"], -h["fused"]),
    )
    return ranked[:max_candidates]


def build_exemplar_payloads(
    exemplars: List[Dict[str, Any]],
    concept_df: pd.DataFrame,
    snomed_rel_df: pd.DataFrame,
) -> List[Dict[str, Any]]:
    payloads: List[Dict[str, Any]] = []
    for ex in exemplars:
        cid = str(ex.get("id", ""))
        rels = extract_snomed_relationships(cid, snomed_rel_df, concept_df)
        scg = ""
        if rels:
            scg, _ = generate_scg_from_table(rels)
        payloads.append(
            {
                "id": cid,
                "label": str(ex.get("label", "")),
                "relationships": rels,
                "scg_expression": scg,
            }
        )
    return payloads


def iter_payloads(
    mapped_rows: List[Dict[str, Any]],
    concept_df: pd.DataFrame,
    snomed_rel_df: pd.DataFrame,
    max_candidates_per_item: int,
    exemplar_k: int,
    min_attrs: int,
) -> List[Dict[str, Any]]:
    id_counts = collect_id_frequency(mapped_rows)
    payloads: List[Dict[str, Any]] = []

    for row in mapped_rows:
        spl_set_id = str(row.get("SPL_SET_ID", ""))
        for item in row.get("items") or []:
            query_text = str(item.get("query_text", ""))
            hits = item.get("hits") or []

            ranked = rank_candidates(
                hits=hits,
                id_counts=id_counts,
                max_candidates=max_candidates_per_item,
            )
            exemplars = select_structured_exemplars(
                candidates=ranked,
                concept_df=concept_df,
                snomed_rel_df=snomed_rel_df,
                id_col="conceptId",
                term_col="term",
                top_k=exemplar_k,
                min_attrs=min_attrs,
            )
            if not exemplars:
                exemplars = ranked[:exemplar_k]

            exemplar_payloads = build_exemplar_payloads(exemplars, concept_df, snomed_rel_df)

            payloads.append(
                {
                    "SPL_SET_ID": spl_set_id,
                    "item_index": item.get("item_index"),
                    "query_text": query_text,
                    "hits_json": json.dumps(exemplar_payloads, ensure_ascii=False, indent=2),
                    "exemplars": exemplars,
                }
            )

    return payloads


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    if isinstance(value, (int, float)):
        return value != 0
    return False


def parse_and_validate(raw_text: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    text = (raw_text or "").replace("<<END_JSON>>", "").strip()
    parsed = extract_json(text)
    if not isinstance(parsed, dict):
        return {
            "query_text": payload["query_text"],
            "pattern_found": False,
            "expression": "",
        }

    pattern_found = parse_bool(parsed.get("pattern_found", False))
    expression = str(parsed.get("expression", "") or "")
    if not pattern_found:
        expression = ""

    return {
        "query_text": payload["query_text"],
        "pattern_found": pattern_found,
        "expression": expression,
    }


def main() -> None:
    def run_single_worker(
        args: argparse.Namespace,
        gpu_id: str,
        worker_idx: int = 0,
        num_workers: int = 1,
    ) -> List[Dict[str, Any]]:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        configure_torch_runtime(args.allow_dynamo)

        print("Loading SNOMED snapshots...")
        concept_df, snomed_rel_df = load_snomed_frames(
            concept_path=args.concept_path,
            description_path=args.description_path,
            relationship_path=args.relationship_path,
        )

        print(f"Reading mapped rows from {args.mapped_jsonl}...")
        mapped_rows = read_jsonl(args.mapped_jsonl)
        all_payloads = iter_payloads(
            mapped_rows=mapped_rows,
            concept_df=concept_df,
            snomed_rel_df=snomed_rel_df,
            max_candidates_per_item=args.max_candidates_per_item,
            exemplar_k=args.exemplar_k,
            min_attrs=args.min_attrs,
        )
        payloads = [
            payload
            for idx, payload in enumerate(all_payloads)
            if (idx % num_workers) == worker_idx
        ]
        chats = build_messages_from_iter(SYSTEM, USER, payloads)

        print(f"Loading model: {args.model_id}")
        model = load_model_local(args.model_id)

        results: List[Dict[str, Any]] = []
        total = len(payloads)
        t0 = time.perf_counter()
        print(f"[Worker {worker_idx} | GPU {gpu_id}] Starting {total} items")
        for shard_idx, (payload, chat) in enumerate(zip(payloads, chats), start=1):
            raw_out = model.generate(
                chat,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
            parsed = parse_and_validate(raw_out, payload)
            global_idx = worker_idx + (shard_idx - 1) * num_workers

            results.append(
                {
                    "SPL_SET_ID": payload["SPL_SET_ID"],
                    "item_index": payload["item_index"],
                    "query_text": payload["query_text"],
                    "pattern_found": parsed["pattern_found"],
                    "expression": parsed["expression"],
                    "num_exemplars": len(payload["exemplars"]),
                    "exemplar_ids": [str(e.get("id", "")) for e in payload["exemplars"]],
                    "_payload_index": global_idx,
                }
            )

            if shard_idx == 1 or shard_idx % 25 == 0 or shard_idx == total:
                elapsed = time.perf_counter() - t0
                pct = (100.0 * shard_idx / total) if total else 100.0
                print(
                    f"[Worker {worker_idx} | GPU {gpu_id}] "
                    f"{shard_idx}/{total} ({pct:.1f}%) elapsed={elapsed:.1f}s"
                )

        return results

    def run_multi_gpu(args: argparse.Namespace, gpu_ids: List[str]) -> None:
        tmp_dir = Path(args.out_jsonl).parent / ".postcord1_tmp"
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
                "--max-candidates-per-item",
                str(args.max_candidates_per_item),
                "--exemplar-k",
                str(args.exemplar_k),
                "--min-attrs",
                str(args.min_attrs),
                "--concept-path",
                args.concept_path,
                "--description-path",
                args.description_path,
                "--relationship-path",
                args.relationship_path,
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
            env["POSTCORD1_WORKER_INDEX"] = str(i)
            env["POSTCORD1_NUM_WORKERS"] = str(len(gpu_ids))
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

    args = parse_args()
    configure_torch_runtime(args.allow_dynamo)
    gpu_ids = parse_gpu_ids(args.gpu_ids)

    if not args._worker and len(gpu_ids) > 1:
        run_multi_gpu(args, gpu_ids)
        return

    worker_idx = int(os.environ.get("POSTCORD1_WORKER_INDEX", "0"))
    num_workers = int(os.environ.get("POSTCORD1_NUM_WORKERS", "1"))
    gpu_id = gpu_ids[0]

    results = run_single_worker(
        args=args,
        gpu_id=gpu_id,
        worker_idx=worker_idx,
        num_workers=num_workers,
    )
    if args._worker:
        if not args._worker_out:
            raise ValueError("Worker mode requires --_worker-out")
        write_jsonl(args._worker_out, results)
        print(f"[Worker {worker_idx}] Wrote {len(results)} rows to {args._worker_out}")
    else:
        for row in results:
            row.pop("_payload_index", None)
        write_jsonl(args.out_jsonl, results)
        print(f"Wrote: {args.out_jsonl}")


if __name__ == "__main__":
    main()
