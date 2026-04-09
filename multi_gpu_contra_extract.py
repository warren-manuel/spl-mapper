#!/usr/bin/env python3
"""
multi_gpu_contra_extract.py

Run LLM contraindication extraction on GPUs 0-3 using 4 worker processes.
Each worker loads the model on a single GPU and processes a shard of inputs.

Outputs: out_gpu{gpu_id}.jsonl in --out_dir
"""

import os
import json
import argparse
import multiprocessing as mp
from typing import Any, List, Tuple

import pandas as pd
import torch

from VaxMapper.src.llm import (
    load_model_local,
    build_messages_from_iter,
    extract_json,
    has_end_json_token,
    trim_after_end_json_token,
)


# -----------------------------
# PROMPTS (edit these)
# -----------------------------

# # SYSTEM_PROMPT = """
# You are a biomedical NLP assistant that identifies CONTRAINDICATIONS in regulatory drug or vaccine documents.
# Your ONLY task is to find text spans that express a contraindication and return them as JSON with character offsets.

# --------------------
# DEFINITIONS
# --------------------
# A contraindication is a condition or situation where the product SHOULD NOT be used or administered.

# An ATOMIC contraindication is the smallest indivisible contraindicated condition or situation.
# Each atomic contraindication must be expressed as its own item.

# --------------------
# CRITICAL INSTRUCTION: SPLITTING COORDINATED CONTRAINDICATIONS
# --------------------
# If a contraindication statement contains coordination (e.g., "or", "and", commas, lists), you MUST ALWAYS split it into MULTIPLE atomic contraindications whenever each item could stand alone.

# Examples:

# Text:
# "known hypersensitivity to drug A or to any of the other ingredients in drug A, or with known hypersensitivity to drug B analogs, including or such as drug C"

# You MUST extract THREE separate contraindications:
# 1) hypersensitivity to drug A
# 2) hypersensitivity to ingredients of drug A
# 3) hypersensitivity to drug B

# DO NOT add a separate contraindication for the example drug (drug C)

# Text:
# "moderate or severe condition X is a contraindication"

# You MUST extract TWO separate contraindications:
# 1) moderate condition X
# 2) severe condition X


# Do NOT merge coordinated items into a single generalized condition.
# Do NOT abstract away specific substances or categories.

# --------------------
# STRUCTURED FIELDS FOR EACH ATOMIC CONTRAINDICATION
# --------------------

# For each atomic contraindication you MUST return a JSON object with the following fields:

# Required fields:
# - "span_text": the full verbatim text span expressing this atomic contraindication.
# - "condition_text": a concise, normalized description of the contraindicated condition or situation.
# - "substance_span":
#   - The exact substring (verbatim from the input text) naming the causative substance, ingredient, product, or drug that the patient is hypersensitive/allergic/intolerant to.
#   - Example values: "drug A", "the active substance", "neomycin", "any component of the vaccine".
#   - If there is no specific substance mentioned, set "substance_span": null.

# - "severity_span":
#   - The exact substring describing severity of the contraindicated condition.
#   - Example values: "moderate", "severe", "life-threatening".
#   - If there is no explicit severity mentioned, set "severity_span": null.

# - "course_span":
#   - The exact substring describing course/onset of the condition.
#   - Example values: "acute", "chronic", "recurrent".
#   - If there is no explicit temporal course, set "course_span": null.

# - "age_constraint":
#   - A normalized brief description of any age restriction mentioned for the contraindicated population.
#   - Example values: "age <= 12 years", "infants < 6 weeks of age", "children and adolescents 2 through 17 years of age".
#   - If no age is mentioned, set "age_constraint": null.

# - "population_span":
#   - The exact substring describing the patient group or population (other than age) to which the contraindication applies.
#   - Example values: "immunocompromised individuals", "pregnant women", "patients with a history of Guillain-Barré syndrome".
#   - If not applicable, set "population_span": null.

# - "other_modifiers":
#   - A SHORT free-text phrase summarizing any additional clinically relevant modifiers that affect how this contraindication should be encoded (e.g., "with history of anaphylaxis", "following previous dose").
#   - If there are no additional modifiers, set "other_modifiers": null.

# Rules:
# - All span_* fields must be copied verbatim from the input text when present.
# - Do NOT invent substances, severities, courses, or age constraints that are not in the text.
# - If a field is not explicitly supported by the text, you MUST set it to null.


# --------------------
# OUTPUT FORMAT
# --------------------
# You MUST return ONLY a single line of MINIFIED JSON. 
# Do NOT include any markdown code fences (no ```).
# Do NOT include any explanations, reasoning, or extra text.
# Do NOT include the word "json".
# Do NOT include trailing commentary.

# The JSON must end with the exact token:

# <<END_JSON>>

# The JSON object MUST have the form:
# {{"items": [ ... ]}}

# Each element of "items" MUST be an object with the fields listed above:
# - "span_text"
# - "condition_text"
# - "substance_span"
# - "severity_span"
# - "course_span"
# - "age_constraint"
# - "population_span"
# - "other_modifiers"

# If there are no contraindications in the text, return: {{"items": []}}

# Rules:
# - "span_text" must be copied verbatim from the input text for that span.
# - "condition_text" should be a concise, normalized description of the contraindicated condition.
# - If there are no contraindications in the text, return: {{"items": []}}
# - The JSON must be valid.
# - The JSON must be minified (no newlines, no indentation).
# - The JSON must immediately be followed by <<END_JSON>>.
# - Nothing may appear after <<END_JSON>>.

# --------------------
# TASK
# --------------------
# The user will provide a block of TEXT (e.g., the CONTRAINDICATIONS section of an SPL).

# Carefully read the TEXT and output ONLY the JSON object described above.
# Do not include any explanations, comments, or additional text outside the JSON. Do not show your reasoning. Provide only the final answer.
# """

SYSTEM_PROMPT = """
You are a biomedical NLP assistant that identifies CONTRAINDICATIONS in regulatory drug or vaccine documents.

Your overall job has TWO STRICT SUBTASKS:

- TASK 1: Identify and list all ATOMIC contraindication spans in the text.
- TASK 2: Convert those ATOMIC contraindications into a structured JSON output with the specified fields.

You must complete TASK 1 fully and accurately before starting TASK 2.


--------------------
DEFINITIONS
--------------------
A contraindication is a condition or situation where the product SHOULD NOT be used or administered.

An ATOMIC contraindication is the smallest indivisible contraindicated condition or situation.
Each atomic contraindication must be expressed as its own item. An atomic contraindication MUST NOT contain any “and”, “or”, or list that can be split into smaller standalone contraindications.




--------------------
TASK 1: FIND ATOMIC CONTRAINDICATION SPANS
--------------------
Given the input TEXT:

1) Carefully read the TEXT and identify EVERY text span that expresses a contraindication.
2) For each span, aggressively split any coordinated phrase into multiple ATOMIC contraindications:
    - If a contraindication statement contains coordination (e.g., "or", "and", commas, bullet lists), 
    you MUST ALWAYS split it into MULTIPLE atomic contraindications whenever each item could stand alone.
        Examples:
            - Text:
                "known hypersensitivity to drug A or to any of the other ingredients in drug A, or 
                with known hypersensitivity to drug B analogs, including or such as drug C"
            You MUST extract THREE separate contraindications:
            1) hypersensitivity to drug A
            2) hypersensitivity to ingredients of drug A
            3) hypersensitivity to drug B
            DO NOT add a separate contraindication for the example drug (drug C).


            Do NOT merge coordinated items into a single generalized condition.
            Do NOT abstract away specific substances or categories.
   - For lists such as "A, B, and C", create one atomic contraindication for each of A, B, and C.
3) Atomic span construction rule (CRITICAL): Each atomic contraindication must be grounded in the TEXT using only words that appear in the TEXT, but it does NOT need to be a single contiguous substring when splitting coordination.
    - When a shared head is coordinated (e.g., “moderate or severe condition X is a contraindication”), you MUST create separate atomic items by recombining the shared head with each coordinated modifier using exact words from the TEXT:
        - moderate condition X
        - severe condition X
    - You may drop coordinating glue (“or”, “and”) and repeated filler words, but you MUST NOT paraphrase or introduce new clinical terms not present in the TEXT.

For the purpose of TASK 2, internally represent the result of TASK 1 as a list of atomic contraindications called ATOMIC_LIST.



--------------------
TASK 1 SELF-CHECK: COORDINATION SPLITTING
--------------------
Before moving to TASK 2, you MUST perform an internal self-check:

- Scan the TEXT again for any "and", "or", comma-separated lists, semicolons, or bullet lists inside contraindication sentences.
- Verify that for EVERY such coordinated phrase, you have created SEPARATE atomic contraindications for each coordinated item.
- If you find any coordinated phrase that is still represented as a single atomic contraindication, FIX your internal list by splitting it into multiple atomic contraindications.
- Especially check phrases like:
  - "moderate or severe ..."
  - "X, Y, and Z ..."
  - "hypersensitivity to A or to any components of B or to C ..."
You may ONLY proceed to TASK 2 once this self-check is satisfied.



--------------------
TASK 2: STRUCTURED FIELDS FOR EACH ATOMIC CONTRAINDICATION
--------------------
TASK 2 INPUT RULE (MOST IMPORTANT):
    - TASK 2 must be generated by iterating over ATOMIC_LIST created in TASK 1.
    - ALL structured fields for each atomic contraindication MUST be based ONLY on the TEXT of that atomic contraindication item. You MUST NOT introduce any information from the original TEXT that is not present in the atomic contraindication item itself.

For each atomic contraindication from TASK 1 you MUST return a JSON object with the following fields:

    - "ci_text": the atomic contraindication copied from the ATOMIC_LIST.
    - "condition_text": a concise, normalized description of ONLY the contraindicated condition or situation.
    - "substance_text":
    - The causative substance, ingredient, product, or drug that the patient is hypersensitive/allergic/intolerant to. Remove words like "any other".
    - Example values: "drug A", "active ingredients of the drug", "neomycin", "component of the vaccine".
    - If there is no specific substance mentioned, set "substance_text": null.

    - "severity_span":
    - The exact substring describing severity of the contraindicated condition.
    - Example values: "moderate", "severe", "life-threatening".
    - If there is no explicit severity mentioned, set "severity_span": null.

    - "course_span":
    - The exact substring describing course/onset of the condition.
    - Example values: "acute", "chronic", "recurrent".
    - If there is no explicit temporal course, set "course_span": null.

    - "age_constraint":
    - A normalized brief description of any age restriction mentioned for the contraindicated population.
    - Example values: "age <= 12 years", "infants < 6 weeks of age", "children and adolescents 2 through 17 years of age".
    - If no age is mentioned, set "age_constraint": null.

    - "population_text":
    - A normalized brief description of the patient group or population (other than age) to which the contraindication applies. Do not use "patient" or "individual" without any other qualifier.
    - Example values: "immunocompromised individuals", "pregnant women", "patients with a history of Guillain-Barré syndrome".
    - If not applicable, set "population_text": null.

    - "other_modifiers":
    - A SHORT free-text phrase summarizing any additional clinically relevant modifiers that affect how this contraindication should be encoded (e.g., "with history of anaphylaxis", "following previous dose").
    - If there are no additional modifiers, set "other_modifiers": null.

Rules:
- Do NOT invent substances, severities, courses, age constraints, or populations that are not in the text.
- If a field is not explicitly supported by the text, you MUST set it to null.

Your ONLY visible output to the user comes from TASK 2.
--------------------
OUTPUT FORMAT (TASK 2 OUTPUT ONLY)
--------------------
You MUST return ONLY a single line of MINIFIED JSON.
Do NOT include any markdown code fences.
Do NOT include any explanations, reasoning, or extra text.
Do NOT include the word "json".
Do NOT include trailing commentary.

The JSON must end with the exact token:

<<END_JSON>>

The JSON object MUST have the form:
{{"items": [ ... ]}}

Each element of "items" MUST be an object with the fields listed above:

- "ci_text"
- "condition_text"
- "substance_text"
- "severity_span"
- "course_span"
- "age_constraint"
- "population_text"
- "other_modifiers"

If there are no contraindications in the text, return: {{"items": []}}

Rules:
- "condition_text" should be a concise, normalized description of the contraindicated condition.
- The JSON must be valid.
- The JSON must be minified (no newlines, no indentation).
- The JSON must immediately be followed by <<END_JSON>>.
- Nothing may appear after <<END_JSON>>.




--------------------
TASK
--------------------
The user will provide a block of TEXT (e.g., the CONTRAINDICATIONS section of an SPL).

First, internally perform TASK 1 and the TASK 1 SELF-CHECK to ensure all coordinated contraindications are split into atomic contraindications.

Then, perform TASK 2 and output ONLY the JSON object described above, followed by <<END_JSON>>.
Do not include any explanations, comments, or additional text outside the JSON. Do not show your reasoning. Provide only the final answer.

"""

USER_PROMPT = """
Here is the CONTRAINDICATIONS section from a vaccine SPL document: \n{text}
"""


def worker_run(
    gpu_id: int,
    model_id: str,
    shard_items: List[Tuple[Any, str]],
    out_path: str,
    max_new_tokens: int = 512,
    retry_max_attempts: int = 2,
    retry_token_increment: int = 256,
):
    """
    Worker process pinned to a single GPU.
    """
    # Pin this process to exactly one GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(
        f"[extract] physical_gpu={gpu_id} visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', '')}"
    )

    # Optional: helps matmul performance on Ampere+ (safe)
    torch.set_float32_matmul_precision("high")

    # Load model onto this GPU
    model = load_model_local(model_id)

    sections = [text for _, text in shard_items]

    # Build messages for this shard
    # NOTE: build_messages_from_iter accepts tuples/dicts/strings.
    # If `sections` is list[str], it will map each to payload={"text": str(item)}.
    msgs = build_messages_from_iter(SYSTEM_PROMPT, USER_PROMPT, sections)

    # Run inference
    outputs = []
    for m in msgs:
        out_text = ""
        used_tokens = max_new_tokens
        attempts = max(1, int(retry_max_attempts))
        for attempt in range(attempts):
            run_tokens = max_new_tokens + (attempt * retry_token_increment)
            out_text = model.generate(m, max_new_tokens=run_tokens)
            used_tokens = run_tokens
            if has_end_json_token(out_text, require_terminal=False):
                break
        outputs.append({"text": out_text, "used_max_new_tokens": used_tokens})

    # Parse + write JSONL
    # We preserve the raw output too for debugging.
    with open(out_path, "w", encoding="utf-8") as f:
        for idx, ((item_index, src_text), out_rec) in enumerate(zip(shard_items, outputs)):
            raw_out = out_rec["text"]
            cleaned_out = trim_after_end_json_token(raw_out)
            parsed = extract_json(cleaned_out)
            rec = {
                "gpu": gpu_id,
                "item_index_local": idx,
                "item_index": item_index,
                "input_text": src_text,
                "raw_output": raw_out,
                "cleaned_output": cleaned_out,
                "has_end_json_token": has_end_json_token(raw_out, require_terminal=False),
                "has_terminal_end_json_token": has_end_json_token(raw_out, require_terminal=True),
                "used_max_new_tokens": out_rec["used_max_new_tokens"],
                "parsed_json": parsed,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def shard_list(items: List[Tuple[Any, str]], num_shards: int, shard_id: int) -> List[Tuple[Any, str]]:
    """
    Deterministic sharding: item i goes to shard (i % num_shards).
    """
    return [x for i, x in enumerate(items) if (i % num_shards) == shard_id]


def to_python_scalar(v: Any) -> Any:
    if pd.isna(v):
        return None
    if hasattr(v, "item"):
        return v.item()
    return v


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="google/medgemma-27b-text-it")
    parser.add_argument("--csv_path", type=str, required=True,
                        help="Path to CSV containing CONTRA_TEXT column")
    parser.add_argument("--text_col", type=str, default="CONTRA_TEXT")
    parser.add_argument("--index_col", type=str, required=True,
                        help="Column used as stable item index; duplicates are dropped (keep first)")
    parser.add_argument("--out_dir", type=str, default="outputs")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument(
        "--retry_max_attempts",
        type=int,
        default=2,
        help="Total attempts per LLM call when <<END_JSON>> is missing.",
    )
    parser.add_argument(
        "--retry_token_increment",
        type=int,
        default=256,
        help="Increase in max_new_tokens for each retry attempt.",
    )
    parser.add_argument("--gpus", type=str, default="0,1,2,3",
                        help="Comma-separated GPU ids to use (default: 0,1,2,3)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load sections
    df = pd.read_csv(args.csv_path)
    if args.index_col not in df.columns:
        raise ValueError(f"Column '{args.index_col}' not found in CSV. Columns: {list(df.columns)}")
    if args.text_col not in df.columns:
        raise ValueError(f"Column '{args.text_col}' not found in CSV. Columns: {list(df.columns)}")

    df = df.drop_duplicates(subset=[args.index_col], keep="first")

    item_indices = [to_python_scalar(v) for v in df[args.index_col].tolist()]
    sections = df[args.text_col].fillna("").astype(str).tolist()
    items = list(zip(item_indices, sections))

    gpu_ids = [int(x.strip()) for x in args.gpus.split(",") if x.strip() != ""]
    num_workers = len(gpu_ids)
    if num_workers == 0:
        raise ValueError("No GPUs specified.")

    # Spawn one worker per GPU
    ctx = mp.get_context("spawn")  # safer with CUDA than fork
    procs = []

    for w, gpu_id in enumerate(gpu_ids):
        shard = shard_list(items, num_workers, w)
        out_path = os.path.join(args.out_dir, f"out_gpu{gpu_id}.jsonl")

        p = ctx.Process(
            target=worker_run,
            args=(
                gpu_id,
                args.model_id,
                shard,
                out_path,
                args.max_new_tokens,
                args.retry_max_attempts,
                args.retry_token_increment,
            ),
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"A worker process exited with code {p.exitcode}")

    print(f"Done. Wrote {num_workers} shard files to: {args.out_dir}")


if __name__ == "__main__":
    main()
