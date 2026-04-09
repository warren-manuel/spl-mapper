# 12-16-2025 First implementation of LLM based contraindication verification  -> Pickle file

import pandas as pd
from VaxMapper.src.llm import load_model_local, build_messages_from_iter, extract_json, flatten, add_hits_json
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import torch
import pickle
torch.set_float32_matmul_precision('high')
import datetime
import time

current_date = datetime.datetime.now().date()

# rx_spl_df = pd.read_csv('results/vaccine_spl_contra_adverse_sections.csv')
similarity_pkl = 'results/2025-12-16_vaccine_spl_contra_rrf_results.pkl'
model = load_model_local("google/medgemma-27b-text-it")

# MINIMAL CONTRAINDICATION MAPPING PROMPT (SNOMED CT)
system = """
You are a biomedical terminology assistant that maps contraindication conditions to SNOMED CT concepts.

Your ONLY task is:
- Decide whether any of the provided SNOMED CT candidate concepts is a DIRECT MATCH for the given contraindication text.
- If there is a direct match, select exactly one SNOMED CT concept from the candidate list.
- If there is no direct match, return "N/A".

--------------------
DEFINITIONS
--------------------
A "contraindication condition" is a disease, clinical finding, physiological state, allergy/hypersensitivity, or patient characteristic that makes use of a product inadvisable.

A "DIRECT MATCH" between the query text and a SNOMED CT candidate means:
- They represent the SAME clinical meaning in typical clinical usage, not just overlapping words.
- Minor wording differences or synonyms are acceptable (e.g., "severe combined immunodeficiency" vs. "Severe combined immunodeficiency (disorder)").
- If the candidate is clearly broader or narrower in a clinically important way (e.g., "immunodeficiency" vs. "HIV infection"), it is NOT a direct match unless the query clearly implies that concept.
- Do NOT invent or use any SNOMED CT concept that is not in the candidate list.

--------------------
OUTPUT FORMAT
--------------------
You MUST return ONLY a single JSON object with exactly the following structure:

{{
  "query_text": "<the original contraindication text>",
  "selected_snomed_id": "<the SNOMED CT conceptId if a direct match exists, otherwise 'N/A'>",
  "selected_snomed_term": "<the SNOMED CT preferred term if a direct match exists, otherwise 'N/A'>"
}}

Rules:
- "selected_snomed_id" MUST be either one of the conceptIds from the candidate list OR the string "N/A".
- Do NOT return multiple IDs. Choose the single best direct match or "N/A".
- Do NOT include any explanations, reasoning, comments, or extra fields.
- Do NOT change the JSON keys or structure.

--------------------
TASK
--------------------
You will be given:
- A "QUERY" containing the contraindication text to be normalized.
- A "CANDIDATES" list that contains possible SNOMED CT concepts.

Carefully compare the QUERY with the CANDIDATES and output ONLY the JSON object described above. Do not show your reasoning. Provide only the final JSON answer.
"""

user = """
QUERY:
"{query_text}"

CANDIDATES (each line is one SNOMED CT concept):
{hits_json}

"""


def print_timing_stats(start_time, end_time, num_sections):
    total_seconds = end_time - start_time
    avg_per_section = total_seconds / num_sections if num_sections else 0
    print(f"Total time: {total_seconds:.2f} seconds")
    print(f"Average time per section: {avg_per_section:.2f} seconds over {num_sections} sections")

with open(similarity_pkl, 'rb') as f:
    output = pickle.load(f)    
records = flatten(output)
payloads = add_hits_json(records, output=['id','label'])
msgs = build_messages_from_iter(system, user, payloads)

start_time = time.perf_counter()
out = [model.generate(m, max_new_tokens=512) for m in msgs]
# out = [model.generate(m, max_new_tokens=512) for m in msgs]
end_time = time.perf_counter()

json_list = []
for i in out:
    res = extract_json(i)
    json_list.append(res)

out_name = f'results/{current_date}_vaccine_spl_contra_verified.pkl'
with open(out_name, 'wb') as f:
    pickle.dump(json_list, f)

print_timing_stats(start_time, end_time, len(payloads))
