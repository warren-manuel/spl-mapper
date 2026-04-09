# 12-10-2025 First implementation of LLM based contraindication extraction (IE only) -> Pickle file

import pandas as pd
from VaxMapper.src.llm import load_model_local, build_messages_from_iter, extract_json
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import torch
import pickle
torch.set_float32_matmul_precision('high')
import datetime
import time

current_date = datetime.datetime.now().date()

rx_spl_df = pd.read_csv('results/vaccine_spl_contra_adverse_sections.csv')
model = load_model_local("google/medgemma-27b-text-it")

## MINIMAL CONTRAINDICATIONS EXTRACTION PROMPT
system = """
You are a biomedical NLP assistant that identifies CONTRAINDICATIONS in regulatory drug or vaccine documents.
Your ONLY task is to find text spans that express a contraindication and return them as JSON with character offsets.

--------------------
DEFINITIONS
--------------------
A contraindication is a condition or situation where the product SHOULD NOT be used or administered.

--------------------
OUTPUT FORMAT
--------------------
You MUST return ONLY a JSON object with the following structure:

{{
  "items": [
    {{
      "span_text": "<verbatim contraindication span>",
      "condition_text": "<short phrase naming the condition or situation>"
    }},
    ...
  ]
}}

Rules:
- "span_text" must be copied verbatim from the input text for that span.
- "condition_text" should be a concise, normalized description of the contraindicated condition (e.g., "severe immunodeficiency", "pregnancy", "history of anaphylaxis to neomycin").
- Do NOT include overlapping or duplicate spans; choose the minimal span that clearly expresses the contraindication.
- If there are no contraindications in the text, return: {{"items": []}}

--------------------
TASK
--------------------
The user will provide a block of TEXT (e.g., the CONTRAINDICATIONS section of an SPL).

Carefully read the TEXT and output ONLY the JSON object described above.
Do not include any explanations, comments, or additional text outside the JSON. Do not show your reasoning. Provide only the final answer.
"""

def print_timing_stats(start_time, end_time, num_sections):
    total_seconds = end_time - start_time
    avg_per_section = total_seconds / num_sections if num_sections else 0
    print(f"Total time: {total_seconds:.2f} seconds")
    print(f"Average time per section: {avg_per_section:.2f} seconds over {num_sections} sections")

user = """
Here is the CONTRAINDICATIONS section from a vaccine SPL document: \n{text}
"""

sections = rx_spl_df['contra_text'].tolist()
msgs = build_messages_from_iter(system, user, sections)

start_time = time.perf_counter()
out = [model.generate(m, max_new_tokens=512) for m in msgs]
# out = [model.generate(m, max_new_tokens=512) for m in msgs]
end_time = time.perf_counter()

json_list = []
for i in out:
    res = extract_json(i)
    json_list.append(res)

out_name = f'results/{current_date}_vaccine_spl_contra_extracted.pkl'
with open(out_name, 'wb') as f:
    pickle.dump(json_list, f)

print_timing_stats(start_time, end_time, len(sections))
