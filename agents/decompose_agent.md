<|think|>
You are decomposing contraindication spans that contain coordinated phrases
into atomic contraindication concepts.

Apply the rules below to determine the items:

---

STAGE 1: SPLITTING
Apply these rules in order:

RULE 1 — SHARED MODIFIER WITH CONJUNCTION
Pattern: "[concept A] and [concept B]" sharing a head noun or modifier
Action: Emit one item per coordinate, distributing the shared modifier.
Example: "viral diseases of the eye and ear"
  → "viral diseases of the eye"
  → "viral diseases of the ear"

RULE 2 — DISJUNCTIVE CAUSATIVE AGENT
Pattern: "[condition] after/following/to [X] or [Y]"
Action: Emit one item per causative agent.
Example: "fever after taking aspirin or other NSAIDs"
  → "fever after taking aspirin"
  → "fever after taking other NSAIDs"

RULE 2 CAUTION — DISTRIBUTED HEAD NOUN: When the last item in the disjunction is a
compound noun ("[Z]-containing [W]", "[Z]-based [W]") and earlier items are bare
modifiers lacking the head noun "[W]", distribute "[W]" to ALL items.
Example: "allergic reaction after diphtheria toxoid, tetanus toxoid, or pertussis-containing vaccine"
  → "allergic reaction after diphtheria toxoid-containing vaccine"
  → "allergic reaction after tetanus toxoid-containing vaccine"
  → "allergic reaction after pertussis-containing vaccine"
NOT: "after diphtheria toxoid" (bare — missing the shared head noun "vaccine")

RULE 3 — ENUMERATED LIST
Pattern: "[concept A], [concept B], [concept C]" as a list
Action: Emit one item per listed concept.

RULE 4 — POPULATION CONJUNCTION
Pattern: "[condition] in [population A] and [population B]"
Action: Emit one item per population if populations are clinically distinct.
Example: "contraindicated in pregnant women and nursing mothers"
  → "use in pregnant women"
  → "use in nursing mothers"

RULE 5 — INGREDIENT LIST EXPANSION
Pattern: ci_text ends with ". Components: <pipe-delimited list>"
Action:
  1. Strip the ". Components: <list>" suffix to obtain the base_span.
  2. Split the suffix on " | " to get ingredient names.
  3. For each ingredient name, emit one item:
     - ci_text = base_span with "any component [of PRODUCT]" or "any ingredient [of PRODUCT]"
       replaced by the ingredient name (keep all surrounding clinical text intact).
     - split_applied = "RULE_5"
  4. Apply DEDUPLICATION GUARD across the emitted items.
  5. If the ingredient list is empty, fall through to RULE 0 on the base_span.

Example:
  Input ci_text: "anaphylaxis after any component of DAPTACEL. Components: BORDETELLA PERTUSSIS TOXOID ANTIGEN (INACTIVATED) | ALUMINUM PHOSPHATE | FORMALDEHYDE"
  base_span: "anaphylaxis after any component of DAPTACEL"
  → ci_text: "anaphylaxis after BORDETELLA PERTUSSIS TOXOID ANTIGEN (INACTIVATED)"  (RULE_5)
  → ci_text: "anaphylaxis after ALUMINUM PHOSPHATE"                                  (RULE_5)
  → ci_text: "anaphylaxis after FORMALDEHYDE"                                        (RULE_5)

RULE 0 — NO SPLIT
If none of the above patterns apply, return the span unchanged as a
single item.

GUARD: After splitting, verify each resulting item is a complete,
self-contained contraindication concept. If a split produces a fragment
(e.g. "severe hepatic" without a noun), do not split — return Rule 0.

Use the source_sentence field to resolve ambiguous modifier scope.

---

STAGE 2: NORMALIZATION

Apply to each item's ci_text. Apply all norms in order.

NORM 1 — PARENTHETICAL RESOLUTION
If "(e.g., X)" or "(including X)" names a MORE SPECIFIC clinical entity
than the head term, replace the head term with X.
  "Severe allergic reaction (e.g., anaphylaxis)" → "anaphylaxis"
If the parenthetical lists SUBTYPES of the head term, keep the head term
and drop the parenthetical.
  "Progressive neurologic disorder (including infantile spasms, uncontrolled
   epilepsy, progressive encephalopathy)" → "progressive neurologic disorder"
If uncertain: keep the head term.

NORM 2 — DROP ADMINISTRATIVE QUALIFIERS
Drop administrative qualifiers: ("known", "documented", "any", "previous")
Preserve as fills: "severe", "moderate", "mild", "acute", "active",
"untreated", "uncorrected", "progressive"

NORM 3 — DO NOT SUBSTITUTE ACROSS SPECIFICITY LEVELS
Do not replace clinical terms ex: "hypersensitivity" with "allergic reaction" or
"allergic reaction" with "anaphylaxis" except via NORM 1.
Use standard medical terminology only when you are confident in the equivalence.

---

DEDUPLICATION GUARD: After applying Stage 2 normalization, if two or more items
produce identical ci_text strings, collapse them to a single item — keep the first
occurrence. Never output duplicate ci_text values in the items array.

---

CRITICAL — OUTPUT FORMAT:
Output ONLY a JSON object in exactly this format, ending with <<END_JSON>>.
Do NOT output any explanation, markdown, bullet points, or text outside the JSON.

{ "items": [
    {
    "ci_text": "<normalized form after NORM 1-4>",
    "split_applied": "<RULE_N or RULE_0>",
    }
]}<<END_JSON>>

The JSON must end with the exact token: <<END_JSON>>
