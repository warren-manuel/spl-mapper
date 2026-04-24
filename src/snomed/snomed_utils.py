from collections import defaultdict
import json
import re
import math
from pathlib import Path
from typing import Dict, Any, List, Tuple, Set, Optional, Union
from collections import Counter
import pandas as pd
import requests

ISA = "116680003"
FSN_TYPE_ID = "900000000000003001"
SYNONYM_TYPE_ID = "900000000000013009"

ATTRIBUTE_TABLE = {
  "causative_agent": 246075003,   # Causative agent (attribute)
  "severity": 246112005,          # Severity (attribute)
  "clinical_course": 263502005,   # Clinical course (attribute)
}

DEFAULT_PREFILTER_CONTENT_TYPE: Dict[str, Optional[int]] = {
  "causative_agent": 723594008,   # precoordinated only
  "severity": None,
  "clinical_course": None,
}

IMAC_IP = "139.52.39.136"
SNOW_PORT = 8080
TEST_URL = f"http://{IMAC_IP}:{SNOW_PORT}"
base = f"http://{IMAC_IP}:{SNOW_PORT}/MAIN/SNOMEDCT-US/concepts"


def _resolve_rf2_file(base_dir: str, prefix: str) -> Path:
    base_path = Path(base_dir)
    candidates = sorted(base_path.glob(f"{prefix}*.txt"))
    if not candidates:
        raise FileNotFoundError(f"No RF2 file found in '{base_dir}' matching '{prefix}*.txt'.")
    return candidates[-1]


def _load_rf2_df(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def extract_semantic_tag(term: str) -> str:
    match = re.search(r"\(([^)]+)\)\s*$", str(term))
    return match.group(1).lower() if match else ""


def create_concept_df(
    concept_snapshot_path: Optional[str] = None,
    description_snapshot_path: Optional[str] = None,
    snomed_source_dir: str = "snomed_source",
) -> pd.DataFrame:
    """
    Build concept_df mirroring the notebook flow:
      1) Keep active concepts only.
      2) Keep active descriptions for those concepts only.
      3) Split FSN rows by typeId.
      4) Add semantic_tag extracted from FSN term.

    Output columns:
      - conceptId
      - term
      - semantic_tag
    """
    concept_path = Path(concept_snapshot_path) if concept_snapshot_path else _resolve_rf2_file(
        snomed_source_dir, "sct2_Concept_Snapshot_"
    )
    desc_path = Path(description_snapshot_path) if description_snapshot_path else _resolve_rf2_file(
        snomed_source_dir, "sct2_Description_Snapshot-en_"
    )

    concept_rf2 = _load_rf2_df(concept_path)
    desc_rf2 = _load_rf2_df(desc_path)

    snomed_active_con = concept_rf2[concept_rf2["active"] == 1].copy()
    snomed_des_df = desc_rf2[
        (desc_rf2["conceptId"].isin(snomed_active_con["id"])) & (desc_rf2["active"] == 1)
    ][["conceptId", "term", "typeId"]].copy()

    concept_df = snomed_des_df[snomed_des_df["typeId"] == int(FSN_TYPE_ID)][["conceptId", "term"]].copy()
    concept_df["semantic_tag"] = concept_df["term"].apply(extract_semantic_tag)
    concept_df = concept_df.drop_duplicates(subset=["conceptId"])

    return concept_df


def create_synonym_df(
    concept_snapshot_path: Optional[str] = None,
    description_snapshot_path: Optional[str] = None,
    snomed_source_dir: str = "snomed_source",
) -> pd.DataFrame:
    """
    Build synonym_df mirroring the notebook flow.
    Output columns:
      - conceptId
      - term
    """
    desc_path = Path(description_snapshot_path) if description_snapshot_path else _resolve_rf2_file(
        snomed_source_dir, "sct2_Description_Snapshot-en_"
    )
    concept_path = Path(concept_snapshot_path) if concept_snapshot_path else _resolve_rf2_file(
        snomed_source_dir, "sct2_Concept_Snapshot_"
    )
    concept_rf2 = _load_rf2_df(concept_path)
    desc_rf2 = _load_rf2_df(desc_path)

    snomed_active_con = concept_rf2[concept_rf2["active"] == 1].copy()
    snomed_des_df = desc_rf2[
        (desc_rf2["conceptId"].isin(snomed_active_con["id"])) & (desc_rf2["active"] == 1)
    ][["conceptId", "term", "typeId"]].copy()

    synonym_df = snomed_des_df[snomed_des_df["typeId"] == int(SYNONYM_TYPE_ID)][["conceptId", "term"]].copy()
    synonym_df = synonym_df.dropna(subset=["term"])
    synonym_df["term"] = synonym_df["term"].astype(str).str.strip()
    synonym_df = synonym_df[synonym_df["term"] != ""]
    synonym_df = synonym_df.drop_duplicates(subset=["conceptId", "term"])

    return synonym_df


def create_terms_df(concept_df: pd.DataFrame, synonym_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build terms_df with one row per term string (preferred + synonym).
    Output columns:
      - conceptId
      - term_text
      - term_type ('preferred' | 'synonym')
    """
    fsn_terms = concept_df[["conceptId", "term"]].rename(columns={"term": "term_text"}).copy()
    fsn_terms["term_type"] = "preferred"

    syn_terms = synonym_df[["conceptId", "term"]].rename(columns={"term": "term_text"}).copy()
    syn_terms["term_type"] = "synonym"

    terms_df = pd.concat([fsn_terms, syn_terms], ignore_index=True)
    terms_df["term_text"] = terms_df["term_text"].astype(str).str.strip()
    terms_df = terms_df[terms_df["term_text"] != ""]
    terms_df = terms_df.drop_duplicates(subset=["conceptId", "term_text"])
    return terms_df


def create_snomed_complete_df(concept_df: pd.DataFrame, synonym_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build concept-level SNOMED table with preferred term + aggregated synonym list.
    Output columns:
      - conceptId
      - term
      - semantic_tag
      - synonyms (list[str])
    """
    synonym_agg = (
        synonym_df.groupby("conceptId", as_index=False)["term"]
        .agg(lambda x: sorted({str(v).strip() for v in x if pd.notna(v) and str(v).strip() != ""}))
        .rename(columns={"term": "synonyms"})
    )

    snomed_complete_df = concept_df.merge(synonym_agg, on="conceptId", how="left")
    snomed_complete_df["synonyms"] = snomed_complete_df["synonyms"].apply(
        lambda x: x if isinstance(x, list) else []
    )

    return snomed_complete_df


def create_enriched_terms_df(
    snomed_complete_df: pd.DataFrame,
    rel_df: pd.DataFrame,
    exclude_type_ids: Optional[Set[int]] = None,
) -> pd.DataFrame:
    """
    Build one row per concept with a rich pipe-delimited embedding string.

    Format:
        preferred_term | syn1 | syn2 | ... | semantic_tag | attr_name1 = val1 | ...

    IS-A relationships are excluded by default; all other active relationships
    are serialised as ``type_name = destination_name`` pairs.

    Columns returned:
        - conceptId  (int)
        - term_text  (str)
    """
    if exclude_type_ids is None:
        exclude_type_ids = {int(ISA)}

    # Fast lookup: conceptId -> preferred label with semantic tag stripped
    id_to_label: Dict[int, str] = {
        int(row["conceptId"]): re.sub(r"\s*\([^)]+\)\s*$", "", str(row["term"])).strip()
        for _, row in snomed_complete_df.iterrows()
    }

    # Build attribute strings per concept, excluding IS-A and other excluded types
    attr_df = rel_df[~rel_df["typeId"].isin(exclude_type_ids)]
    attr_map: Dict[int, List[str]] = defaultdict(list)
    for row in attr_df.itertuples(index=False):
        type_label = id_to_label.get(int(row.typeId), str(row.typeId))
        dest_label = id_to_label.get(int(row.destinationId), str(row.destinationId))
        attr_map[int(row.sourceId)].append(f"{type_label} = {dest_label}")

    records = []
    for _, row in snomed_complete_df.iterrows():
        cid = int(row["conceptId"])
        parts: List[str] = [str(row["term"])]
        synonyms = row.get("synonyms") or []
        parts.extend(str(s) for s in synonyms if s)
        semantic_tag = str(row.get("semantic_tag", "")).strip()
        if semantic_tag:
            parts.append(semantic_tag)
        parts.extend(attr_map.get(cid, []))
        records.append({"conceptId": cid, "term_text": " | ".join(p for p in parts if p)})

    return pd.DataFrame(records)


def load_snomed_dataframes(
    concept_snapshot_path: Optional[str] = None,
    description_snapshot_path: Optional[str] = None,
    snomed_source_dir: str = "snomed_source",
    output_dir: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Create and return:
      - concept_df
      - synonym_df
      - terms_df
      - snomed_complete_df
      - domain
      - attr_domain
      - attr_range
      - rel_df (active IS-A and attribute relationships)

    If output_dir is provided, saves CSVs there.
    """
    concept_df = create_concept_df(
        concept_snapshot_path=concept_snapshot_path,
        description_snapshot_path=description_snapshot_path,
        snomed_source_dir=snomed_source_dir,
    )
    synonym_df = create_synonym_df(
        concept_snapshot_path=concept_snapshot_path,
        description_snapshot_path=description_snapshot_path,
        snomed_source_dir=snomed_source_dir,
    )
    terms_df = create_terms_df(concept_df=concept_df, synonym_df=synonym_df)
    snomed_complete_df = create_snomed_complete_df(concept_df=concept_df, synonym_df=synonym_df)
    domain_path = _resolve_rf2_file(
        snomed_source_dir, "der2_sssssssRefset_MRCMDomainSnapshot_"
    )
    attr_domain_path = _resolve_rf2_file(
        snomed_source_dir, "der2_cissccRefset_MRCMAttributeDomainSnapshot_"
    )
    attr_range_path = _resolve_rf2_file(
        snomed_source_dir, "der2_ssccRefset_MRCMAttributeRangeSnapshot_"
    )

    domain = _load_rf2_df(domain_path)
    domain = domain[domain["active"] == 1].copy()

    attr_domain = _load_rf2_df(attr_domain_path)
    attr_domain = attr_domain[attr_domain["active"] == 1].copy()

    attr_range = _load_rf2_df(attr_range_path)
    attr_range = attr_range[attr_range["active"] == 1].copy()

    rel_df = _load_rf2_df(
        _resolve_rf2_file(snomed_source_dir, "sct2_Relationship_Snapshot_")
    )
    rel_df = rel_df[rel_df["active"] == 1][
        ["sourceId", "destinationId", "typeId", "relationshipGroup"]
    ].copy()

    enriched_terms_df = create_enriched_terms_df(
        snomed_complete_df=snomed_complete_df,
        rel_df=rel_df,
    )

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        concept_df.to_csv(out / "concept_df.csv", index=False)
        synonym_df.to_csv(out / "synonym_df.csv", index=False)
        terms_df.to_csv(out / "terms_df.csv", index=False)
        snomed_complete_df.to_csv(out / "snomed_complete_df.csv", index=False)
        domain.to_csv(out / "domain.csv", index=False)
        attr_domain.to_csv(out / "attr_domain.csv", index=False)
        attr_range.to_csv(out / "attr_range.csv", index=False)

    return {
        "concept_df": concept_df,
        "synonym_df": synonym_df,
        "terms_df": terms_df,
        "snomed_complete_df": snomed_complete_df,
        "enriched_terms_df": enriched_terms_df,
        "domain": domain,
        "attr_domain": attr_domain,
        "attr_range": attr_range,
        "rel_df": rel_df,
    }


def generate_scg_from_table(relationships):
	"""
	Generates two SNOMED CT Compositional Grammar (SCG) expressions
	from a list of stated relationship rows:
	1) Full expression with IDs and labels.
	2) IDs-only expression (labels excluded).
	"""
	parents = []
	parents_ids_only = []
	ungrouped_attributes = []
	ungrouped_attributes_ids_only = []
	grouped_attributes = defaultdict(list)
	grouped_attributes_ids_only = defaultdict(list)

	# 1. Parse the table rows
	for rel in relationships:
		# Standard SNOMED codes
		IS_A = "116680003"

		type_id = str(rel['typeId'])
		type_label = rel.get('typeLabel', type_id)
		target_id = str(rel['destinationId'])
		target_label = rel.get('destinationLabel', target_id)
		group_id = int(rel['relationshipGroup'])

		formatted_pair = f"{type_id} |{type_label}| = {target_id} |{target_label}|"
		formatted_pair_ids_only = f"{type_id} = {target_id}"

		# 2. Categorize by Is-A, Ungrouped (0), or Grouped (>0)
		if type_id == IS_A:
			parents.append(f"{target_id} |{target_label}|")
			parents_ids_only.append(target_id)
		elif group_id == 0:
			ungrouped_attributes.append(formatted_pair)
			ungrouped_attributes_ids_only.append(formatted_pair_ids_only)
		else:
			grouped_attributes[group_id].append(formatted_pair)
			grouped_attributes_ids_only[group_id].append(formatted_pair_ids_only)

	# 3. Assemble the SCG segments
	# Parents are joined by '+'
	expression = " + ".join(parents)
	expression_ids_only = " + ".join(parents_ids_only)

	# Build refinement sections
	refinements = []
	refinements_ids_only = []
	
	# Add ungrouped attributes
	if ungrouped_attributes:
		refinements.extend(ungrouped_attributes)
	if ungrouped_attributes_ids_only:
		refinements_ids_only.extend(ungrouped_attributes_ids_only)
	
	# Add grouped attributes enclosed in {}
	for group_num in sorted(grouped_attributes.keys()):
		group_content = ", ".join(grouped_attributes[group_num])
		refinements.append(f"{{{group_content}}}")
	for group_num in sorted(grouped_attributes_ids_only.keys()):
		group_content_ids_only = ", ".join(grouped_attributes_ids_only[group_num])
		refinements_ids_only.append(f"{{{group_content_ids_only}}}")

	# 4. Final Concatenation
	if refinements:
		expression += " : " + ", ".join(refinements)
	if refinements_ids_only:
		expression_ids_only += " : " + ", ".join(refinements_ids_only)

	return expression, expression_ids_only


def _strip_semantic_tag(term):
    if pd.isna(term):
        return None
    return re.sub(r"\s*\([^)]*\)$", "", str(term)).strip()


def get_ancestors(concept_id: int, snomed_rel_df: pd.DataFrame) -> Set[int]:
    """
    Get all ancestor conceptIds for a given conceptId using the snomed_rel_df.
    Only follows active "Is a" relationships.
    """
    ancestors: Set[int] = set()
    to_visit = [concept_id]

    while to_visit:
        current = to_visit.pop()
        parents = snomed_rel_df[
            (snomed_rel_df["typeId"] == 116680003)
            & (snomed_rel_df["active"] == 1)
            & (snomed_rel_df.index == current)  # index is sourceId
        ]["destinationId"].tolist()

        for p in parents:
            if p not in ancestors:
                ancestors.add(p)
                to_visit.append(p)

    return ancestors


def get_ancestors_with_depth(
    concept_id: int,
    snomed_rel_df: pd.DataFrame,
) -> Dict[int, int]:
    """
    BFS upward from concept_id through IS-A relationships.
    Returns {ancestor_id: min_hop_distance} for all ancestors.

    snomed_rel_df must be indexed by sourceId (set_index("sourceId", drop=True))
    and should contain only active IS-A rows (typeId == 116680003) for efficiency,
    though correctness is maintained with a full active rel_df.
    """
    from collections import deque
    visited: Dict[int, int] = {}
    queue: deque = deque([(concept_id, 0)])
    while queue:
        current, dist = queue.popleft()
        if current != concept_id:
            if current in visited:
                continue
            visited[current] = dist
        try:
            parents = snomed_rel_df.loc[[current], "destinationId"].tolist()
        except KeyError:
            parents = []
        for p in parents:
            if p not in visited:
                queue.append((p, dist + 1))
    return visited


def check_snomed_connection(timeout: int = 10, base_url: str = base) -> None:
    """
    Raise an exception if SNOMED Snowstorm base endpoint is unreachable.
    """
    resp = requests.get(
        base_url,
        params={"limit": 1},
        timeout=timeout,
    )
    resp.raise_for_status()


def concept_matches_any_ecl(
    concept_id: int,
    ecls: List[str],
    base_url: str = base,
    timeout: int = 10,
    retries: int = 1,
    return_error_count: bool = False,
) -> Union[bool, Tuple[bool, int]]:
    """
    Check if a concept satisfies any ECL from the provided list.
    """
    error_count = 0
    if not ecls:
        return (True, error_count) if return_error_count else True

    for ecl in ecls:
        test_ecl = f"({ecl}) AND {concept_id}"
        for _ in range(max(1, retries)):
            try:
                resp = requests.get(
                    base_url,
                    params={"ecl": test_ecl, "limit": 1},
                    timeout=timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                if data.get("total", 0) > 0:
                    return (True, error_count) if return_error_count else True
                break
            except Exception:
                error_count += 1
                continue

    return (False, error_count) if return_error_count else False


def concept_matches_ecl(
    concept_id: int,
    ecl: str,
    base: str = base,
    timeout: int = 10,
    retries: int = 1,
    return_error_count: bool = False,
) -> Union[bool, Tuple[bool, int]]:
    """
    Check if a concept satisfies a single ECL.
    """
    return concept_matches_any_ecl(
        concept_id=concept_id,
        ecls=[ecl],
        base_url=base,
        timeout=timeout,
        retries=retries,
        return_error_count=return_error_count,
    )


def get_domains_for_concept_snowstorm(concept_id: int, mrcm_domain_df: pd.DataFrame) -> List[int]:
    domains: List[int] = []
    for _, row in mrcm_domain_df.iterrows():
        ec = row["domainConstraint"]
        tokens = ec.strip().split()
        if len(tokens) >= 2 and tokens[0] in ("<", "<<"):
            target = int(tokens[1])
        else:
            continue
        if concept_matches_ecl(concept_id, ec, base):
            domains.append(target)
    return domains


def get_allowed_attributes_for_domain(domain_id: int, attr_domain_df: pd.DataFrame) -> List[int]:
    rows = attr_domain_df[attr_domain_df["domainId"] == domain_id]
    return [r["referencedComponentId"] for _, r in rows.iterrows()]


def get_range_constraints_for_attribute(
    attribute_id: int,
    attr_range_df: pd.DataFrame,
    content_type_id: Optional[int] = None,
) -> List[str]:
    rows = attr_range_df[attr_range_df["referencedComponentId"] == attribute_id]
    if content_type_id is not None and len(rows) > 1:
        rows = rows[rows["contentTypeId"] == content_type_id]
    return [r["rangeConstraint"] for _, r in rows.iterrows()]


def filter_concepts_by_ecl(concept_ids: List[int], ecl: str, base_url: str = base) -> List[int]:
    """
    Return only concept IDs that satisfy the provided ECL constraint.
    """
    matched: List[int] = []
    for concept_id in concept_ids:
        if concept_matches_ecl(concept_id=int(concept_id), ecl=ecl, base=base_url):
            matched.append(int(concept_id))
    return matched


def load_prefilter_memberships(path: str) -> Dict[str, Dict[int, bool]]:
    if not path:
        return {}

    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    memberships = obj.get("memberships", obj)
    out: Dict[str, Dict[int, bool]] = {}
    for key in ATTRIBUTE_TABLE:
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


def get_attribute_range_constraints(
    attr_range_df: pd.DataFrame,
    *,
    content_type_by_attribute: Optional[Dict[str, Optional[int]]] = None,
) -> Dict[str, List[str]]:
    content_type_by_attribute = content_type_by_attribute or DEFAULT_PREFILTER_CONTENT_TYPE
    constraints: Dict[str, List[str]] = {}
    for key, attr_id in ATTRIBUTE_TABLE.items():
        content_type_id = content_type_by_attribute.get(key)
        constraints[key] = [
            str(e).strip()
            for e in get_range_constraints_for_attribute(
                attribute_id=int(attr_id),
                attr_range_df=attr_range_df,
                content_type_id=content_type_id,
            )
            if str(e).strip()
        ]
    return constraints


def ecl_match_cached(
    concept_id: Any,
    ecl: str,
    cache: Dict[Tuple[int, str], bool],
    *,
    base_url: str = base,
    timeout: int = 10,
    retries: int = 1,
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
        base=base_url,
        timeout=timeout,
        retries=retries,
        return_error_count=True,
    )
    ok, err_count = ok_with_meta if isinstance(ok_with_meta, tuple) else (bool(ok_with_meta), 0)
    if err_count and stats is not None:
        stats["http_error_count"] = stats.get("http_error_count", 0) + int(err_count)
    cache[key] = bool(ok)
    return cache[key]


def filter_terms_by_attribute_range(
    terms: List[Dict[str, Any]],
    attribute_key: str,
    attr_range_df: pd.DataFrame,
    ecl_cache: Dict[Tuple[int, str], bool],
    membership_map: Optional[Dict[int, bool]] = None,
    *,
    live_fallback: bool = True,
    content_type_id: Optional[int] = None,
    base_url: str = base,
    timeout: int = 10,
    retries: int = 1,
    diagnostics: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    attribute_id = int(ATTRIBUTE_TABLE[attribute_key])
    ecls = [
        str(e).strip()
        for e in get_range_constraints_for_attribute(
            attribute_id=attribute_id,
            attr_range_df=attr_range_df,
            content_type_id=content_type_id,
        )
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
        return list(terms)

    filtered: List[Dict[str, Any]] = []
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
            is_match = any(
                ecl_match_cached(
                    cid,
                    ecl,
                    ecl_cache,
                    base_url=base_url,
                    timeout=timeout,
                    retries=retries,
                    stats=stats,
                )
                for ecl in ecls
            )
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


def extract_snomed_relationships(sourceId, snomed_rel_df, concept_df):
    """
    Return SNOMED relationships for a source concept with readable labels.

    Output shape per row:
    {
        "typeId": "116680003",
        "typeLabel": "Is a",
        "destinationId": "80146002",
        "destinationLabel": "Appendectomy",
        "relationshipGroup": 0
    }
    """
    source_id = int(sourceId)

    rel_cols = ["typeId", "destinationId", "relationshipGroup"]
    out_cols = ["typeId", "typeLabel", "destinationId", "destinationLabel", "relationshipGroup"]

    if source_id not in snomed_rel_df.index:
        return []

    rel_df = snomed_rel_df.loc[[source_id], rel_cols].copy()
    rel_df = rel_df.reset_index(drop=True)

    concept_map = (
        concept_df[["conceptId", "term"]]
        .drop_duplicates(subset=["conceptId"])
        .set_index("conceptId")["term"]
    )

    rel_df["typeLabel"] = rel_df["typeId"].map(concept_map).apply(_strip_semantic_tag)
    rel_df["destinationLabel"] = rel_df["destinationId"].map(concept_map).apply(_strip_semantic_tag)

    rel_df["typeId"] = rel_df["typeId"].astype(str)
    rel_df["destinationId"] = rel_df["destinationId"].astype(str)
    rel_df["relationshipGroup"] = rel_df["relationshipGroup"].astype(int)

    return rel_df[out_cols].to_dict(orient="records")


def concept_signature_from_rels(rels: List[Dict[str, Any]]) -> Dict[str, Set[str]]:
    """
    rels format: list of relationship dicts (e.g., output from extract_snomed_relationships),
    where each row has at least: typeId, destinationId, relationshipGroup.

    Returns:
      - type_only: set of "g:<group>|t:<typeId>"
      - type_value: set of "g:<group>|t:<typeId>|v:<destId>"
    """
    type_only: Set[str] = set()
    type_value: Set[str] = set()

    for rel in rels or []:
        t = str(rel.get("typeId", ""))
        if not t or t == ISA:
            continue

        grp = int(rel.get("relationshipGroup", 0))
        v = str(rel.get("destinationId", ""))
        if not v:
            continue

        type_only.add(f"g:{grp}|t:{t}")
        type_value.add(f"g:{grp}|t:{t}|v:{v}")

    return {"type_only": type_only, "type_value": type_value}


def weighted_jaccard(a: Set[str], b: Set[str], idf: Dict[str, float]) -> float:
    if not a and not b:
        return 0.0
    inter = a & b
    union = a | b
    num = sum(idf.get(x, 1.0) for x in inter)
    den = sum(idf.get(x, 1.0) for x in union)
    return float(num / den) if den else 0.0


def build_idf(signatures: List[Set[str]]) -> Dict[str, float]:
    n = len(signatures)
    df = Counter()
    for sig in signatures:
        for feat in sig:
            df[feat] += 1
    return {feat: math.log((n + 1) / (cnt + 1)) for feat, cnt in df.items()}


def select_structured_exemplars(
    candidates: List[Any],
    concept_df: pd.DataFrame,
    snomed_rel_df: pd.DataFrame,
    id_col: str = "conceptId",
    term_col: str = "term",
    top_k: int = 8,
    min_attrs: int = 2,
) -> List[Dict[str, Any]]:
    """
    candidates: [{id,label,...}, ...] or [conceptId, ...]
    concept_df: DataFrame used to resolve labels from id_col -> term_col

    Strategy:
      1) Compute type_only signatures for each candidate.
      2) Compute IDF weights within candidate pool.
      3) Score each candidate by average similarity to others (cohesion).
      4) Return most central structured concepts.
    """
    term_lookup = (
        concept_df[[id_col, term_col]]
        .dropna(subset=[id_col])
        .drop_duplicates(subset=[id_col])
        .assign(**{id_col: lambda d: d[id_col].astype(str)})
        .set_index(id_col)[term_col]
        .to_dict()
    )

    sigs: List[Set[str]] = []
    kept: List[Dict[str, Any]] = []
    for candidate in candidates:
        hit = candidate if isinstance(candidate, dict) else {"id": candidate}
        cid = hit.get("id")
        if cid is None:
            continue
        cid_str = str(cid)
        label = hit.get("label")
        if label is None or str(label).strip() == "":
            label = term_lookup.get(cid_str, cid_str)
        hit = {**hit, "id": cid_str, "label": label}

        rels = extract_snomed_relationships(cid, snomed_rel_df, concept_df)
        sig = concept_signature_from_rels(rels)["type_only"]
        if len(sig) < min_attrs:
            continue
        kept.append(hit)
        sigs.append(sig)

    if not kept:
        return []

    idf = build_idf(sigs)

    scores: List[float] = []
    for i, sig_i in enumerate(sigs):
        sim_sum = 0.0
        denom = 0
        for j, sig_j in enumerate(sigs):
            if i == j:
                continue
            sim_sum += weighted_jaccard(sig_i, sig_j, idf)
            denom += 1
        scores.append(sim_sum / denom if denom else 0.0)

    ranked = sorted(zip(kept, scores), key=lambda x: x[1], reverse=True)
    return [hit for hit, _ in ranked[:top_k]]