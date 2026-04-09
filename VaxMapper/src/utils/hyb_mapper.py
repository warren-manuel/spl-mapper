from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import faiss
import pandas as pd

from VaxMapper.src.llm import extract_json
from VaxMapper.src.utils.elastisearch_utils import bulk_index, create_index, get_es_client
from VaxMapper.src.utils.embedding_utils import (
    build_and_save_dense_index,
    load_ST_model,
    maybe_move_index_to_gpu,
)
from VaxMapper.src.utils.search_utils import build_is_a_graph, search_query
from VaxMapper.src.utils.snomed_utils import load_snomed_dataframes


DEFAULT_ITEM_TERM_KEYS = ("ci_text", "contraindication_state_text", "substance_text", "severity_span", "clinical_course_span")

SNOMED_CT_SETTINGS = {
    "settings": {
        "index": {
            "number_of_shards": 1,
            "number_of_replicas": 1,
            "similarity": {
                "default": {
                    "type": "BM25",
                    "k1": 1.2,
                    "b": 0.5, #from 0.75 to reduce length normalization pressure for shorter queries (e.g., single-term queries) which are common in this mapping use case.
                }
            },
        },
        "analysis": {
            "filter":{
                "snomed_shingle:": {
                    "type": "shingle",
                    "min_shingle_size": 2,
                    "max_shingle_size": 3,
                    "output_unigrams": True,
            },
            },
            "analyzer": {
                "snomed_text": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": ["lowercase", "asciifolding"],
                },
                "snomed_text_shingle": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": ["lowercase", "asciifolding", "snomed_shingle:"],
                },
            }
        },
    },
    "mappings": {
        "properties": {
            "conceptId": {"type": "keyword"},
            "preferredTerm": {
                "type": "text",
                "analyzer": "snomed_text",
                "search_analyzer": "snomed_text",
                "copy_to": ["all_terms"],
                "fields": {
                    "exact": {"type": "keyword"},  # enables exact match boost at query time
                    "phrase": {
                        "type": "text",
                        "analyzer": "snomed_text",
                        "index_options": "offsets"
                        },
                    "shingle": {
                        "type": "text",
                        "analyzer": "snomed_text_shingle",
                        }
                }
            },
            "synonyms": {
                "type": "text",
                "analyzer": "snomed_text",
                "search_analyzer": "snomed_text",
                "copy_to": ["all_terms"],
            },
            "all_terms": {
                "type": "text",
                "analyzer": "snomed_text",
                "search_analyzer": "snomed_text",
            },
            "semantic_tag": {"type": "keyword"},
        }
    },
}


@dataclass
class MapperResources:
    st_model: Any
    faiss_index: Any
    concept_meta_df: pd.DataFrame
    es: Any
    bm25_index: str
    item_term_keys: Tuple[str, ...]
    k_dense: int
    k_bm25: int
    k_final: int
    n_final: Optional[int] = None
    cross_encoder: Optional[Any] = None
    is_a_graph: Optional[Any] = None


_RESOURCE_CACHE: Dict[Tuple[Any, ...], MapperResources] = {}

# TODO(retrieval-roadmap): Revisit retrieval stack and consider a simpler retrieval/rerank
# pattern similar to the referenced HF Space approach (FAISS + BM25 + reranking):
# https://huggingface.co/spaces/Shriharshan/Autism-RAG
# We will revisit this later.
#
# TODO(retrieval-roadmap): Add support for downloading and wiring a dedicated reranker model
# for retrieval-stage reranking instead of relying only on current fused retrieval.
# We will revisit this later.


def build_es_index(
    es: Any,
    index_name: str,
    snomed_complete_df: pd.DataFrame,
    *,
    rebuild_index: bool = False,
    index_settings: Optional[Dict] = SNOMED_CT_SETTINGS,
) -> None:
    create_index(es, index_name, index_settings, delete_if_exists=rebuild_index)
    if es.indices.exists(index=index_name) and not rebuild_index:
        return
    bulk_index(
        es=es,
        df=snomed_complete_df,
        id_col="conceptId",
        index_name=index_name,
        field_map={
            "conceptId": "conceptId",
            "term": "preferredTerm",
            "synonyms": "synonyms",
            "semantic_tag": "semantic_tag",
        },
    )


def build_or_load_faiss_index(
    terms_df: pd.DataFrame,
    model_name: str,
    device: str,
    index_path: str,
    rebuild_index: bool,
):
    st_model = load_ST_model(model_name, device=device)

    if rebuild_index or not os.path.exists(index_path):
        _ = build_and_save_dense_index(
            df=terms_df,
            model=st_model,
            text_column="term_text",
            id_column="conceptId",
            batch_size=256,
            normalize=True,
            use_gpu_for_queries=True,
            save_index=True,
            index_filename=index_path,
        )

    cpu_index = faiss.read_index(index_path)
    faiss_index = maybe_move_index_to_gpu(cpu_index)
    return st_model, faiss_index


def read_predictions_jsonl_files(
    pred_files: Union[str, Sequence[str]],
    id_field_candidates: Iterable[str] = ("spl_set_id", "SPL_SET_ID", "item_index"),
    item_term_keys: Sequence[str] = DEFAULT_ITEM_TERM_KEYS,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    normalized_term_keys = [k for k in item_term_keys if isinstance(k, str) and k.strip()]
    resolved_pred_files: List[str] = []

    if isinstance(pred_files, str):
        resolved_pred_files = sorted(glob.glob(pred_files))
        if not resolved_pred_files:
            raise FileNotFoundError(f"No files matched pattern: {pred_files}")
    else:
        for path in pred_files:
            if isinstance(path, str):
                matches = sorted(glob.glob(path))
                if matches:
                    resolved_pred_files.extend(matches)
                else:
                    resolved_pred_files.append(path)
        if not resolved_pred_files:
            raise ValueError("pred_files is empty after resolving input paths/patterns")

    for path in resolved_pred_files:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                rec = json.loads(line)
                spl_id = None
                for key in id_field_candidates:
                    if key in rec and rec[key] is not None:
                        spl_id = str(rec[key])
                        break
                if spl_id is None:
                    raise ValueError(f"Prediction record missing SPL_SET_ID/spl_set_id in file {path}")

                raw = rec.get("raw_output") or {}
                parsed = extract_json(raw)
                if not isinstance(parsed, dict):
                    parsed = {}

                items = parsed.get("items") or []
                for idx, item in enumerate(items):
                    if not isinstance(item, dict):
                        continue
                    row: Dict[str, Any] = {"SPL_SET_ID": spl_id, "item_index": idx}
                    for key in normalized_term_keys:
                        value = item.get(key)
                        row[key] = str(value) if value is not None and str(value).strip() != "" else None
                    rows.append(row)

    return pd.DataFrame(rows)


def map_item_terms(
    item: Dict[str, Any],
    resources: MapperResources,
) -> Dict[str, Any]:
    mapped_row: Dict[str, Any] = {
        "SPL_SET_ID": str(item.get("SPL_SET_ID", "")),
        "item_index": item.get("item_index"),
        "ci_text": item.get("ci_text", ""),
    }

    for key in resources.item_term_keys:
        raw_query = item.get(key, None)
        query = str(raw_query).strip() if raw_query is not None else ""
        out_key = f"{key}_terms"
        if not query:
            mapped_row[out_key] = []
            continue

        hits = search_query(
            query_text=query,
            model=resources.st_model,
            faiss_index=resources.faiss_index,
            concept_meta_df=resources.concept_meta_df,
            es=resources.es,
            bm25_index=resources.bm25_index,
            label_column="term",
            bm25_text_field="preferredTerm",
            bm25_id_field="conceptId",
            bm25_label_field="preferredTerm",
            k_dense=resources.k_dense,
            k_bm25=resources.k_bm25,
            k_final=resources.k_final,
            n_final=resources.n_final,
            normalize_query=True,
            cross_encoder=resources.cross_encoder,
            is_a_graph=resources.is_a_graph,
        )
        mapped_row[out_key] = hits

    return mapped_row


def map_terms(
    pred_df: pd.DataFrame,
    item_term_keys: Sequence[str],
    st_model: Any,
    faiss_index: Any,
    concept_meta_df: pd.DataFrame,
    es: Any,
    bm25_index: str,
    k_dense: int,
    k_bm25: int,
    k_final: int,
    n_final: Optional[int] = None,
) -> List[Dict[str, Any]]:
    resources = MapperResources(
        st_model=st_model,
        faiss_index=faiss_index,
        concept_meta_df=concept_meta_df,
        es=es,
        bm25_index=bm25_index,
        item_term_keys=tuple(item_term_keys),
        k_dense=k_dense,
        k_bm25=k_bm25,
        k_final=k_final,
        n_final=n_final,
    )
    return [map_item_terms(row.to_dict(), resources) for _, row in pred_df.iterrows()]


def write_jsonl(records: Sequence[Dict[str, Any]], out_path: str) -> None:
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_mapper_resources(
    *,
    snomed_source_dir: str = "snomed_us_source",
    concept_path: Optional[str] = None,
    description_path: Optional[str] = None,
    es_index: str = "snomed_ct_es_index",
    dense_index_path: str = "results/snomed_terms_dense_test.bin",
    model_name: str = "tavakolih/all-MiniLM-L6-v2-pubmed-full",
    device: str = "cuda",
    k_dense: int = 50,
    k_bm25: int = 50,
    k_final: int = 20,
    n_final: Optional[int] = None,
    rebuild_dense_index: bool = False,
    rebuild_es_index: bool = False,
    item_term_keys: Sequence[str] = DEFAULT_ITEM_TERM_KEYS,
    reranker_model_id: str = "",
    reranker_device: str = "cpu",
) -> MapperResources:
    es = get_es_client()
    snomed_frames = load_snomed_dataframes(
        concept_snapshot_path=concept_path,
        description_snapshot_path=description_path,
        snomed_source_dir=snomed_source_dir,
    )
    build_es_index(
        es,
        es_index,
        snomed_frames["snomed_complete_df"],
        rebuild_index=rebuild_es_index,
    )

    st_model, faiss_index = build_or_load_faiss_index(
        terms_df=snomed_frames["terms_df"],
        model_name=model_name,
        device=device,
        index_path=dense_index_path,
        rebuild_index=rebuild_dense_index,
    )

    cross_encoder = None
    if reranker_model_id:
        from sentence_transformers import CrossEncoder
        cross_encoder = CrossEncoder(reranker_model_id, device=reranker_device)

    is_a_graph = build_is_a_graph(snomed_frames["rel_df"])

    concept_meta_df = snomed_frames["concept_df"].set_index("conceptId")
    return MapperResources(
        st_model=st_model,
        faiss_index=faiss_index,
        concept_meta_df=concept_meta_df,
        es=es,
        bm25_index=es_index,
        item_term_keys=tuple(item_term_keys),
        k_dense=k_dense,
        k_bm25=k_bm25,
        k_final=k_final,
        n_final=n_final,
        cross_encoder=cross_encoder,
        is_a_graph=is_a_graph,
    )


def get_cached_mapper_resources(
    *,
    snomed_source_dir: str = "snomed_us_source",
    concept_path: Optional[str] = None,
    description_path: Optional[str] = None,
    es_index: str = "snomed_ct_es_index",
    dense_index_path: str = "results/snomed_terms_dense_test.bin",
    model_name: str = "tavakolih/all-MiniLM-L6-v2-pubmed-full",
    device: str = "cuda",
    k_dense: int = 50,
    k_bm25: int = 50,
    k_final: int = 20,
    n_final: Optional[int] = None,
    rebuild_dense_index: bool = False,
    rebuild_es_index: bool = False,
    item_term_keys: Sequence[str] = DEFAULT_ITEM_TERM_KEYS,
    reranker_model_id: str = "",
    reranker_device: str = "cpu",
) -> MapperResources:
    cache_key = (
        snomed_source_dir,
        concept_path,
        description_path,
        es_index,
        dense_index_path,
        model_name,
        device,
        k_dense,
        k_bm25,
        k_final,
        n_final,
        rebuild_dense_index,
        rebuild_es_index,
        tuple(item_term_keys),
        reranker_model_id,
        reranker_device,
    )
    resources = _RESOURCE_CACHE.get(cache_key)
    if resources is None:
        resources = load_mapper_resources(
            snomed_source_dir=snomed_source_dir,
            concept_path=concept_path,
            description_path=description_path,
            es_index=es_index,
            dense_index_path=dense_index_path,
            model_name=model_name,
            device=device,
            k_dense=k_dense,
            k_bm25=k_bm25,
            k_final=k_final,
            n_final=n_final,
            rebuild_dense_index=rebuild_dense_index,
            rebuild_es_index=rebuild_es_index,
            item_term_keys=item_term_keys,
            reranker_model_id=reranker_model_id,
            reranker_device=reranker_device,
        )
        _RESOURCE_CACHE[cache_key] = resources
    return resources


def retrieve_candidates_for_item(
    item: Dict[str, Any],
    resources: MapperResources,
) -> Dict[str, Any]:
    mapped_row = map_item_terms(item, resources)
    focus_candidates = mapped_row.get("contraindication_state_text_terms") or mapped_row.get("ci_text_terms") or []
    direct_candidates = mapped_row.get("ci_text_terms") or []

    return {
        "focus_candidates": focus_candidates,
        "direct_candidates": direct_candidates,
        "causative_agent_candidates": mapped_row.get("substance_text_terms", []) or [],
        "severity_candidates": mapped_row.get("severity_span_terms", []) or [],
        "clinical_course_candidates": mapped_row.get("clinical_course_span_terms", []) or [],
        "mapped_row": mapped_row,
    }