# src/utils/search_utils.py

from typing import List, Dict, Optional, Sequence
from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd
from elasticsearch import Elasticsearch
from sentence_transformers import CrossEncoder, SentenceTransformer


# ------------------------------
# 1. Encode query
# ------------------------------

def encode_query(
    model: SentenceTransformer,
    query: str,
    normalize: bool = True,
) -> np.ndarray:
    """
    Encode a single query string (e.g., a contraindication identified by the LLM)
    into a float32 embedding suitable for FAISS.

    Parameters
    ----------
    model : SentenceTransformer
    query : str
    normalize : bool
        If True, L2-normalize embeddings (recommended for IndexFlatIP).

    Returns
    -------
    np.ndarray
        1D vector of shape (dim,), dtype float32.
    """
    emb = model.encode(
        [query],
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
    )
    # emb has shape (1, dim)
    return emb.astype("float32")[0]


# ------------------------------
# 2. Dense (FAISS) candidates
# ------------------------------

# OLD 02/19/2026
# def dense_candidates(
#     faiss_index,
#     q_vec: np.ndarray,
#     meta_df: pd.DataFrame,
#     k: int = 20,
#     id_column: Optional[str] = None,
#     label_column: str = "label",
# ) -> List[Dict]:
#     """
#     Get top-k dense (vector) candidates from a FAISS index.

#     Parameters
#     ----------
#     faiss_index : faiss.Index
#         FAISS index built over your concepts.
#     q_vec : np.ndarray
#         Query vector (dim,), dtype float32.
#     meta_df : pd.DataFrame
#         Metadata table.
#     k : int
#     id_column : Optional[str]
#         Column to use as concept ID in the output. If None, meta_df.index is used.
#     label_column : str
#         Column to use as human-readable label.

#     Returns
#     -------
#     List[Dict]
#         [{"id": ..., "label": ..., "score": ...}, ...]
#     """
#     if q_vec.ndim != 1:
#         raise ValueError(f"q_vec must be 1D, got shape {q_vec.shape}")

#     # 1. FAISS search
#     D, I = faiss_index.search(q_vec.reshape(1, -1), k)
#     ids = I[0]
#     scores = D[0]

#     # Filter out -1 (no more hits)
#     mask = ids >= 0
#     ids = ids[mask]
#     scores = scores[mask]

#     if len(ids) == 0:
#         return []

#     # 2. Map FAISS IDs -> meta_df rows.
#     # Use reindex so we don't crash if some IDs are missing.
#     rows = meta_df.reindex(ids)

#     # Drop rows where label is missing (i.e., IDs not found)
#     rows = rows[rows[label_column].notna()]

#     # If everything is missing, just return empty
#     if rows.empty:
#         print(
#             f"[dense_candidates] WARNING: none of the FAISS IDs were found in "
#             f"meta_df.index. Example IDs: {ids[:10]}"
#         )
#         return []

#     # 3. Determine concept IDs for output
#     if id_column is not None:
#         # concept ID comes from a column
#         concept_ids = rows[id_column].tolist()
#     else:
#         # concept ID is the DataFrame index
#         concept_ids = rows.index.tolist()

#     labels = rows[label_column].astype(str).tolist()
#     # Align scores to the subset of rows we kept
#     # reindex might have changed order; ensure scores correspond to correct ids
#     # Build a mapping id -> score
#     score_map = {int(i): float(s) for i, s in zip(ids, scores)}
#     row_ids_as_int = [int(i) for i in rows.index]
#     aligned_scores = [score_map[i] for i in row_ids_as_int if i in score_map]

#     results = []
#     for cid, label, score in zip(concept_ids, labels, aligned_scores):
#         results.append(
#             {
#                 "id": cid,
#                 "label": label,
#                 "score": float(score),
#             }
#         )

#     return results


def dense_candidates(
    faiss_index,
    q_vec: np.ndarray,
    concept_meta_df: pd.DataFrame,
    k_concepts: int = 20,
    k_vectors: int = 100,
    label_column: str = "preferredTerm",
) -> List[Dict]:
    """
    Dense retrieval when FAISS index contains multiple vectors per concept_id
    (preferred + synonyms), and FAISS IDs are concept_id (duplicates allowed).

    Parameters
    ----------
    faiss_index : faiss.Index
        FAISS index where each vector's ID is a concept_id (may repeat).
    q_vec : np.ndarray
        Query vector (dim,), float32.
    concept_meta_df : pd.DataFrame
        Concept-level metadata with UNIQUE index = concept_id.
        Must contain label_column (e.g., preferredTerm).
    k_concepts : int
        Number of unique concept candidates to return.
    k_vectors : int
        Number of raw vectors to retrieve before aggregating. Should be > k_concepts.
        A common heuristic is 5–20x k_concepts.
    label_column : str
        Column used for display label.

    Returns
    -------
    List[Dict]
        [{"id": concept_id, "label": preferredTerm, "score": dense_score}, ...]
        score is aggregated (max) over all vectors retrieved for that concept.
    """
    if q_vec.ndim != 1:
        raise ValueError(f"q_vec must be 1D, got shape {q_vec.shape}")
    if not concept_meta_df.index.is_unique:
        raise ValueError(
            "concept_meta_df.index must be unique (one row per concept_id). "
            "Deduplicate/aggregate concept metadata first."
        )

    # 1) FAISS search over vectors (may return duplicate concept IDs)
    D, I = faiss_index.search(q_vec.reshape(1, -1), k_vectors)
    vec_scores = D[0]
    vec_ids = I[0]

    # Filter out invalid hits
    mask = vec_ids >= 0
    vec_ids = vec_ids[mask]
    vec_scores = vec_scores[mask]

    if len(vec_ids) == 0:
        return []

    # 2) Aggregate by concept_id (max pooling)
    best_score = defaultdict(lambda: -1e9)
    for cid, s in zip(vec_ids, vec_scores):
        cid = int(cid)
        s = float(s)
        if s > best_score[cid]:
            best_score[cid] = s

    # 3) Rank concepts by aggregated score
    ranked = sorted(best_score.items(), key=lambda x: -x[1])[:k_concepts]

    # 4) Build output with labels
    results: List[Dict] = []
    for cid, s in ranked:
        if cid in concept_meta_df.index:
            label = concept_meta_df.at[cid, label_column]
        else:
            label = ""
        results.append(
            {
                "id": cid,
                "label": str(label),
                "score": float(s),
            }
        )

    return results


# ------------------------------
# 3.1 BM25 Query Builder for SNOMED
# ------------------------------

def build_snomed_query(
    query_text: str,
    text_field: str = "preferredTerm",
    semantic_tags: Optional[List[str]] = None,
) -> Dict:
    """
    Build a multi-tier BM25 query for SNOMED concept retrieval.
    Combines exact, phrase, shingle, and token-level matching with boosts.
    """
    should_clauses = [
        # Tier 1: exact keyword match on preferredTerm.exact — highest boost
        {
            "term": {
                f"{text_field}.exact": {
                    "value": query_text.lower(),
                    "boost": 5.0,
                }
            }
        },
        # Tier 2: phrase match — preserves token order
        {
            "match_phrase": {
                f"{text_field}.phrase": {
                    "query": query_text,
                    "boost": 3.0,
                }
            }
        },
        # Tier 3: shingle match — rewards multi-token substring matches
        {
            "match": {
                f"{text_field}.shingle": {
                    "query": query_text,
                    "boost": 2.0,
                }
            }
        },
        # Tier 4: standard token match on all_terms (preferredTerm + synonyms)
        {
            "match": {
                "all_terms": {
                    "query": query_text,
                    "operator": "or",
                    "boost": 0.6,
                }
            }
        },
    ]

    query: Dict = {"bool": {"should": should_clauses, "minimum_should_match": 1}}

    # Optional: filter by semantic tag if caller provides expected type
    if semantic_tags:
        query["bool"]["filter"] = [
            {"terms": {"semantic_tag": semantic_tags}}
        ]

    return query


# ------------------------------
# 3.2 BM25 (Elasticsearch) candidates
# ------------------------------

def bm25_candidates(
    es: Elasticsearch,
    query_text: str,
    index: str,
    k: int = 20,
    text_field: str = "text",
    id_field: str = "id",
    label_field: Optional[str] = None,
    custom_query: Optional[Dict] = None,
    semantic_tags: Optional[List[str]] = None,
) -> List[Dict]:
    """
    Return top-k lexical hits from an Elasticsearch BM25 index.

    The ES documents are assumed to have at least:
      - an identifier field (id_field),
      - a text field used for matching (text_field),
      - optionally a human-readable label (label_field). If label_field is None,
        text_field is also used as label.

    Parameters
    ----------
    es : Elasticsearch
    query_text : str
        The query string (e.g., contraindication identified by the LLM).
    index : str
        Elasticsearch index name.
    k : int
        Number of hits to return.
    text_field : str
        Field in ES used in the match query.
    id_field : str
        Field in ES representing concept ID.
    label_field : Optional[str]
        Field in ES for human-readable label. Defaults to text_field if None.

    Returns
    -------
    List[Dict]
        List of dicts with keys: "id", "label", "score".
    """
    if label_field is None:
        label_field = text_field

    if custom_query is not None:
        query_obj = custom_query
    else:
        query_obj = {
        "match": {
            text_field: {
                "query": query_text,
                "operator": "or",
            }
        }
    }
    resp = es.search(
        index=index,
        size=k,
        query=query_obj,
    )
    hits = resp.get("hits", {}).get("hits", [])
    results: List[Dict] = []

    for hit in hits:
        src = hit.get("_source", {})
        cid = src.get(id_field)
        label = src.get(label_field, "")
        score = hit.get("_score", 0.0)

        if cid is None:
            # skip malformed documents
            continue

        results.append(
            {
                "id": cid,
                "label": str(label),
                "score": float(score),
            }
        )

    return results


# ------------------------------
# 4. Reciprocal Rank Fusion
# ------------------------------

def fuse_hits_rrf(
    dense_hits: Sequence[Dict],
    bm25_hits: Sequence[Dict],
    k: int = 20,
    rank_bias: int = 60,
) -> List[Dict]:
    """
    Fuse dense and BM25 hits using Reciprocal Rank Fusion (RRF).

    Inputs are two lists of hits in the format:
      {"id": ..., "label": ..., "score": ...}

    Output is a list of fused hits:
      {"id": ..., "label": ..., "fused": <rrf_score>, "from": "bm25,dense"}

    Parameters
    ----------
    dense_hits : Sequence[Dict]
    bm25_hits : Sequence[Dict]
    k : int
        Number of fused hits to return.
    rank_bias : int
        RRF rank_bias parameter (larger values reduce influence of rank).

    Returns
    -------
    List[Dict]
    """
    # 1. sort each list by its native score (descending)
    dense_sorted = sorted(dense_hits, key=lambda x: -x["score"])
    bm25_sorted = sorted(bm25_hits, key=lambda x: -x["score"])

    fused = defaultdict(lambda: {"rrf": 0.0, "origin": set(), "label": None})

    # 2. Add contributions from dense hits
    for rank, hit in enumerate(dense_sorted, start=1):
        key = hit["id"]
        fused[key]["rrf"] += 1.0 / (rank_bias + rank)
        fused[key]["origin"].add("dense")
        if fused[key]["label"] is None:
            fused[key]["label"] = hit.get("label", "")

    # 3. Add contributions from BM25 hits
    for rank, hit in enumerate(bm25_sorted, start=1):
        key = hit["id"]
        fused[key]["rrf"] += 1.0 / (rank_bias + rank)
        fused[key]["origin"].add("bm25")
        if fused[key]["label"] is None:
            fused[key]["label"] = hit.get("label", "")

    # 4. turn dict -> list, sort by fused score
    fused_list: List[Dict] = []
    for key, val in fused.items():
        fused_list.append(
            {
                "id": key,
                "label": val["label"],
                "fused": float(val["rrf"]),
                "from": ",".join(sorted(val["origin"])),  # e.g., "bm25,dense"
            }
        )

    fused_list.sort(key=lambda x: -x["fused"])

    # 5. return top-k
    return fused_list[:k]


# ------------------------------
# 5. SNOMED hierarchy utilities
# ------------------------------

def _concept_term(concept_id: int, concept_meta_df: pd.DataFrame) -> str:
    """O(1) label lookup from concept_meta_df indexed by conceptId."""
    try:
        return str(concept_meta_df.at[concept_id, "term"])
    except KeyError:
        return str(concept_id)


def build_is_a_graph(rel_df: pd.DataFrame) -> nx.DiGraph:
    """Build a directed graph of SNOMED IS-A relationships (child → parent)."""
    IS_A = 116680003
    is_a = rel_df[rel_df["typeId"] == IS_A]
    G = nx.DiGraph()
    G.add_edges_from(zip(is_a["sourceId"], is_a["destinationId"]))
    return G


def get_longest_ancestor_path(
    concept_id: int,
    is_a_graph: nx.DiGraph,
    concept_meta_df: pd.DataFrame,
    max_depth: int = 10,
) -> str:
    """Return a '→'-joined chain from concept to its most distant IS-A ancestor.

    Locates the deepest reachable ancestor via BFS distances, then returns the
    single BFS path to it. Avoids nx.all_simple_paths to prevent exponential
    blowup on dense SNOMED hierarchies.
    """
    if concept_id not in is_a_graph:
        return _concept_term(concept_id, concept_meta_df)
    path_lengths = nx.single_source_shortest_path_length(
        is_a_graph, concept_id, cutoff=max_depth
    )
    if len(path_lengths) <= 1:
        return _concept_term(concept_id, concept_meta_df)
    deepest = max(path_lengths, key=path_lengths.get)
    path = nx.shortest_path(is_a_graph, concept_id, deepest)
    return " → ".join(_concept_term(c, concept_meta_df) for c in path)


# ------------------------------
# 6. Cross-encoder re-ranker
# ------------------------------

def rerank_candidates(
    query_text: str,
    hits: List[Dict],
    cross_encoder: CrossEncoder,
    n_final: int = 10,
) -> List[Dict]:
    """Re-score RRF-fused hits with a cross-encoder and return n_final by rerank score.

    Adds a ``rerank_score`` key to each hit dict (in-place) and returns the
    list sorted descending by that score, truncated to n_final.
    """
    if not hits:
        return hits
    pairs = [(query_text, hit["label"]) for hit in hits]
    scores = cross_encoder.predict(pairs)
    for hit, score in zip(hits, scores):
        hit["rerank_score"] = float(score)
    return sorted(hits, key=lambda h: h["rerank_score"], reverse=True)[:n_final]


# ------------------------------
# 6. End-to-end helper for contraindication queries
# ------------------------------

def search_query(
    query_text: str,
    model: SentenceTransformer,
    faiss_index,
    concept_meta_df: pd.DataFrame,
    es: Elasticsearch,
    bm25_index: str,
    *,
    label_column: str = "label",
    bm25_text_field: str = "text",
    bm25_id_field: str = "id",
    bm25_label_field: Optional[str] = None,
    k_dense: int = 50,
    k_dense_vectors: Optional[int] = None,
    k_bm25: int = 50,
    k_final: int = 20,
    n_final: Optional[int] = None,
    normalize_query: bool = True,
    cross_encoder: Optional[CrossEncoder] = None,
    is_a_graph: Optional[nx.DiGraph] = None,
) -> List[Dict]:
    """
    Convenience wrapper: given a contraindication query string, return fused hits
    from FAISS (dense) + Elasticsearch (BM25) using RRF, with optional cross-encoder
    re-ranking and ancestor path enrichment applied after fusion.

    Parameters
    ----------
    query_text : str
        Contraindication text identified by the LLM.
    model : SentenceTransformer
        Same model used to build the FAISS index.
    faiss_index : faiss.Index
        Dense index over your concepts.
    concept_meta_df : pd.DataFrame
        Metadata table indexed by conceptId; used for label lookups.
    es : Elasticsearch
        Elasticsearch client.
    bm25_index : str
        Elasticsearch index name.
    label_column : str
        Column in concept_meta_df used as label.
    bm25_text_field : str
        ES field used for matching.
    bm25_id_field : str
        ES field representing concept ID.
    bm25_label_field : Optional[str]
        ES field representing label; if None, uses bm25_text_field.
    k_dense : int
        Number of dense candidates to retrieve before fusion.
    k_bm25 : int
        Number of BM25 candidates to retrieve before fusion.
    k_final : int
        Number of fused hits to return from RRF before re-ranking.
    n_final : Optional[int]
        Number of hits returned after cross-encoder re-ranking. Defaults to
        k_final when not set.
    normalize_query : bool
        If True, normalize the query embedding.
    cross_encoder : Optional[CrossEncoder]
        If provided, re-ranks RRF-fused hits before returning.
    is_a_graph : Optional[nx.DiGraph]
        If provided, each returned hit is enriched with an "ancestor_path" key
        containing a '→'-joined IS-A chain from concept to its deepest ancestor.

    Returns
    -------
    List[Dict]
        Hits with keys: "id", "label", "fused", "from",
        and optionally "rerank_score" and "ancestor_path".
    """
    # 1. Encode query
    q_vec = encode_query(model, query_text, normalize=normalize_query)

    # 2. Dense retrieval
    if k_dense_vectors is None:
        k_dense_vectors = max(k_dense * 5, k_dense)
    dense_hits = dense_candidates(
        faiss_index=faiss_index,
        q_vec=q_vec,
        concept_meta_df=concept_meta_df,
        k_concepts=k_dense,
        k_vectors=k_dense_vectors,
        label_column=label_column,
    )

    # 3. BM25 retrieval
    snomed_query = build_snomed_query(
        query_text=query_text,
        text_field=bm25_text_field,
    )
    bm25_hits = bm25_candidates(
        es=es,
        query_text=query_text,
        index=bm25_index,
        k=k_bm25,
        text_field=bm25_text_field,
        id_field=bm25_id_field,
        label_field=bm25_label_field,
        custom_query=snomed_query,
    )

    # 4. Fuse via RRF
    fused_hits = fuse_hits_rrf(
        dense_hits=dense_hits,
        bm25_hits=bm25_hits,
        k=k_final,
    )

    # 5. Optional cross-encoder re-ranking
    _n_final = n_final if n_final is not None else k_final
    if cross_encoder is not None:
        final_hits = rerank_candidates(query_text, fused_hits, cross_encoder, n_final=_n_final)
    else:
        final_hits = fused_hits[:_n_final]

    # 6. Optional ancestor path enrichment
    if is_a_graph is not None:
        for hit in final_hits:
            hit["ancestor_path"] = get_longest_ancestor_path(
                int(hit["id"]), is_a_graph, concept_meta_df
            )
    return final_hits