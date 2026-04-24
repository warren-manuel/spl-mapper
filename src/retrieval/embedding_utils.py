# Contents of /rxnorm-term-getter/rxnorm-term-getter/src/utils/embedding_utils.py
from typing import Optional, Sequence

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def load_ST_model(model_id: str, device: str = 'cuda') -> SentenceTransformer:
    print(f"Loading model {model_id}...")
    model = SentenceTransformer(model_id, device=device)
    print(f"Model {model_id} loaded to {model.device} successfully.")
    return model

def encode_texts(model, texts, batch_size: int = 32, normalize: bool = True) -> np.ndarray:
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
    )
    return embeddings.astype("float32")  # FAISS needs float32

def build_faiss_index(
    embeddings: np.ndarray,
    ids: Optional[Sequence[int]] = None,
    use_gpu: bool = True
) -> faiss.Index:
    # dim = embeddings.shape[1]
    
    n, dim = embeddings.shape
    print(f"Building FAISS IndexFlatIP with {n} vectors of dim {dim}...")
    
    # index = faiss.IndexFlatIP(dim)  # cosine similarity if normalized
    # if use_gpu and faiss.get_num_gpus() > 0:
    #     res = faiss.StandardGpuResources()
    #     index = faiss.index_cpu_to_gpu(res, 0, index)
    # if ids is not None:
    #     id_map = faiss.IndexIDMap(index)
    #     id_map.add_with_ids(embeddings, np.asarray(ids, dtype="int64"))
    #     print(f"Index built with {len(ids)} IDs.")
    #     return id_map
    # else:
    #     index.add(embeddings)
    #     print(f"Index built with {embeddings.shape[0]} vectors.")
    #     return index
    base_index = faiss.IndexFlatIP(dim)  # ~= cosine similarity if normalized

    if ids is not None:
        ids_arr = np.asarray(ids, dtype="int64")
        if ids_arr.shape[0] != n:
            raise ValueError(
                f"Number of ids ({ids_arr.shape[0]}) does not match number of "
                f"embeddings ({n})."
            )
        index = faiss.IndexIDMap(base_index)
        index.add_with_ids(embeddings, ids_arr)
        print(f"Index built with {n} vectors and explicit IDs.")
        return index
    else:
        base_index.add(embeddings)
        print(f"Index built with {n} vectors using implicit IDs [0..{n-1}].")
        return base_index
    
def maybe_move_index_to_gpu(index: faiss.Index) -> faiss.Index:
    """
    If GPUs are available, move a CPU FAISS index to GPU and return it.
    If no GPUs are available, returns the original CPU index.

    Note: This is intended for querying. The on-disk index should remain CPU.
    Device 0 here is the first GPU visible to the current process after any
    CUDA_VISIBLE_DEVICES masking has been applied.
    """
    if faiss.get_num_gpus() == 0:
        print("No GPUs detected by FAISS; keeping index on CPU.")
        return index

    print("Moving FAISS index to GPU (first visible device, FAISS device 0)...")
    res = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
    print("Index moved to GPU successfully.")
    return gpu_index

# OLD 02/19/2026
# def build_and_save_dense_index(
#     df: pd.DataFrame,
#     model: SentenceTransformer,
#     label_column: str = "label",
#     batch_size: int = 64,
#     normalize: bool = True,
#     use_gpu_for_queries: bool = True,
#     save_index: bool = True,
#     index_filename: str = "faiss_index.bin"
# ) -> faiss.Index:
    
#     if label_column not in df.columns:
#         raise KeyError(
#             f"Column '{label_column}' not found in DataFrame. "
#             f"Available columns: {list(df.columns)}"
#         )

#     print(f"Encoding {len(df)} texts from column '{label_column}'...")
#     texts = df[label_column].astype(str).tolist()

#     vecs = encode_texts(
#         model,
#         texts,
#         batch_size=batch_size,
#         normalize=normalize,
#     )

#     faiss_ids = df.index.to_numpy(dtype="int64")

#     cpu_index = build_faiss_index(vecs, ids=faiss_ids)

#     if save_index:
#         print(f"Saving CPU index to '{index_filename}'...")
#         faiss.write_index(cpu_index, index_filename)
#         print(f"Index saved successfully to '{index_filename}'.")

#     if use_gpu_for_queries:
#         return maybe_move_index_to_gpu(cpu_index)
#     else:
#         return cpu_index

def build_and_save_dense_index(
    df: pd.DataFrame,
    model: SentenceTransformer,
    text_column: str = "term_text",     # changed name to be generic
    id_column: str = "concept_id",      # NEW: use concept_id, can be non-unique
    batch_size: int = 64,
    normalize: bool = True,
    use_gpu_for_queries: bool = True,
    save_index: bool = True,
    index_filename: str = "faiss_index.bin",
) -> faiss.Index:
    """
    Build a FAISS dense index over df[text_column], with FAISS IDs = df[id_column].
    This supports multiple rows per concept_id (preferred + synonyms) by allowing
    duplicate IDs.

    IMPORTANT:
      - df[id_column] must be int64 convertible.
      - df[text_column] must be strings.
    """

    if text_column not in df.columns:
        raise KeyError(
            f"Column '{text_column}' not found. Available columns: {list(df.columns)}"
        )
    if id_column not in df.columns:
        raise KeyError(
            f"Column '{id_column}' not found. Available columns: {list(df.columns)}"
        )

    # clean + enforce types
    df = df.copy()
    df[text_column] = df[text_column].astype(str)
    df[id_column] = pd.to_numeric(df[id_column], errors="raise").astype("int64")

    print(f"Encoding {len(df)} texts from column '{text_column}'...")
    texts = df[text_column].tolist()

    vecs = encode_texts(
        model,
        texts,
        batch_size=batch_size,
        normalize=normalize,
    )

    concept_ids = df[id_column].to_numpy(dtype="int64")

    # Build CPU FAISS index with explicit IDs (duplicates allowed)
    dim = vecs.shape[1]
    base = faiss.IndexFlatIP(dim)
    cpu_index = faiss.IndexIDMap2(base)
    cpu_index.add_with_ids(vecs, concept_ids)

    print(f"Index built with {vecs.shape[0]} vectors mapped to {df[id_column].nunique()} unique concept IDs.")

    if save_index:
        print(f"Saving CPU index to '{index_filename}'...")
        faiss.write_index(cpu_index, index_filename)
        print(f"Index saved successfully to '{index_filename}'.")

    if use_gpu_for_queries:
        return maybe_move_index_to_gpu(cpu_index)
    else:
        return cpu_index        
    
    # print(f"Encoding {len(df)} texts from column '{label_column}'...")
    # texts = df[label_column].tolist()
    # vecs = encode_texts(model, texts, batch_size=batch_size, normalize=normalize)
    # faiss_ids = df.index.to_numpy(dtype="int64")
    # faiss_index = build_faiss_index(vecs, ids=faiss_ids, use_gpu=use_gpu)
    # if save_index:
    #     print(f"Saving index to {index_filename}...")
    #     faiss_cpu = faiss.index_gpu_to_cpu(faiss_index) if use_gpu else faiss_index
    #     faiss.write_index(faiss_cpu, index_filename)
    #     print(f"Index saved successfully to {index_filename}")
    # return faiss_index