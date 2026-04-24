# dense_ann.py
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Dict
import numpy as np
import faiss

@dataclass
class Hit:
    id: int           # your RxCUI (int64)
    score: float      # cosine similarity if vectors are L2-normalized
    meta: Optional[dict] = None

def l2_normalize(x: np.ndarray) -> np.ndarray:
    # Safe L2 normalize (no zero division)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (x / norms).astype("float32")

class DenseANN:
    """
    A tiny ANN wrapper:
      - stores normalized float32 vectors
      - uses FAISS inner product (cosine if normalized)
      - sets FAISS IDs to your identifiers (e.g., RxCUIs)
    """
    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu and faiss.get_num_gpus() > 0
        self._index = None
        self._meta: Dict[int, dict] = {}

    @property
    def index(self) -> faiss.Index:
        if self._index is None:
            raise RuntimeError("Index not built yet.")
        return self._index

    def build(self, embeddings: np.ndarray, ids: Sequence[int], meta: Optional[Dict[int, dict]] = None):
        """
        embeddings: [N, D] (unnormalized or normalized)
        ids:        [N] int64 identifiers (e.g., RxCUIs)
        meta:       optional dict id->metadata (kept in Python, not FAISS)
        """
        assert embeddings.ndim == 2, "embeddings must be 2D"
        vecs = l2_normalize(embeddings)        # ensure cosine via IP
        dim = vecs.shape[1]

        base = faiss.IndexFlatIP(dim)          # exact search (fast enough up to ~1-2M vectors on GPU)
        if self.use_gpu:
            res = faiss.StandardGpuResources()
            base = faiss.index_cpu_to_gpu(res, 0, base)

        idmap = faiss.IndexIDMap(base)
        ids64 = np.asarray(ids, dtype="int64")
        idmap.add_with_ids(vecs, ids64)

        self._index = idmap
        if meta:
            # store arbitrary payload keyed by id (not part of FAISS)
            self._meta.update(meta)

    def add(self, embeddings: np.ndarray, ids: Sequence[int], meta: Optional[Dict[int, dict]] = None):
        vecs = l2_normalize(embeddings)
        ids64 = np.asarray(ids, dtype="int64")
        self.index.add_with_ids(vecs, ids64)
        if meta:
            self._meta.update(meta or {})

    def search(self, query_vecs: np.ndarray, k: int = 10, with_meta: bool = False) -> List[List[Hit]]:
        """
        query_vecs: [M, D] (unnormalized or normalized)
        returns:    list of hits per query, each a list[Hit] length <= k
        """
        Q = l2_normalize(query_vecs)
        D, I = self.index.search(Q, k)   # scores and FAISS (== your) IDs
        results: List[List[Hit]] = []
        for qi in range(Q.shape[0]):
            hits = []
            for score, id64 in zip(D[qi], I[qi]):
                if id64 == -1:
                    continue
                meta = self._meta.get(int(id64)) if with_meta else None
                hits.append(Hit(id=int(id64), score=float(score), meta=meta))
            results.append(hits)
        return results

    def save(self, path: str):
        # move to CPU for portability
        idx = self.index
        if self.use_gpu:
            idx = faiss.index_gpu_to_cpu(idx)
        faiss.write_index(idx, path)

    def load(self, path: str):
        idx = faiss.read_index(path)
        if self.use_gpu:
            res = faiss.StandardGpuResources()
            idx = faiss.index_cpu_to_gpu(res, 0, idx)
        self._index = idx

    def set_meta(self, meta: Dict[int, dict]):
        self._meta = meta or {}