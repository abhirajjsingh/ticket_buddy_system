from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class DenseIndex:
    use_faiss: bool = True

    def __post_init__(self):
        self.index = None
        self.embeddings = None  # fallback storage
        self.ids: List[str] = []
        self.dim = 0
        # Try FAISS
        if self.use_faiss:
            try:
                import faiss  # type: ignore
                self._faiss = faiss
            except Exception:
                self.use_faiss = False
                self._faiss = None

    def build(self, vectors: np.ndarray, ids: List[str]):
        self.ids = ids
        self.dim = vectors.shape[1]
        if self.use_faiss and self._faiss is not None:
            index = self._faiss.IndexHNSWFlat(self.dim, 32, self._faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = 80
            index.add(vectors.astype(np.float32))
            self.index = index
        else:
            # fallback exact search with numpy
            self.embeddings = vectors.astype(np.float32)

    def search(self, query_vecs: np.ndarray, top_k: int = 10) -> List[List[Tuple[int, float]]]:
        # returns list per query of (idx, score)
        if self.index is not None:
            self.index.hnsw.efSearch = max(top_k, 50)
            D, I = self.index.search(query_vecs.astype(np.float32), top_k)
            results = []
            for i in range(I.shape[0]):
                results.append([(int(I[i, j]), float(D[i, j])) for j in range(top_k) if I[i, j] != -1])
            return results
        else:
            # cosine similarity since embeddings are normalized to unit norm
            V = self.embeddings  # (N, d)
            results = []
            for q in query_vecs.astype(np.float32):
                scores = (V @ q)
                idx = np.argsort(-scores)[:top_k]
                results.append([(int(i), float(scores[i])) for i in idx])
            return results
