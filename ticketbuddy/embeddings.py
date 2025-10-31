from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import numpy as np

from .config import model_config


@dataclass
class EmbeddingModel:
    name: str = model_config.embedding_model_name
    _model: any = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.name)
        return self._model

    def encode(self, texts: List[str], batch_size: int = 32, show_progress_bar: bool = False) -> np.ndarray:
        model = self._load()
        emb = model.encode(texts, batch_size=batch_size, show_progress_bar=show_progress_bar, normalize_embeddings=True)
        return np.asarray(emb, dtype=np.float32)


class PCAReducer:
    def __init__(self, out_dim: Optional[int]):
        self.out_dim = int(out_dim) if out_dim else 0
        self._pca = None

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        if not self.out_dim or self.out_dim >= X.shape[1]:
            return X
        from sklearn.decomposition import PCA
        self._pca = PCA(n_components=self.out_dim, random_state=42)
        return self._pca.fit_transform(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._pca:
            return X
        return self._pca.transform(X)
