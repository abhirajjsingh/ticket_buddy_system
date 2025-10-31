from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class CrossEncoderReranker:
    model_name: str
    _model: any = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, candidates: List[Tuple[str, str, float]], top_k: int = 3) -> List[Tuple[str, str, float]]:
        # candidates: list of (text, source_id, fused_score)
        model = self._load()
        pairs = [(query, c[0]) for c in candidates]
        scores = model.predict(pairs)
        rescored = [(candidates[i][0], candidates[i][1], float(scores[i])) for i in range(len(candidates))]
        rescored.sort(key=lambda x: x[2], reverse=True)
        return rescored[:top_k]
