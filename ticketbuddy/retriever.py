from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np

from .utils import simple_tokenize


@dataclass
class HybridRetriever:
    texts: List[str]
    metadatas: List[Dict]
    dense_index: any
    bm25: any
    alpha: float = 0.6

    @classmethod
    def build(cls, texts: List[str], metadatas: List[Dict], dense_index, alpha: float = 0.6):
        # Build BM25
        from rank_bm25 import BM25Okapi
        tokenized_corpus = [simple_tokenize(t) for t in texts]
        bm25 = BM25Okapi(tokenized_corpus)
        return cls(texts=texts, metadatas=metadatas, dense_index=dense_index, bm25=bm25, alpha=alpha)

    def query(self, query: str, qvec: np.ndarray, top_k_dense: int = 10, top_k_bm25: int = 10, final_k: int = 8) -> List[Tuple[int, float]]:
        # Dense search
        dense_hits = self.dense_index.search(qvec[None, :], top_k=top_k_dense)[0]  # list of (idx, score)
        dense_scores = {i: s for i, s in dense_hits}

        # BM25
        q_tokens = simple_tokenize(query)
        bm25_scores_list = self.bm25.get_scores(q_tokens)
        bm25_idx = np.argsort(-bm25_scores_list)[:top_k_bm25]
        max_bm25 = float(np.max(bm25_scores_list)) if len(bm25_scores_list) else 1.0
        # Normalize bm25 to 0..1
        bm25_scores = {int(i): float(bm25_scores_list[i] / max_bm25 if max_bm25 else 0.0) for i in bm25_idx}

        # Fuse
        combined: Dict[int, float] = {}
        for i, s in dense_scores.items():
            combined[i] = combined.get(i, 0.0) + self.alpha * s
        for i, s in bm25_scores.items():
            combined[i] = combined.get(i, 0.0) + (1 - self.alpha) * s

        # Top final_k
        top = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:final_k]
        return top
