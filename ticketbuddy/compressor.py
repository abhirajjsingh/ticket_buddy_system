from __future__ import annotations
from typing import List, Dict, Tuple
import numpy as np


def mmr_select(
    query_vec: np.ndarray,
    doc_vecs: np.ndarray,
    candidates: List[int],
    k: int = 6,
    lambda_mult: float = 0.5,
    redundancy_threshold: float = 0.88,
) -> List[int]:
    # MMR selection ensuring diversity and redundancy filtering
    selected: List[int] = []
    cand = candidates.copy()
    if len(cand) <= k:
        return cand

    # Normalize vectors if not already
    def cos(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    while cand and len(selected) < k:
        best_i = None
        best_score = -1e9
        for i in cand:
            rel = cos(query_vec, doc_vecs[i])
            diversity = 0.0
            if selected:
                diversity = max(cos(doc_vecs[i], doc_vecs[j]) for j in selected)
            score = lambda_mult * rel - (1 - lambda_mult) * diversity
            if score > best_score:
                best_score = score
                best_i = i
        if best_i is None:
            break
        # Redundancy filter: if too similar to any selected, skip
        if selected and max(cos(doc_vecs[best_i], doc_vecs[j]) for j in selected) >= redundancy_threshold:
            cand.remove(best_i)
            continue
        selected.append(best_i)
        cand.remove(best_i)
    return selected
