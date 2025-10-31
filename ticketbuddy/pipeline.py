from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any
import json
import numpy as np

from .config import model_config, retrieval_config, paths
from .data_loader import load_it_tickets, load_kb_docs, load_user_queries
from .preprocess import chunk_documents, Chunk
from .embeddings import EmbeddingModel, PCAReducer
from .vectorstore import DenseIndex
from .retriever import HybridRetriever
from .reranker import CrossEncoderReranker
from .compressor import mmr_select
from .generator import LLMGenerator
from .utils import split_multi_issue_query


@dataclass
class TicketBuddyPipeline:
    embedder: EmbeddingModel
    pca: PCAReducer
    dense_index: DenseIndex
    retriever: HybridRetriever
    reranker: CrossEncoderReranker
    generator: LLMGenerator
    chunks: List[Chunk]
    chunk_embeddings: np.ndarray

    @classmethod
    def build(cls) -> "TicketBuddyPipeline":
        # 1) Load data
        tickets = load_it_tickets()
        kb = load_kb_docs()
        docs = tickets + kb

        # 2) Chunk
        chunks = chunk_documents(docs)
        texts = [c.text for c in chunks]
        metadatas = [c.metadata for c in chunks]

        # 3) Embeddings + optional PCA
        embedder = EmbeddingModel()
        X = embedder.encode(texts, show_progress_bar=True)
        pca = PCAReducer(model_config.use_pca_dim)
        Xr = pca.fit_transform(X)

        # 4) Dense index
        dense_index = DenseIndex()
        dense_index.build(Xr, ids=[c.id for c in chunks])

        # 5) Hybrid retriever
        retriever = HybridRetriever.build(
            texts=texts, metadatas=metadatas, dense_index=dense_index, alpha=retrieval_config.fuse_alpha
        )

        # 6) Reranker and generator
        reranker = CrossEncoderReranker(model_name=model_config.reranker_model_name)
        generator = LLMGenerator()

        return cls(
            embedder=embedder,
            pca=pca,
            dense_index=dense_index,
            retriever=retriever,
            reranker=reranker,
            generator=generator,
            chunks=chunks,
            chunk_embeddings=Xr,
        )

    def retrieve_contexts(self, query: str) -> List[Dict[str, Any]]:
        # Dense embedding for query
        qv = self.embedder.encode([query])[0]
        qv = self.pca.transform(np.array([qv]))[0]
        # Hybrid fuse
        fused = self.retriever.query(
            query=query,
            qvec=qv,
            top_k_dense=retrieval_config.top_k_dense,
            top_k_bm25=retrieval_config.top_k_bm25,
            final_k=retrieval_config.final_k,
        )
        cand_idx = [i for i, _ in fused]
        # MMR diversity + redundancy filter
        selected_idx = mmr_select(
            query_vec=qv,
            doc_vecs=self.chunk_embeddings,
            candidates=cand_idx,
            k=retrieval_config.mmr_k,
            lambda_mult=retrieval_config.mmr_lambda,
            redundancy_threshold=retrieval_config.similarity_threshold,
        )
        # Rerank via cross-encoder
        candidates = []  # (text, index_str, score)
        for i, score in fused:
            if i in selected_idx:
                candidates.append((self.chunks[i].text, str(i), float(score)))
        reranked = self.reranker.rerank(query, candidates, top_k=retrieval_config.rerank_top_k)

        contexts: List[Dict[str, Any]] = []
        for text, index_str, score in reranked:
            idx = int(index_str)
            meta = self.chunks[idx].metadata
            source = meta.get("ticket_id") or meta.get("doc_name") or meta.get("parent_id") or self.chunks[idx].id
            contexts.append({
                "text": text,
                "source": str(source),
                "score": float(score),
                "source_meta": meta,
            })
        return contexts

    def answer(self, query: str) -> Dict[str, Any]:
        # Split potential multi-issue queries
        sub_queries = split_multi_issue_query(query)
        merged = {"root_causes": [], "resolution_steps": [], "sources": []}
        seen_src = set()
        for sq in sub_queries:
            contexts = self.retrieve_contexts(sq)
            resp = self.generator.generate(sq, contexts)
            for k in ("root_causes", "resolution_steps"):
                for item in resp.get(k, []):
                    if item and item not in merged[k]:
                        merged[k].append(item)
            for s in resp.get("sources", []):
                sid = json.dumps(s, sort_keys=True)
                if sid not in seen_src:
                    merged["sources"].append(s)
                    seen_src.add(sid)
        return merged

    def run_all(self) -> List[Dict[str, Any]]:
        queries = load_user_queries()
        outputs = []
        for q in queries:
            outputs.append({
                "query": q,
                "answer": self.answer(q),
            })
        with open(paths.results_json, "w", encoding="utf-8") as f:
            json.dump(outputs, f, ensure_ascii=False, indent=2)
        return outputs
