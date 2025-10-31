from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:  # optional dependency
    RecursiveCharacterTextSplitter = None

from .data_loader import Document
from .config import chunking_config


@dataclass
class Chunk:
    id: str
    text: str
    metadata: Dict


def chunk_documents(docs: List[Document]) -> List[Chunk]:
    chunks: List[Chunk] = []
    if RecursiveCharacterTextSplitter is None:
        # Simple fallback splitter by length
        for d in docs:
            text = d.text
            size = chunking_config.chunk_size
            overlap = chunking_config.chunk_overlap
            start = 0
            chunk_id = 0
            while start < len(text):
                end = min(len(text), start + size)
                chunk_text = text[start:end]
                chunks.append(
                    Chunk(
                        id=f"{d.id}#c{chunk_id}",
                        text=chunk_text,
                        metadata={**d.metadata, "parent_id": d.id, "chunk_id": chunk_id},
                    )
                )
                if end == len(text):
                    break
                start = end - overlap
                chunk_id += 1
        return chunks

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunking_config.chunk_size,
        chunk_overlap=chunking_config.chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )

    for d in docs:
        parts = splitter.split_text(d.text)
        for i, t in enumerate(parts):
            chunks.append(
                Chunk(
                    id=f"{d.id}#c{i}",
                    text=t,
                    metadata={**d.metadata, "parent_id": d.id, "chunk_id": i},
                )
            )
    return chunks
