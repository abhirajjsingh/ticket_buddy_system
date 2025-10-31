from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import pandas as pd

from .config import paths
from .utils import clean_text


@dataclass
class Document:
    id: str
    text: str
    metadata: Dict


def load_it_tickets(xlsx_path: Path | None = None) -> List[Document]:
    xlsx = xlsx_path or paths.it_tickets_xlsx
    df = pd.read_excel(xlsx)
    # Expected columns: ticket_id, title, description, root_cause, resolution
    cols = {c.lower(): c for c in df.columns}
    def get(col):
        return df[cols[col]] if col in cols else ""

    docs: List[Document] = []
    for _, row in df.iterrows():
        ticket_id = str(row.get(cols.get("ticket_id", "ticket_id"), "")).strip()
        title = str(row.get(cols.get("title", "title"), ""))
        desc = str(row.get(cols.get("description", "description"), ""))
        root = str(row.get(cols.get("root_cause", "root_cause"), ""))
        res = str(row.get(cols.get("resolution", "resolution"), ""))
        text = "\n".join([
            f"title: {title}",
            f"description: {desc}",
            f"root_cause: {root}",
            f"resolution: {res}",
        ])
        docs.append(
            Document(
                id=f"ticket:{ticket_id}",
                text=clean_text(text),
                metadata={
                    "type": "ticket",
                    "ticket_id": ticket_id,
                    "title": title,
                },
            )
        )
    return docs


def load_kb_docs(kb_dir: Path | None = None) -> List[Document]:
    kb_path = kb_dir or paths.kb_dir
    docs: List[Document] = []
    for p in sorted(kb_path.glob("*.txt")):
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        docs.append(
            Document(
                id=f"kb:{p.name}",
                text=clean_text(text),
                metadata={"type": "kb", "doc_name": p.name},
            )
        )
    return docs


def load_user_queries(xlsx_path: Path | None = None) -> List[str]:
    xlsx = xlsx_path or paths.user_queries_xlsx
    df = pd.read_excel(xlsx)
    col = None
    for c in df.columns:
        if c.lower().strip() == "user_query":
            col = c
            break
    if col is None:
        # fallback to first column
        col = df.columns[0]
    queries = [clean_text(str(x)) for x in df[col].fillna("")]
    return [q for q in queries if q]
