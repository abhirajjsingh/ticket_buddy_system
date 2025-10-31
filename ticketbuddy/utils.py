import re
from typing import List


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # Remove HTML tags and code fences
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"```[\s\S]*?```", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def simple_tokenize(text: str) -> List[str]:
    # Light-weight tokenizer for BM25
    text = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return [t for t in text.split() if t]


def split_multi_issue_query(query: str) -> List[str]:
    # Split multi-intent queries on common conjunctions and punctuation
    if not query:
        return []
    seps = r"[.;]|\band\b|\bplus\b|\balso\b|\b&\b|\b,\b"
    parts = re.split(seps, query, flags=re.IGNORECASE)
    subs = [p.strip() for p in parts if p and p.strip()]
    # Keep at least the original if splitting is too aggressive
    return subs or [query]
