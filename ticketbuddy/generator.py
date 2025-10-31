from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any
import os
import json

from .config import model_config


PROMPT = (
    "You are TicketBuddy, an AI IT assistant. Based on the retrieved context, generate a JSON with: "
    "{\n  \"root_causes\": [\"...\"],\n  \"resolution_steps\": [\"...\"],\n  \"sources\": [{\"ticket_id\": \"...\"}, {\"doc_name\": \"...\"}]\n}. "
    "Be concise, technical, and only use supported evidence."
)


@dataclass
class LLMGenerator:
    openai_model: str = model_config.openai_model
    temperature: float = model_config.temperature
    _client: Any = None

    def _have_openai(self) -> bool:
        return bool(os.environ.get("OPENAI_API_KEY"))

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI()
        return self._client

    def _openai_generate(self, query: str, contexts: List[Dict]) -> Dict:
        client = self._get_client()
        context_blob = "\n\n".join([f"[Source: {c['source']}]\n{c['text']}" for c in contexts])
        messages = [
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": f"Query: {query}\n\nContext:\n{context_blob}"},
        ]
        resp = client.chat.completions.create(
            model=self.openai_model,
            messages=messages,
            temperature=self.temperature,
            response_format={"type": "json_object"},
        )
        try:
            content = resp.choices[0].message.content
            data = json.loads(content)
            # Ensure schema keys exist
            data.setdefault("root_causes", [])
            data.setdefault("resolution_steps", [])
            data.setdefault("sources", [])
            # Normalize sources objects
            norm_sources = []
            for s in data.get("sources", []):
                if not isinstance(s, dict):
                    continue
                if "ticket_id" in s and s["ticket_id"]:
                    norm_sources.append({"ticket_id": str(s["ticket_id"])})
                elif "doc_name" in s and s["doc_name"]:
                    norm_sources.append({"doc_name": str(s["doc_name"])})
            data["sources"] = norm_sources
            return data
        except Exception:
            # fallback minimal structure
            return self._fallback_generate(query, contexts)

    def _fallback_generate(self, query: str, contexts: List[Dict]) -> Dict:
        # Deterministic heuristic synthesis if no LLM available
        root_causes = []
        resolution_steps = []
        sources = []
        for c in contexts:
            txt = c["text"]
            if "root_cause:" in txt:
                # simple extraction
                lines = [l.strip() for l in txt.split("\n") if l.strip()]
                for l in lines:
                    if l.startswith("root_cause:"):
                        rc = l.split("root_cause:", 1)[1].strip()
                        if rc and rc not in root_causes:
                            root_causes.append(rc)
                    if l.startswith("resolution:"):
                        rs = l.split("resolution:", 1)[1].strip()
                        if rs and rs not in resolution_steps:
                            resolution_steps.append(rs)
            # Capture source metadata in normalized form
            meta = c["source_meta"] or {}
            if meta.get("type") == "ticket" and meta.get("ticket_id"):
                sources.append({"ticket_id": str(meta.get("ticket_id"))})
            elif meta.get("type") == "kb" and meta.get("doc_name"):
                sources.append({"doc_name": str(meta.get("doc_name"))})
            else:
                # minimal fallback
                sources.append({"source": str(c.get("source", "unknown"))})
        # Fallback guards
        if not root_causes:
            root_causes = ["No exact historical match found; inferred likely causes based on similar issues."]
        if not resolution_steps:
            resolution_steps = [
                "Verify connectivity and credentials.",
                "Check logs for specific error codes.",
                "Apply the closest known fix from related tickets or KB.",
            ]
        return {
            "root_causes": root_causes[:5],
            "resolution_steps": resolution_steps[:8],
            "sources": sources,
        }

    def generate(self, query: str, contexts: List[Dict]) -> Dict:
        if self._have_openai():
            return self._openai_generate(query, contexts)
        return self._fallback_generate(query, contexts)
