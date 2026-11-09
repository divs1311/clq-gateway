"""Semantic cache for non-personalised answers (FAQ / how-to only).

Design rule: only answers whose root cause is in `eligible_root_causes` are ever stored, so a
merchant-specific diagnosis can never be replayed to another merchant.

The default embedder is an offline TF-IDF (word + char n-gram) model so the project runs without
keys. In production swap in a real embedding model (LiteLLMEmbedder) and pgvector; the interface
is the same: embed(list[str]) -> L2-normalised np.ndarray.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


class TfidfEmbedder:
    def __init__(self, corpus: list[str]):
        self.word = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english").fit(corpus)
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True).fit(corpus)

    def embed(self, texts: list[str]) -> np.ndarray:
        x = hstack([self.word.transform(texts), self.char.transform(texts)]).tocsr()
        return normalize(x).toarray()


class LiteLLMEmbedder:  # pragma: no cover - needs network + keys
    def __init__(self, model: str):
        import litellm
        self.litellm, self.model = litellm, model

    def embed(self, texts: list[str]) -> np.ndarray:
        resp = self.litellm.embedding(model=self.model, input=texts)
        vecs = np.array([d["embedding"] for d in resp.data], dtype=float)
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


@dataclass
class CacheHit:
    faq_id: str
    similarity: float
    output: dict


class SemanticCache:
    def __init__(self, entries: list[dict], threshold: float, embedder=None):
        self.entries = entries
        self.threshold = threshold
        self.embedder = embedder or TfidfEmbedder([e["question"] for e in entries])
        self.matrix = self.embedder.embed([e["question"] for e in entries])

    @classmethod
    def from_seed(cls, path: Path, threshold: float, embedder=None) -> "SemanticCache":
        entries = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
        return cls(entries, threshold, embedder)

    def best(self, query: str) -> tuple[int, float]:
        sims = self.matrix @ self.embedder.embed([query])[0]
        i = int(np.argmax(sims))
        return i, float(sims[i])

    def lookup(self, query: str, context_ids: set[str]) -> Optional[CacheHit]:
        i, sim = self.best(query)
        e = self.entries[i]
        if sim < self.threshold:
            return None
        return CacheHit(e["faq_id"], sim, {
            "root_cause_code": "HOW_TO_QUESTION", "tool_call": None, "escalate_to_human": False,
            "customer_reply": e["answer"],
            "cited_chunks": [e["kb_id"]] if e["kb_id"] in context_ids else [],
            "confidence": 1.0,
        })
