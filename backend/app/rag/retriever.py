from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .vector_store import Document


@dataclass
class RetrievalResult:
    doc: Document
    score: float


def retrieve(query: str, documents: Iterable[Document], top_k: int = 3) -> list[RetrievalResult]:
    results: list[RetrievalResult] = []
    query_lower = query.lower()
    for doc in documents:
        score = 1.0 if query_lower in doc.content.lower() else 0.2
        results.append(RetrievalResult(doc=doc, score=score))
    results.sort(key=lambda item: item.score, reverse=True)
    return results[:top_k]
