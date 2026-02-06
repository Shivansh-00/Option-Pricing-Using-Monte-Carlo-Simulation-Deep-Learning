from __future__ import annotations

from pathlib import Path

from .rag import retriever, vector_store
from .schemas import ExplainRequest


def build_explanation(request: ExplainRequest) -> tuple[str, list[str]]:
    question = request.question.strip()
    context = request.context
    key_factors = ", ".join(sorted(context.keys())) if context else "market context"
    kb_path = Path(__file__).parent / "rag" / "knowledge_base"
    documents = vector_store.load_documents(kb_path)
    retrieved = retriever.retrieve(question or "pricing", documents)
    sources = [item.doc.title for item in retrieved]
    answer = (
        f"The pricing engine considered {key_factors}. "
        "Monte Carlo paths and Black-Scholes analytics were blended, "
        "and the deep learning residual model adjusted for recent regime shifts. "
        f"Retrieved sources: {', '.join(sources) if sources else 'N/A'}."
    )
    if question:
        answer = f"Q: {question}\nA: {answer}"
    return answer, sources
