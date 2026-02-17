"""
Enterprise RAG (Retrieval-Augmented Generation) Subsystem
==========================================================
Production-grade RAG pipeline for option pricing Q&A.

Modules:
- chunking: Multi-strategy document chunking engine
- embeddings: Multi-backend embedding engine with caching
- vector_store: Hybrid vector store (dense + BM25+ sparse)
- retriever: Advanced retrieval with reranking & multi-hop
- prompt_engine: Citation-forcing prompt engineering
- llm_client: LLM integration with retry & circuit breaker
- evaluation: RAG quality metrics & monitoring
- guard_rails: Input validation & robustness
"""

from . import (
    chunking,
    embeddings,
    evaluation,
    guard_rails,
    llm_client,
    prompt_engine,
    retriever,
    vector_store,
)

__all__ = [
    "chunking",
    "embeddings",
    "evaluation",
    "guard_rails",
    "llm_client",
    "prompt_engine",
    "retriever",
    "vector_store",
]
