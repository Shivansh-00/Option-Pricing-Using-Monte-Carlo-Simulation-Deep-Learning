"""
Enterprise Hybrid Vector Store
===============================
Production-grade vector store with:
- Hybrid search (dense embeddings + BM25+ sparse)
- Reciprocal Rank Fusion (RRF)
- Metadata filtering
- Index persistence (save/load)
- Performance telemetry
"""

from __future__ import annotations

import logging
import math
import os
import pickle
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .chunking import Chunk, ChunkMetadata, ChunkStrategy, chunk_document
from .embeddings import EmbeddingEngine, create_embedding_engine

logger = logging.getLogger(__name__)

# ── Data Models ───────────────────────────────────────────────────────────


@dataclass
class Document:
    """A document chunk stored in the vector index."""
    title: str
    content: str
    source: str
    chunk_id: int
    page: int | None = None
    headings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """A single search result with score and provenance."""
    doc: Document
    score: float
    dense_score: float = 0.0
    sparse_score: float = 0.0
    rrf_rank: int = 0


# ── Tokenizer ─────────────────────────────────────────────────────────────

_STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "above", "below", "between", "out", "off", "over",
    "under", "again", "further", "then", "once", "here", "there", "when",
    "where", "why", "how", "all", "each", "every", "both", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only", "own",
    "same", "so", "than", "too", "very", "just", "about", "up", "it",
    "its", "this", "that", "these", "those", "i", "me", "my", "we", "our",
    "you", "your", "he", "him", "his", "she", "her", "they", "them",
    "their", "what", "which", "who", "whom", "and", "but", "if", "or",
}


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+(?:[-'][a-z0-9]+)*", text.lower())
    return [t for t in tokens if t not in _STOP_WORDS and len(t) > 1]


# ── BM25+ Engine ──────────────────────────────────────────────────────────


class BM25:
    """Okapi BM25+ ranking with inverted index for speed."""

    def __init__(
        self,
        corpus: list[list[str]],
        k1: float = 1.5,
        b: float = 0.75,
        delta: float = 0.5,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.delta = delta
        self.corpus_size = len(corpus)
        self.avgdl = sum(len(doc) for doc in corpus) / max(self.corpus_size, 1)
        self.doc_len = [len(doc) for doc in corpus]
        self.corpus = corpus

        self.df: dict[str, int] = {}
        self.tf: list[dict[str, int]] = []
        self.inverted_index: dict[str, set[int]] = {}

        for doc_idx, doc_tokens in enumerate(corpus):
            freq = Counter(doc_tokens)
            self.tf.append(freq)
            for token in freq:
                self.df[token] = self.df.get(token, 0) + 1
                if token not in self.inverted_index:
                    self.inverted_index[token] = set()
                self.inverted_index[token].add(doc_idx)

    def _idf(self, term: str) -> float:
        n = self.df.get(term, 0)
        return math.log((self.corpus_size - n + 0.5) / (n + 0.5) + 1.0)

    def score(self, query_tokens: list[str]) -> np.ndarray:
        scores = np.zeros(self.corpus_size, dtype=np.float64)
        for token in query_tokens:
            if token not in self.inverted_index:
                continue
            idf = self._idf(token)
            for idx in self.inverted_index[token]:
                tf = self.tf[idx].get(token, 0)
                dl = self.doc_len[idx]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[idx] += idf * (numerator / denominator + self.delta)
        return scores


# ── Hybrid Vector Store ──────────────────────────────────────────────────


class VectorStore:
    """
    Enterprise hybrid vector store combining dense and sparse retrieval.

    Features:
    - Dense embeddings (Sentence-Transformer or TF-IDF)
    - BM25+ sparse retrieval with inverted index
    - Reciprocal Rank Fusion (RRF) for result merging
    - Configurable search weights
    - Metadata-based filtering
    - Index persistence
    - Performance telemetry
    """

    def __init__(
        self,
        documents: Iterable[Document],
        embedding_engine: EmbeddingEngine | None = None,
        dense_weight: float = 0.4,
        sparse_weight: float = 0.6,
        use_rrf: bool = True,
        rrf_k: int = 60,
    ) -> None:
        self.documents = list(documents)
        self._build_time = time.time()
        self._dense_weight = dense_weight
        self._sparse_weight = sparse_weight
        self._use_rrf = use_rrf
        self._rrf_k = rrf_k
        self._query_count = 0
        self._total_latency = 0.0
        self._lock = Lock()

        self._embedding_engine = embedding_engine or create_embedding_engine()
        self._dense_matrix: np.ndarray | None = None
        self.bm25: BM25 | None = None
        self.tfidf: TfidfVectorizer | None = None
        self.tfidf_matrix = None

        if self.documents:
            self._build_index()

    def _build_index(self) -> None:
        t0 = time.time()
        corpus_text = [doc.content for doc in self.documents]

        # Sparse index
        corpus_tokens = [tokenize(text) for text in corpus_text]
        self.bm25 = BM25(corpus_tokens)

        # Dense index
        if self._embedding_engine.backend_name == "tfidf":
            self._embedding_engine.fit(corpus_text)
            self.tfidf = self._embedding_engine.tfidf_vectorizer
            if self.tfidf is not None:
                self.tfidf_matrix = self.tfidf.transform(corpus_text)
                self._dense_matrix = self.tfidf_matrix.toarray().astype(np.float32)
        else:
            self._dense_matrix = self._embedding_engine.encode(corpus_text)
            tfidf = TfidfVectorizer(
                stop_words="english", ngram_range=(1, 2),
                max_features=10_000, sublinear_tf=True, min_df=1, max_df=0.95,
            )
            tfidf.fit(corpus_text)
            self.tfidf = tfidf
            self.tfidf_matrix = tfidf.transform(corpus_text)

        logger.info(
            "VectorStore built: %d docs, backend=%s, %.1fms",
            len(self.documents), self._embedding_engine.backend_name,
            (time.time() - t0) * 1000,
        )

    @property
    def doc_count(self) -> int:
        return len(self.documents)

    @property
    def stats(self) -> dict:
        sources = set()
        total_words = 0
        for doc in self.documents:
            sources.add(doc.source)
            total_words += len(doc.content.split())

        avg_latency = (self._total_latency / self._query_count) if self._query_count else 0.0
        return {
            "total_chunks": self.doc_count,
            "unique_sources": len(sources),
            "source_files": sorted(sources),
            "total_words": total_words,
            "avg_chunk_words": total_words // max(self.doc_count, 1),
            "vocab_size": (
                len(self.tfidf.vocabulary_)
                if self.tfidf and hasattr(self.tfidf, "vocabulary_") else 0
            ),
            "embedding_backend": self._embedding_engine.backend_name,
            "embedding_dim": self._embedding_engine.dimension,
            "dense_weight": self._dense_weight,
            "sparse_weight": self._sparse_weight,
            "use_rrf": self._use_rrf,
            "index_built_at": self._build_time,
            "queries_served": self._query_count,
            "avg_search_ms": round(avg_latency * 1000, 2),
        }

    # ── Search ────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.01,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        if not query.strip() or not self.documents:
            return []

        t0 = time.time()
        dense_scores = self._dense_search(query)
        sparse_scores = self._sparse_search(query)

        if self._use_rrf:
            combined = self._reciprocal_rank_fusion(dense_scores, sparse_scores)
        else:
            combined = self._weighted_combination(dense_scores, sparse_scores)

        ranked_idx = np.argsort(combined)[::-1]
        results: list[SearchResult] = []
        for idx in ranked_idx[:top_k * 3]:
            score = float(combined[idx])
            if score < min_score:
                continue
            doc = self.documents[idx]
            if metadata_filter and not self._match_metadata(doc, metadata_filter):
                continue
            results.append(SearchResult(
                doc=doc, score=score,
                dense_score=float(dense_scores[idx]),
                sparse_score=float(sparse_scores[idx]),
            ))
            if len(results) >= top_k:
                break

        with self._lock:
            self._query_count += 1
            self._total_latency += time.time() - t0
        return results

    def _dense_search(self, query: str) -> np.ndarray:
        if self._dense_matrix is None:
            return np.zeros(len(self.documents))
        if self._embedding_engine.backend_name == "tfidf" and self.tfidf is not None:
            qvec = self.tfidf.transform([query]).toarray().astype(np.float32)
        else:
            qvec = self._embedding_engine.encode_query(query).reshape(1, -1)

        norms_doc = np.linalg.norm(self._dense_matrix, axis=1, keepdims=True)
        norms_doc = np.maximum(norms_doc, 1e-10)
        norm_q = max(np.linalg.norm(qvec), 1e-10)
        scores = (self._dense_matrix @ qvec.T).ravel() / (norms_doc.ravel() * norm_q)
        return np.clip(scores, 0, 1)

    def _sparse_search(self, query: str) -> np.ndarray:
        if self.bm25 is None:
            return np.zeros(len(self.documents))
        query_tokens = tokenize(query)
        if not query_tokens:
            return np.zeros(len(self.documents))
        return self.bm25.score(query_tokens)

    def _weighted_combination(self, dense: np.ndarray, sparse: np.ndarray) -> np.ndarray:
        return self._dense_weight * _normalize(dense) + self._sparse_weight * _normalize(sparse)

    def _reciprocal_rank_fusion(
        self, dense: np.ndarray, sparse: np.ndarray, k: int | None = None,
    ) -> np.ndarray:
        k = k or self._rrf_k
        n = len(dense)
        rrf = np.zeros(n, dtype=np.float64)
        dense_rank = np.argsort(np.argsort(-dense))
        sparse_rank = np.argsort(np.argsort(-sparse))
        for i in range(n):
            rrf[i] = (
                self._dense_weight / (k + dense_rank[i] + 1)
                + self._sparse_weight / (k + sparse_rank[i] + 1)
            )
        return rrf

    @staticmethod
    def _match_metadata(doc: Document, filters: dict[str, Any]) -> bool:
        for key, value in filters.items():
            doc_val = doc.metadata.get(key, getattr(doc, key, None))
            if doc_val is None:
                return False
            if isinstance(value, list):
                if doc_val not in value:
                    return False
            elif doc_val != value:
                return False
        return True

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        data = {
            "documents": self.documents,
            "dense_matrix": self._dense_matrix,
            "dense_weight": self._dense_weight,
            "sparse_weight": self._sparse_weight,
            "use_rrf": self._use_rrf,
            "rrf_k": self._rrf_k,
            "build_time": self._build_time,
        }
        with open(path / "index.pkl", "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        if self.tfidf is not None:
            with open(path / "tfidf.pkl", "wb") as f:
                pickle.dump(self.tfidf, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("VectorStore saved to %s", path)

    @classmethod
    def load(cls, path: str | Path, embedding_engine: EmbeddingEngine | None = None) -> VectorStore:
        path = Path(path)
        with open(path / "index.pkl", "rb") as f:
            data = pickle.load(f)  # noqa: S301

        store = cls.__new__(cls)
        store.documents = data["documents"]
        store._dense_matrix = data["dense_matrix"]
        store._dense_weight = data["dense_weight"]
        store._sparse_weight = data["sparse_weight"]
        store._use_rrf = data["use_rrf"]
        store._rrf_k = data["rrf_k"]
        store._build_time = data["build_time"]
        store._query_count = 0
        store._total_latency = 0.0
        store._lock = Lock()
        store._embedding_engine = embedding_engine or create_embedding_engine()

        corpus_tokens = [tokenize(doc.content) for doc in store.documents]
        store.bm25 = BM25(corpus_tokens)

        tfidf_path = path / "tfidf.pkl"
        if tfidf_path.exists():
            with open(tfidf_path, "rb") as f:
                store.tfidf = pickle.load(f)  # noqa: S301
            store.tfidf_matrix = store.tfidf.transform(
                [doc.content for doc in store.documents]
            )
        else:
            store.tfidf = None
            store.tfidf_matrix = None
        logger.info("VectorStore loaded from %s: %d docs", path, len(store.documents))
        return store


# ── Helpers ───────────────────────────────────────────────────────────────


def _normalize(arr: np.ndarray) -> np.ndarray:
    mx = arr.max()
    if mx <= 0:
        return arr
    return arr / mx


# ── File Loaders ──────────────────────────────────────────────────────────


def _load_pdf(file_path: Path) -> list[tuple[int, str]]:
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(str(file_path))
    except Exception:
        return []
    pages: list[tuple[int, str]] = []
    for idx, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((idx, text))
    return pages


def _load_html(file_path: Path) -> str:
    try:
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        text = re.sub(r"<script[^>]*>[\s\S]*?</script>", "", text)
        text = re.sub(r"<style[^>]*>[\s\S]*?</style>", "", text)
        text = re.sub(r"<[^>]+>", " ", text)
        return re.sub(r"\s+", " ", text).strip()
    except Exception:
        return ""


def _load_text(file_path: Path) -> str:
    try:
        return file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


# ── Document Ingestion ────────────────────────────────────────────────────


def load_documents(
    directory: str | Path,
    chunk_strategy: ChunkStrategy = ChunkStrategy.MARKDOWN,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """Load and chunk documents from a directory (recursive)."""
    chunk_size = chunk_size or int(os.getenv("RAG_CHUNK_SIZE", "600"))
    chunk_overlap = chunk_overlap or int(os.getenv("RAG_CHUNK_OVERLAP", "120"))
    documents: list[Document] = []
    supported = {".md", ".txt", ".pdf", ".rst", ".csv", ".html", ".htm"}
    directory = Path(directory)

    if not directory.exists():
        logger.warning("Knowledge base directory does not exist: %s", directory)
        return []

    for file in sorted(directory.rglob("*")):
        if not file.is_file() or file.suffix.lower() not in supported:
            continue
        try:
            if file.suffix.lower() == ".pdf":
                for page_num, text in _load_pdf(file):
                    chunks = chunk_document(
                        text, source_file=file.name,
                        strategy=chunk_strategy,
                        max_size=chunk_size, overlap=chunk_overlap,
                        page_number=page_num,
                    )
                    for c in chunks:
                        documents.append(_chunk_to_doc(c, file, page_num))
            elif file.suffix.lower() in {".html", ".htm"}:
                text = _load_html(file)
                if text:
                    for c in chunk_document(
                        text, source_file=file.name,
                        strategy=ChunkStrategy.RECURSIVE,
                        max_size=chunk_size, overlap=chunk_overlap,
                    ):
                        documents.append(_chunk_to_doc(c, file))
            else:
                text = _load_text(file)
                if not text.strip():
                    continue
                strat = ChunkStrategy.MARKDOWN if file.suffix.lower() == ".md" else chunk_strategy
                for c in chunk_document(
                    text, source_file=file.name, strategy=strat,
                    max_size=chunk_size, overlap=chunk_overlap,
                ):
                    documents.append(_chunk_to_doc(c, file))
        except Exception as exc:
            logger.error("Failed to process %s: %s", file, exc)

    logger.info("Loaded %d chunks from %s", len(documents), directory)
    return documents


def _chunk_to_doc(chunk: Chunk, file: Path, page: int | None = None) -> Document:
    m = chunk.metadata
    return Document(
        title=file.stem.replace("_", " ").title(),
        content=chunk.text,
        source=file.name,
        chunk_id=m.chunk_index,
        page=page or m.page_number,
        headings=m.headings,
        metadata={
            "section_title": m.section_title,
            "section_hierarchy": m.section_hierarchy,
            "word_count": m.word_count,
            "has_code": m.has_code,
            "has_formula": m.has_formula,
            "has_list": m.has_list,
            "has_table": m.has_table,
            "content_hash": m.content_hash,
            "strategy": m.strategy,
        },
    )


# ── Cached Singleton ──────────────────────────────────────────────────────

_STORE: VectorStore | None = None
_STORE_SIGNATURE: tuple[str, int, int] | None = None
_STORE_CREATED: float = 0.0
_CACHE_TTL: int = int(os.getenv("RAG_CACHE_TTL", "300"))


def _latest_mtime(directory: Path) -> int:
    try:
        mtimes = [int(f.stat().st_mtime) for f in directory.rglob("*") if f.is_file()]
        return max(mtimes) if mtimes else 0
    except Exception:
        return 0


def get_store(directory: str | Path, force_rebuild: bool = False) -> VectorStore:
    global _STORE, _STORE_SIGNATURE, _STORE_CREATED
    directory_path = Path(directory)
    file_count = sum(1 for f in directory_path.rglob("*") if f.is_file())
    mtime = _latest_mtime(directory_path)
    sig = (str(directory_path.resolve()), mtime, file_count)
    ttl_expired = (time.time() - _STORE_CREATED) > _CACHE_TTL if _STORE else True

    if force_rebuild or _STORE is None or _STORE_SIGNATURE != sig or ttl_expired:
        logger.info("Building VectorStore from %s (%d files)", directory_path, file_count)
        docs = load_documents(directory_path)
        _STORE = VectorStore(docs)
        _STORE_SIGNATURE = sig
        _STORE_CREATED = time.time()
        logger.info("VectorStore ready: %d chunks indexed", _STORE.doc_count)
    return _STORE
