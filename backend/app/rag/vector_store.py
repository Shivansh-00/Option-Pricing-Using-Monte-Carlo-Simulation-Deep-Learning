from __future__ import annotations

import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Document:
    title: str
    content: str
    source: str
    chunk_id: int
    page: int | None = None
    headings: list[str] = field(default_factory=list)


@dataclass
class SearchResult:
    doc: Document
    score: float

# ---------------------------------------------------------------------------
# BM25 implementation (lightweight, no extra deps)
# ---------------------------------------------------------------------------

class BM25:
    """Okapi BM25 ranking function."""

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.avgdl = sum(len(doc) for doc in corpus) / max(self.corpus_size, 1)
        self.doc_len = [len(doc) for doc in corpus]
        self.corpus = corpus

        self.df: dict[str, int] = {}
        self.tf: list[dict[str, int]] = []
        for doc_tokens in corpus:
            freq = Counter(doc_tokens)
            self.tf.append(freq)
            for token in freq:
                self.df[token] = self.df.get(token, 0) + 1

    def _idf(self, term: str) -> float:
        n = self.df.get(term, 0)
        return math.log((self.corpus_size - n + 0.5) / (n + 0.5) + 1.0)

    def score(self, query_tokens: list[str]) -> np.ndarray:
        scores = np.zeros(self.corpus_size)
        for token in query_tokens:
            idf = self._idf(token)
            for idx in range(self.corpus_size):
                tf = self.tf[idx].get(token, 0)
                dl = self.doc_len[idx]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[idx] += idf * numerator / denominator
        return scores

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Hybrid Vector Store (TF-IDF + BM25)
# ---------------------------------------------------------------------------

class VectorStore:
    def __init__(self, documents: Iterable[Document]) -> None:
        self.documents = list(documents)
        self.tfidf = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=8000,
            sublinear_tf=True,
        )
        self.bm25: BM25 | None = None
        self.tfidf_matrix = None

        if self.documents:
            corpus_text = [doc.content for doc in self.documents]
            self.tfidf_matrix = self.tfidf.fit_transform(corpus_text)
            corpus_tokens = [tokenize(text) for text in corpus_text]
            self.bm25 = BM25(corpus_tokens)

    @property
    def doc_count(self) -> int:
        return len(self.documents)

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.01,
    ) -> list[SearchResult]:
        if not query.strip() or self.tfidf_matrix is None or self.bm25 is None:
            return []

        qvec = self.tfidf.transform([query])
        tfidf_scores = (qvec @ self.tfidf_matrix.T).toarray().ravel()

        query_tokens = tokenize(query)
        bm25_scores = self.bm25.score(query_tokens)

        tfidf_norm = _normalize(tfidf_scores)
        bm25_norm = _normalize(bm25_scores)

        combined = 0.4 * tfidf_norm + 0.6 * bm25_norm

        ranked_idx = np.argsort(combined)[::-1]
        results: list[SearchResult] = []
        for idx in ranked_idx[: top_k * 3]:
            score = float(combined[idx])
            if score < min_score:
                continue
            results.append(SearchResult(doc=self.documents[idx], score=score))
        return results[:top_k]


def _normalize(arr: np.ndarray) -> np.ndarray:
    mx = arr.max()
    if mx <= 0:
        return arr
    return arr / mx

# ---------------------------------------------------------------------------
# Chunking (FIXED overlap)
# ---------------------------------------------------------------------------

def _chunk_text(text: str, chunk_size: int = 600, overlap: int = 120) -> list[str]:
    cleaned = " ".join(text.split())
    if not cleaned:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(len(cleaned), start + chunk_size)
        chunks.append(cleaned[start:end])
        if end >= len(cleaned):
            break
        start = end - overlap
    return chunks


def _extract_headings(text: str) -> list[str]:
    return re.findall(r"^#{1,4}\s+(.+)$", text, re.MULTILINE)

# ---------------------------------------------------------------------------
# File loaders
# ---------------------------------------------------------------------------

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


def _load_text(file_path: Path) -> str:
    try:
        return file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def load_documents(directory: str | Path) -> list[Document]:
    documents: list[Document] = []
    chunk_size = int(os.getenv("RAG_CHUNK_SIZE", "600"))
    overlap = int(os.getenv("RAG_CHUNK_OVERLAP", "120"))

    supported = {".md", ".txt", ".pdf", ".rst", ".csv"}
    for file in sorted(Path(directory).glob("*")):
        if not file.is_file() or file.suffix.lower() not in supported:
            continue

        if file.suffix.lower() == ".pdf":
            for page_num, text in _load_pdf(file):
                headings = _extract_headings(text)
                for cid, chunk in enumerate(_chunk_text(text, chunk_size, overlap)):
                    documents.append(Document(
                        title=file.stem.replace("_", " ").title(),
                        content=chunk,
                        source=file.name,
                        chunk_id=cid,
                        page=page_num,
                        headings=headings,
                    ))
        else:
            text = _load_text(file)
            headings = _extract_headings(text)
            for cid, chunk in enumerate(_chunk_text(text, chunk_size, overlap)):
                documents.append(Document(
                    title=file.stem.replace("_", " ").title(),
                    content=chunk,
                    source=file.name,
                    chunk_id=cid,
                    headings=headings,
                ))
    return documents

# ---------------------------------------------------------------------------
# Cached singleton
# ---------------------------------------------------------------------------

_STORE: VectorStore | None = None
_STORE_SIGNATURE: tuple[str, int, int] | None = None


def get_store(directory: str | Path) -> VectorStore:
    global _STORE, _STORE_SIGNATURE
    directory_path = Path(directory)
    file_count = sum(1 for f in directory_path.glob("*") if f.is_file())
    mtime = _latest_mtime(directory_path)
    sig = (str(directory_path.resolve()), mtime, file_count)
    if _STORE is None or _STORE_SIGNATURE != sig:
        docs = load_documents(directory_path)
        _STORE = VectorStore(docs)
        _STORE_SIGNATURE = sig
    return _STORE


def _latest_mtime(directory: Path) -> int:
    mtimes = [int(f.stat().st_mtime) for f in directory.glob("*") if f.is_file()]
    return max(mtimes) if mtimes else 0
