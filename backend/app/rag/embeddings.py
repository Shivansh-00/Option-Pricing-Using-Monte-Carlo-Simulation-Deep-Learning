"""
Enterprise Embedding Engine
============================
Multi-backend embedding system with caching, batching, and fallbacks.
Supports: TF-IDF (default), Sentence-Transformers (optional), and custom.
"""

from __future__ import annotations

import hashlib
import logging
import os
import pickle
import time
from abc import ABC, abstractmethod
from pathlib import Path
from threading import Lock

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

# ── Abstract Embedding Backend ────────────────────────────────────────────


class EmbeddingBackend(ABC):
    """Interface for embedding backends."""

    @abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode a list of texts into dense vectors."""
        ...

    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        ...

    @property
    def name(self) -> str:
        return self.__class__.__name__


# ── TF-IDF Backend (always available) ────────────────────────────────────


class TfidfBackend(EmbeddingBackend):
    """TF-IDF vectorizer as an embedding backend (zero extra deps)."""

    def __init__(
        self,
        max_features: int = 10_000,
        ngram_range: tuple[int, int] = (1, 2),
        sublinear_tf: bool = True,
    ) -> None:
        self._vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=ngram_range,
            max_features=max_features,
            sublinear_tf=sublinear_tf,
            min_df=1,
            max_df=0.95,
        )
        self._fitted = False
        self._dim = max_features

    def fit(self, corpus: list[str]) -> None:
        self._vectorizer.fit(corpus)
        self._fitted = True
        self._dim = len(self._vectorizer.vocabulary_)

    def encode(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            # Auto-fit on first encode (not ideal but safe)
            self.fit(texts)
        return self._vectorizer.transform(texts).toarray().astype(np.float32)

    def dimension(self) -> int:
        return self._dim

    @property
    def vectorizer(self) -> TfidfVectorizer:
        return self._vectorizer


# ── Sentence-Transformer Backend (optional) ──────────────────────────────


class SentenceTransformerBackend(EmbeddingBackend):
    """
    Dense embedding via sentence-transformers.
    Falls back gracefully if the library is not installed.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
        batch_size: int = 32,
    ) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        self._model = None
        self._dim = 384  # default for MiniLM

        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name, device=device)
            self._dim = self._model.get_sentence_embedding_dimension()
            logger.info(
                "SentenceTransformer loaded: %s (dim=%d, device=%s)",
                model_name, self._dim, device,
            )
        except ImportError:
            logger.warning(
                "sentence-transformers not installed — "
                "SentenceTransformerBackend unavailable. "
                "Install with: pip install sentence-transformers"
            )
        except Exception as exc:
            logger.warning("Failed to load SentenceTransformer: %s", exc)

    @property
    def available(self) -> bool:
        return self._model is not None

    def encode(self, texts: list[str]) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("SentenceTransformer model not loaded")
        embeddings = self._model.encode(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def dimension(self) -> int:
        return self._dim


# ── Embedding Cache ───────────────────────────────────────────────────────


class EmbeddingCache:
    """
    Disk-backed embedding cache to avoid re-computing embeddings.
    Uses content hashes as keys.
    """

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        if cache_dir is None:
            cache_dir = Path(__file__).parent / ".embedding_cache"
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._memory: dict[str, np.ndarray] = {}
        self._lock = Lock()
        self._hits = 0
        self._misses = 0

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def get(self, text: str) -> np.ndarray | None:
        key = self._hash(text)
        with self._lock:
            # Memory cache first
            if key in self._memory:
                self._hits += 1
                return self._memory[key]
            # Disk fallback
            path = self._dir / f"{key}.npy"
            if path.exists():
                try:
                    arr = np.load(path)
                    self._memory[key] = arr
                    self._hits += 1
                    return arr
                except Exception:
                    pass
            self._misses += 1
            return None

    def put(self, text: str, embedding: np.ndarray) -> None:
        key = self._hash(text)
        with self._lock:
            self._memory[key] = embedding
            try:
                np.save(self._dir / f"{key}.npy", embedding)
            except Exception:
                pass

    def get_batch(self, texts: list[str]) -> tuple[list[int], list[np.ndarray], list[int]]:
        """
        Check cache for a batch of texts.
        Returns: (cached_indices, cached_embeddings, uncached_indices)
        """
        cached_idx: list[int] = []
        cached_emb: list[np.ndarray] = []
        uncached_idx: list[int] = []

        for i, text in enumerate(texts):
            result = self.get(text)
            if result is not None:
                cached_idx.append(i)
                cached_emb.append(result)
            else:
                uncached_idx.append(i)

        return cached_idx, cached_emb, uncached_idx

    def put_batch(self, texts: list[str], embeddings: np.ndarray) -> None:
        for text, emb in zip(texts, embeddings):
            self.put(text, emb)

    @property
    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "memory_entries": len(self._memory),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 3) if total else 0.0,
        }

    def clear(self) -> None:
        with self._lock:
            self._memory.clear()
            for f in self._dir.glob("*.npy"):
                f.unlink(missing_ok=True)
            self._hits = 0
            self._misses = 0


# ── Main Embedding Engine ────────────────────────────────────────────────


class EmbeddingEngine:
    """
    Enterprise embedding engine with:
    - Multiple backend support (TF-IDF, SentenceTransformers)
    - Automatic fallback chain
    - Batch processing with caching
    - Performance telemetry
    """

    def __init__(
        self,
        backend: str = "auto",
        model_name: str = "all-MiniLM-L6-v2",
        cache_dir: str | Path | None = None,
        max_features: int = 10_000,
    ) -> None:
        self._cache = EmbeddingCache(cache_dir)
        self._total_encode_time = 0.0
        self._total_encoded = 0

        # Initialize backends
        self._tfidf = TfidfBackend(max_features=max_features)
        self._dense: SentenceTransformerBackend | None = None

        if backend in ("auto", "sentence-transformer", "dense"):
            self._dense = SentenceTransformerBackend(model_name=model_name)
            if not self._dense.available:
                self._dense = None

        # Select active backend
        if backend == "tfidf" or (backend == "auto" and self._dense is None):
            self._active = self._tfidf
            self._backend_name = "tfidf"
        elif self._dense is not None:
            self._active = self._dense
            self._backend_name = "sentence-transformer"
        else:
            self._active = self._tfidf
            self._backend_name = "tfidf"

        logger.info("EmbeddingEngine initialized with backend: %s", self._backend_name)

    @property
    def backend_name(self) -> str:
        return self._backend_name

    @property
    def dimension(self) -> int:
        return self._active.dimension()

    def fit(self, corpus: list[str]) -> None:
        """Fit the TF-IDF backend on corpus (required for TF-IDF)."""
        if isinstance(self._active, TfidfBackend):
            self._active.fit(corpus)

    def encode(
        self,
        texts: list[str],
        use_cache: bool = True,
        batch_size: int = 64,
    ) -> np.ndarray:
        """
        Encode texts into embeddings with caching and batching.

        Parameters
        ----------
        texts : list[str]
            Input texts to embed.
        use_cache : bool
            Whether to use the embedding cache.
        batch_size : int
            Batch size for encoding.

        Returns
        -------
        np.ndarray
            Embedding matrix of shape (n_texts, dim).
        """
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)

        t0 = time.time()

        if use_cache and self._backend_name != "tfidf":
            # TF-IDF is fast enough to skip caching
            cached_idx, cached_emb, uncached_idx = self._cache.get_batch(texts)

            if not uncached_idx:
                # All cached
                embeddings = np.zeros((len(texts), self.dimension), dtype=np.float32)
                for i, emb in zip(cached_idx, cached_emb):
                    embeddings[i] = emb
                return embeddings

            # Encode uncached
            uncached_texts = [texts[i] for i in uncached_idx]
            new_embeddings = self._encode_batched(uncached_texts, batch_size)
            self._cache.put_batch(uncached_texts, new_embeddings)

            # Merge
            embeddings = np.zeros((len(texts), self.dimension), dtype=np.float32)
            for i, emb in zip(cached_idx, cached_emb):
                embeddings[i] = emb
            for i, idx in enumerate(uncached_idx):
                embeddings[idx] = new_embeddings[i]
        else:
            embeddings = self._encode_batched(texts, batch_size)

        elapsed = time.time() - t0
        self._total_encode_time += elapsed
        self._total_encoded += len(texts)
        return embeddings

    def _encode_batched(self, texts: list[str], batch_size: int) -> np.ndarray:
        """Encode in batches to manage memory."""
        if len(texts) <= batch_size:
            return self._active.encode(texts)

        results: list[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            results.append(self._active.encode(batch))
        return np.vstack(results)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query (no caching for queries)."""
        return self._active.encode([query])[0]

    @property
    def stats(self) -> dict:
        avg_time = (
            (self._total_encode_time / self._total_encoded * 1000)
            if self._total_encoded > 0 else 0.0
        )
        return {
            "backend": self._backend_name,
            "dimension": self.dimension,
            "total_encoded": self._total_encoded,
            "total_encode_time_ms": round(self._total_encode_time * 1000, 2),
            "avg_encode_time_ms": round(avg_time, 3),
            "cache": self._cache.stats,
        }

    @property
    def tfidf_vectorizer(self) -> TfidfVectorizer | None:
        """Access the underlying TF-IDF vectorizer if active."""
        if isinstance(self._active, TfidfBackend):
            return self._active.vectorizer
        return self._tfidf.vectorizer if self._tfidf._fitted else None


# ── Factory ───────────────────────────────────────────────────────────────


def create_embedding_engine(
    backend: str | None = None,
    model_name: str | None = None,
    cache_dir: str | Path | None = None,
) -> EmbeddingEngine:
    """
    Factory function to create an EmbeddingEngine with env-var overrides.
    """
    backend = backend or os.getenv("RAG_EMBEDDING_BACKEND", "auto")
    model_name = model_name or os.getenv(
        "RAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2",
    )
    max_features = int(os.getenv("RAG_TFIDF_MAX_FEATURES", "10000"))
    return EmbeddingEngine(
        backend=backend,
        model_name=model_name,
        cache_dir=cache_dir,
        max_features=max_features,
    )
