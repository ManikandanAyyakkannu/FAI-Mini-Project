"""A minimal in-memory vector store (the core idea behind a vector database).

Documents are converted to vectors by a pluggable *embedder*, L2-normalised and
stored in a matrix. A query is embedded the same way and the top-k documents
by cosine similarity are returned.

The default embedder is TF-IDF (word + character n-grams), which needs no
download. Any embedder with ``fit(texts)`` and ``transform(texts)`` can be
plugged in, e.g. a sentence-transformers model, without changing the store.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer


class Embedder(Protocol):
    def fit(self, texts: list[str]) -> "Embedder": ...
    def transform(self, texts: list[str]) -> np.ndarray: ...


class TfidfEmbedder:
    """Sparse lexical embedding: word 1-2 grams + character 3-5 grams (robust to typos)."""

    def __init__(self):
        self.word = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)

    def fit(self, texts):
        self.word.fit(texts)
        self.char.fit(texts)
        return self

    def transform(self, texts):
        m = hstack([self.word.transform(texts), self.char.transform(texts) * 0.7]).toarray()
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        return m / np.where(norms == 0, 1, norms)


class SentenceTransformerEmbedder:
    """Optional dense semantic embedding (pip install sentence-transformers)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # imported lazily
        self.model = SentenceTransformer(model_name)

    def fit(self, texts):
        return self

    def transform(self, texts):
        return np.asarray(self.model.encode(texts, normalize_embeddings=True))


@dataclass
class SearchHit:
    score: float
    text: str
    metadata: dict[str, Any]


class VectorStore:
    def __init__(self, embedder: Embedder | None = None):
        self.embedder = embedder or TfidfEmbedder()
        self.texts: list[str] = []
        self.metadata: list[dict[str, Any]] = []
        self.matrix: np.ndarray | None = None

    def add(self, texts: list[str], metadata: list[dict[str, Any]]) -> None:
        self.texts.extend(texts)
        self.metadata.extend(metadata)
        self.embedder.fit(self.texts)          # re-fit so the vocabulary covers all documents
        self.matrix = self.embedder.transform(self.texts)

    def search(self, query: str, k: int = 3) -> list[SearchHit]:
        if self.matrix is None:
            return []
        q = self.embedder.transform([query])[0]
        scores = self.matrix @ q
        top = np.argsort(-scores)[:k]
        return [SearchHit(float(scores[i]), self.texts[i], self.metadata[i]) for i in top]

    def __len__(self):
        return len(self.texts)
