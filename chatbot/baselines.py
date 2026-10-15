"""Alternative approaches used for comparison.

1. KeywordFAQBot   - classic FAQ-matching chatbot: raw lower-cased text,
                     word TF-IDF, cosine similarity against FAQ questions.
                     No NLP pre-processing, no knowledge representation,
                     no reasoning, no dialogue memory.
2. RetrievalBot    - "NLP + vector store only": the same pre-processing and
                     vector store as the hybrid bot, but every answer is
                     retrieved from the FAQ corpus (no frames, no rules).
"""
from __future__ import annotations

import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .engine import FALLBACK_TEXT, BotResponse
from .faq_builder import build_faq
from .knowledge_base import KnowledgeBase
from .nlp import TextPreprocessor
from .vector_store import VectorStore


class KeywordFAQBot:
    name = "Baseline: keyword FAQ matching"

    def __init__(self, threshold: float = 0.20):
        self.faq = build_faq(KnowledgeBase())
        self.vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 2))
        self.matrix = self.vec.fit_transform([f["question"] for f in self.faq])
        self.threshold = threshold

    def new_session(self):
        return None

    def respond(self, message: str, session=None) -> BotResponse:
        t0 = time.perf_counter()
        sims = (self.matrix @ self.vec.transform([message]).T).toarray().ravel()
        i = int(np.argmax(sims))
        if sims[i] < self.threshold:
            r = BotResponse(FALLBACK_TEXT, "out_of_scope", float(sims[i]), "fallback", "fallback")
        else:
            f = self.faq[i]
            r = BotResponse(f["answer"], f["intent"], float(sims[i]), f["key"], "faq_match")
        r.latency_ms = (time.perf_counter() - t0) * 1000
        return r


class RetrievalBot:
    name = "NLP + vector store retrieval (no KR / reasoning)"

    def __init__(self, threshold: float = 0.30, embedder=None):
        self.pre = TextPreprocessor()
        self.faq = build_faq(KnowledgeBase())
        self.store = VectorStore(embedder)
        self.store.add([self.pre.normalise(f["question"]) for f in self.faq], self.faq)
        self.threshold = threshold

    def new_session(self):
        return None

    def respond(self, message: str, session=None) -> BotResponse:
        t0 = time.perf_counter()
        hit = self.store.search(self.pre.normalise(message), k=1)[0]
        if hit.score < self.threshold:
            r = BotResponse(FALLBACK_TEXT, "out_of_scope", hit.score, "fallback", "fallback")
        else:
            m = hit.metadata
            r = BotResponse(m["answer"], m["intent"], hit.score, m["key"], "vector_store")
        r.latency_ms = (time.perf_counter() - t0) * 1000
        return r
