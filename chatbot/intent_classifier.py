"""Hybrid intent classifier.

score(intent) = W_SIM * similarity + W_KW * keyword evidence

* similarity  - highest cosine similarity between the query and the intent's
                example utterances (TF-IDF vectors of the pre-processed text)
* keyword     - 1.0 if a high-precision trigger word of the intent occurs in
                the query lemmas, otherwise 0

The intent with the best score wins; if even the best score is below
``threshold`` the query is treated as out of scope.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .knowledge_base import DATA_DIR
from .vector_store import TfidfEmbedder

W_SIM, W_KW = 0.6, 0.4


class IntentClassifier:
    def __init__(self, preprocessor, path: str | Path | None = None, threshold: float = 0.30):
        path = Path(path) if path else DATA_DIR / "intents.json"
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self.intents = data["intents"]
        self.smalltalk = data["smalltalk_responses"]
        self.pre = preprocessor
        self.threshold = threshold

        # the vocabulary of the training examples feeds the spelling corrector
        for spec in self.intents.values():
            for ex in spec["examples"]:
                self.pre.add_vocabulary(self.pre.normalise(ex).split())

        self.labels, texts = [], []
        for name, spec in self.intents.items():
            for ex in spec["examples"]:
                self.labels.append(name)
                texts.append(self.pre.process(ex).lemma_text or ex)
        self.labels = np.array(self.labels)
        self.embedder = TfidfEmbedder().fit(texts)
        self.matrix = self.embedder.transform(texts)
        self.keywords = {name: {self.pre.process(k).lemma_text for k in spec["keywords"]} - {""}
                         for name, spec in self.intents.items()}

    def scores(self, processed) -> dict[str, float]:
        text = processed.lemma_text
        if not text:
            return {name: 0.0 for name in self.intents}
        sims = self.matrix @ self.embedder.transform([text])[0]
        lemma_set = set(processed.lemmas)
        joined = " " + text + " "
        out = {}
        for name in self.intents:
            sim = float(sims[self.labels == name].max())
            kw = any((k in lemma_set) if " " not in k else (f" {k} " in joined)
                     for k in self.keywords[name])
            out[name] = W_SIM * sim + W_KW * (1.0 if kw else 0.0)
        return out

    def predict(self, processed) -> tuple[str, float, dict[str, float]]:
        sc = self.scores(processed)
        best = max(sc, key=sc.get)
        if sc[best] < self.threshold:
            return "out_of_scope", sc[best], sc
        return best, sc[best], sc
