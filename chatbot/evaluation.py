"""Evaluation harness: runs any bot over the labelled test set and computes metrics.

A bot only needs ``new_session()`` and ``respond(text, session) -> BotResponse``.
Queries with the same ``conv_id`` are played in one session (multi-turn tests);
every other query gets a fresh session.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from .knowledge_base import DATA_DIR


def load_test_set(path: str | Path | None = None) -> pd.DataFrame:
    df = pd.read_csv(path or DATA_DIR / "test_queries.csv", dtype=str).fillna("")
    return df


def run_bot(bot, df: pd.DataFrame) -> pd.DataFrame:
    rows, sessions = [], {}
    for _, t in df.iterrows():
        if t.conv_id:
            if t.conv_id not in sessions:
                sessions[t.conv_id] = bot.new_session()
            session = sessions[t.conv_id]
        else:
            session = bot.new_session()
        r = bot.respond(t.query, session)
        accepted = t.expected_key.split("|")
        rows.append({
            "id": t.id, "category": t.category, "query": t.query,
            "expected_intent": t.expected_intent, "pred_intent": r.intent,
            "expected_key": accepted[0], "pred_key": r.answer_key,
            "source": r.source, "confidence": round(r.confidence, 3),
            "intent_ok": r.intent == t.expected_intent,
            "answer_ok": r.answer_key in accepted,
            "latency_ms": r.latency_ms, "response": r.text,
        })
    return pd.DataFrame(rows)


def summarise(res: pd.DataFrame) -> dict:
    p, r, f1, _ = precision_recall_fscore_support(res.expected_intent, res.pred_intent,
                                                  average="macro", zero_division=0)
    oos_true = res.expected_key == "fallback"
    oos_pred = res.pred_key == "fallback"
    tp = int((oos_true & oos_pred).sum())
    oos_precision = tp / oos_pred.sum() if oos_pred.sum() else 0.0
    oos_recall = tp / oos_true.sum() if oos_true.sum() else 0.0
    return {
        "Answer accuracy": res.answer_ok.mean(),
        "Intent accuracy": res.intent_ok.mean(),
        "Intent macro precision": p,
        "Intent macro recall": r,
        "Intent macro F1": f1,
        "Out-of-scope precision": oos_precision,
        "Out-of-scope recall": oos_recall,
        "Mean latency (ms)": res.latency_ms.mean(),
    }


def per_category(res: pd.DataFrame) -> pd.Series:
    return res.groupby("category", sort=False).answer_ok.mean()


def retrieval_recall_at_k(store, pre, df: pd.DataFrame, k: int = 3) -> float:
    """Recall@k of the vector store on the free-text FAQ questions."""
    sub = df[df.category == "unstructured_faq"]
    hits = 0
    for _, t in sub.iterrows():
        keys = [h.metadata["key"] for h in store.search(pre.normalise(t.query), k=k)]
        hits += t.expected_key in keys
    return hits / len(sub) if len(sub) else 0.0
