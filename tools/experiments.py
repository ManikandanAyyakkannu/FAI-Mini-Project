import pandas as pd
from chatbot import UniversityChatbot
from chatbot.evaluation import load_test_set, run_bot, summarise, per_category, retrieval_recall_at_k
from chatbot.knowledge_base import DATA_DIR

pd.set_option("display.width", 250)
df = load_test_set()
ho = load_test_set(DATA_DIR / "test_queries_holdout.csv")
configs = {
    "Full hybrid": {},
    "- reasoning": {"use_reasoning": False},
    "- dialogue context": {"use_context": False},
    "- spell correction": {"use_spell_correction": False},
    "- vector fallback": {"use_vector_fallback": False},
}
rows, cats = {}, {}
for name, kw in configs.items():
    res = run_bot(UniversityChatbot(**kw), df)
    rows[name] = summarise(res)
    cats[name] = per_category(res)
print(pd.DataFrame(rows).T.round(3))
print(pd.DataFrame(cats).round(3))

bot = UniversityChatbot()
print("recall@1", retrieval_recall_at_k(bot.store, bot.pre, df, 1), "recall@3", retrieval_recall_at_k(bot.store, bot.pre, df, 3))
for th in [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
    s = summarise(run_bot(UniversityChatbot(intent_threshold=th), df))
    print(th, round(s["Answer accuracy"], 3), round(s["Out-of-scope recall"], 3), round(s["Out-of-scope precision"], 3))
res = run_bot(bot, ho)
print("holdout", summarise(res))
print(res[~res.answer_ok][["id", "query", "expected_key", "pred_key"]])
