import pandas as pd
from chatbot import UniversityChatbot
from chatbot.baselines import KeywordFAQBot, RetrievalBot
from chatbot.evaluation import load_test_set, run_bot, summarise
from chatbot.knowledge_base import DATA_DIR

pd.set_option("display.width", 250)
df = load_test_set(DATA_DIR / "test_queries_holdout.csv")
summ = {}
for name, bot in {"Hybrid": UniversityChatbot(), "Retrieval": RetrievalBot(), "Keyword": KeywordFAQBot()}.items():
    res = run_bot(bot, df)
    summ[name] = summarise(res)
    if name == "Hybrid":
        print(res[~res.answer_ok][["id", "query", "expected_key", "pred_key", "source"]].to_string())
print(pd.DataFrame(summ).round(3))
