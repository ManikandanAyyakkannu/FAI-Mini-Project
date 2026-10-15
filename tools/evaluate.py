import sys
import pandas as pd
from chatbot import UniversityChatbot
from chatbot.baselines import KeywordFAQBot, RetrievalBot
from chatbot.evaluation import load_test_set, run_bot, summarise, per_category

pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 70)
df = load_test_set()
bots = {"Hybrid": UniversityChatbot(), "Retrieval": RetrievalBot(), "Keyword": KeywordFAQBot()}
summ, cats = {}, {}
for name, bot in bots.items():
    res = run_bot(bot, df)
    summ[name] = summarise(res)
    cats[name] = per_category(res)
    if name == "Hybrid" or "-all" in sys.argv:
        bad = res[~res.answer_ok]
        print(f"\n=== {name} failures ({len(bad)}) ===")
        print(bad[["id", "query", "expected_key", "pred_key", "pred_intent", "confidence", "source"]].to_string())
print(pd.DataFrame(summ).round(3))
print(pd.DataFrame(cats).round(3))
