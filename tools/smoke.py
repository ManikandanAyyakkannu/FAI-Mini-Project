import sys
from chatbot import UniversityChatbot

bot = UniversityChatbot()
s = bot.new_session()
qs = sys.argv[1:] or ["hi", "what is the fee for btech cse", "how long is it?", "what about mba?",
      "am i eligible for mtech ai? i did b.e ece with 72%", "Can I join MCA after BSc?", "I got 58 percent",
      "who is the HOD of CSE", "hostle fees?", "is the campus ragging free",
      "i scored 93% in 12th, what will be my fee for btech cse",
      "what is the weather today", "last date to apply for mba", "library timings", "placement for ece",
      "I have 7.2 CGPA in BCA. Am I eligible for MBA?", "which department offers mtech data science"]
verbose = "-v" in qs
qs = [q for q in qs if q != "-v"]
for q in qs:
    r = bot.respond(q, s)
    print(f"Q: {q}\n  [{r.intent} {r.confidence:.2f} | {r.source} | {r.answer_key}]\n  A: {r.text[:300]}")
    if verbose:
        print("  TRACE:\n    " + "\n    ".join(r.trace))
    print()
