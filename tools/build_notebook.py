"""Generates University_Enquiry_Chatbot.ipynb (then executed with nbconvert)."""
import sys
import nbformat as nbf

out_path = sys.argv[1]
nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# AI-Based University Enquiry Chatbot
### Knowledge Representation + Natural Language Processing
**M.Tech - Fundamentals of Artificial Intelligence · Mini Project (Topic 29)**

| | |
|---|---|
| **Team members** | Manikandan Ayyakkannu (RA2612049015035) · Dinakar B (RA2612049015032) · Manoj Kumar R (RA2612049015005) |
| **Technique** | Frame-based knowledge representation (semantic network) + production rules with forward chaining + an NLP pipeline + a vector-store retrieval fallback |
| **Solution files** | this notebook · `chatbot/` package (Python engine) · `index.html` + `web/` (the chatbot website) · `data/` |
| **Live chatbot** | https://manikandanayyakkannu.github.io/FAI-Mini-Project/ |

**How to run:** open the `university_chatbot` folder in VS Code, choose the Python 3.12 kernel and click **Run All**.
The chatbot itself runs in the browser: open the live link above, with nothing to install.

The notebook follows the nine requirements of the assignment:
1. Problem definition · 2. AI problem formulation · 3. Selection and explanation of the AI technique · 4. Python implementation ·
5. Dataset and test-case preparation · 6. Performance evaluation · 7. Comparison with alternative approaches ·
8. Results, discussion and conclusion · 9. Demonstration of the working prototype
""")

code(r"""
import sys, time, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
pd.set_option("display.max_colwidth", 90)
pd.set_option("display.width", 200)

sys.path.insert(0, str(Path.cwd()))
from chatbot import UniversityChatbot
from chatbot.knowledge_base import KnowledgeBase, DATA_DIR
from chatbot.nlp import TextPreprocessor, EntityExtractor
from chatbot.reasoning import ReasoningEngine
from chatbot.vector_store import VectorStore
from chatbot.baselines import KeywordFAQBot, RetrievalBot
from chatbot.evaluation import load_test_set, run_bot, summarise, per_category, retrieval_recall_at_k

FIG = Path("docs/figures"); FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3, "font.size": 10})
COLORS = {"Hybrid KR + NLP (proposed)": "#2a6fdb", "NLP + vector retrieval": "#f2a33a", "Keyword FAQ matching": "#9aa5b1"}
print("Python", sys.version.split()[0], "| pandas", pd.__version__, "| numpy", np.__version__)
""")

# ------------------------------------------------------------------ 1
md(r"""
## 1. Problem Definition

A university answers thousands of repetitive enquiries every year, by phone, e-mail and at many different counters.
Applicants ask *"What is the fee for B.Tech CSE?"* or *"Am I eligible for M.Tech AI with 6.8 CGPA?"*; current students ask
*"When do the end-semester exams start?"*, *"How do I get a bonafide certificate?"* or *"Who should I contact about my hostel room?"*;
alumni ask for transcripts. The answers are spread across the admissions office, the exam cell, the registrar, the accounts office and
the departments, they are not available outside office hours, and they are sometimes inconsistent.

**Goal:** build one intelligent enquiry chatbot for the **whole university** that
* understands free-form English questions, including paraphrases, abbreviations (*B.E*, *M.Tech*, *CGPA*) and spelling mistakes;
* answers factual questions from a structured **knowledge base** of the (fictional) *AI University*: admissions and
  programmes, fees, the academic calendar, exams and results, certificates, university offices, departments, hostel,
  placements, events, research and alumni;
* **reasons** about personal questions, such as eligibility for a programme or the fee after a scholarship, and **explains** each decision;
* keeps track of the **conversation** (follow-up questions such as *"how long is it?"*) and asks for missing details;
* answers free-text FAQs (ragging, Wi-Fi, refunds) through **semantic retrieval**, and politely refuses out-of-scope questions.

**Scope and assumptions:** the knowledge base describes a fictional university, *AI University*, with 20 programmes (engineering,
arts, science, commerce and management), 11 departments, 9 university offices, 7 kinds of certificate, 5 facilities, the
academic calendar, the examination rules, fee payment, events, research, alumni, the admission process and the scholarship policy.
All data is synthetic and created only for this project. The chatbot is text-based, works in English and runs in any web browser.
Anything that is not in the knowledge base (a department the university does not have, or another university) is
answered with a polite "not available" instead of a guess.
""")

# ------------------------------------------------------------------ 2
md(r"""
## 2. AI Problem Formulation

The chatbot is a **knowledge-based agent**: it perceives a sentence, updates its internal state (the dialogue context),
and acts by choosing and producing the answer that its knowledge base and inference rules support.

**PEAS description**

| PEAS | Description |
|---|---|
| **Performance measure** | correct answers (answer accuracy), correct intent (precision / recall / F1), correct refusal of out-of-scope questions, response time, explainability |
| **Environment** | applicants, parents, current students and alumni typing questions; the university's knowledge (frames, rules, FAQ passages) |
| **Actuators** | text answers, clarification questions, explanation traces |
| **Sensors** | the user's typed sentence |

**Environment properties:** partially observable (the user's intention and facts are hidden and must be inferred from text),
stochastic (the same need can be phrased in many ways), sequential (follow-up questions depend on earlier turns),
static, discrete, single-agent.

**Formulation as knowledge representation and reasoning**

* **Knowledge base (KB):** a set of *frames* `F = {University, Program, UGProgram, PGProgram, btech_cse, ..., hostel, ...}` linked by `is_a` edges.
  Slot lookup uses **inheritance**: `value(f, s) = f.s` if the frame stores the slot; otherwise `value(parent(f), s)`.
* **Percept to query mapping:** NLP maps a sentence *x* to a structured query `q = (intent, entities)`,
  where `intent ∈ {program_fee, eligibility, examination, certificates, office_info, ...}` (27 intents) and the entities are programme, qualification, %, CGPA, exam, category and so on.
* **Working memory (WM):** for a personal question, WM = facts about the programme taken from the frames, plus facts about the applicant taken from the sentence.
* **Rules:** production rules `IF c1 ∧ c2 ∧ ... THEN add fact` (16 eligibility rules and 10 scholarship rules).
* **Inference / goal test:** **forward chaining**. Rules fire until a goal fact (`academic_eligible`, `net_fee`) is derived or no rule can fire.
* **Fallback:** if no intent is confident, the query is embedded as a vector and the closest FAQ passage is retrieved (cosine similarity ≥ threshold);
  otherwise the agent answers "out of scope".
""")

code(r"""
kb = KnowledgeBase()
print(f"Frames: {len(kb.frames)} | Programmes: {len(kb.programs)} | Departments: {len(kb.instances_of('Department'))} "
      f"| Offices: {len(kb.instances_of('Office'))} | Certificates: {len(kb.instances_of('Certificate'))} "
      f"| Facilities: {len(kb.instances_of('Facility'))}")
print("\nis_a chain of 'mtech_ai':", " -> ".join(["mtech_ai"] + kb.ancestors("mtech_ai")))
for slot in ["name", "fee_per_year", "duration", "min_percentage", "application_fee", "phone"]:
    val, src = kb.get_with_source("mtech_ai", slot)
    print(f"  mtech_ai.{slot:<16} = {str(val):<42} (stored in frame: {src})")
""")

code(r"""
# The same knowledge seen as a semantic network of (subject, predicate, object) triples
print("Triples with predicate 'offers' (derived inverse of programme.department):")
for t in kb.query(predicate="offers"):
    print("  ", t)
print("\nTotal triples in the semantic network:", len(list(kb.triples())))
""")

code(r"""
# Draw a small part of the semantic network (is_a and department links)
import matplotlib.patches as mpatches
fig, ax = plt.subplots(figsize=(12, 5.6)); ax.axis("off"); ax.grid(False)
pos = {"University": (5.2, 4), "Program": (2.6, 3), "Department": (9.0, 3),
       "UGProgram": (1.2, 2), "ArtsScienceUGProgram": (-0.3, 1.15), "PGProgram": (3.6, 2), "ResearchProgram": (5.8, 2),
       "btech_cse": (1.5, 0.3), "bcom": (-0.3, 0.3), "mtech_ai": (2.9, 0.3), "mba": (4.2, 0.3), "phd": (5.8, 0.3),
       "dept_cse": (7.4, 0.3), "dept_aids": (8.9, 0.3), "dept_mgmt": (10.4, 0.3)}
curve = {("btech_cse", "dept_cse"): 0.14, ("mtech_ai", "dept_aids"): 0.2, ("mba", "dept_mgmt"): 0.2}
edges = [("Program", "University", "is_a"), ("Department", "University", "is_a"), ("UGProgram", "Program", "is_a"),
         ("ArtsScienceUGProgram", "UGProgram", "is_a"),
         ("PGProgram", "Program", "is_a"), ("ResearchProgram", "Program", "is_a"), ("btech_cse", "UGProgram", "is_a"),
         ("bcom", "ArtsScienceUGProgram", "is_a"), ("mtech_ai", "PGProgram", "is_a"), ("mba", "PGProgram", "is_a"),
         ("phd", "ResearchProgram", "is_a"), ("dept_cse", "Department", "is_a"), ("dept_aids", "Department", "is_a"),
         ("dept_mgmt", "Department", "is_a"),
         ("btech_cse", "dept_cse", "department"), ("mtech_ai", "dept_aids", "department"), ("mba", "dept_mgmt", "department")]
for a, b, lab in edges:
    (x1, y1), (x2, y2) = pos[a], pos[b]
    if lab == "is_a":
        ax.annotate("", xy=(x2, y2 - 0.18), xytext=(x1, y1 + 0.18),
                    arrowprops=dict(arrowstyle="->", color="#555", lw=1.2))
    else:   # department relation drawn as a curve below the instance row
        ax.annotate("", xy=(x2, y2 - 0.2), xytext=(x1, y1 - 0.2),
                    arrowprops=dict(arrowstyle="->", color="#2a6fdb", linestyle="--", lw=1.2,
                                    connectionstyle=f"arc3,rad={curve[(a, b)]}"))
for n, (x, y) in pos.items():
    cls = kb.frames[n].get("type") == "class"
    ax.text(x, y, n, ha="center", va="center", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.35", fc="#e8f0fe" if cls else "#fff4e0", ec="#2a6fdb" if cls else "#f2a33a"))
ax.legend(handles=[mpatches.Patch(fc="#e8f0fe", ec="#2a6fdb", label="class frame"),
                   mpatches.Patch(fc="#fff4e0", ec="#f2a33a", label="instance frame"),
                   plt.Line2D([], [], color="#555", label="is_a (inheritance)"),
                   plt.Line2D([], [], color="#2a6fdb", ls="--", label="department relation")],
          loc="upper right", fontsize=8, frameon=False)
ax.set_title("Fragment of the frame-based semantic network", fontsize=11)
ax.set_xlim(-1.4, 11.4); ax.set_ylim(-1.2, 4.4)
plt.tight_layout(); plt.savefig(FIG / "fig_semantic_network.png", dpi=150, bbox_inches="tight"); plt.show()
""")

md(r"""
**University-wide knowledge.** Besides admissions, the knowledge base holds frames for the university's offices, the certificates it
issues, the academic calendar, the examination rules, fee payment, events, research and alumni. Inheritance works here too:
every certificate inherits *how to apply* and the issuing office (the Registrar) from the `Certificate` class, and the provisional
certificate overrides the office with the exam cell. The exam dates are not repeated in the `examinations` frame; they are read
from the `academic_calendar` frame.
""")
code(r"""
for cert in ["cert_bonafide", "cert_tc", "cert_provisional"]:
    office, src = kb.get_with_source(cert, "office")
    print(f"{cert:17} office = {office:17} (from frame '{src}')   fee = Rs. {kb.get(cert, 'fee')}, time = {kb.get(cert, 'time')}")
print()
pd.DataFrame([{"office": o, "name": kb.get(o, "name"), "head": kb.get(o, "head"), "phone": kb.get(o, "phone"),
               "handles": kb.get(o, "handles")} for o in kb.instances_of("Office")])
""")

# ------------------------------------------------------------------ 3
md(r"""
## 3. Selection and Explanation of the AI Technique

The topic asks for **Knowledge Representation + NLP**. The design combines four classical AI components, each chosen for a reason:

| Component | Technique | Why it was chosen |
|---|---|---|
| **Knowledge representation** | Frames with `is_a` inheritance, which also form a semantic network of triples | University data is naturally hierarchical (Programme → UG/PG → B.Tech CSE). Inheritance avoids repetition: duration, documents and deadlines are stored once at the class level. |
| **Reasoning** | Production rules + **forward chaining** with refraction and an explanation trace | Eligibility and scholarship policies *are* IF-THEN rules. Forward chaining is data-driven: start from the applicant's facts and derive conclusions. Each fired rule becomes a human-readable justification. |
| **NLP** | Normalisation → tokenisation → spelling correction (edit-distance similarity) → synonym mapping → stop-word removal → stemming → regex-based entity extraction → hybrid intent classifier (TF-IDF similarity to examples + keyword evidence) | Transparent and explainable, with no training data beyond 112 example sentences and no internet or GPU needed. It handles abbreviations such as *B.E / B.Tech / M.Tech / CGPA*. |
| **Retrieval fallback** | Vector store with cosine similarity over TF-IDF word and character n-gram embeddings | Free-text FAQ passages (ragging, Wi-Fi, refunds) are not worth modelling as frames. A vector store finds them by similarity of meaning. The embedder is pluggable, so a neural sentence embedding can replace TF-IDF without code changes. |
| **Dialogue management** | Context memory + slot filling | Supports follow-up questions (*"how long is it?"*) and asks for missing facts before reasoning. |
| **Closed-world assumption** | Anything that is not a frame is not offered; a list of known missing departments and other institutions helps recognise such questions | *"HOD of the Civil department?"* gets "not available at AI University", and *"HOD of CSE in SRM?"* gets "I only know AI University", instead of a wrong but confident answer. |

**Why not a pure machine-learning or LLM chatbot?** For admissions, answers must be *exactly* correct (fees, cut-offs), *consistent* and *explainable*.
A knowledge-based system guarantees that every answer is traceable to a frame slot or a rule. An LLM may hallucinate numbers and cannot easily justify an eligibility decision.
The cost of the knowledge-based approach is manual knowledge engineering, which is discussed in Section 8.
""")

md("### 3.1 NLP pipeline on example sentences")
code(r"""
bot = UniversityChatbot()
examples = ["Wat is the fess for B.E Mechanical?",
            "I completed B.E. ECE with 6.8 CGPA, am I eligible for M.Tech AI?",
            "hostle accomodation for girls?",
            "I'm an OBC student with 57% in plus two (PCM). Can I join btech mech?"]
rows = []
for s in examples:
    p = bot.pre.process(s)
    e = bot.extractor.extract(p.normalised)
    intent, conf, _ = bot.classifier.predict(p)
    rows.append({"sentence": s, "normalised": p.normalised, "corrections": p.corrections,
                 "lemmas": " ".join(p.lemmas), "intent": f"{intent} ({conf:.2f})", "entities": e.as_dict()})
pd.DataFrame(rows).T
""")

md("### 3.2 Intent classification scores (top 4 intents per query)")
code(r"""
for s in ["How much does MBA cost?", "Who is the head of the ECE department?", "Is the campus ragging free?"]:
    sc = bot.classifier.scores(bot.pre.process(s))
    top = sorted(sc.items(), key=lambda kv: -kv[1])[:4]
    print(f"{s:42} ->", ", ".join(f"{k}={v:.2f}" for k, v in top))
""")

md("### 3.3 Forward chaining with an explanation trace")
code(r"""
reasoner = ReasoningEngine(kb)
applicant = {"qualification": "btech_ece", "cgpa": 6.8, "exam": "gate", "category": None}
result = reasoner.check_eligibility("mtech_ai", applicant)
print("Rules fired (in order):", [r.rule_id for r in result.fired])
for line in result.explanation_lines():
    print("  ", line)
print("\nDerived facts:", {k: result.facts[k] for k in ["percentage", "required_percentage", "qualification_ok",
                                                         "marks_ok", "academic_eligible", "entrance_ok", "stipend"]})
""")

code(r"""
# Scholarship reasoning: merit + sports quota, with the cap at 60%
r = reasoner.compute_fee("btech_cse", {"percentage": 96, "sports": True, "sibling": True})
for line in r.explanation_lines():
    print(line)
""")

md("### 3.4 Vector-store retrieval (fallback for free-text FAQs)")
code(r"""
for q in ["Is there any ragging in the campus?", "Can I get an education loan?", "do students need to wear uniform"]:
    hits = bot.store.search(bot.pre.normalise(q), k=3)
    print(q)
    for h in hits:
        print(f"    {h.score:.2f}  {h.metadata['key']:<28} {h.metadata['question']}")
""")

# ------------------------------------------------------------------ 4
md(r"""
## 4. Python Implementation

The engine is a small Python package, used by this notebook for all the experiments. The chatbot website runs
the **same engine** rewritten in JavaScript (`web/js/`, one file per module) and gives identical answers on all 361 test conversations.

```
university_chatbot/
├── chatbot/
│   ├── knowledge_base.py     frames, inheritance, semantic-network triples
│   ├── nlp.py                normalisation, spelling, synonyms, stemming, entity extraction
│   ├── intent_classifier.py  hybrid TF-IDF similarity + keyword intent classifier
│   ├── reasoning.py          forward-chaining production-rule engine (+ explanations)
│   ├── vector_store.py       in-memory vector store with pluggable embedder
│   ├── answers.py            natural-language answer templates filled from frames
│   ├── faq_builder.py        builds the FAQ corpus (for retrieval and the baselines)
│   ├── engine.py             UniversityChatbot: dialogue manager that ties everything together
│   ├── baselines.py          alternative approaches used for comparison
│   └── evaluation.py         metrics and the test harness
├── data/  knowledge_base.json · rules.json · intents.json · faq_extra.json · test_queries*.csv
├── index.html                the chatbot website
├── web/                      website code (JavaScript version of the engine, styles)
└── tests/engine-test.html    runs the test sets in the browser
```

**Processing pipeline for every message** (`UniversityChatbot.respond`):
1. NLP pre-processing, then entity extraction and intent classification
2. If a task is pending (for example, eligibility is waiting for marks), fill the slot and continue
3. Personal question with facts → **forward chaining** (eligibility or scholarship rules)
4. Dialogue context: resolve *"it"* and elliptical follow-ups (*"what about MBA?"*)
5. Domain gate, and **vector-store retrieval** for unknown or low-confidence questions
6. Otherwise answer from the **frame KB**; return the text plus intent, confidence, source and a full trace
""")

code(r"""
# Architecture diagram
from matplotlib.patches import FancyBboxPatch
fig, ax = plt.subplots(figsize=(13, 4)); ax.axis("off"); ax.grid(False)
W, H = 1.7, 0.9   # box width / height in data units

def box(cx, cy, text, col, w=W):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - H / 2), w, H, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=col, ec="#555", lw=1))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=9)

def arrow(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", color="#555", lw=1.1))

main = [(0.9, "User\nquestion", "#ffffff", 1.2), (3.0, "NLP pipeline\nnormalise · spell\nsynonyms · stem", "#e8f0fe", W),
        (5.1, "Entities +\nintent classifier", "#e8f0fe", W), (7.2, "Dialogue manager\ncontext · slots", "#f3e8ff", W)]
for cx, t, c, w in main:
    box(cx, 1.5, t, c, w)
for (a, _, _, wa), (b, _, _, wb) in zip(main, main[1:]):
    arrow(a + wa / 2, 1.5, b - wb / 2, 1.5)
branches = [(2.7, "Frame KB lookup\n(inheritance)", "#e6f6ea"), (1.5, "Forward-chaining\nrule engine", "#fff4e0"),
            (0.3, "Vector store\nretrieval", "#fde8e8")]
for cy, t, c in branches:
    box(9.6, cy, t, c)
    arrow(7.2 + W / 2, 1.5, 9.6 - W / 2, cy)
    arrow(9.6 + W / 2, cy, 12.0 - W / 2, 1.5)
box(12.0, 1.5, "Answer + intent +\nconfidence + trace", "#ffffff")
ax.set_xlim(0, 13); ax.set_ylim(-0.4, 3.3); ax.set_aspect("auto")
ax.set_title("Architecture of the hybrid KR + NLP enquiry chatbot", fontsize=11)
plt.tight_layout(); plt.savefig(FIG / "fig_architecture.png", dpi=150, bbox_inches="tight"); plt.show()
""")

code(r"""
# The rule base (loaded from data/rules.json)
rules = json.load(open(DATA_DIR / "rules.json"))
fmt = lambda c: f"{c['fact']} {c['op']} {c.get('value', c.get('value_fact', ''))}".strip()
pd.DataFrame([{"set": s, "id": r["id"], "name": r["name"], "IF": " AND ".join(fmt(c) for c in r["if"])}
              for s in ("eligibility_rules", "scholarship_rules") for r in rules[s]])
""")

code(r"""
# Knowledge base summary: all programmes with inherited slots resolved
pd.DataFrame([{"id": p, "name": kb.get(p, "name"), "level": kb.get(p, "level"), "duration": kb.get(p, "duration"),
               "fee/yr (Rs.)": kb.get(p, "fee_per_year"), "seats": kb.get(p, "seats"),
               "min %": kb.get(p, "min_percentage"), "entrance": ", ".join(kb.get(p, "entrance_exams", []))}
              for p in kb.programs])
""")

# ------------------------------------------------------------------ 5
md(r"""
## 5. Dataset and Test-Case Preparation

There is no public dataset for this fictional university, so the knowledge and four test sets were prepared:

1. **Knowledge sources** (`data/`): 70 frames, 26 production rules, 27 intents with 178 example sentences,
   and 175 FAQ passages (160 generated from the frames + 15 free-text passages).
2. **Main test set** (`test_queries.csv`, 105 queries): hand-written to be *different* from the intent examples,
   in 8 categories that stress different abilities: direct factual questions, paraphrases, spelling mistakes, reasoning questions,
   multi-turn conversations (5 dialogues, 14 turns), free-text FAQs, out-of-scope questions and small talk.
   Each query is labelled with the expected **intent** and the expected **answer key**
   (e.g. `program_fee:btech_ece`, `eligibility:mtech_ai:yes`). An answer counts as correct only if the key matches exactly.
   For reasoning questions this means the eligibility decision itself must be right.
3. **Held-out test set** (`test_queries_holdout.csv`, 25 queries): written *after* development finished and evaluated once, without any further changes to the system.
   It gives an unbiased estimate, because the main test set was also used to find bugs during development.
4. **University-services development set** (`test_queries_services.csv`, 40 queries): calendar, exams, results, fee payment,
   certificates, offices, research, events, alumni and two short conversations. It was written after these features were built
   and was then used to improve them, so its score is optimistic (the first run, before any changes, scored 72.5 %).
5. **University-services held-out set** (`test_queries_services_holdout.csv`, 20 queries): written *before* that tuning and scored once.
""")

code(r"""
test = load_test_set()
holdout = load_test_set(DATA_DIR / "test_queries_holdout.csv")
services = load_test_set(DATA_DIR / "test_queries_services.csv")
services_ho = load_test_set(DATA_DIR / "test_queries_services_holdout.csv")
summary = pd.DataFrame({"main test set": test.category.value_counts(sort=False),
                        "held-out set": holdout.category.value_counts(sort=False),
                        "services (dev)": services.category.value_counts(sort=False),
                        "services (held-out)": services_ho.category.value_counts(sort=False)}).fillna(0).astype(int)
summary.loc["TOTAL"] = summary.sum()
summary
""")

code(r"""
test.groupby("category", sort=False).head(2)[["id", "category", "query", "expected_intent", "expected_key"]]
""")

# ------------------------------------------------------------------ 6
md(r"""
## 6. Performance Evaluation

**Metrics**
* **Answer accuracy:** the share of queries whose answer key matches the expected key (the main metric).
* **Intent accuracy** and **macro precision / recall / F1** over intents: how well the system understands the question type.
* **Out-of-scope precision / recall:** whether the bot refuses the right questions and only those.
* **Recall@k of the vector store** on free-text FAQ questions: is the right passage in the top-k?
* **Latency:** mean response time per query in milliseconds.
""")

code(r"""
hybrid = UniversityChatbot()
res_h = run_bot(hybrid, test)
m = summarise(res_h)
pd.DataFrame({"Hybrid KR + NLP": m}).round(3)
""")

code(r"""
cat_h = per_category(res_h)
ax = (cat_h * 100).plot.barh(figsize=(7.5, 3.6), color="#2a6fdb")
for i, v in enumerate(cat_h * 100):
    ax.text(v + 1, i, f"{v:.0f}%", va="center", fontsize=9)
ax.set_xlim(0, 112); ax.set_xlabel("answer accuracy (%)"); ax.invert_yaxis()
ax.set_title("Hybrid chatbot: answer accuracy by test category")
plt.tight_layout(); plt.savefig(FIG / "fig_category_hybrid.png", dpi=150, bbox_inches="tight"); plt.show()
""")

code(r"""
from sklearn.metrics import confusion_matrix
labels = sorted(set(res_h.expected_intent) | set(res_h.pred_intent))
cm = confusion_matrix(res_h.expected_intent, res_h.pred_intent, labels=labels)
fig, ax = plt.subplots(figsize=(8.5, 7)); ax.grid(False)
ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(labels)), labels, rotation=75, fontsize=8); ax.set_yticks(range(len(labels)), labels, fontsize=8)
for i in range(len(labels)):
    for j in range(len(labels)):
        if cm[i, j]:
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8, color="white" if cm[i, j] > 4 else "black")
ax.set_xlabel("predicted intent"); ax.set_ylabel("expected intent"); ax.set_title("Intent confusion matrix (main test set)")
plt.tight_layout(); plt.savefig(FIG / "fig_confusion.png", dpi=150, bbox_inches="tight"); plt.show()
""")

code(r"""
print(f"Vector store Recall@1 on free-text FAQ questions: {retrieval_recall_at_k(hybrid.store, hybrid.pre, test, 1):.2f}")
print(f"Vector store Recall@3 on free-text FAQ questions: {retrieval_recall_at_k(hybrid.store, hybrid.pre, test, 3):.2f}")
print(f"Response source mix: {res_h.source.value_counts().to_dict()}")
print(f"Latency: mean {res_h.latency_ms.mean():.2f} ms, 95th percentile {res_h.latency_ms.quantile(.95):.2f} ms, max {res_h.latency_ms.max():.2f} ms")
""")

md("**Error analysis: every query the hybrid chatbot got wrong**")
code(r"""
res_h[~res_h.answer_ok][["id", "category", "query", "expected_key", "pred_key", "source"]]
""")

# ------------------------------------------------------------------ 7
md(r"""
## 7. Comparison with Alternative Approaches

Two alternative chatbots were built on **exactly the same knowledge** (the same 152 FAQ question-answer pairs, with identical answer wording),
so that only the *approach* differs:

| System | Description |
|---|---|
| **Keyword FAQ matching** (classic baseline) | raw lower-cased text, word TF-IDF, cosine similarity to FAQ questions. No NLP, no KR, no reasoning, no memory. |
| **NLP + vector retrieval** | the same NLP normalisation and vector store as the hybrid bot, but every answer is retrieved from the FAQ. No frames, rules or dialogue state. |
| **Hybrid KR + NLP (proposed)** | frames + forward chaining + NLP + dialogue manager + vector fallback |

An **ablation study** then removes one component of the hybrid system at a time to measure what each part contributes.
""")

code(r"""
systems = {"Keyword FAQ matching": KeywordFAQBot(), "NLP + vector retrieval": RetrievalBot(),
           "Hybrid KR + NLP (proposed)": hybrid}
results, comp, cats = {}, {}, {}
for name, b in systems.items():
    results[name] = run_bot(b, test)
    comp[name] = summarise(results[name])
    cats[name] = per_category(results[name])
comparison = pd.DataFrame(comp)
comparison.round(3)
""")

code(r"""
cat_df = pd.DataFrame(cats) * 100
ax = cat_df.plot.bar(figsize=(11, 4.2), width=0.8, color=[COLORS[c] for c in cat_df.columns])
ax.set_ylabel("answer accuracy (%)"); ax.set_ylim(0, 110); ax.set_xlabel("")
ax.set_title("Answer accuracy by category: proposed system vs. alternative approaches")
plt.xticks(rotation=20); ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
plt.tight_layout(); plt.savefig(FIG / "fig_comparison_category.png", dpi=150, bbox_inches="tight"); plt.show()
""")

code(r"""
keys = ["Answer accuracy", "Intent accuracy", "Intent macro F1", "Out-of-scope recall"]
ax = (comparison.loc[keys] * 100).plot.bar(figsize=(9, 3.8), width=0.8, color=[COLORS[c] for c in comparison.columns])
for cont in ax.containers:
    ax.bar_label(cont, fmt="%.0f", fontsize=8)
ax.set_ylabel("%"); ax.set_ylim(0, 110); plt.xticks(rotation=0)
ax.set_title("Overall metrics"); ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
plt.tight_layout(); plt.savefig(FIG / "fig_comparison_overall.png", dpi=150, bbox_inches="tight"); plt.show()
""")

md("### 7.1 Ablation study: contribution of each component")
code(r"""
ablations = {"Full hybrid": {}, "without reasoning (rules)": {"use_reasoning": False},
             "without dialogue context": {"use_context": False},
             "without spelling correction": {"use_spell_correction": False},
             "without vector-store fallback": {"use_vector_fallback": False}}
abl = {}
for name, kw in ablations.items():
    r = run_bot(UniversityChatbot(**kw), test)
    abl[name] = {"Answer accuracy": r.answer_ok.mean(), **per_category(r).to_dict()}
abl_df = pd.DataFrame(abl).T
abl_df.round(3)
""")

code(r"""
acc = abl_df["Answer accuracy"] * 100
ax = acc.plot.barh(figsize=(7.5, 3), color=["#2a6fdb"] + ["#9aa5b1"] * (len(acc) - 1))
for i, v in enumerate(acc):
    ax.text(v + 0.8, i, f"{v:.1f}%", va="center", fontsize=9)
ax.set_xlim(0, 105); ax.invert_yaxis(); ax.set_xlabel("answer accuracy (%)")
ax.set_title("Ablation: removing one component at a time")
plt.tight_layout(); plt.savefig(FIG / "fig_ablation.png", dpi=150, bbox_inches="tight"); plt.show()
""")

md("### 7.2 Sensitivity to the intent-confidence threshold")
code(r"""
sweep = []
for th in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]:
    s = summarise(run_bot(UniversityChatbot(intent_threshold=th), test))
    sweep.append({"threshold": th, "answer accuracy": s["Answer accuracy"],
                  "out-of-scope recall": s["Out-of-scope recall"], "out-of-scope precision": s["Out-of-scope precision"]})
sweep = pd.DataFrame(sweep).set_index("threshold")
ax = (sweep * 100).plot(marker="o", figsize=(7.5, 3.4), color=["#2a6fdb", "#f2a33a", "#9aa5b1"])
ax.axvline(0.30, ls="--", color="#555", lw=1); ax.text(0.305, 62, "chosen\n(0.30)", fontsize=8)
ax.set_ylabel("%"); ax.set_ylim(55, 105); ax.set_title("Effect of the intent-confidence threshold")
ax.legend(frameon=False, fontsize=8)
plt.tight_layout(); plt.savefig(FIG / "fig_threshold.png", dpi=150, bbox_inches="tight"); plt.show()
sweep.round(3)
""")

md("### 7.3 Held-out test set (evaluated once, after development)\n\n*Note: while checking these results one label was found to be wrong (H14: the knowledge base accepts M.Sc Maths, not B.Sc Maths, for M.Tech Data Science, so the chatbot answer \"not eligible\" was correct). The label was corrected; the system was not changed.*")
code(r"""
ho = {}
for name, b in {"Keyword FAQ matching": KeywordFAQBot(), "NLP + vector retrieval": RetrievalBot(),
                "Hybrid KR + NLP (proposed)": UniversityChatbot()}.items():
    r = run_bot(b, holdout)
    ho[name] = summarise(r)
    if name.startswith("Hybrid"):
        ho_errors = r[~r.answer_ok][["id", "query", "expected_key", "pred_key"]]
display(pd.DataFrame(ho).round(3))
ho_errors
""")

md(r"""
### 7.4 University-services test sets

All three chatbots use the same knowledge here too: the FAQ list is generated from the same frames (one entry for each office,
certificate, the calendar, exams, results, fee payment, events, research and alumni).

*The held-out set scored 80 % for the hybrid chatbot when it was first run. One of its questions ("Can I pay the semester fee using UPI?")
revealed a bug: the spelling corrector turned "using" into "nursing", a word that had been added to the vocabulary for the closed-world
check. The bug was fixed by adding everyday words such as "using" to the list of words that are never corrected; nothing was tuned to
the held-out questions. The numbers below are after that fix.*
""")
code(r"""
sv = {}
for set_name, data in {"services (dev)": services, "services (held-out)": services_ho}.items():
    for name, b in {"Keyword FAQ matching": KeywordFAQBot(), "NLP + vector retrieval": RetrievalBot(),
                    "Hybrid KR + NLP (proposed)": UniversityChatbot()}.items():
        r = run_bot(b, data)
        sv[(set_name, name)] = r.answer_ok.mean()
        if name.startswith("Hybrid") and set_name == "services (held-out)":
            sv_errors = r[~r.answer_ok][["id", "query", "expected_key", "pred_key"]]
display((pd.Series(sv).unstack() * 100).round(1))
sv_errors
""")

# ------------------------------------------------------------------ 8
md(r"""
## 8. Results, Discussion and Conclusion

### Results
* On the 105-query main (admissions) test set the hybrid KR + NLP chatbot gave the correct answer to **94.3 %** of queries
  (intent macro-F1 **0.96**), against **32.4 %** for NLP + vector retrieval and **26.7 %** for keyword FAQ matching.
* On the **held-out** admissions set it reached **80 %**, against 48 % and 32 %.
  The drop from 94 % to 80 % is the expected optimism of a test set that was also used for debugging; the ranking of the three approaches does not change.
* On the **university-services** questions it reached **100 %** on the development set (72.5 % before it was used for tuning) and
  **90 %** on the held-out set (80 % on the first run, before the spelling bug it revealed was fixed), against 40 % for both alternatives.
* Every answer was produced in about **3 ms** on a laptop CPU, with no GPU and no internet.

### Discussion and observations
1. **Reasoning is what a retrieval bot cannot do.** Both baselines scored **0 %** on reasoning questions and **0 %** on multi-turn questions.
   They can return the *general* eligibility text, but not the *decision* for a specific applicant ("72 % in B.E ECE → eligible for M.Tech AI").
   Removing the rule engine alone drops the hybrid from 94.3 % to 74.3 %, so it is the largest single contributor.
2. **Inheritance keeps the KB small and consistent.** `duration`, `documents` and `deadline` are stored once per class
   (UG/PG/Research) and inherited by all programmes. The trace shows where each value came from (e.g. `mba.duration` comes from `PGProgram`).
3. **Explanations come for free.** Each fired rule produces a sentence, e.g. *"[R1] Converted CGPA 6.8 to 64.6 %"* or *"[R3] 5 % relaxation for OBC"*.
   That transparency matters for admission decisions.
4. **NLP robustness:** spelling correction raised typo accuracy from 30 % to 100 % (ablation), and normalisation of *B.E / B.Tech / M.Tech / plus two / CGPA* was essential for entity extraction.
5. **The vector store complements the KB.** With no fallback, free-text FAQ accuracy is 0 %. With it, 67 %, and Recall@3 is 100 %.
   On those questions the plain retrieval baselines actually score *higher* (83-92 %), because the hybrid sometimes routes a question to a confident KB intent:
   *"wifi in the hostel"* goes to the hostel frame. This is a real trade-off between structured and unstructured knowledge.
6. **Out-of-scope detection** (90 % recall) relies on a domain-vocabulary gate. Questions that share domain words
   (*"share **price** of Infosys"*, where price is a synonym of fee) still slip through. The threshold sweep shows that higher thresholds (0.35-0.55) would score better on this test set (up to 96 %),
   but it was *not* adopted, to avoid tuning on test data. It should be validated on new data first.
7. **One knowledge base for the whole university.** Offices, certificates, the calendar and exam rules are frames like the programmes,
   so new topics needed new frames and answer templates but no new kind of reasoning. Links between frames answer follow-ups such as
   *"Which office handles it?"* (examinations → exam cell). The closed-world assumption keeps the bot honest: a department or another
   university that is not in the knowledge base gets "not available" instead of a confident wrong answer.
8. **Limitations:** the knowledge must be engineered by hand; the rule base covers only eligibility and scholarship policies; the lexical
   TF-IDF embeddings miss pure synonyms (*"in charge of"* vs *"HOD"*); there is English only; and the data is synthetic.

### Conclusion
A hybrid **Knowledge Representation + NLP** chatbot for the whole university was designed, implemented and evaluated. Frames with
inheritance hold the university's structured knowledge: programmes, departments, offices, certificates, the academic calendar and exam rules. Forward-chaining production rules answer personalised eligibility and fee questions with explanations.
A lightweight NLP pipeline turns free-form questions into structured queries, a dialogue manager handles follow-ups and missing details,
and a vector store answers unstructured FAQs. The system clearly beats retrieval-only alternatives, especially on reasoning and multi-turn questions,
while staying fast and explainable, and it runs in any web browser.

**Future work:** replace the TF-IDF embedder with a neural sentence embedding (the `Embedder` interface already supports this);
learn intent weights from real chat logs; add backward chaining for "what do I need to become eligible?" questions; add Tamil/Hindi support;
and connect the KB to the university's live databases (admissions, exam results, fee dues).
""")

# ------------------------------------------------------------------ 9
md(r"""
## 9. Demonstration of the Working Prototype

### 9.1 Scripted conversation (outputs saved in the notebook)
The conversation below shows factual answers, follow-up questions using context, slot filling, rule-based reasoning,
spelling correction, the vector-store fallback and an out-of-scope refusal.
""")

code(r"""
demo_bot = UniversityChatbot()
session = demo_bot.new_session()
conversation = ["Hi", "What is the fee for M.Tech AI?", "How long is it?", "What about MBA?",
                "Am I eligible for M.Tech AI?", "I did B.E ECE with 6.8 CGPA and I qualified GATE",
                "I scored 96% in 12th and I am a state level athlete. What will my fee be for B.Tech CSE?",
                "hostle fees for girls?", "Is the campus ragging free?", "Who won the cricket match yesterday?", "Thanks, bye"]
for q in conversation:
    r = demo_bot.respond(q, session)
    print(f"YOU : {q}")
    print(f"BOT : {r.text}")
    print(f"      [intent={r.intent} | source={r.source} | confidence={r.confidence:.2f} | {r.latency_ms:.1f} ms]\n")
""")

md("**University services: exams, an office follow-up, certificates and fees**")
code(r"""
s3 = demo_bot.new_session()
for q in ["When are the end semester exams?", "Which office handles it?", "How do I get a bonafide certificate?",
          "What is the late fee for semester fees?", "When does the semester start?", "Who is the VC?"]:
    r = demo_bot.respond(q, s3)
    print(f"YOU : {q}\nBOT : {r.text}\n      [answer key={r.answer_key} | source={r.source}]\n")
""")

md('**Explanation trace of the eligibility answer above** (how the answer was produced, step by step):')
code(r"""
s2 = demo_bot.new_session()
demo_bot.respond("Am I eligible for M.Tech AI?", s2)
r = demo_bot.respond("I did B.E ECE with 6.8 CGPA and I qualified GATE", s2)
print("\n".join(r.trace))
""")

md(r"""
**Questions outside the knowledge base (closed-world assumption).** The AI department has its own HOD. A department the university
does not have, or a question about another university, gets a polite "not available" instead of a wrong answer.
""")
code(r"""
for q in ["Who is the HOD of the AI department?", "Do you have BBA?", "Who is the HOD of the Civil department?",
          "Do you offer Law?", "hi, do you have available seats in medical?", "Who is the HOD of the CSBS department?",
          "Who is the HOD of CSE in SRM University?", "The answers are not relevant"]:
    r = demo_bot.respond(q, demo_bot.new_session())
    print(f"YOU : {q}\nBOT : {r.text}\n      [answer key={r.answer_key} | source={r.source}]\n")
""")

md(r"""
### 9.2 The chatbot website
The working chatbot is online at **https://manikandanayyakkannu.github.io/FAI-Mini-Project/**. It runs completely in the browser, on a laptop or a phone, with nothing to install.
The replies contain only the answer; how each one was produced is shown in this notebook (the traces above).
The *Demo 1*, *Demo 2* and *Demo 3* buttons (or `?demo=1`, `?demo=2`, `?demo=3` at the end of the link) play a short sample conversation.

![chatbot website](docs/figures/demo_web_1.png)
""")

nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, out_path)
print("written", out_path)
