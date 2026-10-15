"""Builds the project report (.docx). All numbers are computed from a fresh evaluation run."""
import sys
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_COLOR_INDEX
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

from chatbot import UniversityChatbot
from chatbot.baselines import KeywordFAQBot, RetrievalBot
from chatbot.evaluation import load_test_set, per_category, retrieval_recall_at_k, run_bot, summarise
from chatbot.knowledge_base import DATA_DIR

OUT = Path(sys.argv[1])
FIG = Path("docs/figures")
BLUE = RGBColor(0x1F, 0x4E, 0x9B)
GREY = RGBColor(0x55, 0x5F, 0x6B)

# ------------------------------------------------------------- numbers
test = load_test_set()
holdout = load_test_set(DATA_DIR / "test_queries_holdout.csv")
hybrid = UniversityChatbot()
systems = {"Keyword FAQ matching": KeywordFAQBot(), "NLP + vector retrieval": RetrievalBot(),
           "Hybrid KR + NLP (proposed)": hybrid}
res = {n: run_bot(b, test) for n, b in systems.items()}
summ = {n: summarise(r) for n, r in res.items()}
cats = {n: per_category(r) for n, r in res.items()}
res_ho = {n: run_bot(b, holdout) for n, b in {"Keyword FAQ matching": KeywordFAQBot(),
                                               "NLP + vector retrieval": RetrievalBot(),
                                               "Hybrid KR + NLP (proposed)": UniversityChatbot()}.items()}
summ_ho = {n: summarise(r) for n, r in res_ho.items()}
abl_cfg = {"Full hybrid system": {}, "without rule-based reasoning": {"use_reasoning": False},
           "without dialogue context": {"use_context": False},
           "without spelling correction": {"use_spell_correction": False},
           "without vector-store fallback": {"use_vector_fallback": False}}
abl = {}
for name, kw in abl_cfg.items():
    r = run_bot(UniversityChatbot(**kw), test)
    abl[name] = {"acc": r.answer_ok.mean(), **per_category(r).to_dict()}
H = summ["Hybrid KR + NLP (proposed)"]
HO = summ_ho["Hybrid KR + NLP (proposed)"]
pct = lambda v: f"{100 * v:.1f}%"
r1 = retrieval_recall_at_k(hybrid.store, hybrid.pre, test, 1)
r3 = retrieval_recall_at_k(hybrid.store, hybrid.pre, test, 3)
lat = res["Hybrid KR + NLP (proposed)"].latency_ms

# demo transcript (terminal chatbot, same engine)
demo_bot = UniversityChatbot()
s = demo_bot.new_session()
demo_lines = []
for q in ["What is the fee for M.Tech AI?", "How long is it?", "What about MBA?", "Am I eligible for M.Tech AI?",
          "I did B.E ECE with 6.8 CGPA and I qualified GATE", "/why",
          "I scored 96% in 12th and I am a state level athlete. What will my fee be for B.Tech CSE?",
          "hostle fees for girls?", "Is the campus ragging free?", "Who won the cricket match yesterday?"]:
    if q == "/why":
        demo_lines.append("You: /why")
        demo_lines += ["  " + t for t in last.trace if t.startswith("[") or t.startswith("Forward")]
        continue
    last = demo_bot.respond(q, s)
    demo_lines.append(f"You: {q}")
    demo_lines.append(f"Bot: {last.text}")
    demo_lines.append(f"     [{last.intent} | {last.source} | {last.confidence:.2f}]")

# -------------------------------------------------------------- helpers
doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21), Cm(29.7)
for side in ("left_margin", "right_margin"):
    setattr(sec, side, Cm(2.3))
sec.top_margin, sec.bottom_margin = Cm(2.2), Cm(2.2)

st = doc.styles
st["Normal"].font.name = "Calibri"
st["Normal"].font.size = Pt(11)
st["Normal"].element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
st["Normal"].paragraph_format.space_after = Pt(6)
st["Normal"].paragraph_format.line_spacing = 1.12
for lvl, size in ((1, 16), (2, 13), (3, 11.5)):
    h = st[f"Heading {lvl}"]
    h.font.name = "Calibri"
    h.font.size = Pt(size)
    h.font.bold = True
    h.font.color.rgb = BLUE
    h.element.rPr.rFonts.set(qn("w:asciiTheme"), "") if False else None
    h.paragraph_format.space_before = Pt(14 if lvl == 1 else 10)
    h.paragraph_format.space_after = Pt(6)
    rpr = h.element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    for a in ("w:ascii", "w:hAnsi", "w:cs"):
        rfonts.set(qn(a), "Calibri")
    for a in ("w:asciiTheme", "w:hAnsiTheme"):
        if rfonts.get(qn(a)):
            del rfonts.attrib[qn(a)]


def para(text="", bold_lead=None, size=None, color=None, align=None, italic=False, space_after=None):
    p = doc.add_paragraph()
    if bold_lead:
        r = p.add_run(bold_lead)
        r.bold = True
    r = p.add_run(text)
    r.italic = italic
    if size:
        r.font.size = Pt(size)
    if color:
        r.font.color.rgb = color
    if align:
        p.alignment = align
    if space_after is not None:
        p.paragraph_format.space_after = Pt(space_after)
    return p


def bullet(text, bold_lead=None, level=0):
    p = doc.add_paragraph(style="List Bullet" if level == 0 else "List Bullet 2")
    if bold_lead:
        p.add_run(bold_lead).bold = True
    p.add_run(text)
    p.paragraph_format.space_after = Pt(3)
    return p


CURRENT_NUM = [None]


def restart_numbering():
    """Start a new numbered list at 1 (python-docx otherwise continues one global list)."""
    numbering = doc.part.numbering_part.numbering_definitions._numbering
    style_num = doc.styles["List Number"].element.pPr.numPr.numId.val
    abstract = numbering.num_having_numId(style_num).abstractNumId.val
    num = numbering.add_num(abstract)
    num.add_lvlOverride(ilvl=0).add_startOverride(1)
    CURRENT_NUM[0] = num.numId


def numbered(text, bold_lead=None):
    p = doc.add_paragraph(style="List Number")
    if CURRENT_NUM[0] is not None:
        numPr = p._p.get_or_add_pPr().get_or_add_numPr()
        numPr.get_or_add_ilvl().val = 0
        numPr.get_or_add_numId().val = CURRENT_NUM[0]
    if bold_lead:
        p.add_run(bold_lead).bold = True
    p.add_run(text)
    p.paragraph_format.space_after = Pt(3)
    return p


def shade(cell, hex_fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_fill)
    tcPr.append(shd)


def table(header, rows, widths, font=9.5, highlight_last_col=False, bold_first_col=False):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(header):
        c = t.rows[0].cells[i]
        c.text = ""
        r = c.paragraphs[0].add_run(str(h))
        r.bold = True
        r.font.size = Pt(font)
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        shade(c, "1F4E9B")
    for ri, row in enumerate(rows):
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            r = cells[i].paragraphs[0].add_run(str(v))
            r.font.size = Pt(font)
            if (bold_first_col and i == 0) or (highlight_last_col and i == len(row) - 1):
                r.bold = True
            if ri % 2 == 1:
                shade(cells[i], "EEF3FB")
    for row in t.rows:
        for i, w in enumerate(widths):
            row.cells[i].width = Cm(w)
            for p in row.cells[i].paragraphs:
                p.paragraph_format.space_after = Pt(1)
                p.paragraph_format.space_before = Pt(1)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def figure(path, caption, width_cm=16):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(path), width=Cm(width_cm))
    p.paragraph_format.space_after = Pt(2)
    c = para(caption, size=9, color=GREY, align=WD_ALIGN_PARAGRAPH.CENTER, italic=True, space_after=10)
    return c


def code_block(lines, size=8.5):
    t = doc.add_table(rows=1, cols=1)
    t.style = "Table Grid"
    cell = t.rows[0].cells[0]
    shade(cell, "F4F6F8")
    cell.width = Cm(16.4)
    cell.text = ""
    p = cell.paragraphs[0]
    for i, line in enumerate(lines):
        r = p.add_run(line)
        r.font.name = "Consolas"
        r.font.size = Pt(size)
        r._element.rPr.rFonts.set(qn("w:eastAsia"), "Consolas")
        if line.startswith("You:"):
            r.bold = True
        if i < len(lines) - 1:
            r.add_break()
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def page_break():
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


# running header and page numbers (not shown on the cover page)
sec.different_first_page_header_footer = True
hp = sec.header.paragraphs[0]
hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
hr = hp.add_run("AI-Based University Enquiry Chatbot  |  Mini Project Report")
hr.font.size = Pt(8.5)
hr.font.color.rgb = GREY
pbdr = OxmlElement("w:pBdr")
bottom = OxmlElement("w:bottom")
for k, v in (("w:val", "single"), ("w:sz", "4"), ("w:space", "1"), ("w:color", "A0AEC0")):
    bottom.set(qn(k), v)
pbdr.append(bottom)
hp._p.get_or_add_pPr().append(pbdr)
fp = sec.footer.paragraphs[0]
fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
for kind, txt in (("begin", None), (None, "PAGE"), ("end", None)):
    run = fp.add_run()
    if kind:
        el = OxmlElement("w:fldChar")
        el.set(qn("w:fldCharType"), kind)
    else:
        el = OxmlElement("w:instrText")
        el.set(qn("xml:space"), "preserve")
        el.text = txt
    run._r.append(el)
    run.font.size = Pt(9)
    run.font.color.rgb = GREY

# ================================================================ cover
for _ in range(4):
    doc.add_paragraph()
para("M.Tech · Fundamentals of Artificial Intelligence", size=12, color=GREY, align=WD_ALIGN_PARAGRAPH.CENTER)
para("Mini Project Report", size=13, color=GREY, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=24)
p = para("AI-Based University Enquiry Chatbot", size=26, color=BLUE, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=6)
p.runs[0].bold = True
para("using Knowledge Representation and Natural Language Processing", size=14, color=BLUE,
     align=WD_ALIGN_PARAGRAPH.CENTER, space_after=40)
para("Project topic No. 29 · Suggested technique: Knowledge Representation + NLP", size=11, color=GREY,
     align=WD_ALIGN_PARAGRAPH.CENTER, space_after=40)
p = para("Submitted by", size=12, color=BLUE, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=6)
p.runs[0].bold = True
TEAM = [("Manikandan", "35"), ("Manoj Kumar", "05"), ("Dinakar", "32")]
team = doc.add_table(rows=1, cols=2)
team.style = "Table Grid"
team.alignment = WD_TABLE_ALIGNMENT.CENTER
for i, h in enumerate(("Team member", "Roll No.")):
    c = team.rows[0].cells[i]
    c.text = ""
    r = c.paragraphs[0].add_run(h)
    r.bold = True
    r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    shade(c, "1F4E9B")
for name, roll in TEAM:
    cells = team.add_row().cells
    for c, v in zip(cells, (name, roll)):
        c.text = v
        c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
for row in team.rows:
    row.cells[0].width, row.cells[1].width = Cm(6), Cm(3)
    for c in row.cells:
        c.paragraphs[0].paragraph_format.space_before = Pt(2)
        c.paragraphs[0].paragraph_format.space_after = Pt(2)
doc.add_paragraph()
para("Programme: M.Tech - Fundamentals of Artificial Intelligence", size=11, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=2)
para("Date: October 2026", size=11, align=WD_ALIGN_PARAGRAPH.CENTER)
for _ in range(4):
    doc.add_paragraph()
para("Solution files submitted with this report: University_Enquiry_Chatbot.ipynb (solution notebook), chatbot/ (Python "
     "package), chat_cli.py (terminal chatbot), app_streamlit.py (web demo), data/ (knowledge base, rules, test sets).",
     size=9, color=GREY, align=WD_ALIGN_PARAGRAPH.CENTER)
page_break()

# ============================================================== contents
p = para("Abstract", size=16, color=BLUE, space_after=8)
p.runs[0].bold = True
para(f"This project builds an enquiry chatbot for a fictional university. The chatbot answers questions about programmes, fees, "
     f"eligibility, admissions, hostel, scholarships and placements. It combines a frame-based knowledge representation "
     f"(a semantic network with inheritance), forward-chaining production rules for personalised reasoning, a natural-language "
     f"processing pipeline, a dialogue manager and a vector-store retrieval fallback. On a labelled test set of {len(test)} "
     f"queries the hybrid chatbot answered {pct(H['Answer accuracy'])} correctly (intent macro-F1 {H['Intent macro F1']:.2f}), "
     f"against {pct(summ['NLP + vector retrieval']['Answer accuracy'])} for an NLP + vector-retrieval chatbot and "
     f"{pct(summ['Keyword FAQ matching']['Answer accuracy'])} for a keyword FAQ-matching chatbot built on the same knowledge. "
     f"On a held-out set evaluated once after development it reached {pct(HO['Answer accuracy'])}. Every answer is produced in "
     f"a few milliseconds (mean {lat.mean():.1f} ms) on a laptop CPU, works offline and comes with an explanation trace.")
p = para("Keywords: ", size=10, space_after=0)
p.runs[0].bold = True
p.add_run("knowledge representation, frames, semantic network, forward chaining, production rules, natural language "
          "processing, chatbot, vector store").font.size = Pt(10)
page_break()

# Word table of contents (page numbers are filled in when Word updates the field)
p = para("Contents", size=16, color=BLUE, space_after=8)
p.runs[0].bold = True
toc = doc.add_paragraph()
for kind, txt in (("begin", None), (None, 'TOC \\o "1-2" \\h \\z \\u'), ("separate", None),
                  ("text", "Right-click here and choose Update Field to show the table of contents."), ("end", None)):
    run = toc.add_run()
    if kind == "text":
        run.text = txt
        continue
    if kind:
        el = OxmlElement("w:fldChar")
        el.set(qn("w:fldCharType"), kind)
    else:
        el = OxmlElement("w:instrText")
        el.set(qn("xml:space"), "preserve")
        el.text = txt
    run._r.append(el)
page_break()

# ============================================================== 1
doc.add_heading("1. Problem Definition", 1)
para("Every admission season a university receives a very large number of repetitive enquiries by phone, e-mail and at the front "
     "desk, such as \"What is the fee for B.Tech CSE?\", \"Am I eligible for M.Tech AI with 6.8 CGPA?\", \"When is the last "
     "date to apply?\" and \"Is there a hostel for girls?\". Staff spend a lot of time answering the same questions. Answers "
     "are not available outside office hours and are sometimes inconsistent between staff members.")
para("The objective is an intelligent enquiry chatbot that:", space_after=3)
bullet("understands free-form English questions, including paraphrases, abbreviations (B.E, M.Tech, CGPA, plus two) and spelling mistakes;")
bullet("answers factual questions from a structured knowledge base of the university;")
bullet("reasons about personal questions, such as eligibility for a programme or the fee after scholarships, and explains every decision;")
bullet("follows the conversation (\"How long is it?\" after a question about M.Tech AI) and asks for missing details;")
bullet("answers free-text FAQs (ragging, Wi-Fi, refunds) through semantic retrieval, and politely refuses out-of-scope questions.")
para("Scope and assumptions: the knowledge base describes a fictional institution, \"Greenfield University\", with 9 programmes "
     "(B.Tech CSE, AI & DS, ECE and Mechanical; M.Tech AI and Data Science; MBA; MCA; Ph.D.), 6 departments, 5 facilities, the "
     "admission process and the scholarship policy. All data is synthetic and created only for this project. The chatbot is "
     "text-based, works in English and runs fully offline.", space_after=8)

# ============================================================== 2
doc.add_heading("2. AI Problem Formulation", 1)
para("The chatbot is modelled as a knowledge-based agent. It perceives a sentence, updates its internal state (the dialogue "
     "context) and acts by choosing the answer that its knowledge base and inference rules support.")
doc.add_heading("2.1 PEAS description", 2)
table(["PEAS element", "Description"], [
    ["Performance measure", "Answer accuracy, intent precision / recall / F1, correct refusal of out-of-scope questions, response time, explainability"],
    ["Environment", "Prospective students and parents typing questions; the university knowledge (frames, rules, FAQ passages)"],
    ["Actuators", "Text answers, clarification questions, explanation traces"],
    ["Sensors", "The user's typed sentence"]], [4.2, 12.2], bold_first_col=True)
doc.add_heading("2.2 Properties of the task environment", 2)
table(["Property", "Value", "Reason"], [
    ["Observability", "Partially observable", "The user's intention and personal facts are hidden and must be inferred from text"],
    ["Determinism", "Stochastic", "The same need can be phrased in many different ways"],
    ["Episodes", "Sequential", "Follow-up questions depend on earlier turns"],
    ["Dynamics", "Static", "The knowledge does not change during a conversation"],
    ["State space", "Discrete", "Finite sets of intents, entities and facts"],
    ["Agents", "Single agent", "One chatbot serves one user per session"]], [3.3, 3.6, 9.5], bold_first_col=True)
doc.add_heading("2.3 Formal formulation", 2)
bullet(" a set of frames F = {University, Program, UGProgram, PGProgram, btech_cse, ..., hostel, ...} linked by is_a edges. "
       "Slot lookup uses inheritance: value(f, s) = f.s if the frame stores slot s, otherwise value(parent(f), s).",
       "Knowledge base (KB):")
bullet(" the NLP pipeline maps a sentence x to a structured query q = (intent, entities), where intent belongs to 18 "
       "intents (program_fee, eligibility, hostel, ...) and the entities are programme, qualification, percentage, CGPA, entrance "
       "exam, category, subjects and so on.", "Percept to query mapping:")
bullet(" for a personal question, WM = facts about the programme (from the frames) plus facts about the applicant (from the sentence).",
       "Working memory (WM):")
bullet(" production rules of the form IF c1 AND c2 AND ... THEN add/modify fact (16 eligibility rules and 10 scholarship rules).",
       "Operators (rules):")
bullet(" forward chaining fires every applicable rule once (refraction) until a goal fact (academic_eligible, net_fee) is "
       "derived or no rule can fire.", "Inference and goal test:")
bullet(" if no intent is confident, the query is embedded as a vector and the closest FAQ passage is returned when its cosine "
       "similarity is above a threshold; otherwise the agent replies \"out of scope\".", "Fallback:")
figure(FIG / "fig_semantic_network.png", "Figure 1. Fragment of the frame-based semantic network (is_a inheritance and department relations).", 15.5)

# ============================================================== 3
doc.add_heading("3. Selection and Explanation of the AI Technique", 1)
para("The topic asks for Knowledge Representation + NLP. The final design combines five classical AI components. Each was chosen "
     "for a specific reason:")
table(["Component", "Technique used", "Why it was chosen"], [
    ["Knowledge representation", "Frames with is_a inheritance, which also form a semantic network of (subject, predicate, object) triples",
     "University data is naturally hierarchical (Programme > UG/PG > B.Tech CSE). Inheritance stores duration, documents and deadlines once per class, so the KB is small and consistent."],
    ["Reasoning", "Production rules + forward chaining with refraction and an explanation trace",
     "Eligibility and scholarship policies are IF-THEN rules. Forward chaining is data-driven: it starts from the applicant's facts and derives the conclusion. Each fired rule becomes a sentence of explanation."],
    ["NLP", "Normalisation, tokenisation, spelling correction (edit-distance similarity), synonym mapping, stop-word removal, stemming, regex entity extraction, hybrid intent classifier",
     "Transparent and explainable; needs only 109 example sentences; runs offline without a GPU; handles Indian academic abbreviations (B.E, M.Tech, CGPA, plus two)."],
    ["Retrieval fallback", "Vector store: TF-IDF word + character n-gram embeddings, cosine similarity, top-k search",
     "Free-text FAQ passages (ragging, Wi-Fi, refunds) are not worth modelling as frames. The embedder is pluggable, so a neural sentence embedding can be used without code changes."],
    ["Dialogue management", "Context memory + slot filling",
     "Supports follow-up questions and asks for missing facts (marks, qualification) before running the rules."],
], [3.2, 5.6, 7.6], font=9, bold_first_col=True)
doc.add_heading("3.1 Hybrid intent classifier", 2)
para("For each intent i the classifier computes score(i) = 0.6 x max cosine similarity between the query and the intent's example "
     "sentences (TF-IDF vectors of the pre-processed text) + 0.4 x keyword evidence (1 if a high-precision trigger word of the "
     "intent is present). The best intent wins. If even the best score is below 0.30, the query is passed to the vector store "
     "or answered as out of scope. A domain-relevance gate rejects questions that contain no word known to the university "
     "domain, such as \"tell me a joke\".")
doc.add_heading("3.2 Forward chaining with explanations", 2)
para("For the question \"I did B.E ECE with 6.8 CGPA and qualified GATE - am I eligible for M.Tech AI?\" the working memory "
     "starts with the programme facts from the frame (minimum 60%, accepted qualifications, entrance exams) and the applicant "
     "facts from NLP (qualification = btech_ece, cgpa = 6.8, exam = gate). The engine then fires:")
code_block(["[R1]  Converted CGPA 6.8 to 64.6% using the standard formula (CGPA x 9.5).",
            "[R2]  The minimum required aggregate for M.Tech Artificial Intelligence is 60%.",
            "[R4]  Your qualification (B.Tech ECE) is accepted for M.Tech Artificial Intelligence.",
            "[R6]  Your score 64.6% meets the cut-off of 60%.",
            "[R11] There is no specific subject requirement for M.Tech Artificial Intelligence.",
            "[R12] Conclusion: you satisfy the academic eligibility for M.Tech Artificial Intelligence.",
            "[R14] Your GATE score is accepted for admission.",
            "[R16] Because you qualified GATE you are eligible for a stipend: Rs. 12,400 per month."])
doc.add_heading("3.3 Why not a pure machine-learning or LLM chatbot?", 2)
para("For admissions, answers must be exactly correct (fees, cut-offs), consistent and explainable. A knowledge-based system "
     "guarantees that every answer can be traced to a frame slot or a rule. A large language model may produce wrong numbers "
     "and cannot easily justify an eligibility decision. It would also need internet access or a GPU. The cost of the "
     "knowledge-based approach is manual knowledge engineering, which is discussed in Section 8.")

# ============================================================== 4
doc.add_heading("4. Python Implementation", 1)
para("The solution is a small, UI-independent Python 3.12 package. The same engine is used by the solution notebook, the "
     "terminal chatbot (chat_cli.py) and the web demo (app_streamlit.py). Libraries: numpy, pandas, scikit-learn (TF-IDF, "
     "metrics), matplotlib, and streamlit for the web demo. The NLP and reasoning components use only the Python standard library.")
figure(FIG / "fig_architecture.png", "Figure 2. Architecture of the hybrid KR + NLP enquiry chatbot.", 16.4)
table(["Module", "Responsibility"], [
    ["knowledge_base.py", "Frames, is_a inheritance (get, get_with_source, resolved), semantic-network triples and pattern queries"],
    ["nlp.py", "Normalisation of abbreviations, tokenisation, spelling correction, synonyms, stop words, stemming, entity extraction"],
    ["intent_classifier.py", "Hybrid TF-IDF similarity + keyword intent classifier with a confidence threshold"],
    ["reasoning.py", "Generic forward-chaining production-rule engine (conditions, actions, refraction, explanations)"],
    ["vector_store.py", "In-memory vector store with a pluggable embedder (TF-IDF default; sentence-transformers optional)"],
    ["answers.py / faq_builder.py", "Natural-language answer templates filled from frames; FAQ corpus for retrieval and the baselines"],
    ["engine.py", "UniversityChatbot: dialogue manager (context, slot filling), routing to KB / rules / vector store, trace"],
    ["baselines.py / evaluation.py", "Alternative approaches for comparison; test harness and metrics"],
    ["chat_cli.py / app_streamlit.py", "Terminal chatbot (with /why explanations) and web demo"],
    ["data/*.json, data/*.csv", "Knowledge base (29 frames), rules (26), intents (18), free-text FAQs (15), test sets"]],
    [5.0, 11.4], font=9, bold_first_col=True)
para("Processing steps for every message (UniversityChatbot.respond):", space_after=3)
restart_numbering()
for t_ in ["NLP pre-processing, entity extraction and intent classification.",
           "If a task is pending (e.g. eligibility waiting for marks), the new facts fill the slot and the task continues.",
           "Personal questions with applicant facts are sent to the forward-chaining engine (eligibility or scholarship rules).",
           "The dialogue context resolves references (\"it\") and elliptical follow-ups (\"What about MBA?\").",
           "The domain gate and the vector store handle unknown or low-confidence questions.",
           "Otherwise the answer is filled from the frame KB. The response carries the text, intent, confidence, answer source and a full trace."]:
    numbered(t_)

# ============================================================== 5
doc.add_heading("5. Dataset and Test-Case Preparation", 1)
para("There is no public dataset for a fictional university, so the knowledge and the test cases were prepared for this project:")
bullet(" 29 frames, 26 production rules, 18 intents with 109 example sentences, and 90 FAQ passages "
       "(75 generated automatically from the frames and 15 free-text passages).", "Knowledge sources:")
bullet(f" {len(test)} hand-written queries, deliberately different from the intent examples, in 8 categories that test "
       "different abilities. Each query is labelled with the expected intent and the expected answer key (e.g. "
       "program_fee:btech_ece, eligibility:mtech_ai:yes). An answer counts as correct only when the key matches exactly, so "
       "for reasoning questions the eligibility decision itself must be right.", "Main test set:")
bullet(f" {len(holdout)} further queries written after development finished and evaluated only once, without any "
       "change to the system. This gives an unbiased estimate, because the main test set was also used to find bugs.",
       "Held-out test set:")
cnt = test.category.value_counts(sort=False)
cnt_h = holdout.category.value_counts(sort=False)
desc = {"factual": "Direct factual questions", "paraphrase": "Same needs, different wording and synonyms",
        "typo": "Spelling mistakes (\"hostle feez\", \"admision procedure\")",
        "reasoning": "Personal eligibility / fee-after-scholarship decisions",
        "multi_turn": "5 conversations with follow-ups and slot filling",
        "unstructured_faq": "Free-text FAQ topics not modelled as frames", "out_of_scope": "Unrelated questions that must be refused",
        "smalltalk": "Greeting, thanks, goodbye"}
table(["Category", "Main set", "Held-out", "What it tests"],
      [[c, cnt.get(c, 0), cnt_h.get(c, 0), desc[c]] for c in cnt.index] + [["Total", len(test), len(holdout), ""]],
      [3.2, 1.8, 1.8, 9.6], font=9, bold_first_col=True)

# ============================================================== 6
doc.add_heading("6. Performance Evaluation", 1)
para("Metrics used:", space_after=3)
bullet(" share of queries whose answer key matches the expected key (main metric).", "Answer accuracy:")
bullet(" how well the question type is understood.", "Intent accuracy and macro precision / recall / F1:")
bullet(" whether the chatbot refuses the right questions and only those.", "Out-of-scope precision / recall:")
bullet(" whether the right free-text passage is among the top-k results.", "Recall@k of the vector store:")
bullet(" mean response time per query.", "Latency:")
table(["Metric", "Hybrid KR + NLP"], [[k, f"{v:.3f}" if "ms" not in k else f"{v:.2f}"] for k, v in H.items()]
      + [["Vector store Recall@1 / Recall@3", f"{r1:.2f} / {r3:.2f}"],
         ["Latency 95th percentile / max (ms)", f"{lat.quantile(.95):.2f} / {lat.max():.2f}"]],
      [9, 4], bold_first_col=True)
figure(FIG / "fig_category_hybrid.png", "Figure 3. Answer accuracy of the hybrid chatbot by test category.", 13)
figure(FIG / "fig_confusion.png", "Figure 4. Intent confusion matrix on the main test set.", 12.5)
doc.add_heading("6.1 Error analysis", 2)
err = res["Hybrid KR + NLP (proposed)"]
err = err[~err.answer_ok]
reason = {"F18": "\"offers\" sent the query to the programme list instead of the department relation",
          "P18": "the word \"located\" triggered the contact intent",
          "U02": "\"hostel\" is a confident KB intent, so the Wi-Fi FAQ was never consulted",
          "U03": "\"diploma\" was read as an applicant qualification, so the bot asked which programme",
          "U08": "\"uniform\" has no lexical overlap with \"dress code\" (a limit of lexical embeddings)",
          "U09": "\"semester\" pulled the query towards the duration intent",
          "O03": "\"book\" matched the library facility",
          "O07": "\"price\" is a synonym of fee in the domain vocabulary"}
table(["ID", "Query", "Predicted", "Cause"],
      [[r.id, r.query, r.pred_key, reason.get(r.id, "")] for r in err.itertuples()], [1.2, 5.2, 3.6, 6.4], font=8.5)

# ============================================================== 7
doc.add_heading("7. Comparison with Alternative Approaches", 1)
para("Two alternative chatbots were built on exactly the same knowledge: the same 90 FAQ question-answer pairs, with "
     "identical answer wording. Only the approach differs:")
table(["System", "Description"], [
    ["Keyword FAQ matching (classic baseline)", "Raw lower-cased text, word TF-IDF and cosine similarity to FAQ questions. No NLP, no KR, no reasoning, no memory."],
    ["NLP + vector retrieval", "The same NLP normalisation and vector store as the proposed system, but every answer is retrieved from the FAQ corpus. No frames, rules or dialogue state."],
    ["Hybrid KR + NLP (proposed)", "Frames + forward chaining + NLP + dialogue manager + vector-store fallback."]],
    [5, 11.4], font=9, bold_first_col=True)
names = list(summ)
table(["Metric"] + names, [[k] + [f"{summ[n][k]:.3f}" if "ms" not in k else f"{summ[n][k]:.2f}" for n in names] for k in H],
      [5, 3.6, 3.8, 4], font=9, highlight_last_col=True, bold_first_col=True)
figure(FIG / "fig_comparison_category.png", "Figure 5. Answer accuracy by category: proposed system vs. alternative approaches.", 16.4)
doc.add_heading("7.1 Ablation study", 2)
para("To measure what each component contributes, one component was removed at a time:")
abl_cols = ["factual", "paraphrase", "typo", "reasoning", "multi_turn", "unstructured_faq", "out_of_scope"]
table(["Configuration", "Overall"] + ["fact.", "para.", "typo", "reas.", "multi", "FAQ", "OOS"],
      [[n, pct(v["acc"])] + [f"{100 * v[c]:.0f}" for c in abl_cols] for n, v in abl.items()],
      [4.6, 1.7] + [1.45] * 7, font=8.5, bold_first_col=True)
figure(FIG / "fig_ablation.png", "Figure 6. Ablation: answer accuracy when one component is removed.", 12.5)
doc.add_heading("7.2 Sensitivity to the confidence threshold", 2)
para("The intent-confidence threshold was fixed at 0.30 before testing. The sweep in Figure 7 shows that higher thresholds "
     "(0.35-0.55) give slightly higher accuracy and better out-of-scope recall on this test set. This was not adopted, "
     "because choosing a parameter on the test set would make the reported result optimistic. It should be validated on new data.")
figure(FIG / "fig_threshold.png", "Figure 7. Effect of the intent-confidence threshold.", 12)
doc.add_heading("7.3 Held-out test set", 2)
table(["Metric"] + names, [[k] + [f"{summ_ho[n][k]:.3f}" if "ms" not in k else f"{summ_ho[n][k]:.2f}" for n in names]
                           for k in ["Answer accuracy", "Intent accuracy", "Intent macro F1", "Out-of-scope recall"]],
      [5, 3.6, 3.8, 4], font=9, highlight_last_col=True, bold_first_col=True)
para("Note: while checking the held-out results, one label (H14, B.Sc Maths for M.Tech Data Science) was found to be wrong. The "
     "knowledge base accepts M.Sc Maths, not B.Sc, so the chatbot's \"not eligible\" answer was correct. The label was "
     "corrected; the system itself was not changed.", size=9.5, color=GREY, italic=True)

# ============================================================== 8
doc.add_heading("8. Results, Discussion and Conclusion", 1)
doc.add_heading("8.1 Results", 2)
bullet(f" on the {len(test)}-query main test set the hybrid chatbot answered {pct(H['Answer accuracy'])} correctly (intent macro-F1 "
       f"{H['Intent macro F1']:.2f}), against {pct(summ['NLP + vector retrieval']['Answer accuracy'])} for NLP + vector retrieval and "
       f"{pct(summ['Keyword FAQ matching']['Answer accuracy'])} for keyword FAQ matching.", "Main test set:")
bullet(f" {pct(HO['Answer accuracy'])} for the hybrid chatbot, against "
       f"{pct(summ_ho['NLP + vector retrieval']['Answer accuracy'])} and {pct(summ_ho['Keyword FAQ matching']['Answer accuracy'])}. "
       "The drop from the main test set is the expected optimism of a test set that was also used for debugging; the ranking "
       "of the three approaches is the same.", "Held-out set:")
bullet(f" mean {lat.mean():.1f} ms per answer on a laptop CPU, with no GPU and no internet.", "Speed:")
doc.add_heading("8.2 Discussion and individual observations on the outputs", 2)
obs = [
    ("Reasoning is what a retrieval bot cannot do. ",
     f"Both alternatives scored {100 * cats['Keyword FAQ matching']['reasoning']:.0f}% on reasoning questions and about "
     f"{100 * cats['NLP + vector retrieval']['multi_turn']:.0f}% on multi-turn questions. They can return the general eligibility text, "
     "but not the decision for a specific applicant (\"72% in B.E ECE, so eligible for M.Tech AI\"). Removing the rule engine "
     f"alone drops the hybrid from {pct(abl['Full hybrid system']['acc'])} to {pct(abl['without rule-based reasoning']['acc'])}, "
     "the largest single drop."),
    ("Inheritance keeps the knowledge base small and consistent. ",
     "Duration, documents and deadlines are stored once per class (UG / PG / Research) and inherited by every programme. "
     "The trace shows where each value came from, e.g. mba.duration comes from the PGProgram frame."),
    ("Explanations come for free. ",
     "Every fired rule produces a sentence (\"[R3] A 5% relaxation applies for the OBC category, so the cut-off becomes 55%\"), "
     "so a student can see why they are or are not eligible. A statistical chatbot does not give this."),
    ("Small NLP steps matter. ",
     f"Spelling correction raised typo accuracy from {100 * abl['without spelling correction']['typo']:.0f}% to "
     f"{100 * abl['Full hybrid system']['typo']:.0f}%, and normalising B.E / B.Tech / M.Tech / plus two / CGPA was essential for "
     "correct entity extraction."),
    ("Dialogue context is necessary for natural conversation. ",
     f"Without context memory, multi-turn accuracy falls from {100 * abl['Full hybrid system']['multi_turn']:.0f}% to "
     f"{100 * abl['without dialogue context']['multi_turn']:.0f}%. Users naturally ask \"How long is it?\" or answer a "
     "clarification with just \"I have 61 percent\"."),
    ("Structured and unstructured knowledge complement each other, but there is a trade-off. ",
     f"Without the vector store, free-text FAQ accuracy is 0%; with it, {100 * cats['Hybrid KR + NLP (proposed)']['unstructured_faq']:.0f}% "
     f"(Recall@3 = {r3:.2f}). On these questions the plain retrieval baselines score higher "
     f"({100 * cats['NLP + vector retrieval']['unstructured_faq']:.0f}-{100 * cats['Keyword FAQ matching']['unstructured_faq']:.0f}%), "
     "because the hybrid sometimes sends a question to a confident KB intent (\"Wi-Fi in the hostel\" goes to the hostel frame)."),
    ("Out-of-scope detection is the weakest part. ",
     f"The domain gate catches clearly unrelated questions (recall {pct(H['Out-of-scope recall'])}), but questions that share "
     "domain words (\"share price of Infosys\", \"book a movie ticket\") still get a university answer."),
]
restart_numbering()
for lead, text in obs:
    numbered(text, lead)
doc.add_heading("8.3 Limitations", 2)
for t_ in ["Knowledge must be engineered by hand; rules cover only eligibility and scholarship policies.",
           "TF-IDF embeddings are lexical, so pure synonyms (\"in charge of\" vs \"HOD\", \"uniform\" vs \"dress code\") are missed.",
           "English only; the knowledge base is synthetic; the test sets were written by the developer, not collected from real users."]:
    bullet(t_)
doc.add_heading("8.4 Conclusion", 2)
para("A hybrid Knowledge Representation + NLP chatbot was designed, implemented and evaluated. Frames with inheritance hold "
     "the structured university knowledge. Forward-chaining production rules answer personalised eligibility and fee questions "
     "with step-by-step explanations. A lightweight NLP pipeline turns free-form questions into structured queries, a dialogue "
     "manager handles follow-ups and missing details, and a vector store answers unstructured FAQs. The system clearly beats "
     "retrieval-only alternatives, especially on reasoning and multi-turn questions, while staying fast, explainable and offline.")
para("Future work: replace the TF-IDF embedder with a neural sentence embedding (the Embedder interface already supports this), "
     "learn intent weights from real chat logs, add backward chaining for \"what do I need to become eligible?\" questions, "
     "support Tamil and Hindi, and connect the knowledge base to the live admissions database.")

# ============================================================== 9
doc.add_heading("9. Demonstration of the Working Prototype", 1)
para("The prototype has two interfaces on top of the same engine. The terminal chatbot is part of the submission; the web "
     "version is for live demonstration.")
doc.add_heading("9.1 Terminal chatbot (python chat_cli.py)", 2)
para("A real session with the terminal chatbot. The bracketed line after each answer shows intent, answer source and "
     "confidence; /why prints the reasoning trace.")
code_block(["Greenfield University - Enquiry Chatbot  (type /help for examples, /exit to quit)", ""] + demo_lines, size=8)
doc.add_heading("9.2 Web demo (streamlit run app_streamlit.py)", 2)
para("The web demo runs locally at http://localhost:8501. The left panel holds sample questions, the centre shows the "
     "conversation, and the right panel explains each answer: intent, confidence, answer source, the rules that fired "
     "and the extracted entities.")
figure(FIG / "demo_web_1.png", "Figure 8. Web demo: follow-up question resolved from context, then an eligibility decision with the rules that fired.", 16.4)
figure(FIG / "demo_web_2.png", "Figure 9. Web demo: slot filling (CGPA converted to percentage) and a fee calculation with merit scholarship.", 16.4)

# ============================================================ appendix A
doc.add_heading("Appendix A. Software Setup: Steps Followed", 1)
para("All software is free. It was installed for the current Windows user only, without administrator rights:")
restart_numbering()
for lead, t_ in [
    ("Python 3.12.10: ", "winget install --id Python.Python.3.12 -e --scope user"),
    ("Visual Studio Code 1.141: ", "winget install --id Microsoft.VisualStudioCode -e --scope user"),
    ("VS Code extensions: ", "Python (ms-python.python) and Jupyter (ms-toolsai.jupyter), via code --install-extension"),
    ("Python libraries: ", "python -m pip install numpy pandas scikit-learn matplotlib python-docx jupyter ipykernel nbformat nbconvert streamlit"),
    ("PATH: ", "the Python and Python\\Scripts folders were added to the user PATH so that python, pip and streamlit work in any terminal")]:
    numbered(t_, lead)

doc.add_heading("Appendix B. How to Run the Solution", 1)
restart_numbering()
for lead, t_ in [
    ("Open the project: ", "in VS Code choose File > Open Folder and select the university_chatbot folder."),
    ("Notebook: ", "open University_Enquiry_Chatbot.ipynb, select the Python 3.12 kernel (top-right) and click Run All."),
    ("Terminal chatbot: ", "open Terminal > New Terminal and run:  python chat_cli.py   (commands: /why, /reset, /help, /exit)"),
    ("Web demo: ", "in the terminal run:  streamlit run app_streamlit.py   then open http://localhost:8501 (add ?demo=1 for a sample conversation)."),
    ("Re-run the evaluation: ", "the notebook recomputes every metric and figure from data/test_queries.csv.")]:
    numbered(t_, lead)

doc.add_heading("References", 1)
restart_numbering()
for ref in ["S. Russell and P. Norvig, Artificial Intelligence: A Modern Approach, 4th ed., Pearson, 2021 (Ch. 7-10: knowledge-based agents, inference, knowledge representation).",
            "M. Minsky, \"A Framework for Representing Knowledge,\" MIT AI Laboratory Memo 306, 1974.",
            "D. Jurafsky and J. H. Martin, Speech and Language Processing, 3rd ed. draft, 2024 (chatbots and dialogue systems; vector semantics).",
            "C. D. Manning, P. Raghavan and H. Schütze, Introduction to Information Retrieval, Cambridge University Press, 2008 (TF-IDF, cosine similarity).",
            "F. Pedregosa et al., \"Scikit-learn: Machine Learning in Python,\" Journal of Machine Learning Research, 12, pp. 2825-2830, 2011."]:
    numbered(ref)

doc.save(OUT)
print("saved", OUT)
print({k: round(v, 3) for k, v in H.items()})
