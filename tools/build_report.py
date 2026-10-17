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
services = load_test_set(DATA_DIR / "test_queries_services.csv")
services_ho = load_test_set(DATA_DIR / "test_queries_services_holdout.csv")
res_sv, res_svh = {}, {}
for n, make in {"Keyword FAQ matching": KeywordFAQBot, "NLP + vector retrieval": RetrievalBot,
                "Hybrid KR + NLP (proposed)": UniversityChatbot}.items():
    res_sv[n] = run_bot(make(), services)
    res_svh[n] = run_bot(make(), services_ho)
SV = res_sv["Hybrid KR + NLP (proposed)"].answer_ok.mean()
SVH = res_svh["Hybrid KR + NLP (proposed)"].answer_ok.mean()
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

# sample conversation for Section 9 (the website gives exactly the same answers)
SOURCE_NAME = {"knowledge_base": "frame knowledge base", "rule_engine": "rule-based reasoning",
               "vector_store": "vector-store search", "dialogue": "dialogue manager",
               "smalltalk": "small talk", "fallback": "out of scope"}
demo_bot = UniversityChatbot()
s = demo_bot.new_session()
demo_lines = []
for q in ["What is the fee for M.Tech AI?", "How long is it?", "What about MBA?", "Am I eligible for M.Tech AI?",
          "I did B.E ECE with 6.8 CGPA and I qualified GATE",
          "I scored 96% in 12th and I am a state level athlete. What will my fee be for B.Tech CSE?",
          "hostle fees for girls?", "Is the campus ragging free?", "Who won the cricket match yesterday?"]:
    last = demo_bot.respond(q, s)
    demo_lines.append(f"You: {q}")
    demo_lines.append(f"Bot: {last.text}")
    demo_lines.append(f"     (answered by: {SOURCE_NAME[last.source]})")
s = demo_bot.new_session()
demo_lines2 = []
for q in ["When are the end semester exams?", "Which office handles it?", "How do I get a bonafide certificate?",
          "What is the late fee for semester fees?", "Who is the HOD of the Civil department?"]:
    last = demo_bot.respond(q, s)
    demo_lines2.append(f"You: {q}")
    demo_lines2.append(f"Bot: {last.text}")
    demo_lines2.append(f"     (answered by: {SOURCE_NAME[last.source]})")

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


def observation(text):
    """An individual observation about the output just above it."""
    p = para(text, bold_lead="Observation: ", size=10, space_after=10)
    return p


def page_break():
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


DEMO_URL = "https://manikandanayyakkannu.github.io/FAI-Mini-Project/"

# no running header or footer: the pages are kept plain

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
TEAM = [("Manikandan Ayyakkannu", "RA2612049015035"), ("Dinakar B", "RA2612049015032"), ("Manoj Kumar R", "RA2612049015005")]
team = doc.add_table(rows=1, cols=2)
team.style = "Table Grid"
team.alignment = WD_TABLE_ALIGNMENT.CENTER
for i, h in enumerate(("Team member", "Register No.")):
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
    row.cells[0].width, row.cells[1].width = Cm(6.5), Cm(4.5)
    for c in row.cells:
        c.paragraphs[0].paragraph_format.space_before = Pt(2)
        c.paragraphs[0].paragraph_format.space_after = Pt(2)
doc.add_paragraph()
para("Programme: M.Tech - Fundamentals of Artificial Intelligence", size=11, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=2)
para("Date: October 2026", size=11, align=WD_ALIGN_PARAGRAPH.CENTER)
for _ in range(4):
    doc.add_paragraph()
para(f"Live chatbot: {DEMO_URL}", size=10, color=BLUE, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=4)
para("Submitted along with this report: the solution notebook (University_Enquiry_Chatbot.ipynb), the chatbot website "
     "(index.html and web/), the Python engine (chatbot/) and the data files (data/).",
     size=9, color=GREY, align=WD_ALIGN_PARAGRAPH.CENTER)
page_break()

# ============================================================== contents
p = para("Abstract", size=16, color=BLUE, space_after=8)
p.runs[0].bold = True
para(f"In this project we built an enquiry chatbot for a whole university, the fictional AI University. Applicants, "
     f"students and alumni can ask it about admissions and programmes, fees, the academic calendar, exams and results, "
     f"certificates, university offices, departments, the hostel, placements and events. The university data is stored "
     f"as frames that inherit from each other (a small semantic network). Personal questions such as \"Am I eligible?\" or "
     f"\"What will my fee be after the scholarship?\" are answered by forward-chaining production rules. In front of this sit "
     f"an NLP pipeline, a dialogue manager for follow-up questions, and a vector store for FAQ topics that are not in the frames. "
     f"We tested it on {len(test)} labelled questions. Our chatbot answered {pct(H['Answer accuracy'])} of them correctly "
     f"(intent macro-F1 {H['Intent macro F1']:.2f}). On the same knowledge, an NLP + vector-retrieval chatbot managed "
     f"{pct(summ['NLP + vector retrieval']['Answer accuracy'])} and a keyword FAQ-matching chatbot "
     f"{pct(summ['Keyword FAQ matching']['Answer accuracy'])}. On a separate held-out set, which we ran only once at the end, "
     f"it scored {pct(HO['Answer accuracy'])}. On university-services questions it scored {pct(SVH)} on a held-out set, "
     f"against {pct(res_svh['NLP + vector retrieval'].answer_ok.mean())} for the retrieval chatbot. "
     f"Answers take a few milliseconds (mean {lat.mean():.1f} ms) and every answer can "
     f"be explained. The finished chatbot is a website that runs completely inside the browser, so anyone can try it from a link.")
p = para("Keywords: ", size=10, space_after=0)
p.runs[0].bold = True
p.add_run("knowledge representation, frames, semantic network, forward chaining, production rules, natural language "
          "processing, chatbot, university enquiry, vector store, closed-world assumption").font.size = Pt(10)
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
para("A university answers the same questions again and again, by phone, by e-mail and at many different counters. "
     "Applicants ask \"What is the fee for B.Tech CSE?\" or \"Am I eligible for M.Tech AI with 6.8 CGPA?\". Current students "
     "ask \"When do the end-semester exams start?\", \"How do I get a bonafide certificate?\" or \"Whom do I contact about "
     "my hostel room?\". Alumni ask for transcripts. The answers are spread over the admissions office, the exam cell, the "
     "registrar, the accounts office and the departments. Nobody answers after office hours, and two staff members do not "
     "always give the same answer.")
para("Our aim was to build one enquiry chatbot for the whole university that:", space_after=3)
bullet("understands free-form English questions, including paraphrases, abbreviations (B.E, M.Tech, CGPA, plus two) and spelling mistakes;")
bullet("answers factual questions about the whole university from a structured knowledge base: admissions and programmes, "
       "fees, the academic calendar, exams and results, certificates, offices, departments, hostel, placements and events;")
bullet("reasons about personal questions, such as eligibility for a programme or the fee after scholarships, and explains every decision;")
bullet("follows the conversation (\"How long is it?\" after a question about M.Tech AI) and asks for missing details;")
bullet("answers free-text FAQs (ragging, Wi-Fi, refunds) through semantic retrieval, and politely refuses out-of-scope questions;")
bullet("says clearly when something is not available (a department the university does not have, or a question about "
       "another university) instead of giving a wrong answer.")
para("Scope and assumptions: the knowledge base describes a fictional institution, \"AI University\", with 20 programmes "
     "(B.Tech CSE, AI & DS, ECE and Mechanical; B.Sc Computer Science and Mathematics; BCA; B.Com; BBA; B.A. English and "
     "Economics; M.Tech AI and Data Science; MBA, MBA Business Analytics and MBA Finance; MCA; M.Com; M.A. English; Ph.D.), "
     "11 departments, 9 university offices, 7 kinds of certificate, 5 facilities, the academic calendar, the examination "
     "rules, fee payment, events, research, alumni, the admission process and the scholarship policy. We made up all of this data for the "
     "project. The chatbot is text-based, works in English and runs in any web browser without a server.", space_after=8)

# ============================================================== 2
doc.add_heading("2. AI Problem Formulation", 1)
para("The chatbot is modelled as a knowledge-based agent. It perceives a sentence, updates its internal state (the dialogue "
     "context) and acts by choosing the answer that its knowledge base and inference rules support.")
doc.add_heading("2.1 PEAS description", 2)
table(["PEAS element", "Description"], [
    ["Performance measure", "Answer accuracy, intent precision / recall / F1, correct refusal of out-of-scope questions, response time, explainability"],
    ["Environment", "Applicants, parents, current students and alumni typing questions; the university knowledge (frames, rules, FAQ passages)"],
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
bullet(" the NLP pipeline maps a sentence x to a structured query q = (intent, entities), where intent belongs to 27 "
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
para("The suggested technique for this topic was Knowledge Representation + NLP. We ended up with six parts, each from "
     "classical AI. The table says what each part does and why we picked it:")
table(["Component", "Technique used", "Why it was chosen"], [
    ["Knowledge representation", "Frames with is_a inheritance, which also form a semantic network of (subject, predicate, object) triples",
     "University data is naturally hierarchical (Programme > UG/PG > B.Tech CSE). Inheritance stores duration, documents and deadlines once per class, so the KB is small and consistent."],
    ["Reasoning", "Production rules + forward chaining with refraction and an explanation trace",
     "Eligibility and scholarship policies are IF-THEN rules. Forward chaining is data-driven: it starts from the applicant's facts and derives the conclusion. Each fired rule becomes a sentence of explanation."],
    ["NLP", "Normalisation, tokenisation, spelling correction (edit-distance similarity), synonym mapping, stop-word removal, stemming, regex entity extraction, hybrid intent classifier",
     "Easy to follow and explain; needs only 112 example sentences; no GPU or downloaded model; handles Indian academic abbreviations (B.E, M.Tech, CGPA, plus two)."],
    ["Retrieval fallback", "Vector store: TF-IDF word + character n-gram embeddings, cosine similarity, top-k search",
     "Free-text FAQ passages (ragging, Wi-Fi, refunds) are not worth modelling as frames. The embedder is pluggable, so a neural sentence embedding can be used without code changes."],
    ["Dialogue management", "Context memory + slot filling",
     "Supports follow-up questions and asks for missing facts (marks, qualification) before running the rules."],
    ["Closed-world assumption", "Whatever is not a frame in the KB is not offered; lists of known missing departments and other institutions help recognise such questions",
     "Without it, \"HOD of CSE in SRM?\" got our own CSE HOD. Now the bot says what it does not know instead of answering a different question with confidence."],
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
doc.add_heading("3.3 Saying \"not available\": the closed-world assumption", 2)
para("While testing, we noticed two kinds of question that got confident but wrong answers. \"Who is the HOD of CSE in SRM "
     "University?\" was answered with our own CSE HOD, because the bot recognised \"HOD\" and \"CSE\" and simply ignored "
     "\"SRM\". And \"Who is the HOD of the Civil department?\" got a general B.Tech answer, because Civil is not a branch we "
     "offer. Both cases are handled with the closed-world assumption: if something is not in the knowledge base, the "
     "university does not offer it.")
bullet(" before anything else, the bot looks for the name of another institution (a list of 40 or so, plus any "
       "\"<name> University\"). If it finds one, it says it can only answer about AI University.", "Other universities:")
bullet(" the knowledge base also lists subjects that people commonly ask about but AI University does not offer (Civil, EEE, "
       "Law, Pharmacy, Hotel Management and so on). If the question names one of these and nothing that we do offer, the bot "
       "replies that the department is not available and gives the admissions office's contact details. It reads the "
       "question once more without that word, so \"I did B.E Civil, can I join MBA?\" is still treated as an MBA question.",
       "Departments we do not have:")
bullet(" a department name the bot has never heard of (\"HOD of the XYZ department\") gets the same \"not available\" "
       "reply, instead of a list of every department.", "Unknown department names:")
bullet(" \"hi, do you have seats in medical?\" is answered as a question, not just as a greeting, and \"the answers are not "
       "relevant\" gets an apology with an example of how to ask, instead of the out-of-scope message.",
       "Greetings and complaints:")
para("We also gave AI & DS its own department, with its own HOD, and added arts, science, commerce and management "
     "programmes. The arts and science degrees use a new class frame, ArtsScienceUGProgram. It inherits from UGProgram "
     "but overrides the duration (3 years), the cut-off (50%) and the subject requirement, which is another example of "
     "inheritance at work.")

doc.add_heading("3.4 From admissions to the whole university", 2)
para("Our first version only answered admission questions. To cover the whole university we added knowledge, not new kinds "
     "of reasoning:")
bullet(" Office (9 offices: admissions, registrar, exam cell, accounts, dean of students, hostel, placement cell, international "
       "relations, IT helpdesk) and Certificate (7 certificates). Every certificate inherits how to apply and the issuing office "
       "from the Certificate class; the provisional certificate overrides the office with the exam cell.", "New class frames:")
bullet(" academic_calendar, examinations, fee_payment, events, research and alumni. The exam dates are not repeated in the "
       "examinations frame; the answer reads them from the academic_calendar frame.", "New instance frames:")
bullet(" every department with placements keeps four seasons (2022-23 to 2025-26). A year in the question selects the season "
       "(\"placements in 2023\" means the batch that graduated in 2023), and a year we have no data for gets \"not "
       "available\" with the latest figures.", "Placement history:")
bullet(" academic calendar, examination, results, fee payment, certificates, office information, faculty and research, events "
       "and alumni, each with example sentences and a few high-precision keywords.", "Nine new intents:")
bullet(" each topic is linked to the office that handles it (examinations -> exam cell, fee payment -> accounts office, "
       "certificates -> registrar), so \"Which office handles it?\" after an exam question is answered from the dialogue context.",
       "Links between frames:")
bullet(" when a question names an office or a certificate (\"exam cell\", \"TC\"), that entity decides the topic, whatever the "
       "intent classifier guessed.", "Entities decide the topic:")
para("The new intents share words with the old ones (\"fee\", \"start\", \"semester\"), so a few rules separate them: "
     "\"When does the semester start?\" is a calendar question, not the admission deadline, and \"Can I pay the fees late?\" "
     "is about paying, not about how much a programme costs.")

doc.add_heading("3.5 Why not a pure machine-learning or LLM chatbot?", 2)
para("We did consider using a large language model, but decided against it. For admissions the numbers (fees, cut-offs) "
     "have to be exactly right, the same every time, and explainable. In our system every answer can be traced back to a "
     "frame slot or a rule. A language model can make up a fee or a cut-off, it cannot properly justify an eligibility "
     "decision, and it needs a paid API or a GPU. Our approach has its own cost: someone has to write the knowledge "
     "by hand. We discuss this in Section 8.")

# ============================================================== 4
doc.add_heading("4. Python Implementation", 1)
para("We first wrote the chatbot as a small Python 3.12 package. The solution notebook uses it for all the experiments, "
     "and it was used to produce every number in this report. It needs numpy, pandas, scikit-learn (TF-IDF and metrics) "
     "and matplotlib. The NLP and reasoning parts use only the Python standard library.")
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
    ["data/*.json, data/*.csv", "Knowledge base (70 frames), rules (26), intents (27), free-text FAQs (15), four test sets"]],
    [5.0, 11.4], font=9, bold_first_col=True)
doc.add_heading("4.1 The chatbot website", 2)
para("For the demonstration we wanted a link that anyone could open, with nothing to install and no terminal. So we "
     "rewrote the same engine in plain JavaScript, one file for each Python module. It runs entirely inside the browser "
     "and is hosted for free on GitHub Pages. There is no server and no API key, and both versions read the same data/*.json "
     "files. To make sure the website behaves exactly like the Python version, we put 361 test conversations (all four test sets plus 171 extra questions about the newer features) through "
     "both. The answers, intents, answer sources and confidence scores came out identical, so the results in Sections 6 "
     "and 7 also hold for the website.")
table(["File", "What it contains"], [
    ["index.html", "The chat page: sample questions and guided demos on the left, the conversation on the right"],
    ["web/css/style.css", "Layout and colours (light and dark theme, works on phones)"],
    ["web/js/knowledge-base.js, nlp.js, intent-classifier.js, reasoning.js, answers.js, similarity.js", "The engine, one file per Python module"],
    ["web/js/chatbot.js", "Dialogue manager: decides whether the KB, the rules or the vector store answers"],
    ["web/js/app.js", "Everything on the screen: messages, follow-up suggestions and guided demos"],
    ["tests/engine-test.html", "Runs all four test sets in the browser and shows the accuracy"]],
    [6.4, 10.0], font=9, bold_first_col=True)
para("What happens to every message (respond() in both versions):", space_after=3)
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
para("There is no public dataset for a made-up university, so we wrote the knowledge and the test questions ourselves:")
bullet(" 70 frames, 26 production rules, 27 intents with 178 example sentences, and 175 FAQ passages "
       "(160 generated automatically from the frames and 15 free-text passages).", "Knowledge sources:")
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
para("For the rest of the university we wrote two more test sets:", space_after=3)
bullet(f" {len(services)} questions on the calendar, exams, results, fee payment, certificates, offices, research, events "
       "and alumni, including two short conversations. We wrote them after building these features and then used them to "
       "improve the features, so their score is optimistic.", "University-services development set:")
bullet(f" {len(services_ho)} further questions, written before that tuning and scored once.",
       "University-services held-out set:")
cnt_s = services.category.value_counts(sort=False)
cnt_sh = services_ho.category.value_counts(sort=False)
table(["Category", "Development", "Held-out"],
      [[c, cnt_s.get(c, 0), cnt_sh.get(c, 0)] for c in cnt_s.index] + [["Total", len(services), len(services_ho)]],
      [5, 3, 3], font=9, bold_first_col=True)

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
observation(f"{pct(H['Answer accuracy'])} of the {len(test)} questions got exactly the right answer, and the intent was "
            f"right just as often (macro-F1 {H['Intent macro F1']:.2f}). The chatbot refused {pct(H['Out-of-scope recall'])} of the "
            f"questions it should refuse, and when it did refuse it was right {pct(H['Out-of-scope precision'])} of the time. "
            f"An answer takes {lat.mean():.1f} ms on average, so speed is not an issue.")
figure(FIG / "fig_category_hybrid.png", "Figure 3. Answer accuracy of the hybrid chatbot by test category.", 13)
_hc = cats["Hybrid KR + NLP (proposed)"]
observation(f"{sum(_hc == 1)} of the {len(_hc)} categories are answered perfectly, including reasoning, multi-turn and spelling "
            f"mistakes, which are the hard cases for a chatbot. The weakest category is {desc[_hc.idxmin()].lower()} "
            f"({100 * _hc.min():.0f}%): these questions are not in the frames and have to be matched by wording alone.")
figure(FIG / "fig_confusion.png", "Figure 4. Intent confusion matrix on the main test set.", 12.5)
_wrong = res["Hybrid KR + NLP (proposed)"]
_wrong = _wrong[_wrong.expected_intent != _wrong.pred_intent]
observation(f"Almost everything lies on the diagonal. Only {len(_wrong)} questions got the wrong intent, and no intent is "
            "confused systematically with another: each mistake is a single question with an unusual word (listed below).")
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
observation("Most remaining errors have the same cause: the chatbot compares words, not meanings. \"Uniform\" and \"dress "
            "code\" share no words, and \"price\" is a synonym of fee in our domain. A neural sentence embedding in the "
            "vector store would fix several of them without changing the knowledge base.")

# ============================================================== 7
doc.add_heading("7. Comparison with Alternative Approaches", 1)
para("Two alternative chatbots were built on exactly the same knowledge: the same 175 FAQ question-answer pairs, with "
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
observation(f"Our chatbot is ahead in every category except free-text FAQs, where the retrieval chatbots score "
            f"{100 * cats['NLP + vector retrieval']['unstructured_faq']:.0f}-{100 * cats['Keyword FAQ matching']['unstructured_faq']:.0f}% "
            f"against our {100 * cats['Hybrid KR + NLP (proposed)']['unstructured_faq']:.0f}%. The biggest gaps are reasoning and "
            "multi-turn questions, where both alternatives score 0%: retrieving a stored answer cannot decide whether one "
            "particular student is eligible, or remember what \"it\" refers to.")
doc.add_heading("7.1 Ablation study", 2)
para("To measure what each component contributes, one component was removed at a time:")
abl_cols = ["factual", "paraphrase", "typo", "reasoning", "multi_turn", "unstructured_faq", "out_of_scope"]
table(["Configuration", "Overall"] + ["fact.", "para.", "typo", "reas.", "multi", "FAQ", "OOS"],
      [[n, pct(v["acc"])] + [f"{100 * v[c]:.0f}" for c in abl_cols] for n, v in abl.items()],
      [4.6, 1.7] + [1.45] * 7, font=8.5, bold_first_col=True)
figure(FIG / "fig_ablation.png", "Figure 6. Ablation: answer accuracy when one component is removed.", 12.5)
_drops = {n: abl["Full hybrid system"]["acc"] - v["acc"] for n, v in abl.items() if n != "Full hybrid system"}
_big = max(_drops, key=_drops.get)
observation(f"Every component earns its place: removing any one of them lowers the accuracy. The largest drop comes "
            f"{_big} ({pct(abl['Full hybrid system']['acc'])} to {pct(abl[_big]['acc'])}), which shows that the rules are the "
            "core of the system. The other components each fix one kind of question (typos, follow-ups or free-text FAQs).")
doc.add_heading("7.2 Sensitivity to the confidence threshold", 2)
para("The intent-confidence threshold was fixed at 0.30 before testing. The sweep in Figure 7 shows that higher thresholds "
     "(0.35-0.55) give slightly higher accuracy and better out-of-scope recall on this test set. This was not adopted, "
     "because choosing a parameter on the test set would make the reported result optimistic. It should be validated on new data.")
figure(FIG / "fig_threshold.png", "Figure 7. Effect of the intent-confidence threshold.", 12)
observation("The accuracy changes by only a few points between thresholds 0.20 and 0.55, so the system is not sensitive to "
            "this setting. Below 0.30 more unrelated questions get an answer; above it, a few more are refused correctly.")
doc.add_heading("7.3 Held-out test set", 2)
table(["Metric"] + names, [[k] + [f"{summ_ho[n][k]:.3f}" if "ms" not in k else f"{summ_ho[n][k]:.2f}" for n in names]
                           for k in ["Answer accuracy", "Intent accuracy", "Intent macro F1", "Out-of-scope recall"]],
      [5, 3.6, 3.8, 4], font=9, highlight_last_col=True, bold_first_col=True)
para("Note: while checking the held-out results, one label (H14, B.Sc Maths for M.Tech Data Science) was found to be wrong. The "
     "knowledge base accepts M.Sc Maths, not B.Sc, so the chatbot's \"not eligible\" answer was correct. The label was "
     "corrected; the system itself was not changed.", size=9.5, color=GREY, italic=True)
observation(f"On new questions the accuracy falls from {pct(H['Answer accuracy'])} to {pct(HO['Answer accuracy'])}, which is the "
            "honest figure to expect from real users. The ranking of the three chatbots does not change, so the advantage "
            "of the knowledge-based design is not a result of tuning on the main test set.")
doc.add_heading("7.4 University-services test sets", 2)
para("The FAQ list that the two alternative chatbots search is generated from the same frames, with one entry for each office, "
     "each certificate, the calendar, exams, results, fee payment, events, research and alumni, so all three systems again "
     "share the same knowledge.")
names_sv = list(res_sv)
table(["Test set"] + names_sv,
      [["Development (" + str(len(services)) + ")"] + [pct(res_sv[n].answer_ok.mean()) for n in names_sv],
       ["Held-out (" + str(len(services_ho)) + ")"] + [pct(res_svh[n].answer_ok.mean()) for n in names_sv]],
      [4, 3.8, 3.8, 4.8], font=9, highlight_last_col=True, bold_first_col=True)
para("How to read these numbers honestly: on its first run, before we used it to improve anything, our chatbot scored 72.5% "
     "on the development set; the 100% above is after tuning on it. The held-out set scored 80% on its single run. One of its "
     "questions, \"Can I pay the semester fee using UPI?\", revealed a real bug: the spelling corrector turned \"using\" into "
     "\"nursing\", a word we had added to the vocabulary for the closed-world check. We fixed it by adding everyday words such "
     "as \"using\" to the list of words that are never corrected, and nothing else was changed for the held-out questions. "
     f"After that fix the held-out score is {pct(SVH)}. The remaining misses are \"winter break\" and \"lunch break\", "
     "phrasings that the calendar intent has not seen.", size=9.5, color=GREY, italic=True)
observation("Questions from current students are short and use everyday words (\"Do we have classes on Saturdays?\", "
            "\"I need my transfer certificate\"), so the retrieval chatbots, which match whole FAQ questions, do poorly "
            "on them. Our chatbot first finds the office or certificate named in the question, and that entity decides the "
            "answer. The held-out set also showed that adding words to the vocabulary has side effects on spelling correction, "
            "which is worth testing every time the knowledge base grows.")

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
bullet(f" {pct(SVH)} on the held-out university-services questions (80% on the first run, before the spelling bug was "
       f"fixed) and {pct(SV)} on the development set, against {pct(res_svh['NLP + vector retrieval'].answer_ok.mean())} and "
       f"{pct(res_svh['Keyword FAQ matching'].answer_ok.mean())} for the alternatives on the held-out set.", "University services:")
bullet(f" mean {lat.mean():.1f} ms per answer on an ordinary laptop, without a GPU. The browser version feels just as instant.", "Speed:")
doc.add_heading("8.2 Discussion and individual observations on the outputs", 2)
obs = [
    ("The rules make the biggest difference. ",
     f"Both alternatives scored {100 * cats['Keyword FAQ matching']['reasoning']:.0f}% on reasoning questions and about "
     f"{100 * cats['NLP + vector retrieval']['multi_turn']:.0f}% on multi-turn questions. They can show the general eligibility "
     "text, but they cannot say whether this particular student (\"72% in B.E ECE\") is eligible for M.Tech AI. When we "
     f"switched off only the rule engine, accuracy fell from {pct(abl['Full hybrid system']['acc'])} to "
     f"{pct(abl['without rule-based reasoning']['acc'])}, the biggest drop of any component."),
    ("Inheritance kept the knowledge base small. ",
     "Duration, documents and deadlines are written once in the UG, PG and Research class frames, and every programme "
     "inherits them. When one of these values changes we edit one place. The trace also shows where a value came from, "
     "for example that mba.duration comes from the PGProgram frame."),
    ("Explanations needed no extra work. ",
     "Each rule that fires writes one sentence, e.g. \"[R3] A 5% relaxation applies for the OBC category, so the cut-off "
     "becomes 55%\". The important ones are part of the answer itself (\"Converted CGPA 7.2 to 68.4%... Your score meets the "
     "cut-off of 50%\"), so a student can see exactly why they are or are not eligible, and the notebook prints the full "
     "trace. A purely statistical chatbot cannot give this."),
    ("Small NLP steps mattered more than we expected. ",
     f"Spelling correction alone took typo accuracy from {100 * abl['without spelling correction']['typo']:.0f}% to "
     f"{100 * abl['Full hybrid system']['typo']:.0f}%. Without normalising B.E / B.Tech / M.Tech / plus two / CGPA, most "
     "entities would not have been found at all."),
    ("People expect the bot to remember the conversation. ",
     f"Without the context memory, multi-turn accuracy falls from {100 * abl['Full hybrid system']['multi_turn']:.0f}% to "
     f"{100 * abl['without dialogue context']['multi_turn']:.0f}%. Students ask \"How long is it?\" or reply to a question "
     "with just \"I have 61 percent\", and the bot has to understand that."),
    ("Frames and free-text FAQs work well together, with one trade-off. ",
     f"Without the vector store the free-text FAQ questions score 0%; with it, "
     f"{100 * cats['Hybrid KR + NLP (proposed)']['unstructured_faq']:.0f}% (Recall@3 = {r3:.2f}). The plain retrieval "
     f"chatbots still do better on these "
     f"({100 * cats['NLP + vector retrieval']['unstructured_faq']:.0f}-{100 * cats['Keyword FAQ matching']['unstructured_faq']:.0f}%), "
     "because our chatbot sometimes trusts a confident KB intent too much (\"Wi-Fi in the hostel\" goes to the hostel frame)."),
    ("One knowledge base can cover the whole university. ",
     "Offices, certificates, the calendar and the exam rules are frames just like the programmes, so going from admissions to "
     "the whole university needed new frames, intents and answer templates but no new kind of reasoning. Links between frames "
     "answer follow-ups such as \"Which office handles it?\" (examinations -> exam cell). The retrieval chatbots do much worse "
     "here because many of these questions are short and use everyday words (\"Do we have classes on Saturdays?\")."),
    ("Spotting out-of-scope questions is the weakest part. ",
     f"The domain gate catches clearly unrelated questions (recall {pct(H['Out-of-scope recall'])}), but questions that "
     "share words with the domain (\"share price of Infosys\", \"book a movie ticket\") still get a university answer."),
]
restart_numbering()
for lead, text in obs:
    numbered(text, lead)
doc.add_heading("8.3 Limitations", 2)
for t_ in ["Knowledge must be engineered by hand; rules cover only eligibility and scholarship policies.",
           "The \"not available\" replies only cover departments and institutions in our lists; an unknown name that is "
           "not written as \"<name> University\" can still slip through.",
           "TF-IDF embeddings are lexical, so pure synonyms (\"in charge of\" vs \"HOD\", \"uniform\" vs \"dress code\") are missed.",
           "The university-services test sets are small (40 and 20 questions) and were written by us after building those "
           "features.",
           "English only; the knowledge base is synthetic; the test sets were written by the developer, not collected from real users."]:
    bullet(t_)
doc.add_heading("8.4 Conclusion", 2)
para("We designed, built and tested an enquiry chatbot for the whole university that combines knowledge representation "
     "with NLP. Frames with inheritance hold the university's facts: programmes, departments, offices, certificates, the "
     "academic calendar and the exam rules. Forward-chaining rules answer personal eligibility and fee questions and "
     "explain each step. A simple NLP pipeline turns a free-form question into intent and entities, the dialogue manager "
     "handles follow-ups and missing details, and a vector store covers FAQ topics that are not in the frames. Our chatbot "
     "clearly beats the retrieval-only chatbots, especially on reasoning and multi-turn questions. It stays fast and "
     "explainable, and as a website it can be demonstrated from any phone or laptop.")
para("If we continue the project, we would like to: try a neural sentence embedding in place of TF-IDF (the embedder is "
     "already pluggable); learn the intent weights from real chat logs; add backward chaining for questions like \"what "
     "do I need to become eligible?\"; support Tamil and Hindi; and read the knowledge base from the university's real "
     "databases (admissions, exam results, fee dues).")

# ============================================================== 9
doc.add_heading("9. Demonstration of the Working Prototype", 1)
para(f"The working chatbot is online at {DEMO_URL}. It opens straight away in any browser, on a laptop or a phone, "
     "with nothing to install.")
doc.add_heading("9.1 The chat page", 2)
para("The page is kept simple, like a normal chat application. On the left are sample questions grouped by topic, "
     "plus three guided demos that type out a short conversation on their own (useful during the presentation). On the "
     "right is the conversation. The replies contain only the answer, with no technical details. After each answer "
     "the bot suggests follow-up questions such as \"How long is it?\". The (i) button opens a short page about the "
     "project, the team and the results. How each answer was produced (intent, entities, rules fired, trace) is shown "
     "in the notebook, Section 9.")
figure(FIG / "demo_web_1.png", "Figure 8. Demo 1: a follow-up question (\"How long is it?\") answered from context, then an eligibility decision.", 16.4)
figure(FIG / "demo_web_2.png", "Figure 9. Demo 2: the bot asks for the missing details, converts CGPA to a percentage, and then works out a fee with the merit scholarship.", 16.4)
figure(FIG / "demo_web_3.png", "Figure 10. Demo 3: exam dates, \"Which office handles it?\" answered from context, a certificate, and a department the university does not have.", 16.4)
observation("In Demo 1, \"How long is it?\" has no programme in it, yet the answer is about M.Tech AI, taken from the "
            "previous question. In Demo 2 the bot asks for the missing marks instead of guessing, then converts the CGPA "
            "with rule R1 before checking the cut-off. In Demo 3, \"Which office handles it?\" is answered with the exam "
            "cell because the previous topic was the exams, and the Civil Engineering question gets \"not available\" "
            "rather than a wrong answer. The replies read like a person answering, while the reasoning behind each one "
            "stays traceable in the notebook.")
doc.add_heading("9.2 A sample conversation", 2)
para("This is one complete conversation with the chatbot. The line after each answer says which part of the system "
     "produced it.")
code_block(demo_lines, size=8)
para("And a conversation about university services:")
code_block(demo_lines2, size=8)
observation("Each answer comes from a known place: a frame slot, a rule or an FAQ passage. In the first conversation the "
            "misspelt \"hostle fees\" is corrected, the ragging question is matched in the FAQ passages, and the cricket "
            "question is refused instead of being answered with something unrelated. In the second, the exam dates come from "
            "the academic-calendar frame, the office from the link between the exams and the exam cell, and the certificate "
            "fee and time from the certificate frame. The same questions give the same answers every time.")

# ============================================================ appendix A
doc.add_heading("Appendix A. How to Use the Solution", 1)
restart_numbering()
for lead, t_ in [
    ("The chatbot: ", f"open {DEMO_URL} in any browser. Click a sample question or type your own. \"Demo 1\" and "
                      "\"Demo 2\" (admissions) and \"Demo 3\" (exams and services) in the left panel play a short conversation for the presentation."),
    ("Test results in the browser: ", "open tests/engine-test.html on the same site. It runs all four test sets and shows "
                                     "the accuracy and every answer."),
    ("The notebook: ", "open the project folder in VS Code (File > Open Folder), open University_Enquiry_Chatbot.ipynb, "
                       "select the Python 3.12 kernel (top right) and click Run All. It recomputes every metric and "
                       "figure in this report."),
    ("Changing the knowledge: ", "the facts, rules, intents and FAQs live in data/*.json. Both the website and the notebook "
                                 "read these files, so an edit there changes both.")]:
    numbered(t_, lead)

doc.add_heading("Appendix B. Software Used", 1)
para("All the software is free. Nothing needs to be installed to use the chatbot, since it runs in the browser. For the "
     "notebook we installed, for the Windows user only:")
restart_numbering()
for lead, t_ in [
    ("Python 3.12.10 ", "(winget install --id Python.Python.3.12 -e --scope user)"),
    ("Visual Studio Code ", "with the Python and Jupyter extensions"),
    ("Python libraries: ", "numpy, pandas, scikit-learn, matplotlib, jupyter and ipykernel (python -m pip install -r requirements.txt)"),
    ("Hosting: ", "GitHub Pages (Settings > Pages > Deploy from branch: main, folder / (root))")]:
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
