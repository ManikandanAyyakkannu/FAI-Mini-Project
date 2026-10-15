"""The hybrid Knowledge-Representation + NLP chatbot engine.

Pipeline for every user message:

    text -> NLP pre-processing -> entity extraction -> intent classification
         -> dialogue context (coreference / slot filling)
         -> reasoning (forward chaining) or frame-KB lookup
         -> vector-store retrieval fallback for unknown questions
         -> natural-language answer + explanation trace

The engine is UI-independent: the terminal chatbot, the notebook and the web
demo all call ``UniversityChatbot.respond(message, session)``.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

from .answers import Answers, rs
from .faq_builder import build_faq
from .intent_classifier import IntentClassifier
from .knowledge_base import KnowledgeBase
from .nlp import EntityExtractor, TextPreprocessor, qualification_label
from .reasoning import ReasoningEngine
from .vector_store import VectorStore

PROGRAM_INTENTS = {"program_fee", "program_duration", "program_seats", "eligibility",
                   "placement", "department_info", "admission_deadline", "documents_required"}
SMALLTALK = {"greeting", "goodbye", "thanks"}
PERSONAL_RE = re.compile(r"\b(am i|can i|could i|will i|would i|do i|i have|i've|i did|i got|i scored|"
                         r"i secured|i completed|i finished|i am|i'm|my|me|i)\b")
GENERIC_WORDS = {"what", "how", "who", "which", "when", "why", "have", "good", "take", "much", "many",
                 "s", "per", "under", "kind", "like", "best", "help", "give", "make", "show", "list",
                 "anyone", "all", "there", "well", "one", "need", "provide", "available"}
FALLBACK_TEXT = ("Sorry, I could not find an answer to that. I can help with programmes, fees, eligibility, "
                 "admissions, deadlines, documents, hostel, scholarships, placements and campus facilities. "
                 "For anything else please contact the admissions office at +91 44 2345 6789.")


@dataclass
class Session:
    context: dict[str, Any] = field(default_factory=dict)   # last programme / intent
    pending: dict[str, Any] | None = None                    # unfinished task waiting for facts
    history: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class BotResponse:
    text: str
    intent: str
    confidence: float
    answer_key: str
    source: str                      # knowledge_base | rule_engine | vector_store | smalltalk | fallback | dialogue
    entities: dict[str, Any] = field(default_factory=dict)
    trace: list[str] = field(default_factory=list)
    latency_ms: float = 0.0


class UniversityChatbot:
    """Hybrid KR + NLP chatbot. The flags allow ablation experiments."""

    def __init__(self, use_reasoning: bool = True, use_context: bool = True,
                 use_vector_fallback: bool = True, use_spell_correction: bool = True,
                 embedder=None, intent_threshold: float = 0.30, vector_threshold: float = 0.35):
        self.kb = KnowledgeBase()
        self.pre = TextPreprocessor()
        for aliases in self.kb.program_aliases.values():
            for a in aliases:
                self.pre.add_vocabulary(a.split())
        if not use_spell_correction:
            self.pre.correct = lambda toks: (toks, {})
        self.extractor = EntityExtractor(self.kb)
        self.classifier = IntentClassifier(self.pre, threshold=intent_threshold)
        self.reasoner = ReasoningEngine(self.kb)
        self.answers = Answers(self.kb)
        self.faq = build_faq(self.kb)
        for f in self.faq:                       # FAQ wording also counts as known vocabulary
            self.pre.add_vocabulary(self.pre.normalise(f["question"]).split())
        self.store = VectorStore(embedder)
        self.store.add([self.pre.normalise(f["question"]) for f in self.faq], self.faq)
        # domain vocabulary = every content word the knowledge sources know about
        texts = [ex for spec in self.classifier.intents.values() for ex in spec["examples"]]
        texts += [f["question"] for f in self.faq]
        texts += [a for al in self.kb.program_aliases.values() for a in al]
        self.domain_vocab = {lem for t in texts for lem in self.pre.process(t).lemmas} - GENERIC_WORDS
        self.use_reasoning = use_reasoning
        self.use_context = use_context
        self.use_vector_fallback = use_vector_fallback
        self.vector_threshold = vector_threshold

    def new_session(self) -> Session:
        return Session()

    # ================================================================ main
    def respond(self, message: str, session: Session | None = None) -> BotResponse:
        t0 = time.perf_counter()
        session = session if session is not None else Session()
        resp = self._respond(message, session)
        resp.latency_ms = (time.perf_counter() - t0) * 1000
        session.history.append((message, resp.text))
        if resp.entities.get("program") in self.kb.programs:
            session.context["program"] = resp.entities["program"]
        if resp.intent not in SMALLTALK | {"out_of_scope"}:
            session.context["last_intent"] = resp.intent
        return resp

    def _respond(self, message: str, session: Session) -> BotResponse:
        p = self.pre.process(message)
        ents = self.extractor.extract(p.normalised, p.lemmas)
        intent, conf, _ = self.classifier.predict(p)
        trace = [f"Normalised text: '{p.normalised}'", f"Lemmas: {p.lemmas}"]
        if p.corrections:
            trace.append(f"Spelling corrected: {p.corrections}")
        trace.append(f"Intent classifier -> {intent} (score {conf:.2f})")
        if ents.as_dict():
            trace.append(f"Entities: {ents.as_dict()}")

        lemmas = set(p.lemmas)
        personal = bool(PERSONAL_RE.search(p.normalised))
        applicant = {"qualification": ents.qualification, "percentage": ents.percentage, "cgpa": ents.cgpa,
                     "exam": ents.exam, "category": ents.category, "subjects": ents.subjects,
                     "sports": ents.sports, "sibling": ents.sibling}
        has_facts = ents.has_applicant_facts or ents.sports or ents.sibling

        # ---- 1. finish a pending task (multi-turn slot filling)
        answers_program_question = (session.pending and session.pending["program"] is None
                                    and ents.program in self.kb.programs)
        if self.use_context and session.pending and (has_facts or answers_program_question) \
                and intent not in SMALLTALK:
            task = session.pending
            merged = {k: v for k, v in task["facts"].items() if v not in (None, False)}
            merged.update({k: v for k, v in applicant.items() if v not in (None, False)})
            pid = ents.program if (ents.program and not ents.program.startswith("group")) else task["program"]
            trace.append(f"Dialogue: continuing pending '{task['task']}' task with new facts")
            if task["task"] == "eligibility":
                return self._eligibility(pid, merged, session, trace, conf)
            return self._fee_calc(pid, merged, session, trace, conf)

        # ---- 2. personalised questions -> rule-based reasoning
        if self.use_reasoning:
            money = intent in ("program_fee", "scholarship") or {"fee", "scholarship"} & lemmas
            if money and (ents.percentage or ents.cgpa or ents.sports or ents.sibling):
                pid = ents.program or (session.context.get("program") if self.use_context else None)
                trace.append("Reasoning: personal fee/scholarship question -> scholarship rules")
                return self._fee_calc(pid, applicant, session, trace, conf)
            if not money and (ents.has_applicant_facts or (personal and intent == "eligibility")):
                if intent in ("eligibility", "out_of_scope", "admission_process", "program_list") or ents.program:
                    pid = ents.program or (session.context.get("program") if self.use_context else None)
                    trace.append("Reasoning: personal eligibility question -> eligibility rules")
                    return self._eligibility(pid, applicant, session, trace, conf)

        # ---- 3. dialogue context (coreference: "its fee", "what about MCA?")
        if self.use_context:
            last = session.context.get("last_intent")
            if ents.program and last in PROGRAM_INTENTS and (intent == "out_of_scope" or conf < 0.42) \
                    and len(p.lemmas) <= 4:
                trace.append(f"Dialogue: elliptical follow-up -> reusing previous intent '{last}'")
                intent = last
            if intent in PROGRAM_INTENTS and not ents.program and not ents.department \
                    and session.context.get("program"):
                ents.program = session.context["program"]
                trace.append(f"Dialogue: programme taken from context -> {ents.program}")

        # ---- 4a. domain-relevance gate: no known domain word and no entity -> out of scope
        known = set(p.lemmas) & self.domain_vocab
        if not known and not (ents.program or ents.department or ents.facility or has_facts):
            trace.append("Domain gate: no university-related word found -> out of scope")
            return BotResponse(FALLBACK_TEXT, "out_of_scope", conf, "fallback", "fallback", ents.as_dict(), trace)

        # ---- 4b. vector store retrieval for unknown / low-confidence questions, or when a
        #          free-text FAQ passage (a topic with no frame in the KB) matches strongly
        if self.use_vector_fallback and intent not in SMALLTALK:
            hits = self.store.search(p.normalised, k=3)
            top = hits[0] if hits else None
            free_text_wins = (top is not None and top.metadata["intent"] == "faq"
                              and top.score >= 0.5 and top.score > conf)
            if intent == "out_of_scope" or conf < 0.40 or free_text_wins:
                trace.append("Vector store top-3: " + "; ".join(f"{h.metadata['key']} ({h.score:.2f})" for h in hits))
                if top and top.score >= self.vector_threshold and (intent == "out_of_scope" or top.score > conf):
                    return BotResponse(top.metadata["answer"], top.metadata["intent"], top.score,
                                       top.metadata["key"], "vector_store", ents.as_dict(), trace)
        if intent == "out_of_scope" or conf < 0.40:
            if intent == "out_of_scope":
                return BotResponse(FALLBACK_TEXT, "out_of_scope", conf, "fallback", "fallback", ents.as_dict(), trace)

        # ---- 5. answer from the frame knowledge base
        if intent in SMALLTALK:
            return BotResponse(self.classifier.smalltalk[intent], intent, conf, intent, "smalltalk",
                               ents.as_dict(), trace)
        if intent == "eligibility" and personal and self.use_reasoning and ents.program:
            return self._eligibility(ents.program, applicant, session, trace, conf)
        text, key = self._kb_answer(intent, ents, p, trace)
        return BotResponse(text, intent, conf, key, "knowledge_base", ents.as_dict(), trace)

    # ============================================================ KB lookup
    def _level(self, ents) -> str | None:
        if ents.program and not ents.program.startswith("group"):
            return self.kb.program_level(ents.program)
        if ents.program in ("group:btech",):
            return "UG"
        if ents.program in ("group:mtech", "group:pg"):
            return "PG"
        return ents.level_filter

    def _kb_answer(self, intent: str, ents, p, trace) -> tuple[str, str]:
        A, pid = self.answers, ents.program
        trace.append(f"Knowledge base lookup for intent '{intent}'")
        if intent == "program_list":
            return A.program_list(ents.level_filter), f"program_list:{ents.level_filter or 'all'}"
        if intent in ("program_fee", "program_duration", "program_seats"):
            fn = {"program_fee": A.program_fee, "program_duration": A.program_duration,
                  "program_seats": A.program_seats}[intent]
            if pid:
                if not pid.startswith("group"):
                    slot = {"program_fee": "fee_per_year", "program_duration": "duration",
                            "program_seats": "seats"}[intent]
                    val, src = self.kb.get_with_source(pid, slot)
                    trace.append(f"Frame lookup: {pid}.{slot} = {val} (from frame '{src}')")
                return fn(pid), f"{intent}:{pid}"
            return fn("all"), f"{intent}:all"
        if intent == "eligibility":
            if pid:
                return A.eligibility_info(pid), f"eligibility:{pid}"
            return A.eligibility_info("group:btech") + "\n" + A.eligibility_info("mtech_ai") + "\n" + \
                A.eligibility_info("mba"), "eligibility:all"
        if intent == "admission_process":
            return A.admission_process(), "admission_process"
        if intent == "admission_deadline":
            lvl = self._level(ents)
            return A.admission_deadline(lvl), f"admission_deadline:{lvl or 'all'}"
        if intent == "documents_required":
            lvl = self._level(ents) or "UG"
            return A.documents_required(lvl), f"documents_required:{lvl}"
        if intent == "hostel":
            return A.hostel(), "hostel"
        if intent == "facility":
            if ents.facility == "hostel":
                return A.hostel(), "hostel"
            if ents.facility:
                return A.facility(ents.facility), f"facility:{ents.facility}"
            return A.facilities_overview(), "facility:all"
        if intent == "scholarship":
            return A.scholarship_info(), "scholarship"
        if intent == "placement":
            dept = ents.department or (self.kb.get(pid, "department") if pid and not pid.startswith("group") else None)
            return A.placement(dept), f"placement:{dept or 'all'}"
        if intent == "department_info":
            dept = ents.department
            if pid and not pid.startswith("group") and (not dept or "offer" in p.lemmas):
                dept = self.kb.get(pid, "department")
                trace.append(f"Semantic network: {pid} --department--> {dept}")
                if "offer" in p.lemmas or not ents.department:
                    return A.department_of_program(pid), f"department_info:{dept}"
            if dept:
                return A.department_info(dept), f"department_info:{dept}"
            return "\n".join(A.department_info(d) for d in self.kb.instances_of("Department")), "department_info:all"
        if intent == "contact":
            return A.contact(), "contact"
        if intent == "about_university":
            return A.about(), "about_university"
        return FALLBACK_TEXT, "fallback"

    # =========================================================== reasoning
    def _resolve_program(self, pid):
        if pid == "group:btech":
            return "btech_cse", "B.Tech (all branches share the same eligibility)"
        if pid == "group:mtech":
            return "mtech_ai", None
        return pid, None

    def _eligibility(self, pid, applicant, session, trace, conf) -> BotResponse:
        if pid in (None, "group:pg"):
            session.pending = {"task": "eligibility", "program": None, "facts": applicant}
            return BotResponse("Which programme would you like me to check your eligibility for? "
                               "(e.g. B.Tech CSE, M.Tech AI, MBA, MCA, Ph.D)",
                               "eligibility", conf, "eligibility:ask_program", "dialogue", {}, trace)
        pid, note = self._resolve_program(pid)
        level = self.kb.program_level(pid)
        facts = {k: v for k, v in applicant.items() if v not in (None, False)}
        if level == "UG" and "qualification" not in facts and ("percentage" in facts or "cgpa" in facts):
            facts["qualification"] = "12th"
            trace.append("Assumption: for B.Tech the marks refer to 12th standard")
        missing = []
        if "qualification" not in facts:
            missing.append("your qualification (e.g. 'B.Tech ECE', 'B.Sc', '12th')")
        if "percentage" not in facts and "cgpa" not in facts:
            missing.append("your percentage or CGPA")
        ents = {"program": pid, **facts}
        if missing:
            session.pending = {"task": "eligibility", "program": pid, "facts": facts}
            trace.append(f"Slot filling: missing {missing}")
            return BotResponse(f"To check your eligibility for {self.answers.name(pid)}, please tell me "
                               + " and ".join(missing) + ".", "eligibility", conf, f"eligibility:{pid}:ask",
                               "dialogue", ents, trace)
        session.pending = None
        result = self.reasoner.check_eligibility(pid, facts)
        trace.append("Forward chaining fired: " + ", ".join(r.rule_id for r in result.fired))
        trace.extend(result.explanation_lines())
        wm = result.facts
        name = note or self.answers.name(pid)
        if wm.get("academic_eligible") is True:
            text = f"Yes, you are eligible for {name}. "
            lines = [r.explanation for r in result.fired if r.rule_id in ("R1", "R3", "R4", "R6", "R9")]
            text += " ".join(lines)
            if wm.get("entrance_ok") is True:
                text += f" Your {wm['exam'].upper()} score is accepted."
            elif wm.get("entrance_ok") == "pending":
                text += f" You also need a valid entrance score: {wm['entrance_list']}."
            if wm.get("stipend"):
                text += f" You will also get a GATE stipend of {self.kb.get(pid, 'gate_stipend')}."
            key = f"eligibility:{pid}:yes"
        else:
            reasons = "; ".join(wm.get("reasons", [])) or "the criteria are not met"
            text = (f"Sorry, you are not eligible for {name} because {reasons}. "
                    + self.answers.eligibility_info(pid))
            key = f"eligibility:{pid}:no"
        return BotResponse(text, "eligibility", conf, key, "rule_engine", ents, trace)

    def _fee_calc(self, pid, applicant, session, trace, conf) -> BotResponse:
        facts = {k: v for k, v in applicant.items() if v not in (None, False)}
        if pid is None or pid.startswith("group"):
            session.pending = {"task": "fee_calc", "program": None, "facts": facts}
            return BotResponse("Which programme are you interested in? I will calculate your fee after the "
                               "scholarship.", "scholarship", conf, "scholarship:ask_program", "dialogue", facts, trace)
        session.pending = None
        result = self.reasoner.compute_fee(pid, facts)
        trace.append("Forward chaining fired: " + ", ".join(r.rule_id for r in result.fired))
        trace.extend(result.explanation_lines())
        wm = result.facts
        steps = " ".join(r.explanation for r in result.fired if r.rule_id != "S0")
        text = (f"Fee estimate for {self.answers.name(pid)}: tuition {rs(wm['fee_per_year'])} per year. {steps} "
                f"Your estimated tuition is {rs(wm['net_fee'])} per year (scholarship {wm['waiver']}%).")
        return BotResponse(text, "scholarship", conf, f"scholarship:{pid}:calc", "rule_engine",
                           {"program": pid, **facts}, trace)
