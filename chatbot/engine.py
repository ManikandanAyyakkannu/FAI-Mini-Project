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
# "hi, do you have seats in medical?" - a greeting in front of a real question
GREETING_RE = re.compile(r"^\s*(?:hi+|hai|hey+|hello+|helo|good (?:morning|afternoon|evening)|namaste|vanakkam|greetings)\b"
                         r"[\s,.!:;-]*(?:(?:sir|madam|mam|there|team|bot|all)\b[\s,.!:;-]*)?", re.I)
# the user says the last answer was not useful
COMPLAINT_RE = re.compile(r"\b(?:not (?:relevant|relavant|correct|right|helpful|useful|related|clear)|irrelevant|irrelavant|"
                          r"wrong (?:answer|reply|response|information|info)|(?:is|are|was|were) wrong|useless|"
                          r"does not make sense|doesn t make sense|did not (?:answer|understand)|didn t (?:answer|understand)|"
                          r"not what i asked|you (?:are|r) wrong)\b")
# words around "department" that are not a department's name
NOT_A_DEPT = {"the", "of", "which", "what", "your", "this", "that", "each", "all", "every", "hod", "head", "who", "is",
              "a", "an", "any", "in", "for", "about", "tell", "me", "details", "info", "information", "name", "list",
              "ai", "university", "college", "our", "my", "whose", "and", "or", "to", "there", "do", "you",
              "have", "has", "are", "where", "how", "many", "contact", "email", "phone", "number", "s"}
# intents whose topic is decided by a named office or certificate
PASS_RE = re.compile(r"\b(?:pass mark|pass marks|passing mark|passing marks|to pass (?:a|the|each|my|in)?\s*(?:subject|course|paper|exam))")
PAYMENT_RE = re.compile(r"\b(?:late|due|draft|upi|net banking|online|instal+ments?|receipt|reciept|portal|card|two parts)\b")
VC_RE = re.compile(r"\b(?:vice chancellor|vc)\b")
# who to contact for each topic (the office frame that handles it)
TOPIC_OFFICE = {"hostel": "office_hostel", "placement": "office_placement", "examination": "office_coe",
                "results": "office_coe", "fee_payment": "office_accounts", "certificates": "office_registrar",
                "events": "office_dsw", "admission_process": "office_admissions"}
CONTACT_WORDS = {"contact", "whom", "office", "phone", "email", "handle", "responsible", "incharge"}
ENTITY_CONFIDENCE = 0.6      # a named office / certificate is strong evidence for its topic
OFFICE_OR_CERT_INTENTS = {"office_info", "certificates", "contact", "out_of_scope", "about_university"}
STUDENT_COUNT_RE = re.compile(r"\b(?:how many|number of|no of|total|strength|count|enrolment|enrollment)\b")
SEAT_STATUS_RE = re.compile(r"\b(?:available|availability|open|vacant|vacancy|vacancies|left|remaining|still|empty|free)\b")
AVAILABILITY_RE = re.compile(r"\b(?:do you (?:have|offer)|do we have|is there|are there|offered|available|"
                             r"tell me about)\b")
# words that can stand before "university" without naming another institution ("the university", "this university")
NOT_A_NAME = {"the", "this", "that", "your", "our", "my", "their", "its", "a", "an", "ai", "which", "what",
              "whose", "any", "same", "other", "another", "good", "best", "top", "great", "famous", "deemed",
              "private", "government", "state", "central", "open", "nearby", "near", "whole", "entire", "new", "old",
              "of", "for", "about", "in", "at", "to", "from", "and", "or", "is", "s", "each", "every", "one", "no"}
FALLBACK_TEXT = ("Sorry, I could not find an answer to that. I can help with admissions, programmes, fees, exams and "
                 "results, the academic calendar, certificates, university offices, hostel, scholarships, placements, "
                 "events and campus facilities. For anything else please call the university at +91 44 2345 6789.")


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
        # names the closed-world checks look for must not be "corrected" into other words
        for item in self.kb.not_offered:
            for a in item["aliases"]:
                self.pre.add_vocabulary(a.split())
        for name in self.kb.other_institutions:
            self.pre.add_vocabulary(name.split())
        for aliases in list(self.kb.office_aliases.values()) + list(self.kb.certificate_aliases.values()):
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
        # words of greetings / thanks / goodbye, so "list all departments" is not mistaken for "that's all"
        talk = [t for n in SMALLTALK for t in self.classifier.intents[n]["examples"] + self.classifier.intents[n]["keywords"]]
        self.smalltalk_vocab = {lem for t in talk for lem in self.pre.process(t).lemmas}
        # closed-world knowledge, longest names first so "anna university" wins over shorter matches
        self.other_names = sorted(self.kb.other_institutions.items(), key=lambda x: -len(x[0]))
        self.not_offered = [(a, item["name"], item.get("unless", []))
                            for item in self.kb.not_offered for a in item["aliases"]]
        self.not_offered.sort(key=lambda x: -len(x[0]))
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
        # "hi, what is the fee for MBA?": answer the question and just say hello back
        greet = GREETING_RE.match(message)
        rest = message[greet.end():].strip() if greet else ""
        if rest and set(self.pre.process(rest).lemmas) & self.domain_vocab:
            resp = self._respond(rest, session)
            resp.text = "Hello! " + resp.text
            resp.trace.insert(0, f"Greeting '{greet.group(0).strip()}' noted; answering the rest: '{rest}'")
        else:
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
        intent, conf, scores = self.classifier.predict(p)
        trace = [f"Normalised text: '{p.normalised}'", f"Lemmas: {p.lemmas}"]
        # small talk only wins when the message has no university words in it
        if intent in SMALLTALK and (set(p.lemmas) & self.domain_vocab) - self.smalltalk_vocab:
            best = max((n for n in scores if n not in SMALLTALK), key=scores.get)
            intent, conf = (best if scores[best] >= self.classifier.threshold else "out_of_scope"), scores[best]
            trace.append("Small talk skipped: the message also contains university words")
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
        # ---- 0. the user says the last answer did not help
        if COMPLAINT_RE.search(TextPreprocessor.normalise(message)) and not (
                ents.program or ents.department or ents.facility or ents.has_applicant_facts):
            trace.append("Feedback: the user says the previous answer was not useful")
            return BotResponse(self._complaint_text(), "out_of_scope", conf, "feedback", "smalltalk", {}, trace)

        # a question about another university: we only know AI University, so do not guess
        if intent not in SMALLTALK and (other := self._other_institution(message)):
            trace.append(f"Closed world: '{other}' is another institution, not in the knowledge base")
            return BotResponse(self._other_institution_text(other), "out_of_scope", conf, "other_institution",
                               "fallback", ents.as_dict(), trace)

        # a bare degree ("BCA") right after the bot asked for the student's qualification is their degree
        if (session.pending and session.pending["program"] and ents.ambiguous_qualification
                and ents.program and not ents.qualification):
            ents.qualification, ents.program = ents.ambiguous_qualification, None
            applicant["qualification"] = ents.qualification
            trace.append(f"Dialogue: '{ents.qualification}' answers the question about your qualification")
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

        # ---- 1b. a department / programme that is not in the knowledge base (closed-world assumption)
        if intent not in SMALLTALK and (missing := self._not_offered(p.normalised, ents)):
            alias, name = missing
            trace.append(f"Closed world: '{alias}' is not a department or programme in the knowledge base")
            return BotResponse(self._not_offered_text(name), intent if intent != "out_of_scope" else "department_info",
                               conf, "not_offered:" + name.lower().replace(" ", "_"), "knowledge_base",
                               ents.as_dict(), trace)

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

        # ---- 2b. "how many students study here?": the University frame's 'students' slot
        if (STUDENT_COUNT_RE.search(p.normalised) and re.search(r"\bstudents?\b", p.normalised)
                and not (ents.program or ents.department or ents.facility or has_facts)):
            trace.append("Frame lookup: University.students")
            return BotResponse(self.answers.student_strength(), "about_university", conf, "about_university:students",
                               "knowledge_base", ents.as_dict(), trace)

        # ---- 2c. "number of departments", "list the departments": the list of departments
        if re.search(r"\bdepartments\b", p.normalised) and not (ents.program or ents.department or has_facts):
            trace.append("Question about the departments in general -> list of departments")
            text, key = self._kb_answer("department_info", ents, p, trace)
            return BotResponse(text, "department_info", conf, key, "knowledge_base", ents.as_dict(), trace)

        # ---- 3. dialogue context (coreference: "its fee", "what about MCA?")
        if self.use_context:
            last = session.context.get("last_intent")
            # "what about 2023?" right after a placement question
            if ents.year and last == "placement" and (intent == "out_of_scope" or conf < 0.42) and len(p.lemmas) <= 4:
                trace.append("Dialogue: follow-up about another year -> reusing the previous intent 'placement'")
                intent, conf = last, max(conf, ENTITY_CONFIDENCE)
            # a short follow-up ("and MBA?") - but "do you have BBA?" is a new question, not a follow-up
            if ents.program and last in PROGRAM_INTENTS and (intent == "out_of_scope" or conf < 0.42) \
                    and len(p.lemmas) <= 4 and not AVAILABILITY_RE.search(p.normalised):
                trace.append(f"Dialogue: elliptical follow-up -> reusing previous intent '{last}'")
                intent = last
            if intent in PROGRAM_INTENTS and not ents.program and not ents.department \
                    and session.context.get("program"):
                ents.program = session.context["program"]
                trace.append(f"Dialogue: programme taken from context -> {ents.program}")

        # "where is the DS department?" / "email of the CSE department": that department's own details
        if intent == "contact" and ents.department:
            trace.append(f"Contact question about a department -> details of {ents.department}")
            intent = "department_info"

        # ---- 3c. topics of current students that the classifier mixes up with admission topics
        words = set(p.lemmas)
        # "when does the semester start?" is about the academic calendar, not the admission deadline;
        # "when do the end sem exams start?" is about the exams
        if intent in ("admission_deadline", "academic_calendar") and {"semester", "class", "college", "reopen"} & words \
                and not {"apply", "admission", "register", "deadline", "entrance"} & words:
            intent = "examination" if "exam" in words else "academic_calendar"
            trace.append(f"Question about the semester, not about applying -> {intent}")
        # "minimum marks to pass a subject" is an exam rule, not admission eligibility
        if PASS_RE.search(p.normalised) and not (ents.program or has_facts):
            trace.append("Pass marks of a course -> examination")
            intent = "examination"
        # "can I pay the fees by demand draft / late?" is about paying, not about how much a programme costs
        if intent in ("program_fee", "out_of_scope") and not ents.program and PAYMENT_RE.search(p.normalised) \
                and ("fee" in words or "pay" in p.tokens):
            trace.append("Question about paying fees -> fee_payment")
            intent = "fee_payment"
        # "who is the VC?"
        if VC_RE.search(p.normalised) and not ents.office:
            trace.append("Frame lookup: University.vice_chancellor")
            return BotResponse(self.answers.vice_chancellor(), "office_info", conf, "office_info:vice_chancellor",
                               "knowledge_base", ents.as_dict(), trace)
        # "whom should I contact for a hostel room change?" / "which office handles it?":
        # the office responsible for the topic (the topic of this question, or of the previous one)
        if not ents.office and CONTACT_WORDS & words:
            topic = intent if intent in TOPIC_OFFICE else (
                session.context.get("last_intent") if intent in ("out_of_scope", "office_info", "contact") else None)
            if topic in TOPIC_OFFICE:
                ents.office = TOPIC_OFFICE[topic]
                trace.append(f"Semantic network: {topic} --handled by--> {ents.office}")
                intent, conf = "office_info", max(conf, ENTITY_CONFIDENCE)

        # a named certificate or office decides the topic ("TC", "exam cell"), whatever the classifier guessed
        if ents.certificate and intent in OFFICE_OR_CERT_INTENTS | {"documents_required", "admission_process", "office_info"}:
            if intent != "certificates":
                trace.append(f"Certificate mentioned ({ents.certificate}) -> certificates")
            intent, conf = "certificates", max(conf, ENTITY_CONFIDENCE)
        elif ents.office and intent in OFFICE_OR_CERT_INTENTS | {"hostel", "placement", "facility", "department_info"} \
                and not (ents.office == "office_admissions" and intent == "contact"):
            if intent != "office_info":
                trace.append(f"University office mentioned ({ents.office}) -> office_info")
            intent, conf = "office_info", max(conf, ENTITY_CONFIDENCE)

        # ---- 3b. "do you have BBA?" / "tell me about BBA": a short summary of that programme
        pid = ents.program
        if (pid in self.kb.programs and not has_facts and intent in ("out_of_scope", "program_list", "about_university")
                and (ask := AVAILABILITY_RE.search(p.normalised))):
            trace.append(f"Knowledge base lookup: summary of programme '{pid}'")
            return BotResponse(self.answers.program_info(pid, confirm=ask.group(0) != "tell me about"), "program_list",
                               conf, f"program_info:{pid}", "knowledge_base", ents.as_dict(), trace)

        # ---- 4a. domain-relevance gate: no known domain word and no entity -> out of scope
        known = set(p.lemmas) & self.domain_vocab
        pure_talk = intent in SMALLTALK and set(p.lemmas) <= self.smalltalk_vocab     # "that's all", "thanks a lot"
        if not known and not pure_talk and not (ents.program or ents.department or ents.facility or ents.office
                                                or ents.certificate or has_facts or (ents.year and intent == "placement")):
            trace.append("Domain gate: no university-related word found -> out of scope")
            return BotResponse(FALLBACK_TEXT, "out_of_scope", conf, "fallback", "fallback", ents.as_dict(), trace)

        # ---- 4b. vector store retrieval for unknown / low-confidence questions, or when a
        #          free-text FAQ passage (a topic with no frame in the KB) matches strongly
        if self.use_vector_fallback and intent not in SMALLTALK:
            hits = self.store.search(p.normalised, k=3)
            if ents.program in self.kb.programs:          # "students in MBA" must not become "seats in MBA Finance"
                hits = [h for h in hits if h.metadata["key"].partition(":")[2] in ("", ents.program)
                        or h.metadata["key"].partition(":")[2] not in self.kb.programs]
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

    # ======================================================= closed world
    def _other_institution(self, message: str) -> str | None:
        """Name of another university/college mentioned in the message, if any."""
        text = TextPreprocessor.normalise(message)
        for alias, name in self.other_names:
            if re.search(rf"\b{re.escape(alias)}\b", text):
                return name
        m = re.search(r"\b([a-z]+) university\b", text)
        if m and m.group(1) not in NOT_A_NAME:
            return m.group(1).title() + " University"
        return None

    def _other_institution_text(self, name: str) -> str:
        return (f"I can only answer questions about AI University, so I don't have information about {name}. "
                f"Please check {name}'s official website or contact their admissions office.")

    def _not_offered(self, text: str, ents) -> tuple[str, str] | None:
        """A department or programme the university does not have, e.g. 'civil' or 'law'."""
        for alias, name, unless in self.not_offered:
            if not re.search(rf"\b{re.escape(alias)}\b", text):
                continue
            if any(re.search(rf"\b{re.escape(u)}\b", text) for u in unless):
                continue                         # "medical facility" is about the campus, not a medical course
            # read the question again without that word: if it still names something we do offer
            # ("I did B.E Civil, can I join MBA?"), or the word was part of the student's own
            # degree, this is not a question about the missing department
            rest = self.extractor.extract(re.sub(rf"\b{re.escape(alias)}\b", " ", text))
            if (rest.program and not rest.program.startswith("group")) or rest.department:
                continue
            if ents.qualification and rest.qualification != ents.qualification:
                continue
            return alias, name
        return None

    @staticmethod
    def _unknown_department(text: str) -> str | None:
        """The name in "<name> department" or "HOD of <name>", when the KB found no such department."""
        m = (re.search(r"((?:[a-z]+ ){1,4})(?:department|dept)\b", text + " ")
             or re.search(r"\b(?:hod|head) of ((?:[a-z]+ ?){1,4})$", text))
        if not m:
            return None
        words = m.group(1).split()
        name = []
        for w in reversed(words):                 # keep the words right before "department"
            if w in NOT_A_DEPT:
                break
            name.insert(0, w)
        if not name:
            return None
        name = " ".join(name)
        return name.upper() if len(name) <= 5 else name.title()

    def _complaint_text(self) -> str:
        u = self.kb.frames["University"]
        return ("Sorry, that was not the answer you were looking for. I only know about AI University, so please "
                "try asking in a different way, for example \"What is the fee for B.Tech CSE?\" or \"Who is the HOD of "
                f"the ECE department?\". You can also contact the admissions office at {u['phone']} or {u['email']}.")

    def _not_offered_text(self, name: str) -> str:
        u = self.kb.frames["University"]
        return (f"Sorry, the {name} department is not available at {u['name']}. For more details, please contact "
                f"the admissions office at {u['phone']} or {u['email']}.")

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
            target = pid or "all"
            text = fn(target)
            # "are seats still open?": the KB knows the intake, not today's vacancies, so say so
            if intent == "program_seats" and SEAT_STATUS_RE.search(p.normalised):
                text += "\n" + A.seat_status(target)
                trace.append("Seat availability asked: live vacancies are not in the knowledge base")
            return text, f"{intent}:{target}"
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
            year = ents.year
            if not year and re.search(r"\b(?:last|previous) year\b", p.normalised):
                year = A.placement_years()[-2]
            if year:
                trace.append(f"Placement season asked: {year} (seasons in the KB: {', '.join(A.placement_years())})")
            return A.placement(dept, year), f"placement:{dept or 'all'}" + (f":{year}" if year else "")
        if intent == "department_info":
            dept = ents.department
            if pid and not pid.startswith("group") and (not dept or "offer" in p.lemmas):
                dept = self.kb.get(pid, "department")
                trace.append(f"Semantic network: {pid} --department--> {dept}")
                # "which department offers M.Tech DS?" names the department; "HOD of DS dept" wants its details
                if "offer" in p.lemmas or not (ents.department or {"hod", "department"} & set(p.lemmas)):
                    return A.department_of_program(pid), f"department_info:{dept}"
            if dept:
                return A.department_info(dept), f"department_info:{dept}"
            # "HOD of the XYZ department": a department we do not know is not offered (closed world)
            if unknown := self._unknown_department(p.normalised):
                trace.append(f"Closed world: no department called '{unknown}' in the knowledge base")
                return self._not_offered_text(unknown), "not_offered:" + unknown.lower().replace(" ", "_")
            return "\n".join(A.department_info(d) for d in self.kb.instances_of("Department")), "department_info:all"
        if intent == "contact":
            return A.contact(), "contact"
        if intent == "about_university":
            return A.about(), "about_university"

        # ---- university life beyond admissions
        words = set(p.lemmas) | set(p.tokens)
        if intent == "academic_calendar":
            return A.academic_calendar(words), "academic_calendar"
        if intent == "examination":
            return A.examination(words), "examination"
        if intent == "results":
            return A.results(words), "results"
        if intent == "fee_payment":
            return A.fee_payment(words), "fee_payment"
        if intent == "certificates":
            trace.append(f"Frame lookup: {ents.certificate or 'all certificates'} (office from the Certificate class)")
            return A.certificate(ents.certificate), f"certificates:{ents.certificate or 'all'}"
        if intent == "office_info":
            if not ents.office and {"vice", "chancellor", "vc"} & words:
                return A.vice_chancellor(), "office_info:vice_chancellor"
            return A.office(ents.office), f"office_info:{ents.office or 'all'}"
        if intent == "faculty_research":
            return A.faculty_research(words), "faculty_research"
        if intent == "events":
            return A.events(words), "events"
        if intent == "alumni":
            return A.alumni(), "alumni"
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
