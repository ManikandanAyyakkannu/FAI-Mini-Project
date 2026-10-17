// The hybrid Knowledge Representation + NLP chatbot engine.
//
// Every message goes through these steps:
//
//   text -> NLP pre-processing -> entity extraction -> intent classification
//        -> dialogue context (follow-up questions, missing details)
//        -> rule-based reasoning (forward chaining) or frame knowledge-base lookup
//        -> vector-store search for questions the knowledge base does not cover
//        -> answer text + explanation trace
//
// The page (app.js) only calls  bot.respond(message, session).

import { KnowledgeBase } from "./knowledge-base.js";
import { TextPreprocessor, EntityExtractor, entitiesAsDict, hasApplicantFacts } from "./nlp.js";
import { IntentClassifier } from "./intent-classifier.js";
import { ReasoningEngine, keepGiven } from "./reasoning.js";
import { Answers, buildFaq, rs } from "./answers.js";
import { VectorStore } from "./similarity.js";

const PROGRAM_INTENTS = new Set(["program_fee", "program_duration", "program_seats", "eligibility",
  "placement", "department_info", "admission_deadline", "documents_required"]);
const SMALLTALK = new Set(["greeting", "goodbye", "thanks"]);
const PERSONAL_RE = /\b(am i|can i|could i|will i|would i|do i|i have|i've|i did|i got|i scored|i secured|i completed|i finished|i am|i'm|my|me|i)\b/;
const GENERIC_WORDS = new Set(["what", "how", "who", "which", "when", "why", "have", "good", "take", "much", "many",
  "s", "per", "under", "kind", "like", "best", "help", "give", "make", "show", "list",
  "anyone", "all", "there", "well", "one", "need", "provide", "available"]);
// "hi, do you have seats in medical?" - a greeting in front of a real question
const GREETING_RE = /^\s*(?:hi+|hai|hey+|hello+|helo|good (?:morning|afternoon|evening)|namaste|vanakkam|greetings)\b[\s,.!:;-]*(?:(?:sir|madam|mam|there|team|bot|all)\b[\s,.!:;-]*)?/i;
// the user says the last answer was not useful
const COMPLAINT_RE = new RegExp("\\b(?:not (?:relevant|relavant|correct|right|helpful|useful|related|clear)|irrelevant|irrelavant|" +
  "wrong (?:answer|reply|response|information|info)|(?:is|are|was|were) wrong|useless|" +
  "does not make sense|doesn t make sense|did not (?:answer|understand)|didn t (?:answer|understand)|" +
  "not what i asked|you (?:are|r) wrong)\\b");
// words around "department" that are not a department's name
const NOT_A_DEPT = new Set(["the", "of", "which", "what", "your", "this", "that", "each", "all", "every", "hod", "head", "who",
  "is", "a", "an", "any", "in", "for", "about", "tell", "me", "details", "info", "information", "name", "list",
  "ai", "university", "college", "our", "my", "whose", "and", "or", "to", "there", "do", "you",
  "have", "has", "are", "where", "how", "many", "contact", "email", "phone", "number", "s"]);
const PASS_RE = /\b(?:pass mark|pass marks|passing mark|passing marks|to pass (?:a|the|each|my|in)?\s*(?:subject|course|paper|exam))/;
const PAYMENT_RE = /\b(?:late|due|draft|upi|net banking|online|instal+ments?|receipt|reciept|portal|card|two parts)\b/;
const VC_RE = /\b(?:vice chancellor|vc)\b/;
// who to contact for each topic (the office frame that handles it)
const TOPIC_OFFICE = { hostel: "office_hostel", placement: "office_placement", examination: "office_coe",
  results: "office_coe", fee_payment: "office_accounts", certificates: "office_registrar",
  events: "office_dsw", admission_process: "office_admissions" };
const CONTACT_WORDS = new Set(["contact", "whom", "office", "phone", "email", "handle", "responsible", "incharge"]);
const ENTITY_CONFIDENCE = 0.6;      // a named office / certificate is strong evidence for its topic
// intents whose topic is decided by a named office or certificate
const OFFICE_OR_CERT_INTENTS = new Set(["office_info", "certificates", "contact", "out_of_scope", "about_university"]);
const STUDENT_COUNT_RE = /\b(?:how many|number of|no of|total|strength|count|enrolment|enrollment)\b/;
const SEAT_STATUS_RE = /\b(?:available|availability|open|vacant|vacancy|vacancies|left|remaining|still|empty|free)\b/;
const AVAILABILITY_RE = /\b(?:do you (?:have|offer)|do we have|is there|are there|offered|available|tell me about)\b/;
// words that can stand before "university" without naming another institution ("the university", "this university")
const NOT_A_NAME = new Set(["the", "this", "that", "your", "our", "my", "their", "its", "a", "an", "ai", "which",
  "what", "whose", "any", "same", "other", "another", "good", "best", "top", "great", "famous", "deemed",
  "private", "government", "state", "central", "open", "nearby", "near", "whole", "entire", "new", "old",
  "of", "for", "about", "in", "at", "to", "from", "and", "or", "is", "s", "each", "every", "one", "no"]);
const escapeRe = s => s.replace(/[.*+?^${}()|[\]\\-]/g, "\\$&");
const wordRe = (w, flags = "") => new RegExp(`\\b${escapeRe(w)}\\b`, flags);

export const FALLBACK_TEXT = "Sorry, I could not find an answer to that. I can help with admissions, programmes, fees, exams and " +
  "results, the academic calendar, certificates, university offices, hostel, scholarships, placements, " +
  "events and campus facilities. For anything else please call the university at +91 44 2345 6789.";


export function newSession() {
  return { context: {}, pending: null, history: [] };
}

const fmt = n => n.toFixed(2);

export class UniversityChatbot {
  // data = { knowledgeBase, rules, intents, faqExtra } (the JSON files in data/)
  constructor(data, options = {}) {
    this.kb = new KnowledgeBase(data.knowledgeBase);
    this.pre = new TextPreprocessor();
    for (const aliases of Object.values(this.kb.programAliases))
      for (const a of aliases) this.pre.addVocabulary(a.split(" "));
    // names the closed-world checks look for must not be "corrected" into other words
    for (const item of this.kb.notOffered)
      for (const a of item.aliases) this.pre.addVocabulary(a.split(" "));
    for (const name of Object.keys(this.kb.otherInstitutions)) this.pre.addVocabulary(name.split(" "));
    for (const aliases of [...Object.values(this.kb.officeAliases), ...Object.values(this.kb.certificateAliases)])
      for (const a of aliases) this.pre.addVocabulary(a.split(" "));
    if (options.spellCheck === false) this.pre.spellCheck = false;
    this.extractor = new EntityExtractor(this.kb);
    this.classifier = new IntentClassifier(this.pre, data.intents, options.intentThreshold ?? 0.30);
    this.reasoner = new ReasoningEngine(this.kb, data.rules);
    this.answers = new Answers(this.kb);
    this.faq = buildFaq(this.kb, this.answers, data.faqExtra.faqs);
    for (const f of this.faq) this.pre.addVocabulary(TextPreprocessor.normalise(f.question).split(" "));
    this.store = new VectorStore();
    this.store.add(this.faq.map(f => TextPreprocessor.normalise(f.question)), this.faq);

    // domain vocabulary = every content word the knowledge sources know about
    const texts = [
      ...Object.values(this.classifier.intents).flatMap(spec => spec.examples),
      ...this.faq.map(f => f.question),
      ...Object.values(this.kb.programAliases).flat(),
    ];
    this.domainVocab = new Set();
    for (const t of texts)
      for (const lem of this.pre.process(t).lemmas) if (!GENERIC_WORDS.has(lem)) this.domainVocab.add(lem);
    // words of greetings / thanks / goodbye, so "list all departments" is not mistaken for "that's all"
    this.smalltalkVocab = new Set();
    for (const n of SMALLTALK)
      for (const t of [...this.classifier.intents[n].examples, ...this.classifier.intents[n].keywords])
        for (const lem of this.pre.process(t).lemmas) this.smalltalkVocab.add(lem);

    // closed-world knowledge, longest names first so "anna university" wins over shorter matches
    this.otherNames = Object.entries(this.kb.otherInstitutions).sort((a, b) => b[0].length - a[0].length);
    this.notOffered = this.kb.notOffered.flatMap(item => item.aliases.map(a => [a, item.name, item.unless || []]))
      .sort((a, b) => b[0].length - a[0].length);

    this.useReasoning = options.useReasoning ?? true;
    this.useContext = options.useContext ?? true;
    this.useVectorFallback = options.useVectorFallback ?? true;
    this.vectorThreshold = options.vectorThreshold ?? 0.35;
  }

  get stats() {
    return {
      frames: Object.keys(this.kb.frames).length,
      programmes: this.kb.programs.length,
      rules: this.reasoner.eligibilityEngine.rules.length + this.reasoner.scholarshipEngine.rules.length,
      intents: Object.keys(this.classifier.intents).length,
      passages: this.store.size,
    };
  }

  // ================================================================ main
  respond(message, session = newSession()) {
    const t0 = performance.now();
    // "hi, what is the fee for MBA?": answer the question and just say hello back
    const greet = message.match(GREETING_RE);
    const rest = greet ? message.slice(greet[0].length).trim() : "";
    let resp;
    if (rest && this.pre.process(rest).lemmas.some(l => this.domainVocab.has(l))) {
      resp = this.#respond(rest, session);
      resp.text = "Hello! " + resp.text;
      resp.trace.unshift(`Greeting '${greet[0].trim()}' noted; answering the rest: '${rest}'`);
    } else {
      resp = this.#respond(message, session);
    }
    resp.latencyMs = performance.now() - t0;
    session.history.push([message, resp.text]);
    if (this.kb.programs.includes(resp.entities.program)) session.context.program = resp.entities.program;
    if (!SMALLTALK.has(resp.intent) && resp.intent !== "out_of_scope") session.context.lastIntent = resp.intent;
    return resp;
  }

  #reply(text, intent, confidence, answerKey, source, entities, trace) {
    return { text, intent, confidence, answerKey, source, entities, trace, ...this.lastAnalysis };
  }

  #respond(message, session) {
    const p = this.pre.process(message);
    const ents = this.extractor.extract(p.normalised);
    let { intent, confidence: conf, scores } = this.classifier.predict(p);
    const topIntents = Object.entries(scores).sort((a, b) => b[1] - a[1]).slice(0, 3);
    this.lastAnalysis = { normalised: p.normalised, lemmas: p.lemmas, corrections: p.corrections, topIntents };

    const trace = [`Normalised text: '${p.normalised}'`, `Lemmas: [${p.lemmas.join(", ")}]`];
    // small talk only wins when the message has no university words in it
    if (SMALLTALK.has(intent) && p.lemmas.some(l => this.domainVocab.has(l) && !this.smalltalkVocab.has(l))) {
      let best = null;
      for (const n in scores) if (!SMALLTALK.has(n) && (best === null || scores[n] > scores[best])) best = n;
      intent = scores[best] >= this.classifier.threshold ? best : "out_of_scope";
      conf = scores[best];
      trace.push("Small talk skipped: the message also contains university words");
    }
    if (Object.keys(p.corrections).length) {
      trace.push("Spelling corrected: " + Object.entries(p.corrections).map(([a, b]) => `${a} -> ${b}`).join(", "));
    }
    trace.push(`Intent classifier -> ${intent} (score ${fmt(conf)})`);
    if (Object.keys(entitiesAsDict(ents)).length) trace.push("Entities: " + JSON.stringify(entitiesAsDict(ents)));

    const lemmas = new Set(p.lemmas);
    const personal = PERSONAL_RE.test(p.normalised);
    const applicant = {
      qualification: ents.qualification, percentage: ents.percentage, cgpa: ents.cgpa, exam: ents.exam,
      category: ents.category, subjects: ents.subjects, sports: ents.sports, sibling: ents.sibling,
    };
    // ---- 0. the user says the last answer did not help
    if (COMPLAINT_RE.test(TextPreprocessor.normalise(message)) &&
        !(ents.program || ents.department || ents.facility || hasApplicantFacts(ents))) {
      trace.push("Feedback: the user says the previous answer was not useful");
      return this.#reply(this.#complaintText(), "out_of_scope", conf, "feedback", "smalltalk", {}, trace);
    }

    // a question about another university: we only know AI University, so do not guess
    const other = SMALLTALK.has(intent) ? null : this.#otherInstitution(message);
    if (other) {
      trace.push(`Closed world: '${other}' is another institution, not in the knowledge base`);
      return this.#reply(this.#otherInstitutionText(other), "out_of_scope", conf, "other_institution", "fallback",
        entitiesAsDict(ents), trace);
    }

    // a bare degree ("BCA") right after the bot asked for the student's qualification is their degree
    if (session.pending?.program && ents.ambiguousQualification && ents.program && !ents.qualification) {
      ents.qualification = ents.ambiguousQualification;
      ents.program = null;
      applicant.qualification = ents.qualification;
      trace.push(`Dialogue: '${ents.qualification}' answers the question about your qualification`);
    }
    const hasFacts = hasApplicantFacts(ents) || ents.sports || ents.sibling;
    const isProgram = pid => this.kb.programs.includes(pid);

    // ---- 1. finish a task that is waiting for details (multi-turn slot filling)
    const task = session.pending;
    const answersProgramQuestion = task && task.program === null && isProgram(ents.program);
    if (this.useContext && task && (hasFacts || answersProgramQuestion) && !SMALLTALK.has(intent)) {
      const merged = { ...keepGiven(task.facts), ...keepGiven(applicant) };
      const pid = ents.program && !ents.program.startsWith("group") ? ents.program : task.program;
      trace.push(`Dialogue: continuing the pending '${task.task}' task with the new details`);
      if (task.task === "eligibility") return this.#eligibility(pid, merged, session, trace, conf);
      return this.#feeCalc(pid, merged, session, trace, conf);
    }

    // ---- 1b. a department / programme that is not in the knowledge base (closed-world assumption)
    const missing = SMALLTALK.has(intent) ? null : this.#notOfferedIn(p.normalised, ents);
    if (missing) {
      const [alias, name] = missing;
      trace.push(`Closed world: '${alias}' is not a department or programme in the knowledge base`);
      return this.#reply(this.#notOfferedText(name), intent !== "out_of_scope" ? intent : "department_info", conf,
        "not_offered:" + name.toLowerCase().replaceAll(" ", "_"), "knowledge_base", entitiesAsDict(ents), trace);
    }

    // ---- 2. personal questions -> rule-based reasoning
    if (this.useReasoning) {
      const money = intent === "program_fee" || intent === "scholarship" || lemmas.has("fee") || lemmas.has("scholarship");
      const fromContext = this.useContext ? session.context.program ?? null : null;
      if (money && (ents.percentage || ents.cgpa || ents.sports || ents.sibling)) {
        trace.push("Reasoning: personal fee/scholarship question -> scholarship rules");
        return this.#feeCalc(ents.program || fromContext, applicant, session, trace, conf);
      }
      if (!money && (hasApplicantFacts(ents) || (personal && intent === "eligibility"))) {
        if (["eligibility", "out_of_scope", "admission_process", "program_list"].includes(intent) || ents.program) {
          trace.push("Reasoning: personal eligibility question -> eligibility rules");
          return this.#eligibility(ents.program || fromContext, applicant, session, trace, conf);
        }
      }
    }

    // ---- 2b. "how many students study here?": the University frame's 'students' slot
    if (STUDENT_COUNT_RE.test(p.normalised) && /\bstudents?\b/.test(p.normalised) &&
        !(ents.program || ents.department || ents.facility || hasFacts)) {
      trace.push("Frame lookup: University.students");
      return this.#reply(this.answers.studentStrength(), "about_university", conf, "about_university:students",
        "knowledge_base", entitiesAsDict(ents), trace);
    }

    // ---- 2c. "number of departments", "list the departments": the list of departments
    if (/\bdepartments\b/.test(p.normalised) && !(ents.program || ents.department || hasFacts)) {
      trace.push("Question about the departments in general -> list of departments");
      const [text, key] = this.#kbAnswer("department_info", ents, p, trace);
      return this.#reply(text, "department_info", conf, key, "knowledge_base", entitiesAsDict(ents), trace);
    }

    // ---- 3. dialogue context ("how long is it?", "what about MCA?")
    if (this.useContext) {
      const last = session.context.lastIntent;
      // "what about 2023?" right after a placement question
      if (ents.year && last === "placement" && (intent === "out_of_scope" || conf < 0.42) && p.lemmas.length <= 4) {
        trace.push("Dialogue: follow-up about another year -> reusing the previous intent 'placement'");
        intent = last;
        conf = Math.max(conf, ENTITY_CONFIDENCE);
      }
      // a short follow-up ("and MBA?") - but "do you have BBA?" is a new question, not a follow-up
      if (ents.program && PROGRAM_INTENTS.has(last) && (intent === "out_of_scope" || conf < 0.42) &&
          p.lemmas.length <= 4 && !AVAILABILITY_RE.test(p.normalised)) {
        trace.push(`Dialogue: short follow-up -> reusing the previous intent '${last}'`);
        intent = last;
      }
      if (PROGRAM_INTENTS.has(intent) && !ents.program && !ents.department && session.context.program) {
        ents.program = session.context.program;
        trace.push(`Dialogue: programme taken from the conversation -> ${ents.program}`);
      }
    }

    // "where is the DS department?" / "email of the CSE department": that department's own details
    if (intent === "contact" && ents.department) {
      trace.push(`Contact question about a department -> details of ${ents.department}`);
      intent = "department_info";
    }

    // ---- 3c. topics of current students that the classifier mixes up with admission topics
    const words = new Set(p.lemmas);
    // "when does the semester start?" is the academic calendar; "when do the end sem exams start?" the exams
    if (["admission_deadline", "academic_calendar"].includes(intent) &&
        ["semester", "class", "college", "reopen"].some(w => words.has(w)) &&
        !["apply", "admission", "register", "deadline", "entrance"].some(w => words.has(w))) {
      intent = words.has("exam") ? "examination" : "academic_calendar";
      trace.push(`Question about the semester, not about applying -> ${intent}`);
    }
    // "minimum marks to pass a subject" is an exam rule, not admission eligibility
    if (PASS_RE.test(p.normalised) && !(ents.program || hasFacts)) {
      trace.push("Pass marks of a course -> examination");
      intent = "examination";
    }
    // "can I pay the fees by demand draft / late?" is about paying, not about how much a programme costs
    if (["program_fee", "out_of_scope"].includes(intent) && !ents.program && PAYMENT_RE.test(p.normalised) &&
        (words.has("fee") || p.tokens.includes("pay"))) {
      trace.push("Question about paying fees -> fee_payment");
      intent = "fee_payment";
    }
    // "who is the VC?"
    if (VC_RE.test(p.normalised) && !ents.office) {
      trace.push("Frame lookup: University.vice_chancellor");
      return this.#reply(this.answers.viceChancellor(), "office_info", conf, "office_info:vice_chancellor",
        "knowledge_base", entitiesAsDict(ents), trace);
    }
    // "whom should I contact for a hostel room change?" / "which office handles it?":
    // the office responsible for the topic (the topic of this question, or of the previous one)
    if (!ents.office && [...CONTACT_WORDS].some(w => words.has(w))) {
      const topic = intent in TOPIC_OFFICE ? intent :
        (["out_of_scope", "office_info", "contact"].includes(intent) ? session.context.lastIntent : null);
      if (topic in TOPIC_OFFICE) {
        ents.office = TOPIC_OFFICE[topic];
        trace.push(`Semantic network: ${topic} --handled by--> ${ents.office}`);
        intent = "office_info";
        conf = Math.max(conf, ENTITY_CONFIDENCE);
      }
    }

    // a named certificate or office decides the topic ("TC", "exam cell"), whatever the classifier guessed
    if (ents.certificate && (OFFICE_OR_CERT_INTENTS.has(intent) ||
        ["documents_required", "admission_process", "office_info"].includes(intent))) {
      if (intent !== "certificates") trace.push(`Certificate mentioned (${ents.certificate}) -> certificates`);
      intent = "certificates";
      conf = Math.max(conf, ENTITY_CONFIDENCE);
    } else if (ents.office && (OFFICE_OR_CERT_INTENTS.has(intent) ||
        ["hostel", "placement", "facility", "department_info"].includes(intent)) &&
        !(ents.office === "office_admissions" && intent === "contact")) {
      if (intent !== "office_info") trace.push(`University office mentioned (${ents.office}) -> office_info`);
      intent = "office_info";
      conf = Math.max(conf, ENTITY_CONFIDENCE);
    }

    // ---- 3b. "do you have BBA?" / "tell me about BBA": a short summary of that programme
    const ask = AVAILABILITY_RE.exec(p.normalised);
    if (isProgram(ents.program) && !hasFacts && ["out_of_scope", "program_list", "about_university"].includes(intent) && ask) {
      trace.push(`Knowledge base lookup: summary of programme '${ents.program}'`);
      return this.#reply(this.answers.programInfo(ents.program, ask[0] !== "tell me about"), "program_list", conf,
        `program_info:${ents.program}`, "knowledge_base", entitiesAsDict(ents), trace);
    }

    // ---- 4a. domain gate: no university word and no entity -> out of scope
    const known = p.lemmas.some(l => this.domainVocab.has(l));
    const pureTalk = SMALLTALK.has(intent) && p.lemmas.every(l => this.smalltalkVocab.has(l));   // "that's all"
    if (!known && !pureTalk && !(ents.program || ents.department || ents.facility || ents.office ||
        ents.certificate || hasFacts || (ents.year && intent === "placement"))) {
      trace.push("Domain gate: no university-related word found -> out of scope");
      return this.#reply(FALLBACK_TEXT, "out_of_scope", conf, "fallback", "fallback", entitiesAsDict(ents), trace);
    }

    // ---- 4b. vector-store search for unknown / low-confidence questions, or when a
    //          free-text FAQ (a topic with no frame in the KB) matches strongly
    if (this.useVectorFallback && !SMALLTALK.has(intent)) {
      let hits = this.store.search(p.normalised, 3);
      if (isProgram(ents.program)) {          // "students in MBA" must not become "seats in MBA Finance"
        hits = hits.filter(h => {
          const i = h.metadata.key.indexOf(":");
          const about = i < 0 ? "" : h.metadata.key.slice(i + 1);
          return about === "" || about === ents.program || !isProgram(about);
        });
      }
      const top = hits[0];
      const freeTextWins = top && top.metadata.intent === "faq" && top.score >= 0.5 && top.score > conf;
      if (intent === "out_of_scope" || conf < 0.40 || freeTextWins) {
        trace.push("Vector store top-3: " + hits.map(h => `${h.metadata.key} (${fmt(h.score)})`).join("; "));
        if (top && top.score >= this.vectorThreshold && (intent === "out_of_scope" || top.score > conf)) {
          return this.#reply(top.metadata.answer, top.metadata.intent, top.score, top.metadata.key,
            "vector_store", entitiesAsDict(ents), trace);
        }
      }
    }
    if (intent === "out_of_scope") {
      return this.#reply(FALLBACK_TEXT, "out_of_scope", conf, "fallback", "fallback", entitiesAsDict(ents), trace);
    }

    // ---- 5. answer from the frame knowledge base
    if (SMALLTALK.has(intent)) {
      return this.#reply(this.classifier.smalltalk[intent], intent, conf, intent, "smalltalk", entitiesAsDict(ents), trace);
    }
    if (intent === "eligibility" && personal && this.useReasoning && ents.program) {
      return this.#eligibility(ents.program, applicant, session, trace, conf);
    }
    const [text, key] = this.#kbAnswer(intent, ents, p, trace);
    return this.#reply(text, intent, conf, key, "knowledge_base", entitiesAsDict(ents), trace);
  }

  // ======================================================== closed world
  // name of another university / college mentioned in the message, if any
  #otherInstitution(message) {
    const text = TextPreprocessor.normalise(message);
    for (const [alias, name] of this.otherNames) if (wordRe(alias).test(text)) return name;
    const m = text.match(/\b([a-z]+) university\b/);
    if (m && !NOT_A_NAME.has(m[1])) return m[1][0].toUpperCase() + m[1].slice(1) + " University";
    return null;
  }

  #otherInstitutionText(name) {
    return `I can only answer questions about AI University, so I don't have information about ${name}. ` +
      `Please check ${name}'s official website or contact their admissions office.`;
  }

  // a department or programme the university does not have, e.g. "civil" or "law"
  #notOfferedIn(text, ents) {
    for (const [alias, name, unless] of this.notOffered) {
      if (!wordRe(alias).test(text)) continue;
      if (unless.some(u => wordRe(u).test(text))) continue;   // "medical facility" is about the campus, not a course
      // read the question again without that word: if it still names something we do offer
      // ("I did B.E Civil, can I join MBA?"), or the word was part of the student's own
      // degree, this is not a question about the missing department
      const rest = this.extractor.extract(text.replace(wordRe(alias, "g"), " "));
      if ((rest.program && !rest.program.startsWith("group")) || rest.department) continue;
      if (ents.qualification && rest.qualification !== ents.qualification) continue;
      return [alias, name];
    }
    return null;
  }

  // the name in "<name> department" or "HOD of <name>", when the KB found no such department
  static unknownDepartment(text) {
    const m = (text + " ").match(/((?:[a-z]+ ){1,4})(?:department|dept)\b/) ||
      text.match(/\b(?:hod|head) of ((?:[a-z]+ ?){1,4})$/);
    if (!m) return null;
    const words = m[1].split(/\s+/).filter(Boolean);
    const name = [];
    for (let i = words.length - 1; i >= 0; i--) {       // keep the words right before "department"
      if (NOT_A_DEPT.has(words[i])) break;
      name.unshift(words[i]);
    }
    if (!name.length) return null;
    const joined = name.join(" ");
    return joined.length <= 5 ? joined.toUpperCase() : name.map(w => w[0].toUpperCase() + w.slice(1)).join(" ");
  }

  #complaintText() {
    const u = this.kb.frames.University;
    return "Sorry, that was not the answer you were looking for. I only know about AI University, so please " +
      "try asking in a different way, for example \"What is the fee for B.Tech CSE?\" or \"Who is the HOD of " +
      `the ECE department?". You can also contact the admissions office at ${u.phone} or ${u.email}.`;
  }

  #notOfferedText(name) {
    const u = this.kb.frames.University;
    return `Sorry, the ${name} department is not available at ${u.name}. For more details, please contact ` +
      `the admissions office at ${u.phone} or ${u.email}.`;
  }

  // ============================================================ KB lookup
  #level(ents) {
    const pid = ents.program;
    if (pid && !pid.startsWith("group")) return this.kb.programLevel(pid);
    if (pid === "group:btech") return "UG";
    if (pid === "group:mtech" || pid === "group:pg") return "PG";
    return ents.levelFilter;
  }

  #kbAnswer(intent, ents, p, trace) {
    const A = this.answers, kb = this.kb, pid = ents.program;
    const single = pid && !pid.startsWith("group");
    trace.push(`Knowledge base lookup for intent '${intent}'`);
    switch (intent) {
      case "program_list":
        return [A.programList(ents.levelFilter), `program_list:${ents.levelFilter || "all"}`];
      case "program_fee": case "program_duration": case "program_seats": {
        const fn = { program_fee: "programFee", program_duration: "programDuration", program_seats: "programSeats" }[intent];
        if (single) {
          const slot = { program_fee: "fee_per_year", program_duration: "duration", program_seats: "seats" }[intent];
          const [val, src] = kb.getWithSource(pid, slot);
          trace.push(`Frame lookup: ${pid}.${slot} = ${val} (from frame '${src}')`);
        }
        const target = pid || "all";
        let text = A[fn](target);
        // "are seats still open?": the KB knows the intake, not today's vacancies, so say so
        if (intent === "program_seats" && SEAT_STATUS_RE.test(p.normalised)) {
          text += "\n" + A.seatStatus(target);
          trace.push("Seat availability asked: live vacancies are not in the knowledge base");
        }
        return [text, `${intent}:${target}`];
      }
      case "eligibility":
        if (pid) return [A.eligibilityInfo(pid), `eligibility:${pid}`];
        return [[A.eligibilityInfo("group:btech"), A.eligibilityInfo("mtech_ai"), A.eligibilityInfo("mba")].join("\n"),
          "eligibility:all"];
      case "admission_process":
        return [A.admissionProcess(), "admission_process"];
      case "admission_deadline": {
        const lvl = this.#level(ents);
        return [A.admissionDeadline(lvl), `admission_deadline:${lvl || "all"}`];
      }
      case "documents_required": {
        const lvl = this.#level(ents) || "UG";
        return [A.documentsRequired(lvl), `documents_required:${lvl}`];
      }
      case "hostel":
        return [A.hostel(), "hostel"];
      case "facility":
        if (ents.facility === "hostel") return [A.hostel(), "hostel"];
        if (ents.facility) return [A.facility(ents.facility), `facility:${ents.facility}`];
        return [A.facilitiesOverview(), "facility:all"];
      case "scholarship":
        return [A.scholarshipInfo(), "scholarship"];
      case "placement": {
        const dept = ents.department || (single ? kb.get(pid, "department") : null);
        let year = ents.year;
        if (!year && /\b(?:last|previous) year\b/.test(p.normalised)) year = A.placementYears().at(-2);
        if (year) trace.push(`Placement season asked: ${year} (seasons in the KB: ${A.placementYears().join(", ")})`);
        return [A.placement(dept, year), `placement:${dept || "all"}` + (year ? `:${year}` : "")];
      }
      case "department_info": {
        let dept = ents.department;
        const asksOffer = p.lemmas.includes("offer");
        if (single && (!dept || asksOffer)) {
          dept = kb.get(pid, "department");
          trace.push(`Semantic network: ${pid} --department--> ${dept}`);
          // "which department offers M.Tech DS?" names the department; "HOD of DS dept" wants its details
          if (asksOffer || !(ents.department || p.lemmas.includes("hod") || p.lemmas.includes("department")))
            return [A.departmentOfProgram(pid), `department_info:${dept}`];
        }
        if (dept) return [A.departmentInfo(dept), `department_info:${dept}`];
        // "HOD of the XYZ department": a department we do not know is not offered (closed world)
        const unknown = UniversityChatbot.unknownDepartment(p.normalised);
        if (unknown) {
          trace.push(`Closed world: no department called '${unknown}' in the knowledge base`);
          return [this.#notOfferedText(unknown), "not_offered:" + unknown.toLowerCase().replaceAll(" ", "_")];
        }
        return [kb.instancesOf("Department").map(d => A.departmentInfo(d)).join("\n"), "department_info:all"];
      }
      case "contact":
        return [A.contact(), "contact"];
      case "about_university":
        return [A.about(), "about_university"];
    }

    // ---- university life beyond admissions
    const words = new Set([...p.lemmas, ...p.tokens]);
    switch (intent) {
      case "academic_calendar":
        return [A.academicCalendar(words), "academic_calendar"];
      case "examination":
        return [A.examination(words), "examination"];
      case "results":
        return [A.results(words), "results"];
      case "fee_payment":
        return [A.feePayment(words), "fee_payment"];
      case "certificates":
        trace.push(`Frame lookup: ${ents.certificate || "all certificates"} (office from the Certificate class)`);
        return [A.certificate(ents.certificate), `certificates:${ents.certificate || "all"}`];
      case "office_info":
        if (!ents.office && ["vice", "chancellor", "vc"].some(w => words.has(w)))
          return [A.viceChancellor(), "office_info:vice_chancellor"];
        return [A.office(ents.office), `office_info:${ents.office || "all"}`];
      case "faculty_research":
        return [A.facultyResearch(words), "faculty_research"];
      case "events":
        return [A.events(words), "events"];
      case "alumni":
        return [A.alumni(), "alumni"];
    }
    return [FALLBACK_TEXT, "fallback"];
  }

  // =========================================================== reasoning
  #eligibility(pid, applicant, session, trace, conf) {
    if (pid === null || pid === undefined || pid === "group:pg") {
      session.pending = { task: "eligibility", program: null, facts: applicant };
      return this.#reply("Which programme would you like me to check your eligibility for? " +
        "(e.g. B.Tech CSE, M.Tech AI, MBA, MCA, Ph.D)", "eligibility", conf, "eligibility:ask_program", "dialogue", {}, trace);
    }
    let note = null;
    if (pid === "group:btech") { pid = "btech_cse"; note = "B.Tech (all branches share the same eligibility)"; }
    else if (pid === "group:mtech") pid = "mtech_ai";

    const facts = keepGiven(applicant);
    if (this.kb.programLevel(pid) === "UG" && !("qualification" in facts) && ("percentage" in facts || "cgpa" in facts)) {
      facts.qualification = "12th";
      trace.push("Assumption: for B.Tech the marks refer to 12th standard");
    }
    const missing = [];
    if (!("qualification" in facts)) missing.push("your qualification (e.g. 'B.Tech ECE', 'B.Sc', '12th')");
    if (!("percentage" in facts) && !("cgpa" in facts)) missing.push("your percentage or CGPA");
    const ents = { program: pid, ...facts };
    if (missing.length) {
      session.pending = { task: "eligibility", program: pid, facts };
      trace.push(`Slot filling: still missing ${missing.join(" and ")}`);
      return this.#reply(`To check your eligibility for ${this.answers.name(pid)}, please tell me ` + missing.join(" and ") + ".",
        "eligibility", conf, `eligibility:${pid}:ask`, "dialogue", ents, trace);
    }

    session.pending = null;
    const result = this.reasoner.checkEligibility(pid, facts);
    trace.push("Forward chaining fired: " + result.fired.map(r => r.id).join(", "));
    trace.push(...result.lines);
    const wm = result.facts;
    const name = note || this.answers.name(pid);
    let text, key;
    if (wm.academic_eligible === true) {
      text = `Yes, you are eligible for ${name}. ` +
        result.fired.filter(r => ["R1", "R3", "R4", "R6", "R9"].includes(r.id)).map(r => r.explanation).join(" ");
      if (wm.entrance_ok === true) text += ` Your ${wm.exam.toUpperCase()} score is accepted.`;
      else if (wm.entrance_ok === "pending") text += ` You also need a valid entrance score: ${wm.entrance_list}.`;
      if (wm.stipend) text += ` You will also get a GATE stipend of ${this.kb.get(pid, "gate_stipend")}.`;
      key = `eligibility:${pid}:yes`;
    } else {
      const reasons = (wm.reasons || []).join("; ") || "the criteria are not met";
      text = `Sorry, you are not eligible for ${name} because ${reasons}. ` + this.answers.eligibilityInfo(pid);
      key = `eligibility:${pid}:no`;
    }
    return this.#reply(text, "eligibility", conf, key, "rule_engine", ents, trace);
  }

  #feeCalc(pid, applicant, session, trace, conf) {
    const facts = keepGiven(applicant);
    if (!pid || pid.startsWith("group")) {
      session.pending = { task: "fee_calc", program: null, facts };
      return this.#reply("Which programme are you interested in? I will calculate your fee after the scholarship.",
        "scholarship", conf, "scholarship:ask_program", "dialogue", facts, trace);
    }
    session.pending = null;
    const result = this.reasoner.computeFee(pid, facts);
    trace.push("Forward chaining fired: " + result.fired.map(r => r.id).join(", "));
    trace.push(...result.lines);
    const wm = result.facts;
    const steps = result.fired.filter(r => r.id !== "S0").map(r => r.explanation).join(" ");
    const text = `Fee estimate for ${this.answers.name(pid)}: tuition ${rs(wm.fee_per_year)} per year. ${steps} ` +
      `Your estimated tuition is ${rs(wm.net_fee)} per year (scholarship ${wm.waiver}%).`;
    return this.#reply(text, "scholarship", conf, `scholarship:${pid}:calc`, "rule_engine", { program: pid, ...facts }, trace);
  }
}

// loads the JSON files from data/ and builds the bot
export async function loadChatbot(base = "data/") {
  const get = async f => {
    const r = await fetch(base + f, { cache: "no-cache" });    // always pick up edited data files
    if (!r.ok) throw new Error(`Could not load ${base + f} (${r.status})`);
    return r.json();
  };
  const [knowledgeBase, rules, intents, faqExtra] = await Promise.all(
    ["knowledge_base.json", "rules.json", "intents.json", "faq_extra.json"].map(get));
  return new UniversityChatbot({ knowledgeBase, rules, intents, faqExtra });
}
