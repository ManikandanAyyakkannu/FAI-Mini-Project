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
export const FALLBACK_TEXT = "Sorry, I could not find an answer to that. I can help with programmes, fees, eligibility, " +
  "admissions, deadlines, documents, hostel, scholarships, placements and campus facilities. " +
  "For anything else please contact the admissions office at +91 44 2345 6789.";

// where an answer came from (shown in the "How I answered" panel)
export const SOURCES = {
  knowledge_base: "Frame knowledge base",
  rule_engine: "Rule-based reasoning",
  vector_store: "Vector-store search",
  dialogue: "Dialogue manager",
  smalltalk: "Small talk",
  fallback: "Out of scope",
};

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
    const resp = this.#respond(message, session);
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

    // ---- 3. dialogue context ("how long is it?", "what about MCA?")
    if (this.useContext) {
      const last = session.context.lastIntent;
      if (ents.program && PROGRAM_INTENTS.has(last) && (intent === "out_of_scope" || conf < 0.42) && p.lemmas.length <= 4) {
        trace.push(`Dialogue: short follow-up -> reusing the previous intent '${last}'`);
        intent = last;
      }
      if (PROGRAM_INTENTS.has(intent) && !ents.program && !ents.department && session.context.program) {
        ents.program = session.context.program;
        trace.push(`Dialogue: programme taken from the conversation -> ${ents.program}`);
      }
    }

    // ---- 4a. domain gate: no university word and no entity -> out of scope
    const known = p.lemmas.some(l => this.domainVocab.has(l));
    if (!known && !(ents.program || ents.department || ents.facility || hasFacts)) {
      trace.push("Domain gate: no university-related word found -> out of scope");
      return this.#reply(FALLBACK_TEXT, "out_of_scope", conf, "fallback", "fallback", entitiesAsDict(ents), trace);
    }

    // ---- 4b. vector-store search for unknown / low-confidence questions, or when a
    //          free-text FAQ (a topic with no frame in the KB) matches strongly
    if (this.useVectorFallback && !SMALLTALK.has(intent)) {
      const hits = this.store.search(p.normalised, 3);
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
        if (!pid) return [A[fn]("all"), `${intent}:all`];
        if (single) {
          const slot = { program_fee: "fee_per_year", program_duration: "duration", program_seats: "seats" }[intent];
          const [val, src] = kb.getWithSource(pid, slot);
          trace.push(`Frame lookup: ${pid}.${slot} = ${val} (from frame '${src}')`);
        }
        return [A[fn](pid), `${intent}:${pid}`];
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
        return [A.placement(dept), `placement:${dept || "all"}`];
      }
      case "department_info": {
        let dept = ents.department;
        const asksOffer = p.lemmas.includes("offer");
        if (single && (!dept || asksOffer)) {
          dept = kb.get(pid, "department");
          trace.push(`Semantic network: ${pid} --department--> ${dept}`);
          if (asksOffer || !ents.department) return [A.departmentOfProgram(pid), `department_info:${dept}`];
        }
        if (dept) return [A.departmentInfo(dept), `department_info:${dept}`];
        return [kb.instancesOf("Department").map(d => A.departmentInfo(d)).join("\n"), "department_info:all"];
      }
      case "contact":
        return [A.contact(), "contact"];
      case "about_university":
        return [A.about(), "about_university"];
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
    const r = await fetch(base + f);
    if (!r.ok) throw new Error(`Could not load ${base + f} (${r.status})`);
    return r.json();
  };
  const [knowledgeBase, rules, intents, faqExtra] = await Promise.all(
    ["knowledge_base.json", "rules.json", "intents.json", "faq_extra.json"].map(get));
  return new UniversityChatbot({ knowledgeBase, rules, intents, faqExtra });
}
