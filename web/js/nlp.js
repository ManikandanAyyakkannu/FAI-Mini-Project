// NLP pipeline: normalisation, tokenisation, spelling correction, synonym mapping,
// stop-word removal, light stemming and entity extraction.
// Everything is hand-written rules - no pretrained model is used.

import { closestWord } from "./similarity.js";

// ------------------------------------------------------------------------
// 1. Normalisation: unify the many ways students write degrees, marks etc.
// ------------------------------------------------------------------------
const NORMALISE = [
  [/\bb\.\s?tech\b|\bb\s?tech\b|\bb-tech\b/g, "btech"],
  [/\bm\.\s?tech\b|\bm\s?tech\b|\bm-tech\b/g, "mtech"],
  [/\bb\.\s?e\.?(?=\s|$)/g, "btech "],
  [/\bbe(?=\s+(?:in\s+)?(?:cse|ece|eee|it|mech|mechanical|civil|computer|electronics|electrical)\b)/g, "btech"],
  [/\bm\.\s?e\.?(?=\s|$)/g, "mtech "],
  [/\bph\.?\s?d\.?/g, "phd"],
  [/\bm\.?\s?b\.?\s?a\b\.?/g, "mba"],
  [/\bm\.?\s?c\.?\s?a\b\.?/g, "mca"],
  [/\bb\.?\s?c\.?\s?a\b\.?/g, "bca"],
  [/\bb\.\s?sc\b\.?|\bb\s?sc\b/g, "bsc"],
  [/\bm\.\s?sc\b\.?|\bm\s?sc\b/g, "msc"],
  [/\bb\.\s?com\b\.?/g, "bcom"],
  [/\bm\.\s?com\b\.?/g, "mcom"],
  [/\bb\.b\.a\b\.?/g, "bba"],
  [/\bb\.\s?a\b\.?/g, "ba"],
  [/\bm\.\s?a\b\.?/g, "ma"],
  [/\bplus\s?two\b|\+\s?2\b|\bhsc\b|\bclass\s?12\b|\b12th\s+(?:std|standard|grade)\b|\bhigher secondary\b|\btwelfth\b/g, "12th"],
  [/\ba\.i\.?/g, "ai"],
  [/\bai\s+university\b/g, "university"],        // the university's own name, not the AI programme
  [/\bai\s*(?:&|and|-|\/)\s*ds\b/g, "ai ds"],
  [/\bai\s*(?:&|and|-|\/)\s*data science\b/g, "ai ds"],
  [/\bartificial intelligence (?:&|and) data science\b/g, "ai ds"],
  [/\bcomputer science (?:and|&) engineering\b/g, "computer science"],
  [/\belectronics (?:and|&) communication(?: engineering)?\b/g, "electronics"],
  [/\bc\.?\s?g\.?\s?p\.?\s?a\b/g, "cgpa"],
  [/(\d+(?:\.\d+)?)\s*(?:%|per\s?cent\b|percentage\b)/g, "$1 percent"],
  [/(\d+(?:\.\d+)?)\s*(?:lpa|lakhs? per annum)/g, "$1 lpa"],
  [/\bjee\s*mains?\b/g, "jee"],
  [/\bugc[\s-]?net\b/g, "net"],
];

// ------------------------------------------------------------------------
// 2. Synonyms -> one canonical word per concept (multi-word phrases first)
// ------------------------------------------------------------------------
const PHRASE_SYNONYMS = {
  "last date": "deadline", "closing date": "deadline", "due date": "deadline",
  "how much": "fee", "how long": "duration", "how many years": "duration",
  "how many semesters": "duration", "how many seats": "seat",
  "head of department": "hod", "head of the department": "hod",
  "fee structure": "fee", "cut off": "cutoff", "cut-off": "cutoff",
  "get into": "join", "get admission": "admission", "take admission": "admission",
  "financial aid": "scholarship", "fee waiver": "scholarship",
  "fee concession": "scholarship", "sports quota": "sports scholarship",
  "mess food": "mess", "salary package": "placement", "get a job": "placement",
  "contact number": "phone", "phone number": "phone", "mobile number": "phone",
  "email id": "email", "mail id": "email",
};

const WORD_SYNONYMS = {
  // fees
  fees: "fee", cost: "fee", costs: "fee", charges: "fee", charge: "fee",
  tuition: "fee", price: "fee", expense: "fee", expenses: "fee", pay: "fee", payable: "fee",
  // eligibility
  eligible: "eligibility", eligiblity: "eligibility", qualify: "eligibility",
  qualified: "eligibility", criteria: "eligibility", criterion: "eligibility",
  requirement: "eligibility", requirements: "eligibility", prerequisite: "eligibility",
  cutoff: "cutoff",
  // admission
  apply: "apply", application: "apply", applying: "apply", enroll: "apply",
  enrol: "apply", enrollment: "apply", registration: "register", procedure: "procedure",
  admissions: "admission",
  // hostel
  dorm: "hostel", dormitory: "hostel", accommodation: "hostel", accomodation: "hostel",
  hostels: "hostel", stay: "hostel", lodging: "hostel", rooms: "room",
  // placement
  placements: "placement", job: "placement", jobs: "placement", recruit: "placement",
  recruitment: "placement", recruiters: "placement", recruiter: "placement",
  package: "placement", salary: "placement", companies: "placement", company: "placement",
  hiring: "placement", placed: "placement", ctc: "placement", lpa: "placement",
  // scholarship
  scholarships: "scholarship", concession: "scholarship", waiver: "scholarship",
  discount: "scholarship", stipend: "scholarship", freeship: "scholarship",
  // department
  hod: "hod", head: "hod", dept: "department", departments: "department",
  // duration
  duration: "duration", years: "year", semesters: "semester", length: "duration",
  // seats
  seats: "seat", intake: "seat", vacancy: "seat", vacancies: "seat",
  // programmes
  courses: "course", programs: "program", programmes: "program", programme: "program",
  branches: "branch", degrees: "degree", specialisations: "branch", specializations: "branch",
  // documents
  documents: "document", certificates: "certificate", papers: "paper", docs: "document",
  // facilities
  buses: "bus", "college bus": "bus", labs: "lab", laboratory: "lab", laboratories: "lab",
  gymnasium: "gym", facilities: "facility", amenities: "facility",
  // contact
  telephone: "phone", mobile: "phone", mail: "email", reach: "contact",
  where: "location", situated: "location", located: "location",
  // greetings
  hii: "hi", hai: "hi", helo: "hello", thx: "thanks", thankyou: "thanks",
  // student short forms
  sem: "semester", sems: "semester", exams: "exam", dd: "draft",
};
const synonym = w => (Object.hasOwn(WORD_SYNONYMS, w) ? WORD_SYNONYMS[w] : w);

const words = s => s.trim().split(/\s+/);

// the words the synonym table is built on; preferred when spelling correction finds a tie
const CORE_WORDS = new Set([...Object.keys(WORD_SYNONYMS), ...Object.values(WORD_SYNONYMS)]);

const STOPWORDS = new Set(words(`
a an the is are am was were be been being do does did doing to of in on at for from by
with about as into through over i me my we our you your he she it its they them their
this that these those and or but if then so than too very can could would should will
shall may might must please tell let know want need get got also any some just there here
ok okay sir madam mam kindly regarding u ur pls plz`));

// everyday English words that are fine even though the university domain does not use them
const COMMON_WORDS = new Set(words(`
weather today tomorrow movie movies music cricket score news price prices pizza order food
which where there their these those about after before again against while would could should
other others every great thing things think thought people really still never always maybe
house money world water phone write right wrong little small large young older video videos
free freely trees three where whose whom please hello thanks thank sorry bring brought
match matches team teams player won lost meaning capital joke jokes laptop mobile ticket
recipe cook cooking travel train flight hotel
using used uses useful possible parts break lunch week weeks month months today daily every`));

// words that must never be "corrected"
const PROTECTED = new Set(["btech", "mtech", "mba", "mca", "phd", "bsc", "msc", "bca", "bcom", "cse", "ece",
  "eee", "mech", "ai", "ds", "it", "gate", "jee", "cat", "mat", "gmat", "tancet",
  "guet", "net", "sc", "st", "obc", "bc", "mbc", "pcm", "pcb", "cgpa", "hod", "lpa",
  "naac", "nri", "ug", "pg", "12th", "wifi"]);

const isAlpha = w => /^[a-z]+$/.test(w);
const escapeRe = s => s.replace(/[.*+?^${}()|[\]\\-]/g, "\\$&");

// very light suffix stripping (enough for plurals and verb forms)
export function stem(word) {
  if (PROTECTED.has(word) || word.length <= 4 || /^\d+$/.test(word)) return word;
  for (const [suf, rep] of [["ies", "y"], ["ing", ""], ["ed", ""], ["es", ""], ["s", ""]]) {
    if (word.endsWith(suf) && word.length - suf.length >= 3) {
      if (suf === "es" && !["ches", "shes", "sses", "xes"].some(e => word.endsWith(e))) continue;
      if (suf === "s" && word.endsWith("ss")) continue;
      return word.slice(0, -suf.length) + rep;
    }
  }
  return word;
}

export class TextPreprocessor {
  constructor() {
    this.vocabulary = new Set([...PROTECTED, ...Object.keys(WORD_SYNONYMS), ...Object.values(WORD_SYNONYMS)]);
    this.spellCheck = true;
  }

  addVocabulary(list) {
    for (const w of list) if (isAlpha(w) || PROTECTED.has(w)) this.vocabulary.add(w);
  }

  static normalise(text) {
    let t = " " + text.toLowerCase().trim() + " ";
    t = t.replaceAll("’", "'").replaceAll("`", "'");
    for (const [pattern, replacement] of NORMALISE) t = t.replace(pattern, replacement);
    t = t.replace(/[^a-z0-9.%\s]/g, " ");
    t = t.replace(/(?<!\d)\.|\.(?!\d)/g, " ");       // keep decimal points only
    return t.replace(/\s+/g, " ").trim();
  }

  correct(tokens) {
    const corrections = {};
    if (!this.spellCheck) return [tokens, corrections];
    const fixed = tokens.map(tok => {
      const known = this.vocabulary.has(tok) || this.vocabulary.has(stem(tok)) || this.vocabulary.has(tok + "s");
      if (tok.length >= 4 && isAlpha(tok) && !known && !STOPWORDS.has(tok) && !COMMON_WORDS.has(tok)) {
        // short words must keep the same first letter, to avoid wild corrections
        const long = tok.length >= 5;
        const candidates = long ? this.vocabulary : [...this.vocabulary].filter(w => w[0] === tok[0]);
        const match = closestWord(tok, candidates, long ? 0.8 : 0.75, CORE_WORDS);
        if (match) {
          corrections[tok] = match;
          return match;
        }
      }
      return tok;
    });
    return [fixed, corrections];
  }

  process(text) {
    const norm = TextPreprocessor.normalise(text);
    const [tokens, corrections] = this.correct(norm ? norm.split(" ") : []);
    let joined = tokens.join(" ");
    for (const [phrase, canon] of Object.entries(PHRASE_SYNONYMS)) {
      joined = joined.replace(new RegExp(`\\b${escapeRe(phrase)}\\b`, "g"), canon);
    }
    const lemmas = [];
    for (const tok of joined.split(" ").filter(Boolean)) {
      for (const part of synonym(tok).split(" ")) {
        if (STOPWORDS.has(part)) continue;
        lemmas.push(synonym(stem(part)));
      }
    }
    return { raw: text, normalised: tokens.join(" "), tokens, lemmas, corrections, lemmaText: lemmas.join(" ") };
  }
}

// ------------------------------------------------------------------------
// 3. Entity extraction
// ------------------------------------------------------------------------
const LEVEL_PATTERNS = [
  ["btech", "btech|undergraduate|ug|bachelor of engineering|bachelor of technology|engineering degree"],
  ["mtech", "mtech|master of technology|master of engineering"],
  ["mba", "mba|master of business administration|business administration"],
  ["mca", "mca|master of computer applications"],
  ["phd", "phd|doctorate|doctoral"],
  ["bsc", "bsc"], ["msc", "msc"], ["bca", "bca"], ["bcom", "bcom"], ["mcom", "mcom"],
  ["ba", "ba"], ["ma", "ma"], ["bba", "bba"], ["diploma", "diploma"], ["twelfth", "12th"],
  ["pg", "pg|postgraduate|post graduate|masters"],
];
const BRANCH_PATTERNS = [
  ["aids", "ai ds|aids|ai and ds"],
  ["cse", "cse|computer science|computer engineering|cs"],
  ["it", "it(?= |$)|information technology"],
  ["ece", "ece|electronics|electronic communication"],
  ["eee", "eee|electrical"],
  ["mech", "mech|mechanical"],
  ["civil", "civil"],
  ["ai", "ai|artificial intelligence"],
  ["ds", "ds|data science"],
  ["maths", "maths|mathematics|math"],
  ["stats", "statistics|stats"],
  ["physics", "physics"],
  ["chemistry", "chemistry"],
  ["commerce", "commerce"],
  ["english", "english literature|english"],
  ["economics", "economics"],
  ["analytics", "business analytics|analytics"],
  ["finance", "finance"],
];
const groupRegex = (prefix, list) =>
  new RegExp("\\b(?:" + list.map(([n, p]) => `(?<${prefix}${n}>${p})`).join("|") + ")\\b", "g");
const LEVELS_RE = groupRegex("L_", LEVEL_PATTERNS);
const BRANCH_RE = groupRegex("B_", BRANCH_PATTERNS);
const matchedGroup = m => Object.keys(m.groups).find(k => m.groups[k] !== undefined).slice(2);

const OFFERED = {
  "btech|cse": "btech_cse", "btech|aids": "btech_aids", "btech|ai": "btech_aids",
  "btech|ds": "btech_aids", "btech|ece": "btech_ece", "btech|mech": "btech_mech",
  "mtech|ai": "mtech_ai", "mtech|aids": "mtech_ai", "mtech|ds": "mtech_ds", "mtech|cse": "mtech_ai",
  "mba|": "mba", "mca|": "mca", "phd|": "phd",
  "mba|analytics": "mba_ba", "mba|finance": "mba_finance",
  "bsc|cse": "bsc_cs", "bsc|maths": "bsc_maths", "bca|": "bca",
  "bcom|": "bcom", "bba|": "bba", "mcom|": "mcom",
  "ba|english": "ba_english", "ba|economics": "ba_economics", "ma|english": "ma_english",
};
// degrees that are either a programme we offer or the student's own qualification
const FLEX_LEVELS = new Set(["bsc", "bca", "bcom", "ba", "bba", "msc", "ma", "mcom"]);
// branches a level may be combined with when it is read as one degree
const LEVEL_BRANCHES = { mba: new Set(["analytics", "finance"]), mca: new Set(), phd: new Set(), "12th": new Set(), diploma: new Set() };
const BARE_BRANCH = { cse: "btech_cse", ece: "btech_ece", mech: "btech_mech", aids: "btech_aids", ai: "mtech_ai", ds: "mtech_ds" };
const GROUPS = { btech: "group:btech", mtech: "group:mtech", pg: "group:pg" };

const QUAL_CUES = new Set(["have", "had", "did", "done", "completed", "complete", "finished", "studied", "studying",
  "with", "having", "holding", "graduate", "graduated", "pursuing", "doing", "my", "after",
  "from", "passed", "hold", "own", "scored", "got", "secured", "am", "cgpa", "percent"]);
const TARGET_CUES = new Set(["for", "into", "join", "joining", "admission", "apply", "applying", "pursue",
  "study", "get", "take", "seat", "in", "to", "enrol", "enroll"]);

const FACILITIES = {
  hostel: "hostel|mess|room|dorm|accommodation",
  library: "library|books|journal",
  transport: "transport|bus|buses|route",
  sports: "sports?|gym|ground|swimming|pool|cricket|football|badminton",
  labs: "labs?|laboratory|laboratories|gpu|workshop",
};
const EXAMS = ["gate", "jee", "cat", "mat", "gmat", "tancet", "guet", "net"];
const CATEGORY_RE = /\b(sc|st|obc|mbc|bc|general|oc)\b(?:\s+(?:category|student|candidate|quota|community))?/;

// one degree mentioned in the sentence, e.g. "btech ece"
class DegreeMention {
  constructor(level, branch, start, end) {
    // role: target | qualification; ambiguous = no cue word, could be either
    Object.assign(this, { level, branch, start, end, role: "unknown", ambiguous: false });
  }

  get program() {
    if (this.level === null) return this.branch ? BARE_BRANCH[this.branch] || null : null;
    const key = `${this.level}|${this.branch}`;
    if (this.branch && key in OFFERED) return OFFERED[key];
    if (this.level === "btech" || this.level === "mtech") return null;   // a branch we do not offer, or none
    return OFFERED[`${this.level}|`] || null;
  }

  get group() {
    return this.program === null ? GROUPS[this.level] || null : null;
  }

  get qualification() {
    if (this.level === null || this.level === "pg") return null;
    let br = this.branch;
    if (this.level === "btech" && (br === "ai" || br === "ds")) br = "aids";
    if ((this.level === "bsc" || this.level === "msc") && (br === "cse" || br === "it")) br = "cs";
    return br ? `${this.level}_${br}` : this.level;
  }
}

const QUAL_NAMES = { btech: "B.Tech", mtech: "M.Tech", bsc: "B.Sc", msc: "M.Sc", bca: "BCA",
  bcom: "B.Com", ba: "B.A", ma: "M.A", bba: "BBA", mba: "MBA", mca: "MCA",
  phd: "Ph.D", diploma: "Diploma", "12th": "12th standard" };

export function qualificationLabel(q) {
  if (!q) return "-";
  const i = q.indexOf("_");
  const [lvl, br] = i < 0 ? [q, ""] : [q.slice(0, i), q.slice(i + 1)];
  return `${QUAL_NAMES[lvl] || lvl.toUpperCase()} ${br.toUpperCase()}`.trim();
}

const ENTITY_ORDER = ["program", "qualification", "department", "facility", "office", "certificate", "percentage", "cgpa",
  "exam", "category", "subjects", "sports", "sibling", "levelFilter", "year"];

// only the entities that were actually found
export function entitiesAsDict(ent) {
  const d = {};
  for (const k of ENTITY_ORDER) {
    const v = ent[k];
    if (v === null || v === undefined || v === false || v === 0 || (Array.isArray(v) && !v.length)) continue;
    d[k] = v;
  }
  return d;
}

export class EntityExtractor {
  constructor(kb) {
    const byLength = aliases => Object.entries(aliases)
      .flatMap(([id, list]) => list.map(a => [a, id]))
      .sort((x, y) => y[0].length - x[0].length);
    this.deptAliases = byLength(kb.departmentAliases);
    this.officeAliases = byLength(kb.officeAliases);
    this.certAliases = byLength(kb.certificateAliases);
  }

  degrees(text) {
    const tokens = text.split(" ");
    const offsets = [];
    let pos = 0;
    for (const tok of tokens) { offsets.push(pos); pos += tok.length + 1; }
    const tokIndex = ch => {
      let idx = 0;
      offsets.forEach((o, i) => { if (o <= ch) idx = i; });
      return idx;
    };
    const find = re => [...text.matchAll(re)].map(m => [
      matchedGroup(m).replace("twelfth", "12th"), tokIndex(m.index), tokIndex(m.index + m[0].length - 1)]);

    const levels = find(LEVELS_RE);
    const branches = find(BRANCH_RE);
    const used = new Set();
    const mentions = [];
    for (const [lvl, s, e] of levels) {
      let best = null;
      branches.forEach(([br, bs, be], bi) => {
        if (used.has(bi) || (LEVEL_BRANCHES[lvl] && !LEVEL_BRANCHES[lvl].has(br))) return;
        const gapAfter = bs - e;           // branch after level: "btech in cse"
        const gapBefore = s - be;          // branch before level: "cse btech"
        if ((gapAfter > 0 && gapAfter <= 3) || (gapBefore > 0 && gapBefore <= 1)) {
          const dist = gapAfter > 0 ? gapAfter : gapBefore + 0.5;
          if (best === null || dist < best[0]) best = [dist, bi];
        }
      });
      if (best) {
        used.add(best[1]);
        const [br, bs, be] = branches[best[1]];
        mentions.push(new DegreeMention(lvl, br, Math.min(s, bs), Math.max(e, be)));
      } else {
        mentions.push(new DegreeMention(lvl, null, s, e));
      }
    }
    branches.forEach(([br, bs, be], bi) => {
      if (!used.has(bi) && br in BARE_BRANCH) mentions.push(new DegreeMention(null, br, bs, be));
    });
    mentions.sort((a, b) => a.start - b.start);

    // is this degree the one the student wants, or the one they already have?
    for (const m of mentions) {
      // the 3 words before the degree; "do you have B.Tech Civil" is about the university, not the student
      const lo = Math.max(0, m.start - 3);
      const before = tokens.slice(lo, m.start).filter((t, k) =>
        !((t === "have" || t === "had") && lo + k > 0 && ["you", "u", "we"].includes(tokens[lo + k - 1])));
      const justBefore = tokens.slice(Math.max(0, m.start - 1), m.start);
      if (m.level === "12th" || m.level === "diploma") {
        m.role = "qualification";
      } else if (FLEX_LEVELS.has(m.level)) {
        // "fee for BCA" asks about the programme; "I did BCA" / "after BCA" is the student's degree
        if (before.some(t => QUAL_CUES.has(t)) || m.program === null) m.role = "qualification";
        else if (before.some(t => TARGET_CUES.has(t))) m.role = "target";
        else { m.role = "qualification"; m.ambiguous = true; }
      } else if (before.some(t => QUAL_CUES.has(t)) && !justBefore.some(t => ["for", "into", "join"].includes(t))) {
        m.role = "qualification";
      } else if (before.some(t => TARGET_CUES.has(t))) {
        m.role = "target";
      }
    }
    return mentions;
  }

  extract(text) {
    const ent = {
      program: null, qualification: null, department: null, facility: null, office: null, certificate: null, percentage: null,
      cgpa: null, exam: null, category: null, subjects: null, sports: false, sibling: false, levelFilter: null,
      year: null,                       // academic season, e.g. "2022-23"
      ambiguousQualification: null,     // degree read as the programme, but it may be the student's own
    };
    const degrees = this.degrees(text);

    const targets = degrees.filter(m => m.role === "target" && (m.program || m.group));
    const others = degrees.filter(m => m.role === "unknown" && (m.program || m.group));
    let quals = degrees.filter(m => m.role === "qualification");
    let chosen = targets[0] || others[0] || null;
    if (chosen === null && quals.length >= 2) {
      // "btech cse ... mtech ai": the higher degree is the target
      chosen = quals[quals.length - 1];
      quals = quals.slice(0, -1);
    }
    if (chosen) ent.program = chosen.program || chosen.group;
    // the student's own qualification: an explicit cue first, otherwise another degree mention
    let qual = quals[0] || null;
    if (qual === null) qual = degrees.find(m => m !== chosen && m.qualification) || null;
    if (qual && qual.qualification) ent.qualification = qual.qualification;

    for (const [alias, dept] of this.deptAliases) {
      if (new RegExp(`\\b${escapeRe(alias)}\\b`).test(text) &&
          (text.includes("department") || text.includes("dept") || text.includes("hod") || text.includes("head") ||
           ["management", "school of management", "research centre"].includes(alias))) {
        ent.department = dept;
        break;
      }
    }

    // academic season: "2023-24" as written, or "2023" = the batch that graduated in 2023 (2022-23)
    const pair = text.match(/\b(20\d{2})\s+(?:to\s+)?(\d{2}|20\d{2})\b/);
    const single = text.match(/\b(20\d{2})\b/);
    if (pair && Number(pair[2].slice(-2)) === (Number(pair[1]) + 1) % 100) {
      ent.year = `${pair[1]}-${pair[2].slice(-2)}`;
    } else if (single) {
      const y = Number(single[1]);
      ent.year = `${y - 1}-${String(y % 100).padStart(2, "0")}`;
    }

    // university offices and certificates (longest alias first, so "exam cell" beats "cell")
    const named = list => (list.find(([a]) => new RegExp(`\\b${escapeRe(a)}\\b`).test(text)) || [null, null])[1];
    ent.office = named(this.officeAliases);
    ent.certificate = named(this.certAliases);

    for (const [fac, pat] of Object.entries(FACILITIES)) {
      if (new RegExp(`\\b(?:${pat})\\b`).test(text)) { ent.facility = fac; break; }
    }

    if (/\b(?:ug|undergraduate|bachelor)\b/.test(text)) ent.levelFilter = "UG";
    else if (/\b(?:pg|postgraduate|post graduate|masters)\b/.test(text)) ent.levelFilter = "PG";

    // marks: CGPA first, then percentage
    let m = text.match(/(\d+(?:\.\d+)?)\s*cgpa|cgpa\s*(?:of|is|was|:)?\s*(\d+(?:\.\d+)?)/);
    if (m) {
      const val = parseFloat(m[1] || m[2]);
      if (val > 0 && val <= 10) ent.cgpa = val;
    }
    m = text.match(/(\d{2,3}(?:\.\d+)?)\s*(?:percent|marks)/);
    let m2;
    if (m && parseFloat(m[1]) >= 30 && parseFloat(m[1]) <= 100) {
      ent.percentage = parseFloat(m[1]);
    } else if (ent.cgpa === null &&
               (m2 = text.match(/\b(?:percentage|percent|marks|aggregate)\s*(?:is|was|of|:)?\s*(\d{2}(?:\.\d+)?)\b/))) {
      ent.percentage = parseFloat(m2[1]);
    } else if (ent.cgpa === null) {
      m = text.match(/\b(?:scored|got|secured|with|have|of|aggregate)\s+(\d{2}(?:\.\d+)?)\b(?!\s*(?:lpa|lakh|years?|seats?|k\b|rank))/);
      if (m && parseFloat(m[1]) >= 35 && parseFloat(m[1]) <= 100) ent.percentage = parseFloat(m[1]);
    }

    ent.exam = EXAMS.find(ex => new RegExp(`\\b${ex}\\b`).test(text)) || null;
    m = text.match(CATEGORY_RE);
    if (m && m[1] !== "general" && m[1] !== "oc") ent.category = m[1];

    if (/\bpcm\b|physics.{0,15}chemistry.{0,15}math|math.{0,15}physics.{0,15}chemistry/.test(text)) {
      ent.subjects = ["pcm"];
    } else if (/\b(?:pcb|biology|bio)\b/.test(text)) {
      ent.subjects = ["pcb"];
    } else if (/\b(?:commerce|arts|humanities)\b/.test(text) && (ent.qualification || "").startsWith("12th")) {
      ent.subjects = ["commerce"];
    }

    ent.sports = /\b(?:sports? (?:quota|scholarship)|state level|national level|district level|sportsperson|athlete|player)\b/.test(text);
    ent.sibling = /\b(?:sibling|brother|sister)\b/.test(text);

    // "BCA fee?" - a degree with no cue word and no marks is the programme being asked about
    const own = degrees.filter(d => d.qualification);
    if (ent.program === null && own.length === 1 && own[0].ambiguous && own[0].program &&
        ent.percentage === null && ent.cgpa === null) {
      ent.program = own[0].program;
      ent.ambiguousQualification = ent.qualification;
      ent.qualification = null;
    }
    return ent;
  }
}

export const hasApplicantFacts = ent =>
  ent.percentage !== null || ent.cgpa !== null || ent.qualification !== null;
