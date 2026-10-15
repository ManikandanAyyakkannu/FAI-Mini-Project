"""NLP pipeline: normalisation, tokenisation, spelling correction, synonym
mapping, stop-word removal, light stemming and entity extraction.

Only the Python standard library is used here, so the pipeline is easy to
explain and has no hidden pretrained model.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Any

# --------------------------------------------------------------------------
# 1. Normalisation: unify the many ways students write degree names, marks...
# --------------------------------------------------------------------------
_NORMALISE = [
    (r"\bb\.\s?tech\b|\bb\s?tech\b|\bb-tech\b", "btech"),
    (r"\bm\.\s?tech\b|\bm\s?tech\b|\bm-tech\b", "mtech"),
    (r"\bb\.\s?e\.?(?=\s|$)", "btech "),
    (r"\bbe(?=\s+(?:in\s+)?(?:cse|ece|eee|it|mech|mechanical|civil|computer|electronics|electrical)\b)", "btech"),
    (r"\bm\.\s?e\.?(?=\s|$)", "mtech "),
    (r"\bph\.?\s?d\.?", "phd"),
    (r"\bm\.?\s?b\.?\s?a\b\.?", "mba"),
    (r"\bm\.?\s?c\.?\s?a\b\.?", "mca"),
    (r"\bb\.?\s?c\.?\s?a\b\.?", "bca"),
    (r"\bb\.\s?sc\b\.?|\bb\s?sc\b", "bsc"),
    (r"\bm\.\s?sc\b\.?|\bm\s?sc\b", "msc"),
    (r"\bb\.\s?com\b\.?", "bcom"),
    (r"\bb\.\s?a\b\.?", "ba"),
    (r"\bplus\s?two\b|\+\s?2\b|\bhsc\b|\bclass\s?12\b|\b12th\s+(?:std|standard|grade)\b|\bhigher secondary\b|\btwelfth\b", "12th"),
    (r"\ba\.i\.?", "ai"),
    (r"\bai\s*(?:&|and|-|/)\s*ds\b", "ai ds"),
    (r"\bai\s*(?:&|and|-|/)\s*data science\b", "ai ds"),
    (r"\bartificial intelligence (?:&|and) data science\b", "ai ds"),
    (r"\bcomputer science (?:and|&) engineering\b", "computer science"),
    (r"\belectronics (?:and|&) communication(?: engineering)?\b", "electronics"),
    (r"\bc\.?\s?g\.?\s?p\.?\s?a\b", "cgpa"),
    (r"(\d+(?:\.\d+)?)\s*(?:%|per\s?cent\b|percentage\b)", r"\1 percent"),
    (r"(\d+(?:\.\d+)?)\s*(?:lpa|lakhs? per annum)", r"\1 lpa"),
    (r"\bjee\s*mains?\b", "jee"),
    (r"\bugc[\s-]?net\b", "net"),
]
_NORMALISE = [(re.compile(p), r) for p, r in _NORMALISE]

# --------------------------------------------------------------------------
# 2. Synonyms -> canonical concept words (multi-word phrases first)
# --------------------------------------------------------------------------
PHRASE_SYNONYMS = {
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
}
WORD_SYNONYMS = {
    # fees
    "fees": "fee", "cost": "fee", "costs": "fee", "charges": "fee", "charge": "fee",
    "tuition": "fee", "price": "fee", "expense": "fee", "expenses": "fee", "pay": "fee",
    "payable": "fee",
    # eligibility
    "eligible": "eligibility", "eligiblity": "eligibility", "qualify": "eligibility",
    "qualified": "eligibility", "criteria": "eligibility", "criterion": "eligibility",
    "requirement": "eligibility", "requirements": "eligibility", "prerequisite": "eligibility",
    "cutoff": "cutoff",
    # admission
    "apply": "apply", "application": "apply", "applying": "apply", "enroll": "apply",
    "enrol": "apply", "enrollment": "apply", "registration": "register", "procedure": "procedure",
    "admissions": "admission",
    # hostel
    "dorm": "hostel", "dormitory": "hostel", "accommodation": "hostel", "accomodation": "hostel",
    "hostels": "hostel", "stay": "hostel", "lodging": "hostel", "rooms": "room",
    # placement
    "placements": "placement", "job": "placement", "jobs": "placement", "recruit": "placement",
    "recruitment": "placement", "recruiters": "placement", "recruiter": "placement",
    "package": "placement", "salary": "placement", "companies": "placement", "company": "placement",
    "hiring": "placement", "placed": "placement", "ctc": "placement", "lpa": "placement",
    # scholarship
    "scholarships": "scholarship", "concession": "scholarship", "waiver": "scholarship",
    "discount": "scholarship", "stipend": "scholarship", "freeship": "scholarship",
    # department
    "hod": "hod", "head": "hod", "dept": "department", "departments": "department",
    # duration
    "duration": "duration", "years": "year", "semesters": "semester", "length": "duration",
    # seats
    "seats": "seat", "intake": "seat", "vacancy": "seat", "vacancies": "seat",
    # programmes
    "courses": "course", "programs": "program", "programmes": "program", "programme": "program",
    "branches": "branch", "degrees": "degree", "specialisations": "branch", "specializations": "branch",
    # documents
    "documents": "document", "certificates": "certificate", "papers": "paper", "docs": "document",
    # facilities
    "buses": "bus", "college bus": "bus", "labs": "lab", "laboratory": "lab", "laboratories": "lab",
    "gymnasium": "gym", "facilities": "facility", "amenities": "facility",
    # contact
    "telephone": "phone", "mobile": "phone", "mail": "email", "reach": "contact",
    "where": "location", "situated": "location", "located": "location",
    # greetings
    "hii": "hi", "hai": "hi", "helo": "hello", "thx": "thanks", "thankyou": "thanks",
}

STOPWORDS = set("""
a an the is are am was were be been being do does did doing to of in on at for from by
with about as into through over i me my we our you your he she it its they them their
this that these those and or but if then so than too very can could would should will
shall may might must please tell let know want need get got also any some just there here
ok okay sir madam mam kindly regarding u ur pls plz
""".split())

# everyday English words that are valid even though they are not in the domain vocabulary
COMMON_WORDS = set("""
weather today tomorrow movie movies music cricket score news price prices pizza order food
which where there their these those about after before again against while would could should
other others every great thing things think thought people really still never always maybe
house money world water phone write right wrong little small large young older video videos
free freely trees three where whose whom please hello thanks thank sorry bring brought
match matches team teams player won lost meaning capital joke jokes laptop mobile ticket
recipe cook cooking travel train flight hotel
""".split())

# words that should never be "corrected"
_PROTECTED = {"btech", "mtech", "mba", "mca", "phd", "bsc", "msc", "bca", "bcom", "cse", "ece",
              "eee", "mech", "ai", "ds", "it", "gate", "jee", "cat", "mat", "gmat", "tancet",
              "guet", "net", "sc", "st", "obc", "bc", "mbc", "pcm", "pcb", "cgpa", "hod", "lpa",
              "naac", "nri", "ug", "pg", "12th", "wifi"}


def _stem(word: str) -> str:
    """Very light suffix stripping (enough for plural/verb forms in queries)."""
    if word in _PROTECTED or len(word) <= 4 or word.isdigit():
        return word
    for suf, rep in (("ies", "y"), ("ing", ""), ("ed", ""), ("es", ""), ("s", "")):
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            if suf == "es" and not word.endswith(("ches", "shes", "sses", "xes")):
                continue
            if suf == "s" and word.endswith("ss"):
                continue
            return word[: -len(suf)] + rep
    return word


@dataclass
class Processed:
    raw: str
    normalised: str
    tokens: list[str]
    lemmas: list[str]
    corrections: dict[str, str] = field(default_factory=dict)

    @property
    def lemma_text(self) -> str:
        return " ".join(self.lemmas)


class TextPreprocessor:
    def __init__(self, vocabulary: set[str] | None = None):
        self.vocabulary: set[str] = set(vocabulary or ())
        self.vocabulary |= _PROTECTED | set(WORD_SYNONYMS) | set(WORD_SYNONYMS.values())

    def add_vocabulary(self, words) -> None:
        self.vocabulary |= {w for w in words if w.isalpha() or w in _PROTECTED}

    @staticmethod
    def normalise(text: str) -> str:
        t = " " + text.lower().strip() + " "
        t = t.replace("’", "'").replace("`", "'")
        for pat, rep in _NORMALISE:
            t = pat.sub(rep, t)
        t = re.sub(r"[^a-z0-9.%\s]", " ", t)
        t = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", t)       # keep decimal points only
        return re.sub(r"\s+", " ", t).strip()

    def correct(self, tokens: list[str]) -> tuple[list[str], dict[str, str]]:
        fixed, corrections = [], {}
        for tok in tokens:
            known = (tok in self.vocabulary or _stem(tok) in self.vocabulary or tok + "s" in self.vocabulary)
            if (len(tok) >= 4 and tok.isalpha() and not known
                    and tok not in STOPWORDS and tok not in COMMON_WORDS):
                # short words need the same first letter to avoid wild corrections
                cands = self.vocabulary if len(tok) >= 5 else {w for w in self.vocabulary if w[:1] == tok[:1]}
                match = difflib.get_close_matches(tok, cands, n=1, cutoff=0.8 if len(tok) >= 5 else 0.75)
                if match:
                    corrections[tok] = match[0]
                    tok = match[0]
            fixed.append(tok)
        return fixed, corrections

    def process(self, text: str) -> Processed:
        norm = self.normalise(text)
        tokens = norm.split()
        tokens, corrections = self.correct(tokens)
        joined = " ".join(tokens)
        for phrase, canon in PHRASE_SYNONYMS.items():
            joined = re.sub(rf"\b{re.escape(phrase)}\b", canon, joined)
        lemmas = []
        for tok in joined.split():
            tok = WORD_SYNONYMS.get(tok, tok)
            for part in tok.split():
                if part in STOPWORDS:
                    continue
                lemmas.append(WORD_SYNONYMS.get(_stem(part), _stem(part)))
        return Processed(text, " ".join(tokens), tokens, lemmas, corrections)


# --------------------------------------------------------------------------
# 3. Entity extraction
# --------------------------------------------------------------------------
LEVEL_PATTERNS = [
    ("btech", r"btech|undergraduate|ug|bachelor of engineering|bachelor of technology|engineering degree"),
    ("mtech", r"mtech|master of technology|master of engineering"),
    ("mba", r"mba|master of business administration|business administration"),
    ("mca", r"mca|master of computer applications"),
    ("phd", r"phd|doctorate|doctoral"),
    ("bsc", r"bsc"), ("msc", r"msc"), ("bca", r"bca"), ("bcom", r"bcom"),
    ("ba", r"ba"), ("ma", r"ma"), ("bba", r"bba"), ("diploma", r"diploma"), ("12th", r"12th"),
    ("pg", r"pg|postgraduate|post graduate|masters"),
]
BRANCH_PATTERNS = [
    ("aids", r"ai ds|aids|ai and ds"),
    ("cse", r"cse|computer science|computer engineering|cs"),
    ("it", r"it(?= |$)|information technology"),
    ("ece", r"ece|electronics|electronic communication"),
    ("eee", r"eee|electrical"),
    ("mech", r"mech|mechanical"),
    ("civil", r"civil"),
    ("ai", r"ai|artificial intelligence"),
    ("ds", r"ds|data science"),
    ("maths", r"maths|mathematics|math"),
    ("stats", r"statistics|stats"),
    ("physics", r"physics"),
    ("chemistry", r"chemistry"),
    ("commerce", r"commerce"),
]
LEVELS_RE = re.compile(r"\b(?:" + "|".join(f"(?P<L_{n.replace('12th', 'twelfth')}>{p})"
                                           for n, p in LEVEL_PATTERNS) + r")\b")
BRANCH_RE = re.compile(r"\b(?:" + "|".join(f"(?P<B_{n}>{p})" for n, p in BRANCH_PATTERNS) + r")\b")

OFFERED = {
    ("btech", "cse"): "btech_cse", ("btech", "aids"): "btech_aids", ("btech", "ai"): "btech_aids",
    ("btech", "ds"): "btech_aids", ("btech", "ece"): "btech_ece", ("btech", "mech"): "btech_mech",
    ("mtech", "ai"): "mtech_ai", ("mtech", "aids"): "mtech_ai", ("mtech", "ds"): "mtech_ds",
    ("mtech", "cse"): "mtech_ai",
    ("mba", None): "mba", ("mca", None): "mca", ("phd", None): "phd",
}
BARE_BRANCH = {"cse": "btech_cse", "ece": "btech_ece", "mech": "btech_mech", "aids": "btech_aids",
               "ai": "mtech_ai", "ds": "mtech_ds"}
GROUPS = {"btech": "group:btech", "mtech": "group:mtech", "pg": "group:pg"}

QUAL_CUES = {"have", "had", "did", "done", "completed", "complete", "finished", "studied", "studying",
             "with", "having", "holding", "graduate", "graduated", "pursuing", "doing", "my", "after",
             "from", "passed", "hold", "own", "scored", "got", "secured", "am", "cgpa", "percent"}
TARGET_CUES = {"for", "into", "join", "joining", "admission", "apply", "applying", "pursue",
               "study", "get", "take", "seat", "in", "to", "enrol", "enroll"}

FACILITIES = {
    "hostel": r"hostel|mess|room|dorm|accommodation",
    "library": r"library|books|journal",
    "transport": r"transport|bus|buses|route",
    "sports": r"sports?|gym|ground|swimming|pool|cricket|football|badminton",
    "labs": r"labs?|laboratory|laboratories|gpu|workshop",
}
EXAMS = {"gate": r"gate", "jee": r"jee", "cat": r"cat", "mat": r"mat", "gmat": r"gmat",
         "tancet": r"tancet", "guet": r"guet", "net": r"net"}
CATEGORY_RE = re.compile(r"\b(sc|st|obc|mbc|bc|general|oc)\b(?:\s+(?:category|student|candidate|quota|community))?")


@dataclass
class DegreeMention:
    level: str
    branch: str | None
    start: int
    end: int
    role: str = "unknown"        # "target" | "qualification" | "unknown"

    @property
    def program(self) -> str | None:
        if self.level in ("btech", "mtech") and self.branch:
            return OFFERED.get((self.level, self.branch))
        if self.level in ("mba", "mca", "phd"):
            return OFFERED[(self.level, None)]
        if self.level is None and self.branch:
            return BARE_BRANCH.get(self.branch)
        return None

    @property
    def group(self) -> str | None:
        return GROUPS.get(self.level) if self.program is None else None

    @property
    def qualification(self) -> str | None:
        if self.level is None or self.level == "pg":
            return None
        lvl = self.level
        br = self.branch
        if lvl == "btech" and br in ("ai", "ds"):
            br = "aids"
        if lvl in ("bsc", "msc") and br in ("cse", "it"):
            br = "cs"
        return f"{lvl}_{br}" if br else lvl


def qualification_label(q: str | None) -> str:
    if not q:
        return "-"
    names = {"btech": "B.Tech", "mtech": "M.Tech", "bsc": "B.Sc", "msc": "M.Sc", "bca": "BCA",
             "bcom": "B.Com", "ba": "B.A", "ma": "M.A", "bba": "BBA", "mba": "MBA", "mca": "MCA",
             "phd": "Ph.D", "diploma": "Diploma", "12th": "12th standard"}
    lvl, _, br = q.partition("_")
    return f"{names.get(lvl, lvl.upper())} {br.upper() if br else ''}".strip()


@dataclass
class Entities:
    degrees: list[DegreeMention] = field(default_factory=list)
    program: str | None = None            # target programme id (or group:...)
    qualification: str | None = None       # applicant's qualification e.g. btech_ece
    department: str | None = None
    facility: str | None = None
    percentage: float | None = None
    cgpa: float | None = None
    exam: str | None = None
    category: str | None = None
    subjects: list[str] | None = None
    sports: bool = False
    sibling: bool = False
    level_filter: str | None = None        # "UG" / "PG" for programme listing

    def as_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k != "degrees" and v not in (None, False, [])}
        return d

    @property
    def has_applicant_facts(self) -> bool:
        return any(v is not None for v in (self.percentage, self.cgpa, self.qualification))


class EntityExtractor:
    def __init__(self, kb):
        self.kb = kb
        self.dept_alias = sorted(((a, d) for d, al in kb.department_aliases.items() for a in al),
                                 key=lambda x: -len(x[0]))

    def _degrees(self, text: str) -> list[DegreeMention]:
        tokens = text.split()
        # character offset -> token index
        offsets, pos = [], 0
        for tok in tokens:
            offsets.append(pos)
            pos += len(tok) + 1

        def tok_index(ch):
            idx = 0
            for i, o in enumerate(offsets):
                if o <= ch:
                    idx = i
            return idx

        levels = [(m.lastgroup[2:].replace("twelfth", "12th"), tok_index(m.start()), tok_index(m.end() - 1))
                  for m in LEVELS_RE.finditer(text)]
        branches = [(m.lastgroup[2:], tok_index(m.start()), tok_index(m.end() - 1))
                    for m in BRANCH_RE.finditer(text)]
        used, mentions = set(), []
        for lvl, s, e in levels:
            best = None
            for bi, (br, bs, be) in enumerate(branches):
                if bi in used:
                    continue
                gap_after = bs - e          # branch after level: "btech in cse"
                gap_before = s - be         # branch before level: "cse btech"
                if 0 < gap_after <= 3 or 0 < gap_before <= 1:
                    dist = gap_after if gap_after > 0 else gap_before + 0.5
                    if best is None or dist < best[0]:
                        best = (dist, bi)
            if best and lvl not in ("mba", "mca", "phd", "12th", "diploma"):
                used.add(best[1])
                br, bs, be = branches[best[1]]
                mentions.append(DegreeMention(lvl, br, min(s, bs), max(e, be)))
            else:
                mentions.append(DegreeMention(lvl, None, s, e))
        for bi, (br, bs, be) in enumerate(branches):
            if bi not in used and br in BARE_BRANCH:
                mentions.append(DegreeMention(None, br, bs, be))
        mentions.sort(key=lambda m: m.start)

        for m in mentions:
            before = set(tokens[max(0, m.start - 3): m.start])
            if m.level in ("12th", "bsc", "bca", "bcom", "ba", "bba", "diploma", "msc", "ma"):
                m.role = "qualification"
            elif before & QUAL_CUES and not (set(tokens[max(0, m.start - 1): m.start]) & {"for", "into", "join"}):
                m.role = "qualification"
            elif before & TARGET_CUES:
                m.role = "target"
        return mentions

    def extract(self, text: str, lemmas: list[str] | None = None) -> Entities:
        ent = Entities()
        ent.degrees = self._degrees(text)

        targets = [m for m in ent.degrees if m.role == "target" and (m.program or m.group)]
        others = [m for m in ent.degrees if m.role == "unknown" and (m.program or m.group)]
        quals = [m for m in ent.degrees if m.role == "qualification"]
        chosen = (targets or others or [None])[0]
        if chosen is None and len(quals) >= 2:
            # "btech cse ... mtech ai": the higher degree is the target
            chosen = quals[-1]
            quals = quals[:-1]
        if chosen:
            ent.program = chosen.program or chosen.group
        # applicant qualification: explicit cue first, otherwise a second degree mention
        qual_m = quals[0] if quals else None
        if qual_m is None:
            rest = [m for m in ent.degrees if m is not chosen and m.qualification]
            qual_m = rest[0] if rest else None
        if qual_m and qual_m.qualification:
            ent.qualification = qual_m.qualification

        # department
        for alias, dept in self.dept_alias:
            if re.search(rf"\b{re.escape(alias)}\b", text) and (
                    "department" in text or "dept" in text or "hod" in text or "head" in text
                    or alias in ("management", "school of management", "research centre")):
                ent.department = dept
                break

        for fac, pat in FACILITIES.items():
            if re.search(rf"\b(?:{pat})\b", text):
                ent.facility = fac
                break

        if re.search(r"\b(?:ug|undergraduate|bachelor)\b", text):
            ent.level_filter = "UG"
        elif re.search(r"\b(?:pg|postgraduate|post graduate|masters)\b", text):
            ent.level_filter = "PG"

        m = re.search(r"(\d+(?:\.\d+)?)\s*cgpa|cgpa\s*(?:of|is|was|:)?\s*(\d+(?:\.\d+)?)", text)
        if m:
            val = float(m.group(1) or m.group(2))
            if 0 < val <= 10:
                ent.cgpa = val
        m = re.search(r"(\d{2,3}(?:\.\d+)?)\s*(?:percent|marks)", text)
        if m and 30 <= float(m.group(1)) <= 100:
            ent.percentage = float(m.group(1))
        elif ent.cgpa is None and (m := re.search(r"\b(?:percentage|percent|marks|aggregate)\s*(?:is|was|of|:)?\s*"
                                                  r"(\d{2}(?:\.\d+)?)\b", text)):
            ent.percentage = float(m.group(1))
        elif ent.cgpa is None:
            m = re.search(r"\b(?:scored|got|secured|with|have|of|aggregate)\s+(\d{2}(?:\.\d+)?)\b"
                          r"(?!\s*(?:lpa|lakh|years?|seats?|k\b|rank))", text)
            if m and 35 <= float(m.group(1)) <= 100:
                ent.percentage = float(m.group(1))

        for ex, pat in EXAMS.items():
            if re.search(rf"\b{pat}\b", text):
                ent.exam = ex
                break
        m = CATEGORY_RE.search(text)
        if m and m.group(1) not in ("general", "oc"):
            ent.category = m.group(1)

        if re.search(r"\bpcm\b|physics.{0,15}chemistry.{0,15}math|math.{0,15}physics.{0,15}chemistry", text):
            ent.subjects = ["pcm"]
        elif re.search(r"\b(?:pcb|biology|bio)\b", text):
            ent.subjects = ["pcb"]
        elif re.search(r"\b(?:commerce|arts|humanities)\b", text) and (ent.qualification or "").startswith("12th"):
            ent.subjects = ["commerce"]

        ent.sports = bool(re.search(r"\b(?:sports? (?:quota|scholarship)|state level|national level|"
                                    r"district level|sportsperson|athlete|player)\b", text))
        ent.sibling = bool(re.search(r"\b(?:sibling|brother|sister)\b", text))
        return ent
