// Natural-language answer templates, filled from the frame knowledge base.
// buildFaq() turns the same answers into a question/answer list for the vector store.

import { qualificationLabel } from "./nlp.js";

// Indian-style rupee formatting: 150000 -> "Rs. 1,50,000"
export function rs(amount) {
  let s = String(Math.trunc(amount));
  if (s.length > 3) {
    let head = s.slice(0, -3);
    const tail = s.slice(-3);
    const parts = [];
    while (head.length > 2) { parts.unshift(head.slice(-2)); head = head.slice(0, -2); }
    if (head) parts.unshift(head);
    s = parts.join(",") + "," + tail;
  }
  return `Rs. ${s}`;
}

export class Answers {
  constructor(kb) {
    this.kb = kb;
  }

  name(pid) {
    return this.kb.get(pid, "name", pid);
  }

  groupMembers(group) {
    const kb = this.kb;
    if (group === "group:btech") return kb.programs.filter(p => p.startsWith("btech"));
    if (group === "group:mtech") return kb.programs.filter(p => p.startsWith("mtech"));
    if (group === "group:pg") return kb.programs.filter(p => kb.programLevel(p) === "PG");
    return kb.programs;
  }

  isMany(pid) {
    return pid === "all" || pid.startsWith("group:");
  }

  // ------------------------------------------------------------ programmes
  programList(level = null) {
    const kb = this.kb;
    const progs = kb.programs.filter(p => level === null || kb.programLevel(p) === level);
    const title = { UG: "Undergraduate", PG: "Postgraduate" }[level] || "All";
    return [`${title} programmes at Greenfield University:`,
      ...progs.map(p => `  - ${this.name(p)} (${kb.get(p, "duration")}, ${kb.get(p, "seats")} seats)`)].join("\n");
  }

  programFee(pid) {
    const kb = this.kb;
    const hostelFee = rs(kb.get("hostel", "fee_per_year"));
    if (this.isMany(pid)) {
      return ["Annual tuition fees:",
        ...this.groupMembers(pid).map(p => `  - ${this.name(p)}: ${rs(kb.get(p, "fee_per_year"))} per year`),
        `Hostel (optional): ${hostelFee} per year including mess.`].join("\n");
    }
    const fee = kb.get(pid, "fee_per_year");
    const dur = kb.get(pid, "duration");
    const first = dur.split(" ")[0];
    const total = /^\d+$/.test(first)
      ? ` The total tuition for ${dur.split("(")[0].trim()} is about ${rs(fee * Number(first))}.` : "";
    return `The tuition fee for ${this.name(pid)} is ${rs(fee)} per year.${total} ` +
      `A one-time application fee of ${rs(kb.get(pid, "application_fee"))} and a refundable caution ` +
      `deposit of ${rs(kb.get(pid, "caution_deposit"))} are also payable. ` +
      `Hostel with mess costs ${hostelFee} per year extra.`;
  }

  programDuration(pid) {
    if (this.isMany(pid)) {
      return ["Programme durations:",
        ...this.groupMembers(pid).map(p => `  - ${this.name(p)}: ${this.kb.get(p, "duration")}`)].join("\n");
    }
    return `${this.name(pid)} is a ${this.kb.get(pid, "duration")} ${this.kb.get(pid, "mode").toLowerCase()} programme.`;
  }

  programSeats(pid) {
    if (this.isMany(pid)) {
      return ["Seat intake:",
        ...this.groupMembers(pid).map(p => `  - ${this.name(p)}: ${this.kb.get(p, "seats")} seats`)].join("\n");
    }
    return `${this.name(pid)} has an intake of ${this.kb.get(pid, "seats")} seats per year.`;
  }

  eligibilityInfo(pid) {
    const kb = this.kb;
    let label;
    if (pid.startsWith("group:")) {
      pid = this.groupMembers(pid)[0];
      label = pid.startsWith("btech") ? "B.Tech (all branches)" : this.name(pid);
    } else {
      label = this.name(pid);
    }
    const mp = kb.get(pid, "min_percentage");
    const exams = kb.get(pid, "entrance_exams", []).join(", ");
    let crit;
    if (kb.programLevel(pid) === "UG") {
      crit = `a pass in 12th standard with Physics, Chemistry and Mathematics and at least ${mp}% aggregate ` +
        `(${mp - 5}% for SC/ST/OBC)`;
    } else {
      const allowed = kb.get(pid, "allowed_qualifications", []);
      const quals = allowed.length === 1 && allowed[0] === "any_degree"
        ? "any bachelor's degree" : allowed.map(qualificationLabel).join(", ");
      crit = `${quals} with at least ${mp}% aggregate (${mp - 5}% for SC/ST/OBC; CGPA x 9.5 is used to convert)`;
    }
    return `Eligibility for ${label}: ${crit}, and a valid score in one of: ${exams}.`;
  }

  // ------------------------------------------------------------- admission
  admissionProcess() {
    const steps = this.kb.get("admission_process", "steps");
    return "Admission process:\n" + steps.map((s, i) => `  ${i + 1}. ${s}`).join("\n");
  }

  admissionDeadline(level) {
    const kb = this.kb;
    const opens = kb.get("admission_process", "applications_open");
    const guet = kb.get("admission_process", "guet_date");
    const cls = { UG: "UGProgram", PG: "PGProgram", Research: "ResearchProgram" }[level];
    if (cls) {
      return `Applications for ${level} programmes open on ${opens} and close on ` +
        `${kb.get(cls, "deadline")}. The GUET entrance exam is on ${guet}.`;
    }
    return `Applications open on ${opens}. Last dates: UG - ${kb.get("UGProgram", "deadline")}; ` +
      `PG - ${kb.get("PGProgram", "deadline")}; Ph.D - ${kb.get("ResearchProgram", "deadline")}. ` +
      `The GUET entrance exam is on ${guet}.`;
  }

  documentsRequired(level) {
    const cls = { PG: "PGProgram", Research: "ResearchProgram" }[level] || "UGProgram";
    const label = { PG: "PG", Research: "Ph.D" }[level] || "UG";
    return `Documents required for ${label} admission:\n` +
      this.kb.get(cls, "documents").map(d => `  - ${d}`).join("\n");
  }

  // ------------------------------------------------------------ facilities
  hostel() {
    const h = this.kb.resolved("hostel");
    return `${h.description} Rooms: ${h.room_types}. The fee is ${rs(h.fee_per_year)} per year, ` +
      `which includes ${h.includes}. AC rooms cost ${rs(h.ac_extra)} extra per year. ${h.timings}.`;
  }

  facility(fid) {
    if (fid === "hostel") return this.hostel();
    const f = this.kb.resolved(fid);
    const parts = [`${f.name}: ${f.description}`];
    if ("timings" in f) parts.push(`Timings: ${f.timings}.`);
    if ("fee_per_year" in f) parts.push(`Fee: ${rs(f.fee_per_year)} per year.`);
    return parts.join(" ");
  }

  facilitiesOverview() {
    return "Campus facilities:\n" + this.kb.instancesOf("Facility")
      .map(f => `  - ${this.kb.get(f, "name")}: ${this.kb.get(f, "description")}`).join("\n");
  }

  // ---------------------------------------------------------- scholarships
  scholarshipInfo() {
    const s = this.kb.resolved("scholarship_policy");
    const tiers = s.merit.map(t => `${t.min}% and above - ${t.waiver}% waiver`).join("; ");
    return `Scholarships (on tuition fee): merit - ${tiers}. Sports quota (state/national level) - ` +
      `${s.sports_waiver}%. Sibling concession - ${s.sibling_waiver}%. Waivers can be combined up to ` +
      `${s.max_waiver}%. ${s.government} M.Tech students with GATE get a stipend of Rs. 12,400 per month.`;
  }

  // ------------------------------------------------------------ placements
  placement(dept) {
    const kb = this.kb;
    const p = dept ? kb.get(dept, "placement") : null;
    if (p) {
      return `Placements 2025-26, ${kb.get(dept, "name")}: ${p.percent_placed}% of students placed, ` +
        `average package ${p.average_lpa} LPA, highest ${p.highest_lpa} LPA. ` +
        `Top recruiters: ${p.top_recruiters.join(", ")}.`;
    }
    const lines = ["Placements 2025-26 (percentage placed / average / highest):"];
    for (const d of kb.instancesOf("Department")) {
      const q = kb.get(d, "placement");
      if (q) lines.push(`  - ${kb.get(d, "name")}: ${q.percent_placed}% / ${q.average_lpa} LPA / ${q.highest_lpa} LPA`);
    }
    return lines.join("\n");
  }

  // ----------------------------------------------------------- departments
  departmentInfo(dept) {
    const d = this.kb.resolved(dept);
    const progs = this.kb.programsOfDepartment(dept).map(p => this.name(p)).join(", ");
    return `${d.name} - Head of Department: ${d.hod}. Located in ${d.block}. Email: ${d.dept_email}.` +
      (progs ? ` Programmes offered: ${progs}.` : "");
  }

  departmentOfProgram(pid) {
    const dept = this.kb.get(pid, "department");
    return `${this.name(pid)} is offered by the ${this.kb.get(dept, "name")} (HOD: ${this.kb.get(dept, "hod")}).`;
  }

  // ------------------------------------------------------------------ misc
  contact() {
    const u = this.kb.frames.University;
    return `Admissions office, ${u.name}, ${u.location}. Phone: ${u.phone}. Email: ${u.email}. ` +
      `Website: ${u.website}. Office hours: ${u.office_hours}.`;
  }

  about() {
    const u = this.kb.frames.University;
    return `${u.name} is located in ${u.location}. It was established in ${u.established} and is ` +
      `accredited ${u.accreditation}. It offers ${this.kb.programs.length} programmes across UG, PG and ` +
      `research levels.`;
  }
}

// A flat FAQ list (question, answer, key) built from the knowledge base, plus the
// free-text FAQs in data/faq_extra.json. The vector store searches this list.
export function buildFaq(kb, A, extraFaqs) {
  const faq = [];
  const add = (key, intent, question, answer) => faq.push({ key, intent, question, answer });

  add("program_list:all", "program_list", "What programmes and courses are offered?", A.programList());
  add("program_list:UG", "program_list", "What undergraduate (UG) B.Tech courses are offered?", A.programList("UG"));
  add("program_list:PG", "program_list", "What postgraduate (PG) courses are offered?", A.programList("PG"));

  for (const p of kb.programs) {
    const n = kb.get(p, "name");
    add(`program_fee:${p}`, "program_fee", `What is the fee for ${n}?`, A.programFee(p));
    add(`program_duration:${p}`, "program_duration", `What is the duration of ${n}?`, A.programDuration(p));
    add(`program_seats:${p}`, "program_seats", `How many seats are there in ${n}?`, A.programSeats(p));
    add(`eligibility:${p}`, "eligibility", `What is the eligibility for ${n}?`, A.eligibilityInfo(p));
    add(`department_info:${kb.get(p, "department")}`, "department_info", `Which department offers ${n}?`,
      A.departmentOfProgram(p));
  }

  add("program_fee:group:btech", "program_fee", "What is the fee for B.Tech courses?", A.programFee("group:btech"));
  add("admission_process", "admission_process", "What is the admission process? How do I apply?", A.admissionProcess());
  add("admission_deadline:all", "admission_deadline", "What is the last date to apply for admission?",
    A.admissionDeadline(null));
  for (const lvl of ["UG", "PG"]) {
    add(`admission_deadline:${lvl}`, "admission_deadline", `What is the ${lvl} application deadline?`,
      A.admissionDeadline(lvl));
    add(`documents_required:${lvl}`, "documents_required", `What documents are required for ${lvl} admission?`,
      A.documentsRequired(lvl));
  }
  add("hostel", "hostel", "Is hostel accommodation available? What is the hostel fee?", A.hostel());
  for (const f of ["library", "transport", "sports", "labs"]) {
    add(`facility:${f}`, "facility", `Tell me about the ${kb.get(f, "name").toLowerCase()} facility.`, A.facility(f));
  }
  add("scholarship", "scholarship", "What scholarships are available?", A.scholarshipInfo());
  add("placement:all", "placement", "How are the placements?", A.placement(null));
  for (const d of kb.instancesOf("Department")) {
    const dn = kb.get(d, "name");
    add(`department_info:${d}`, "department_info", `Who is the HOD of the ${dn}?`, A.departmentInfo(d));
    if (kb.get(d, "placement")) {
      add(`placement:${d}`, "placement", `What is the placement record of the ${dn}?`, A.placement(d));
    }
  }
  add("contact", "contact", "How can I contact the admissions office?", A.contact());
  add("about_university", "about_university", "Tell me about the university.", A.about());

  for (const item of extraFaqs) add(item.key, "faq", item.question, item.answer);
  return faq;
}
