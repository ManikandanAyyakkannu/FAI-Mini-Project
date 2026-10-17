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
    return [`${title} programmes at AI University:`,
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

  // short summary of one programme ("Do you have BBA?", "Tell me about BBA")
  programInfo(pid, confirm = false) {
    const kb = this.kb;
    const dept = kb.get(pid, "department");
    return (confirm ? "Yes. " : "") +
      `${this.name(pid)} is offered by the ${kb.get(dept, "name")}. It is a ${kb.get(pid, "duration")} ` +
      `programme with ${kb.get(pid, "seats")} seats, and the tuition fee is ` +
      `${rs(kb.get(pid, "fee_per_year"))} per year. Ask me about its eligibility, fees or placements ` +
      `for more details.`;
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

  // added when someone asks whether seats are still open: we only know the intake, not live numbers
  seatStatus(pid) {
    const kb = this.kb;
    const level = this.isMany(pid) ? null : kb.programLevel(pid);
    const cls = { UG: "UGProgram", PG: "PGProgram", Research: "ResearchProgram" }[level];
    const closes = cls
      ? `Applications for ${level} programmes close on ${kb.get(cls, "deadline")}. `
      : `Applications close on ${kb.get("UGProgram", "deadline")} (UG) and ${kb.get("PGProgram", "deadline")} (PG). `;
    return "I don't have live information on how many seats are still open. " + closes +
      `For the current seat position, please contact the admissions office at ${kb.get("University", "phone")}.`;
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
      const stream = kb.get(pid, "required_subjects", []).length ? "with Physics, Chemistry and Mathematics" : "in any stream";
      crit = `a pass in 12th standard ${stream} and at least ${mp}% aggregate (${mp - 5}% for SC/ST/OBC)`;
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
  // seasons with placement figures, oldest first (the "placement" slot is the latest season)
  placementYears() {
    const kb = this.kb;
    const latest = kb.get("University", "placement_year");
    const withHistory = kb.instancesOf("Department").find(d => kb.get(d, "placement_history"));
    const older = withHistory ? Object.keys(kb.get(withHistory, "placement_history")).sort() : [];
    return [...older, latest];
  }

  #placementOf(dept, year) {
    if (year === this.kb.get("University", "placement_year")) return this.kb.get(dept, "placement");
    return (this.kb.get(dept, "placement_history") || {})[year] || null;
  }

  placement(dept, year = null) {
    const kb = this.kb;
    const years = this.placementYears();
    let note = "";
    if (year && !years.includes(year)) {
      note = `Placement figures for ${year} are not available; I have them for ${years[0]} to ${years.at(-1)}. ` +
        `Here are the latest (${years.at(-1)}):\n`;
      year = null;
    }
    year = year || years.at(-1);
    const p = dept ? this.#placementOf(dept, year) : null;
    if (p) {
      return note + `Placements ${year}, ${kb.get(dept, "name")}: ${p.percent_placed}% of students placed, ` +
        `average package ${p.average_lpa} LPA, highest ${p.highest_lpa} LPA. ` +
        `Top recruiters: ${kb.get(dept, "placement").top_recruiters.join(", ")}.`;
    }
    if (dept) {
      return `Placement figures for the ${kb.get(dept, "name")} are not published yet. For details, please ` +
        `contact the admissions office at ${kb.get("University", "phone")}.`;
    }
    const lines = [note + `Placements ${year} (percentage placed / average / highest):`];
    for (const d of kb.instancesOf("Department")) {
      const q = this.#placementOf(d, year);
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

  // -------------------------------------------------------- university life
  // Each answer picks the parts of its frame that the question asks about (by its key words).
  academicCalendar(words) {
    const c = this.kb.resolved("academic_calendar");
    const has = (...ws) => ws.some(w => words.has(w));
    let parts = [];
    if (has("holiday")) parts.push(`University holidays: ${c.holidays}.`);
    if (has("vacation", "summer", "winter")) parts.push(`Winter vacation: ${c.winter_vacation}. Summer vacation: ${c.summer_vacation}.`);
    if (has("timing", "time", "hour", "working", "saturday"))
      parts.push(`Classes run from ${c.class_timings}. Working days: ${c.working_days}.`);
    if (!parts.length) {
      parts = [`Academic calendar 2026-27: odd semester - ${c.odd_semester}; winter vacation ${c.winter_vacation}; ` +
        `even semester - ${c.even_semester}; summer vacation ${c.summer_vacation}.`];
    }
    return parts.join(" ");
  }

  examination(words) {
    const e = this.kb.resolved("examinations"), c = this.kb.resolved("academic_calendar");
    const has = (...ws) => ws.some(w => words.has(w));
    const parts = [];
    if (has("hall", "ticket")) parts.push(`${e.hall_ticket}.`);
    if (has("grade", "grading", "cgpa", "gpa")) {
      const grades = e.grades.map(([g, r, p]) => `${g} (${r}) = ${p}`).join(", ");
      parts.push(`Grades and grade points: ${grades}. ${e.cgpa}.`);
    }
    if (has("pass", "internal", "pattern", "mark", "assessment")) parts.push(`${e.pattern}. ${e.pass_rule}.`);
    if (has("timetable", "schedule", "when", "date", "start") || !parts.length) {
      // the exam dates live in the academic-calendar frame (a link in the semantic network)
      const dates = ["odd_semester", "even_semester"].map(s => c[s].split(", ")[1].replace("end-semester exams ", ""));
      parts.push(`End-semester exams: odd semester ${dates[0]}; even semester ${dates[1]}. ${e.timetable}.`);
    }
    return parts.join(" ") + ` For anything else about exams, contact the ${this.kb.get(e.office, "name")}.`;
  }

  results(words) {
    const e = this.kb.resolved("examinations");
    const has = (...ws) => ws.some(w => words.has(w));
    let parts = [];
    if (has("revaluation", "photocopy", "recheck", "rechecking", "retotal")) parts.push(`${e.revaluation}.`);
    if (has("supplementary", "arrear", "fail", "reappear", "backlog")) parts.push(`${e.supplementary}.`);
    if (!parts.length) parts = [`${e.results}. ${e.revaluation}.`];
    return parts.join(" ") + ` Results are handled by the ${this.kb.get(e.office, "name")}.`;
  }

  feePayment(words) {
    const f = this.kb.resolved("fee_payment");
    const has = (...ws) => ws.some(w => words.has(w));
    const parts = [];
    if (has("late")) parts.push(`Late fee: ${f.late_fee}.`);
    if (has("instalment", "installment")) parts.push(`${f.instalments}.`);
    if (has("receipt")) parts.push(`${f.receipt}.`);
    if (!parts.length || has("due", "date", "last", "deadline", "when")) parts.unshift(`Semester fees are due by ${f.due_dates}.`);
    if (parts.length === 1 && !has("late", "receipt", "instalment", "installment"))
      parts.push(`You can pay ${f.modes}. Late fee: ${f.late_fee}.`);
    return parts.join(" ") + ` Questions about payments go to the ${this.kb.get(f.office, "name")}.`;
  }

  certificate(cid) {
    const kb = this.kb;
    if (!cid) {
      const items = kb.instancesOf("Certificate")
        .map(c => `${kb.get(c, "name")} - ${this.#certFee(c)}, ${kb.get(c, "time")}`).join("; ");
      return `Certificates you can request: ${items}. ${kb.get("Certificate", "how")}.`;
    }
    const c = kb.resolved(cid);
    const text = `${c.name}: ${c.how}. Fee: ${this.#certFee(cid)}. It takes ${c.time}. ` +
      `It is issued by the ${kb.get(c.office, "name")}.`;
    return text + ("note" in c ? ` ${c.note}` : "");
  }

  #certFee(cid) {
    const fee = this.kb.get(cid, "fee");
    return fee ? rs(fee) : "free";
  }

  office(oid) {
    const kb = this.kb;
    if (!oid) {
      return "University offices:\n" + kb.instancesOf("Office").map(o =>
        `  - ${kb.get(o, "name")}: ${kb.get(o, "head")}, ${kb.get(o, "phone")} - handles ${kb.get(o, "handles")}`).join("\n");
    }
    const o = kb.resolved(oid);
    return `${o.name} - ${o.head}. Location: ${o.location}. Phone: ${o.phone}, e-mail: ${o.email}. ` +
      `Office hours: ${o.office_hours}. It handles ${o.handles}.`;
  }

  viceChancellor() {
    const u = this.kb.frames.University;
    return `The Vice-Chancellor of ${u.name} is ${u.vice_chancellor}. ` +
      `The Registrar is ${this.kb.get("office_registrar", "head").split(" (")[0]}.`;
  }

  facultyResearch(words) {
    const u = this.kb.frames.University, r = this.kb.resolved("research");
    const has = (...ws) => ws.some(w => words.has(w));
    const faculty = `${u.name} has ${u.faculty} The student-teacher ratio is ${u.student_teacher_ratio}.`;
    const research = `Research: ${r.centres}. There are ${r.projects}, and ${r.output}. ` +
      "Ph.D. admissions are open in all departments.";
    if (has("research", "patent", "publication", "project", "centre", "paper", "funded"))
      return has("faculty", "professor", "teacher", "staff") ? faculty + " " + research : research;
    return has("faculty", "professor", "teacher", "staff", "ratio") ? faculty : faculty + " " + research;
  }

  events(words) {
    const e = this.kb.resolved("events");
    const picks = [["convocation", `The convocation is held ${e.convocation}.`],
      ["tech", `Technical fest: ${e.tech_fest}.`], ["technical", `Technical fest: ${e.tech_fest}.`],
      ["cultural", `Cultural fest: ${e.cultural_fest}.`], ["sport", `Sports: ${e.sports_meet}.`]];
    let parts = [...new Set(picks.filter(([w]) => words.has(w)).map(([, t]) => t))];
    if (!parts.length) {
      parts = [`Main events: ${e.tech_fest}; ${e.cultural_fest}; ${e.sports_meet}; and the convocation ` +
        `${e.convocation}. ${e.other}.`];
    }
    return parts.join(" ") + ` Events are organised by the ${this.kb.get(e.office, "name")}.`;
  }

  alumni() {
    const a = this.kb.resolved("alumni");
    return `Our graduates are part of ${a.association}. ${a.benefits}. They meet at ${a.meet}. Contact: ${a.contact}.`;
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
      `research levels, and has ${u.students}.`;
  }

  studentStrength() {
    const u = this.kb.frames.University;
    return `${u.name} has ${u.students}, across ${this.kb.programs.length} programmes in ` +
      `${this.kb.instancesOf("Department").length} departments.`;
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

  // university life beyond admissions
  const none = new Set();
  add("academic_calendar", "academic_calendar", "What is the academic calendar? When do semesters and vacations start?",
    A.academicCalendar(none));
  add("examination", "examination", "When are the end-semester exams and what is the exam pattern?", A.examination(none));
  add("results", "results", "When are the results published and how do I apply for revaluation?", A.results(none));
  add("fee_payment", "fee_payment", "How and when do I pay the semester fee?", A.feePayment(none));
  for (const c of kb.instancesOf("Certificate")) {
    add(`certificates:${c}`, "certificates", `How do I get a ${kb.get(c, "name").toLowerCase()}?`, A.certificate(c));
  }
  for (const o of kb.instancesOf("Office")) {
    add(`office_info:${o}`, "office_info", `How can I contact the ${kb.get(o, "name").split(" (")[0]}?`, A.office(o));
  }
  add("faculty_research", "faculty_research", "How many faculty members are there, and what research is done?",
    A.facultyResearch(none));
  add("events", "events", "What is the date of the convocation and the cultural and technical events?", A.events(none));
  add("alumni", "alumni", "Is there an alumni association?", A.alumni());

  for (const item of extraFaqs) add(item.key, "faq", item.question, item.answer);
  return faq;
}
