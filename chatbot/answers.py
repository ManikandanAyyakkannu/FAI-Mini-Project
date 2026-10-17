"""Natural-language answer templates filled from the frame knowledge base.

Both the hybrid chatbot and the FAQ builder use these functions, so the two
approaches compared in the evaluation return exactly the same wording; only the
way they *choose* an answer differs.
"""
from __future__ import annotations


def rs(amount: int) -> str:
    """Indian-style rupee formatting: 150000 -> Rs. 1,50,000"""
    s = str(int(amount))
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        head = ",".join([head[max(0, i - 2):i] for i in range(len(head), 0, -2)][::-1])
        s = head + "," + tail
    return f"Rs. {s}"


class Answers:
    def __init__(self, kb):
        self.kb = kb

    # ----------------------------------------------------------- helpers
    def name(self, pid):
        return self.kb.get(pid, "name", pid)

    def group_members(self, group: str) -> list[str]:
        if group == "group:btech":
            return [p for p in self.kb.programs if p.startswith("btech")]
        if group == "group:mtech":
            return [p for p in self.kb.programs if p.startswith("mtech")]
        if group == "group:pg":
            return [p for p in self.kb.programs if self.kb.program_level(p) == "PG"]
        return self.kb.programs

    # --------------------------------------------------------- programmes
    def program_list(self, level: str | None = None) -> str:
        progs = [p for p in self.kb.programs if level is None or self.kb.program_level(p) == level]
        title = {"UG": "Undergraduate", "PG": "Postgraduate"}.get(level, "All")
        lines = [f"{title} programmes at AI University:"]
        for p in progs:
            lines.append(f"  - {self.name(p)} ({self.kb.get(p, 'duration')}, {self.kb.get(p, 'seats')} seats)")
        return "\n".join(lines)

    def program_fee(self, pid: str) -> str:
        if pid.startswith("group:") or pid == "all":
            lines = ["Annual tuition fees:"]
            for p in self.group_members(pid):
                lines.append(f"  - {self.name(p)}: {rs(self.kb.get(p, 'fee_per_year'))} per year")
            lines.append(f"Hostel (optional): {rs(self.kb.get('hostel', 'fee_per_year'))} per year including mess.")
            return "\n".join(lines)
        fee = self.kb.get(pid, "fee_per_year")
        dur = self.kb.get(pid, "duration")
        years = int(dur.split()[0]) if dur.split()[0].isdigit() else None
        total = f" The total tuition for {dur.split('(')[0].strip()} is about {rs(fee * years)}." if years else ""
        return (f"The tuition fee for {self.name(pid)} is {rs(fee)} per year.{total} "
                f"A one-time application fee of {rs(self.kb.get(pid, 'application_fee'))} and a refundable caution "
                f"deposit of {rs(self.kb.get(pid, 'caution_deposit'))} are also payable. "
                f"Hostel with mess costs {rs(self.kb.get('hostel', 'fee_per_year'))} per year extra.")

    def program_info(self, pid: str, confirm: bool = False) -> str:
        """Short summary of one programme ("Do you have BBA?", "Tell me about BBA")."""
        dept = self.kb.get(pid, "department")
        return (("Yes. " if confirm else "") +
                f"{self.name(pid)} is offered by the {self.kb.get(dept, 'name')}. It is a {self.kb.get(pid, 'duration')} "
                f"programme with {self.kb.get(pid, 'seats')} seats, and the tuition fee is "
                f"{rs(self.kb.get(pid, 'fee_per_year'))} per year. Ask me about its eligibility, fees or placements "
                f"for more details.")

    def program_duration(self, pid: str) -> str:
        if pid.startswith("group:") or pid == "all":
            return "\n".join(["Programme durations:"] + [f"  - {self.name(p)}: {self.kb.get(p, 'duration')}"
                                                         for p in self.group_members(pid)])
        dur, src = self.kb.get_with_source(pid, "duration")
        return f"{self.name(pid)} is a {dur} {self.kb.get(pid, 'mode').lower()} programme."

    def program_seats(self, pid: str) -> str:
        if pid.startswith("group:") or pid == "all":
            return "\n".join(["Seat intake:"] + [f"  - {self.name(p)}: {self.kb.get(p, 'seats')} seats"
                                                for p in self.group_members(pid)])
        return f"{self.name(pid)} has an intake of {self.kb.get(pid, 'seats')} seats per year."

    def seat_status(self, pid: str) -> str:
        """Added when someone asks whether seats are still open: we only know the intake, not live numbers."""
        level = self.kb.program_level(pid) if pid not in ("all",) and not pid.startswith("group:") else None
        cls = {"UG": "UGProgram", "PG": "PGProgram", "Research": "ResearchProgram"}.get(level)
        closes = (f"Applications for {level} programmes close on {self.kb.get(cls, 'deadline')}. " if cls
                  else f"Applications close on {self.kb.get('UGProgram', 'deadline')} (UG) and "
                       f"{self.kb.get('PGProgram', 'deadline')} (PG). ")
        return ("I don't have live information on how many seats are still open. " + closes +
                f"For the current seat position, please contact the admissions office at {self.kb.get('University', 'phone')}.")

    def eligibility_info(self, pid: str) -> str:
        if pid.startswith("group:"):
            members = self.group_members(pid)
            pid = members[0]
            label = "B.Tech (all branches)" if pid.startswith("btech") else self.name(pid)
        else:
            label = self.name(pid)
        level = self.kb.program_level(pid)
        mp = self.kb.get(pid, "min_percentage")
        exams = ", ".join(self.kb.get(pid, "entrance_exams", []))
        if level == "UG":
            stream = "with Physics, Chemistry and Mathematics" if self.kb.get(pid, "required_subjects") else "in any stream"
            crit = f"a pass in 12th standard {stream} and at least {mp}% aggregate ({mp - 5}% for SC/ST/OBC)"
        else:
            allowed = self.kb.get(pid, "allowed_qualifications", [])
            from .nlp import qualification_label
            if allowed == ["any_degree"]:
                quals = "any bachelor's degree"
            else:
                quals = ", ".join(qualification_label(a) for a in allowed)
            crit = f"{quals} with at least {mp}% aggregate ({mp - 5}% for SC/ST/OBC; CGPA x 9.5 is used to convert)"
        return f"Eligibility for {label}: {crit}, and a valid score in one of: {exams}."

    # ---------------------------------------------------------- admission
    def admission_process(self) -> str:
        steps = self.kb.get("admission_process", "steps")
        return "Admission process:\n" + "\n".join(f"  {i}. {s}" for i, s in enumerate(steps, 1))

    def admission_deadline(self, level: str | None) -> str:
        opens = self.kb.get("admission_process", "applications_open")
        guet = self.kb.get("admission_process", "guet_date")
        if level in ("UG", "PG", "Research"):
            cls = {"UG": "UGProgram", "PG": "PGProgram", "Research": "ResearchProgram"}[level]
            return (f"Applications for {level} programmes open on {opens} and close on "
                    f"{self.kb.get(cls, 'deadline')}. The GUET entrance exam is on {guet}.")
        return (f"Applications open on {opens}. Last dates: UG - {self.kb.get('UGProgram', 'deadline')}; "
                f"PG - {self.kb.get('PGProgram', 'deadline')}; Ph.D - {self.kb.get('ResearchProgram', 'deadline')}. "
                f"The GUET entrance exam is on {guet}.")

    def documents_required(self, level: str | None) -> str:
        cls = {"PG": "PGProgram", "Research": "ResearchProgram"}.get(level, "UGProgram")
        docs = self.kb.get(cls, "documents")
        lab = {"PG": "PG", "Research": "Ph.D"}.get(level, "UG")
        return f"Documents required for {lab} admission:\n" + "\n".join(f"  - {d}" for d in docs)

    # --------------------------------------------------------- facilities
    def hostel(self) -> str:
        h = self.kb.resolved("hostel")
        return (f"{h['description']} Rooms: {h['room_types']}. The fee is {rs(h['fee_per_year'])} per year, "
                f"which includes {h['includes']}. AC rooms cost {rs(h['ac_extra'])} extra per year. {h['timings']}.")

    def facility(self, fid: str) -> str:
        if fid == "hostel":
            return self.hostel()
        f = self.kb.resolved(fid)
        parts = [f"{f['name']}: {f['description']}"]
        if "timings" in f:
            parts.append(f"Timings: {f['timings']}.")
        if "fee_per_year" in f:
            parts.append(f"Fee: {rs(f['fee_per_year'])} per year.")
        return " ".join(parts)

    def facilities_overview(self) -> str:
        return "Campus facilities:\n" + "\n".join(
            f"  - {self.kb.get(f, 'name')}: {self.kb.get(f, 'description')}"
            for f in self.kb.instances_of("Facility"))

    # ------------------------------------------------- scholarships / money
    def scholarship_info(self) -> str:
        s = self.kb.resolved("scholarship_policy")
        tiers = "; ".join(f"{t['min']}% and above - {t['waiver']}% waiver" for t in s["merit"])
        return (f"Scholarships (on tuition fee): merit - {tiers}. Sports quota (state/national level) - "
                f"{s['sports_waiver']}%. Sibling concession - {s['sibling_waiver']}%. Waivers can be combined up to "
                f"{s['max_waiver']}%. {s['government']} M.Tech students with GATE get a stipend of Rs. 12,400 per month.")

    # --------------------------------------------------------- placements
    def placement_years(self) -> list[str]:
        """Seasons with placement figures, oldest first (the 'placement' slot is the latest season)."""
        latest = self.kb.get("University", "placement_year")
        older = sorted(next((self.kb.get(d, "placement_history") for d in self.kb.instances_of("Department")
                             if self.kb.get(d, "placement_history")), {}))
        return older + [latest]

    def _placement_of(self, dept: str, year: str) -> dict | None:
        if year == self.kb.get("University", "placement_year"):
            return self.kb.get(dept, "placement")
        return (self.kb.get(dept, "placement_history") or {}).get(year)

    def placement(self, dept: str | None, year: str | None = None) -> str:
        years = self.placement_years()
        note = ""
        if year and year not in years:
            note = (f"Placement figures for {year} are not available; I have them for {years[0]} to {years[-1]}. "
                    f"Here are the latest ({years[-1]}):\n")
            year = None
        year = year or years[-1]
        if dept and self._placement_of(dept, year):
            p = self._placement_of(dept, year)
            recruiters = self.kb.get(dept, "placement")["top_recruiters"]
            return note + (f"Placements {year}, {self.kb.get(dept, 'name')}: {p['percent_placed']}% of students placed, "
                           f"average package {p['average_lpa']} LPA, highest {p['highest_lpa']} LPA. "
                           f"Top recruiters: {', '.join(recruiters)}.")
        if dept:
            return (f"Placement figures for the {self.kb.get(dept, 'name')} are not published yet. For details, please "
                    f"contact the admissions office at {self.kb.get('University', 'phone')}.")
        lines = [note + f"Placements {year} (percentage placed / average / highest):"]
        for d in self.kb.instances_of("Department"):
            p = self._placement_of(d, year)
            if p:
                lines.append(f"  - {self.kb.get(d, 'name')}: {p['percent_placed']}% / {p['average_lpa']} LPA / "
                             f"{p['highest_lpa']} LPA")
        return "\n".join(lines)

    # -------------------------------------------------------- departments
    def department_info(self, dept: str) -> str:
        d = self.kb.resolved(dept)
        progs = ", ".join(self.name(p) for p in self.kb.programs_of_department(dept))
        return (f"{d['name']} - Head of Department: {d['hod']}. Located in {d['block']}. "
                f"Email: {d['dept_email']}." + (f" Programmes offered: {progs}." if progs else ""))

    def department_of_program(self, pid: str) -> str:
        dept = self.kb.get(pid, "department")
        return f"{self.name(pid)} is offered by the {self.kb.get(dept, 'name')} (HOD: {self.kb.get(dept, 'hod')})."

    # ---------------------------------------------------- university life
    # Each answer picks the parts of its frame that the question asks about (by its key words).
    def academic_calendar(self, words: set[str]) -> str:
        c = self.kb.resolved("academic_calendar")
        parts = []
        if "holiday" in words:
            parts.append(f"University holidays: {c['holidays']}.")
        if "vacation" in words or "summer" in words or "winter" in words:
            parts.append(f"Winter vacation: {c['winter_vacation']}. Summer vacation: {c['summer_vacation']}.")
        if {"timing", "time", "hour", "working", "saturday"} & words:
            parts.append(f"Classes run from {c['class_timings']}. Working days: {c['working_days']}.")
        if not parts:
            parts = [f"Academic calendar 2026-27: odd semester - {c['odd_semester']}; winter vacation {c['winter_vacation']}; "
                     f"even semester - {c['even_semester']}; summer vacation {c['summer_vacation']}."]
        return " ".join(parts)

    def examination(self, words: set[str]) -> str:
        e, c = self.kb.resolved("examinations"), self.kb.resolved("academic_calendar")
        parts = []
        if {"hall", "ticket"} & words:
            parts.append(f"{e['hall_ticket']}.")
        if {"grade", "grading", "cgpa", "gpa"} & words:
            grades = ", ".join(f"{g} ({r}) = {p}" for g, r, p in e["grades"])
            parts.append(f"Grades and grade points: {grades}. {e['cgpa']}.")
        if {"pass", "internal", "pattern", "mark", "assessment"} & words:
            parts.append(f"{e['pattern']}. {e['pass_rule']}.")
        if {"timetable", "schedule", "when", "date", "start"} & words or not parts:
            # the exam dates live in the academic-calendar frame (a link in the semantic network)
            dates = [c[s].split(", ")[1].replace("end-semester exams ", "") for s in ("odd_semester", "even_semester")]
            parts.append(f"End-semester exams: odd semester {dates[0]}; even semester {dates[1]}. {e['timetable']}.")
        office = self.kb.get(e["office"], "name")
        return " ".join(parts) + f" For anything else about exams, contact the {office}."

    def results(self, words: set[str]) -> str:
        e = self.kb.resolved("examinations")
        parts = []
        if {"revaluation", "photocopy", "recheck", "rechecking", "retotal"} & words:
            parts.append(f"{e['revaluation']}.")
        if {"supplementary", "arrear", "fail", "reappear", "backlog"} & words:
            parts.append(f"{e['supplementary']}.")
        if not parts:
            parts = [f"{e['results']}. {e['revaluation']}."]
        return " ".join(parts) + f" Results are handled by the {self.kb.get(e['office'], 'name')}."

    def fee_payment(self, words: set[str]) -> str:
        f = self.kb.resolved("fee_payment")
        parts = []
        if "late" in words:
            parts.append(f"Late fee: {f['late_fee']}.")
        if {"instalment", "installment"} & words:
            parts.append(f"{f['instalments']}.")
        if "receipt" in words:
            parts.append(f"{f['receipt']}.")
        if not parts or {"due", "date", "last", "deadline", "when"} & words:
            parts.insert(0, f"Semester fees are due by {f['due_dates']}.")
        if len(parts) == 1 and not {"late", "receipt", "instalment", "installment"} & words:
            parts.append(f"You can pay {f['modes']}. Late fee: {f['late_fee']}.")
        return " ".join(parts) + f" Questions about payments go to the {self.kb.get(f['office'], 'name')}."

    def certificate(self, cid: str | None) -> str:
        if not cid:
            items = "; ".join(f"{self.kb.get(c, 'name')} - {self._cert_fee(c)}, {self.kb.get(c, 'time')}"
                              for c in self.kb.instances_of("Certificate"))
            return f"Certificates you can request: {items}. {self.kb.get('Certificate', 'how')}."
        c = self.kb.resolved(cid)
        text = (f"{c['name']}: {c['how']}. Fee: {self._cert_fee(cid)}. It takes {c['time']}. "
                f"It is issued by the {self.kb.get(c['office'], 'name')}.")
        return text + (f" {c['note']}" if "note" in c else "")

    def _cert_fee(self, cid: str) -> str:
        fee = self.kb.get(cid, "fee")
        return rs(fee) if fee else "free"

    def office(self, oid: str | None) -> str:
        if not oid:
            return "University offices:\n" + "\n".join(
                f"  - {self.kb.get(o, 'name')}: {self.kb.get(o, 'head')}, {self.kb.get(o, 'phone')} - handles "
                f"{self.kb.get(o, 'handles')}" for o in self.kb.instances_of("Office"))
        o = self.kb.resolved(oid)
        return (f"{o['name']} - {o['head']}. Location: {o['location']}. Phone: {o['phone']}, e-mail: {o['email']}. "
                f"Office hours: {o['office_hours']}. It handles {o['handles']}.")

    def vice_chancellor(self) -> str:
        u = self.kb.frames["University"]
        return (f"The Vice-Chancellor of {u['name']} is {u['vice_chancellor']}. "
                f"The Registrar is {self.kb.get('office_registrar', 'head').split(' (')[0]}.")

    def faculty_research(self, words: set[str]) -> str:
        u, r = self.kb.frames["University"], self.kb.resolved("research")
        faculty = (f"{u['name']} has {u['faculty']} The student-teacher ratio is {u['student_teacher_ratio']}.")
        research = (f"Research: {r['centres']}. There are {r['projects']}, and {r['output']}. "
                    f"Ph.D. admissions are open in all departments.")
        if {"research", "patent", "publication", "project", "centre", "paper", "funded"} & words:
            return research if not {"faculty", "professor", "teacher", "staff"} & words else faculty + " " + research
        return faculty if {"faculty", "professor", "teacher", "staff", "ratio"} & words else faculty + " " + research

    def events(self, words: set[str]) -> str:
        e = self.kb.resolved("events")
        picks = [("convocation", f"The convocation is held {e['convocation']}."),
                 ("tech", f"Technical fest: {e['tech_fest']}."), ("technical", f"Technical fest: {e['tech_fest']}."),
                 ("cultural", f"Cultural fest: {e['cultural_fest']}."), ("sport", f"Sports: {e['sports_meet']}.")]
        parts = list(dict.fromkeys(text for w, text in picks if w in words))
        if not parts:
            parts = [f"Main events: {e['tech_fest']}; {e['cultural_fest']}; {e['sports_meet']}; and the convocation "
                     f"{e['convocation']}. {e['other']}."]
        return " ".join(parts) + f" Events are organised by the {self.kb.get(e['office'], 'name')}."

    def alumni(self) -> str:
        a = self.kb.resolved("alumni")
        return (f"Our graduates are part of {a['association']}. {a['benefits']}. They meet at {a['meet']}. "
                f"Contact: {a['contact']}.")

    # -------------------------------------------------------------- misc
    def contact(self) -> str:
        u = self.kb.frames["University"]
        return (f"Admissions office, {u['name']}, {u['location']}. Phone: {u['phone']}. Email: {u['email']}. "
                f"Website: {u['website']}. Office hours: {u['office_hours']}.")

    def student_strength(self) -> str:
        u = self.kb.frames["University"]
        return (f"{u['name']} has {u['students']}, across {len(self.kb.programs)} programmes in "
                f"{len(self.kb.instances_of('Department'))} departments.")

    def about(self) -> str:
        u = self.kb.frames["University"]
        return (f"{u['name']} is located in {u['location']}. It was established in {u['established']} and is "
                f"accredited {u['accreditation']}. It offers {len(self.kb.programs)} programmes across UG, PG and "
                f"research levels, and has {u['students']}.")
