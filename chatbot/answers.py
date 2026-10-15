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
        lines = [f"{title} programmes at Greenfield University:"]
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
            crit = (f"a pass in 12th standard with Physics, Chemistry and Mathematics and at least {mp}% aggregate "
                    f"({mp - 5}% for SC/ST/OBC)")
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
    def placement(self, dept: str | None) -> str:
        if dept and self.kb.get(dept, "placement"):
            p = self.kb.get(dept, "placement")
            return (f"Placements 2025-26, {self.kb.get(dept, 'name')}: {p['percent_placed']}% of students placed, "
                    f"average package {p['average_lpa']} LPA, highest {p['highest_lpa']} LPA. "
                    f"Top recruiters: {', '.join(p['top_recruiters'])}.")
        lines = ["Placements 2025-26 (percentage placed / average / highest):"]
        for d in self.kb.instances_of("Department"):
            p = self.kb.get(d, "placement")
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

    # -------------------------------------------------------------- misc
    def contact(self) -> str:
        u = self.kb.frames["University"]
        return (f"Admissions office, {u['name']}, {u['location']}. Phone: {u['phone']}. Email: {u['email']}. "
                f"Website: {u['website']}. Office hours: {u['office_hours']}.")

    def about(self) -> str:
        u = self.kb.frames["University"]
        return (f"{u['name']} is located in {u['location']}. It was established in {u['established']} and is "
                f"accredited {u['accreditation']}. It offers {len(self.kb.programs)} programmes across UG, PG and "
                f"research levels.")
