"""Builds a flat FAQ corpus (question, answer, key) from the knowledge base.

This corpus is what a typical "FAQ page" would contain. It is used
  * by the baseline retrieval chatbot (the alternative approach), and
  * as the vector-store fallback of the hybrid chatbot.
"""
from __future__ import annotations

import json

from .answers import Answers
from .knowledge_base import DATA_DIR


def build_faq(kb) -> list[dict]:
    A = Answers(kb)
    faq: list[dict] = []

    def add(key, intent, questions, answer):
        for q in questions:
            faq.append({"key": key, "intent": intent, "question": q, "answer": answer})

    add("program_list:all", "program_list", ["What programmes and courses are offered?"], A.program_list())
    add("program_list:UG", "program_list", ["What undergraduate (UG) B.Tech courses are offered?"], A.program_list("UG"))
    add("program_list:PG", "program_list", ["What postgraduate (PG) courses are offered?"], A.program_list("PG"))

    for p in kb.programs:
        n = kb.get(p, "name")
        add(f"program_fee:{p}", "program_fee", [f"What is the fee for {n}?"], A.program_fee(p))
        add(f"program_duration:{p}", "program_duration", [f"What is the duration of {n}?"], A.program_duration(p))
        add(f"program_seats:{p}", "program_seats", [f"How many seats are there in {n}?"], A.program_seats(p))
        add(f"eligibility:{p}", "eligibility", [f"What is the eligibility for {n}?"], A.eligibility_info(p))
        add(f"department_info:{kb.get(p, 'department')}", "department_info",
            [f"Which department offers {n}?"], A.department_of_program(p))

    add("program_fee:group:btech", "program_fee", ["What is the fee for B.Tech courses?"], A.program_fee("group:btech"))
    add("admission_process", "admission_process", ["What is the admission process? How do I apply?"],
        A.admission_process())
    add("admission_deadline:all", "admission_deadline", ["What is the last date to apply for admission?"],
        A.admission_deadline(None))
    for lvl in ("UG", "PG"):
        add(f"admission_deadline:{lvl}", "admission_deadline", [f"What is the {lvl} application deadline?"],
            A.admission_deadline(lvl))
        add(f"documents_required:{lvl}", "documents_required", [f"What documents are required for {lvl} admission?"],
            A.documents_required(lvl))
    add("hostel", "hostel", ["Is hostel accommodation available? What is the hostel fee?"], A.hostel())
    for f in ("library", "transport", "sports", "labs"):
        add(f"facility:{f}", "facility", [f"Tell me about the {kb.get(f, 'name').lower()} facility."], A.facility(f))
    add("scholarship", "scholarship", ["What scholarships are available?"], A.scholarship_info())
    add("placement:all", "placement", ["How are the placements?"], A.placement(None))
    for d in kb.instances_of("Department"):
        dn = kb.get(d, "name")
        add(f"department_info:{d}", "department_info", [f"Who is the HOD of the {dn}?"], A.department_info(d))
        if kb.get(d, "placement"):
            add(f"placement:{d}", "placement", [f"What is the placement record of the {dn}?"], A.placement(d))
    add("contact", "contact", ["How can I contact the admissions office?"], A.contact())
    add("about_university", "about_university", ["Tell me about the university."], A.about())

    # university life beyond admissions
    add("academic_calendar", "academic_calendar", ["What is the academic calendar? When do semesters and vacations start?"],
        A.academic_calendar(set()))
    add("examination", "examination", ["When are the end-semester exams and what is the exam pattern?"], A.examination(set()))
    add("results", "results", ["When are the results published and how do I apply for revaluation?"], A.results(set()))
    add("fee_payment", "fee_payment", ["How and when do I pay the semester fee?"], A.fee_payment(set()))
    for c in kb.instances_of("Certificate"):
        add(f"certificates:{c}", "certificates", [f"How do I get a {kb.get(c, 'name').lower()}?"], A.certificate(c))
    for o in kb.instances_of("Office"):
        add(f"office_info:{o}", "office_info", [f"How can I contact the {kb.get(o, 'name').split(' (')[0]}?"], A.office(o))
    add("faculty_research", "faculty_research", ["How many faculty members are there, and what research is done?"],
        A.faculty_research(set()))
    add("events", "events", ["What is the date of the convocation and the cultural and technical events?"], A.events(set()))
    add("alumni", "alumni", ["Is there an alumni association?"], A.alumni())

    with open(DATA_DIR / "faq_extra.json", encoding="utf-8") as f:
        for item in json.load(f)["faqs"]:
            add(item["key"], "faq", [item["question"]], item["answer"])
    return faq
