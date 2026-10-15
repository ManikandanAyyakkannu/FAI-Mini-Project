"""Forward-chaining production-rule inference engine.

Working memory is a dict of facts. Rules (loaded from data/rules.json) have a
list of conditions (IF) and a list of actions (THEN). The engine repeatedly
scans the rule base in order and fires every rule whose conditions hold and
which has not fired before (refraction), until no rule can fire. Each firing is
recorded so the chatbot can explain *why* it reached its conclusion.
"""
from __future__ import annotations

import json
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .answers import rs
from .knowledge_base import DATA_DIR

BACHELORS = {"btech", "bsc", "bca", "bcom", "ba", "bba"}
MASTERS = {"mtech", "msc", "mca", "mba", "ma", "me"}


def qualification_matches(q: str, allowed: list[str]) -> bool:
    """Does the applicant's qualification q (e.g. 'btech_ece') satisfy any allowed entry?"""
    degree = q.split("_")[0]
    for a in allowed:
        if a == q or a == degree:
            return True
        if a == "any_degree" and degree in BACHELORS | MASTERS:
            return True
        # branch not given ("btech") but some branch of that degree is allowed -> benefit of doubt
        if "_" not in q and a.startswith(degree + "_"):
            return True
    return False


class _SafeDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"


class _Formatter(string.Formatter):
    def format_field(self, value, spec):
        try:
            return super().format_field(value, spec)
        except (ValueError, TypeError):
            return str(value)


_FMT = _Formatter()


@dataclass
class FiredRule:
    rule_id: str
    name: str
    explanation: str


@dataclass
class InferenceResult:
    facts: dict[str, Any]
    fired: list[FiredRule] = field(default_factory=list)

    def explanation_lines(self) -> list[str]:
        return [f"[{r.rule_id}] {r.explanation}" for r in self.fired]


class ForwardChainingEngine:
    def __init__(self, rules: list[dict]):
        self.rules = rules

    # ------------------------------------------------------------ conditions
    @staticmethod
    def _test(cond: dict, wm: dict) -> bool:
        fact, op = cond["fact"], cond["op"]
        present = fact in wm and wm[fact] is not None
        val = wm.get(fact)
        target = wm.get(cond["value_fact"]) if "value_fact" in cond else cond.get("value")
        if op == "exists":
            return present
        if op == "not_exists":
            return not present
        if op == "non_empty":
            return bool(val)
        if op == "empty":
            return present and not val
        if not present:
            return False
        if op == "eq":
            return val == target
        if op == "ne":
            return val != target
        if op == "in":
            return val in target
        if op == "in_fact":
            return target is not None and val in target
        if op in ("ge", "ge_fact"):
            return target is not None and val >= target
        if op in ("lt", "lt_fact"):
            return target is not None and val < target
        if op == "gt":
            return val > target
        if op == "contains_all":
            return target is not None and all(t in val for t in target)
        if op == "not_contains_all":
            return target is not None and not all(t in val for t in target)
        if op == "qual_in":
            return target is not None and qualification_matches(val, target)
        if op == "qual_not_in":
            return target is not None and not qualification_matches(val, target)
        raise ValueError(f"Unknown operator {op}")

    # --------------------------------------------------------------- actions
    @staticmethod
    def _apply(action: dict, wm: dict) -> None:
        if "append" in action:
            wm.setdefault(action["append"], []).append(action["value"])
            return
        key = action["set"]
        if "value" in action:
            wm[key] = action["value"]
        elif "from" in action:
            wm[key] = wm[action["from"]]
        elif "mul" in action:
            a, b = action["mul"]
            wm[key] = round(wm[a] * b, 2)
        elif "add" in action:
            a, b = action["add"]
            wm[key] = wm[a] + b
        elif "net" in action:
            fee, pct = action["net"]
            wm[key] = int(round(wm[fee] * (100 - wm[pct]) / 100))

    # ------------------------------------------------------------ inference
    def run(self, facts: dict[str, Any], max_cycles: int = 20) -> InferenceResult:
        wm = dict(facts)
        fired_ids: set[str] = set()
        result = InferenceResult(wm)
        for _ in range(max_cycles):
            fired_this_cycle = False
            for rule in self.rules:
                if rule["id"] in fired_ids:
                    continue
                if all(self._test(c, wm) for c in rule["if"]):
                    for act in rule["then"]:
                        self._apply(act, wm)
                    fired_ids.add(rule["id"])
                    fired_this_cycle = True
                    view = _SafeDict(wm)
                    view.update({k + "_upper": str(v).upper() for k, v in wm.items() if isinstance(v, str)})
                    view.update({k + "_rs": rs(v) for k, v in wm.items()
                                 if isinstance(v, (int, float)) and not isinstance(v, bool)})
                    text = _FMT.vformat(rule.get("explain", rule["name"]), (), view)
                    result.fired.append(FiredRule(rule["id"], rule["name"], text))
            if not fired_this_cycle:
                break
        return result


class ReasoningEngine:
    """Builds working memory from the frame KB + user facts and runs the rule sets."""

    def __init__(self, kb, rules_path: str | Path | None = None):
        self.kb = kb
        path = Path(rules_path) if rules_path else DATA_DIR / "rules.json"
        with open(path, encoding="utf-8") as f:
            rules = json.load(f)
        self.eligibility_engine = ForwardChainingEngine(rules["eligibility_rules"])
        self.scholarship_engine = ForwardChainingEngine(rules["scholarship_rules"])

    def _program_facts(self, pid: str) -> dict[str, Any]:
        exams = self.kb.get(pid, "entrance_exams", [])
        return {
            "program": pid,
            "program_name": self.kb.get(pid, "name"),
            "min_percentage": self.kb.get(pid, "min_percentage"),
            "allowed_qualifications": self.kb.get(pid, "allowed_qualifications", []),
            "required_subjects": self.kb.get(pid, "required_subjects", []),
            "entrance_exams_lower": [e.lower().split()[0].split("-")[0] for e in exams],
            "entrance_list": ", ".join(exams),
            "gate_stipend": self.kb.get(pid, "gate_stipend"),
            "fee_per_year": self.kb.get(pid, "fee_per_year"),
        }

    @staticmethod
    def _user_facts(applicant: dict[str, Any]) -> dict[str, Any]:
        from .nlp import qualification_label
        facts = {k: v for k, v in applicant.items() if v not in (None, False)}
        if "qualification" in facts:
            facts["qualification_label"] = qualification_label(facts["qualification"])
        return facts

    def check_eligibility(self, pid: str, applicant: dict[str, Any]) -> InferenceResult:
        facts = self._program_facts(pid) | self._user_facts(applicant)
        return self.eligibility_engine.run(facts)

    def compute_fee(self, pid: str, applicant: dict[str, Any]) -> InferenceResult:
        facts = self._program_facts(pid) | self._user_facts(applicant)
        return self.scholarship_engine.run(facts)
