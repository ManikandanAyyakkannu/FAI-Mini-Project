// Forward-chaining production-rule inference engine.
//
// Working memory is an object of facts. Each rule (from data/rules.json) has a list
// of conditions (IF) and a list of actions (THEN). The engine scans the rules in
// order and fires every rule whose conditions hold and that has not fired before
// (refraction), until no rule can fire. Every firing is recorded so the chatbot
// can explain *why* it reached its conclusion.

import { rs } from "./answers.js";
import { qualificationLabel } from "./nlp.js";

const BACHELORS = ["btech", "bsc", "bca", "bcom", "ba", "bba"];
const MASTERS = ["mtech", "msc", "mca", "mba", "ma", "me"];

// does the applicant's qualification (e.g. "btech_ece") satisfy any allowed entry?
export function qualificationMatches(q, allowed) {
  const degree = q.split("_")[0];
  for (const a of allowed) {
    if (a === q || a === degree) return true;
    if (a === "any_degree" && (BACHELORS.includes(degree) || MASTERS.includes(degree))) return true;
    // branch not given ("btech") but some branch of that degree is allowed: benefit of the doubt
    if (!q.includes("_") && a.startsWith(degree + "_")) return true;
  }
  return false;
}

const present = (wm, fact) => wm[fact] !== undefined && wm[fact] !== null;
const truthy = v => !(v === undefined || v === null || v === false || v === 0 || v === "" ||
                      (Array.isArray(v) && v.length === 0));
const contains = (list, v) => (Array.isArray(list) ? list.includes(v) : String(list).includes(v));

function test(cond, wm) {
  const { fact, op } = cond;
  const val = wm[fact];
  const target = "value_fact" in cond ? wm[cond.value_fact] : cond.value;
  const has = present(wm, fact);
  const hasTarget = target !== undefined && target !== null;
  switch (op) {
    case "exists": return has;
    case "not_exists": return !has;
    case "non_empty": return truthy(val);
    case "empty": return has && !truthy(val);
  }
  if (!has) return false;
  switch (op) {
    case "eq": return val === target;
    case "ne": return val !== target;
    case "in": return contains(target, val);
    case "in_fact": return hasTarget && contains(target, val);
    case "ge": case "ge_fact": return hasTarget && val >= target;
    case "lt": case "lt_fact": return hasTarget && val < target;
    case "gt": return val > target;
    case "contains_all": return hasTarget && target.every(t => val.includes(t));
    case "not_contains_all": return hasTarget && !target.every(t => val.includes(t));
    case "qual_in": return hasTarget && qualificationMatches(val, target);
    case "qual_not_in": return hasTarget && !qualificationMatches(val, target);
  }
  throw new Error(`Unknown operator ${op}`);
}

function apply(action, wm) {
  if ("append" in action) {
    (wm[action.append] ||= []).push(action.value);
    return;
  }
  const key = action.set;
  if ("value" in action) wm[key] = action.value;
  else if ("from" in action) wm[key] = wm[action.from];
  else if ("mul" in action) wm[key] = Math.round(wm[action.mul[0]] * action.mul[1] * 100) / 100;
  else if ("add" in action) wm[key] = wm[action.add[0]] + action.add[1];
  else if ("net" in action) wm[key] = Math.round(wm[action.net[0]] * (100 - wm[action.net[1]]) / 100);
}

// Fills a rule's explanation template, e.g. "Converted CGPA {cgpa} to {percentage:.1f}%".
// {x_upper} gives the fact in capitals and {x_rs} formats it as rupees.
const DECIMAL_FACTS = new Set(["cgpa", "percentage"]);      // always shown with a decimal point

// toFixed(), but an exact tie rounds to the even digit (71.25 -> "71.2"), like Python does
function toFixedEven(value, digits) {
  const [whole, frac = ""] = Math.abs(value).toPrecision(20).split(".");
  if (value >= 0 && /^50*$/.test(frac.slice(digits))) {
    const kept = digits ? `${whole}.${frac.slice(0, digits)}` : whole;
    if (Number(kept.at(-1)) % 2 === 0) return kept;
  }
  return value.toFixed(digits);
}

function fillTemplate(template, wm) {
  return template.replace(/\{(\w+)(?::\.(\d)f)?\}/g, (whole, name, digits) => {
    let key = name, value;
    if (name.endsWith("_upper") && typeof wm[name.slice(0, -6)] === "string") {
      value = wm[name.slice(0, -6)].toUpperCase();
    } else if (name.endsWith("_rs") && typeof wm[name.slice(0, -3)] === "number") {
      value = rs(wm[name.slice(0, -3)]);
    } else if (name in wm) {
      value = wm[name];
    } else {
      return `{${name}}`;
    }
    if (typeof value === "number") {
      if (digits !== undefined) return toFixedEven(value, Number(digits));
      if (DECIMAL_FACTS.has(key) && Number.isInteger(value)) return value.toFixed(1);
    }
    return String(value);
  });
}

export class ForwardChainingEngine {
  constructor(rules) {
    this.rules = rules;
  }

  run(facts, maxCycles = 20) {
    const wm = { ...facts };
    const firedIds = new Set();
    const fired = [];
    for (let cycle = 0; cycle < maxCycles; cycle++) {
      let firedThisCycle = false;
      for (const rule of this.rules) {
        if (firedIds.has(rule.id)) continue;
        if (rule.if.every(c => test(c, wm))) {
          rule.then.forEach(a => apply(a, wm));
          firedIds.add(rule.id);
          firedThisCycle = true;
          fired.push({ id: rule.id, name: rule.name, explanation: fillTemplate(rule.explain || rule.name, wm) });
        }
      }
      if (!firedThisCycle) break;
    }
    return { facts: wm, fired, lines: fired.map(r => `[${r.id}] ${r.explanation}`) };
  }
}

// builds working memory from the frame KB + the applicant's facts and runs a rule set
export class ReasoningEngine {
  constructor(kb, rules) {
    this.kb = kb;
    this.eligibilityEngine = new ForwardChainingEngine(rules.eligibility_rules);
    this.scholarshipEngine = new ForwardChainingEngine(rules.scholarship_rules);
  }

  programFacts(pid) {
    const kb = this.kb;
    const exams = kb.get(pid, "entrance_exams", []);
    return {
      program: pid,
      program_name: kb.get(pid, "name"),
      min_percentage: kb.get(pid, "min_percentage"),
      allowed_qualifications: kb.get(pid, "allowed_qualifications", []),
      required_subjects: kb.get(pid, "required_subjects", []),
      entrance_exams_lower: exams.map(e => e.toLowerCase().split(" ")[0].split("-")[0]),
      entrance_list: exams.join(", "),
      gate_stipend: kb.get(pid, "gate_stipend"),
      fee_per_year: kb.get(pid, "fee_per_year"),
    };
  }

  static userFacts(applicant) {
    const facts = keepGiven(applicant);
    if ("qualification" in facts) facts.qualification_label = qualificationLabel(facts.qualification);
    return facts;
  }

  checkEligibility(pid, applicant) {
    return this.eligibilityEngine.run({ ...this.programFacts(pid), ...ReasoningEngine.userFacts(applicant) });
  }

  computeFee(pid, applicant) {
    return this.scholarshipEngine.run({ ...this.programFacts(pid), ...ReasoningEngine.userFacts(applicant) });
  }
}

// drops facts that were not given (null / false / 0)
export function keepGiven(obj) {
  const out = {};
  for (const [k, v] of Object.entries(obj)) {
    if (v !== null && v !== undefined && v !== false && v !== 0) out[k] = v;
  }
  return out;
}
