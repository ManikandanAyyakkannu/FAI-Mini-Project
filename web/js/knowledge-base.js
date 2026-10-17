// Frame-based knowledge representation with inheritance (a small semantic network).
//
// Every entity (programme, department, facility, policy) is a frame: a set of
// slot -> value pairs. Frames are linked by "is_a" edges, and a slot that a frame
// does not store is inherited from its parent. Example: btech_cse does not store
// "duration"; it inherits "4 years (8 semesters)" from UGProgram.

export class KnowledgeBase {
  constructor(data) {
    this.frames = data.frames;
    this.programAliases = data.program_aliases;
    this.departmentAliases = data.department_aliases;
    this.officeAliases = data.office_aliases || {};
    this.certificateAliases = data.certificate_aliases || {};
    // closed world: subjects the university does not offer, and names of other institutions
    this.notOffered = data.not_offered || [];
    this.otherInstitutions = data.other_institutions || {};
  }

  // the is_a chain of a frame, nearest parent first
  ancestors(id) {
    const chain = [];
    let cur = this.frames[id]?.is_a;
    while (cur) {
      chain.push(cur);
      cur = this.frames[cur]?.is_a;
    }
    return chain;
  }

  // slot lookup with inheritance
  get(id, slot, fallback = null) {
    for (const fid of [id, ...this.ancestors(id)]) {
      const frame = this.frames[fid] || {};
      if (slot in frame) return frame[slot];
    }
    return fallback;
  }

  // like get(), but also says which frame supplied the value (used in explanations)
  getWithSource(id, slot) {
    for (const fid of [id, ...this.ancestors(id)]) {
      if (slot in (this.frames[fid] || {})) return [this.frames[fid][slot], fid];
    }
    return [null, null];
  }

  // all slots of a frame, including inherited ones (the child overrides the parent)
  resolved(id) {
    const out = {};
    for (const fid of [id, ...this.ancestors(id)].reverse()) {
      for (const [k, v] of Object.entries(this.frames[fid])) {
        if (k !== "is_a" && k !== "type") out[k] = v;
      }
    }
    return out;
  }

  // every instance frame that inherits (directly or indirectly) from a class
  instancesOf(classId) {
    return Object.keys(this.frames).filter(
      id => this.frames[id].type === "instance" && this.ancestors(id).includes(classId));
  }

  get programs() {
    if (!this._programs) this._programs = this.instancesOf("Program");
    return this._programs;
  }

  programLevel(pid) {
    return this.get(pid, "level");
  }

  programsOfDepartment(dept) {
    return this.programs.filter(p => this.get(p, "department") === dept);
  }
}
