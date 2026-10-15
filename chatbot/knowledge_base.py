"""Frame-based knowledge representation with inheritance (a small semantic network).

Every entity (programme, department, facility, policy) is a *frame*: a set of
slot -> value pairs. Frames are linked with ``is_a`` edges, and a slot that is
not stored on a frame is inherited from its ancestors. For example
``btech_cse`` does not store ``duration``; it inherits "4 years" from
``UGProgram``.

The same data is also exposed as (subject, predicate, object) triples so it can
be queried like a semantic network ("which programmes does dept_cse offer?").
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


class KnowledgeBase:
    def __init__(self, path: str | Path | None = None):
        path = Path(path) if path else DATA_DIR / "knowledge_base.json"
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        self.frames: dict[str, dict[str, Any]] = raw["frames"]
        self.program_aliases: dict[str, list[str]] = raw["program_aliases"]
        self.department_aliases: dict[str, list[str]] = raw["department_aliases"]

    # ------------------------------------------------------------------ frames
    def ancestors(self, frame_id: str) -> list[str]:
        """Return the is_a chain, nearest parent first."""
        chain, cur = [], self.frames.get(frame_id, {}).get("is_a")
        while cur:
            chain.append(cur)
            cur = self.frames.get(cur, {}).get("is_a")
        return chain

    def get(self, frame_id: str, slot: str, default: Any = None) -> Any:
        """Slot lookup with inheritance along is_a links."""
        for fid in [frame_id, *self.ancestors(frame_id)]:
            frame = self.frames.get(fid, {})
            if slot in frame:
                return frame[slot]
        return default

    def get_with_source(self, frame_id: str, slot: str) -> tuple[Any, str | None]:
        """Like get(), but also returns which frame supplied the value (for explanations)."""
        for fid in [frame_id, *self.ancestors(frame_id)]:
            if slot in self.frames.get(fid, {}):
                return self.frames[fid][slot], fid
        return None, None

    def resolved(self, frame_id: str) -> dict[str, Any]:
        """All slots of a frame including inherited ones (child overrides parent)."""
        out: dict[str, Any] = {}
        for fid in reversed([frame_id, *self.ancestors(frame_id)]):
            out.update({k: v for k, v in self.frames[fid].items() if k not in ("is_a", "type")})
        return out

    def instances_of(self, class_id: str) -> list[str]:
        """All instance frames that inherit (directly or indirectly) from class_id."""
        return [fid for fid, fr in self.frames.items()
                if fr.get("type") == "instance" and class_id in self.ancestors(fid)]

    def is_a(self, frame_id: str, class_id: str) -> bool:
        return class_id in self.ancestors(frame_id)

    # ---------------------------------------------------------------- programs
    @property
    def programs(self) -> list[str]:
        return self.instances_of("Program")

    def program_level(self, pid: str) -> str:
        return self.get(pid, "level")

    def programs_of_department(self, dept_id: str) -> list[str]:
        return [p for p in self.programs if self.get(p, "department") == dept_id]

    # ----------------------------------------------------------------- triples
    def triples(self) -> Iterable[tuple[str, str, Any]]:
        """Expose the frames as semantic-network triples (subject, predicate, object)."""
        for fid, frame in self.frames.items():
            for slot, val in frame.items():
                if slot.startswith("_") or slot == "type":
                    continue
                yield (fid, slot, val)
        # derived inverse relation: department --offers--> programme
        for p in self.programs:
            yield (self.get(p, "department"), "offers", p)

    def query(self, subject: str | None = None, predicate: str | None = None,
              obj: Any = None) -> list[tuple[str, str, Any]]:
        """Pattern match over triples; None acts as a wildcard."""
        return [t for t in self.triples()
                if (subject is None or t[0] == subject)
                and (predicate is None or t[1] == predicate)
                and (obj is None or t[2] == obj)]
