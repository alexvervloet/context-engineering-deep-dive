"""
context/assemble.py — fit the right things in the window, in the right order.
=============================================================================

A real request's context is assembled from competing sources: the system prompt,
tool definitions, retrieved documents, long-term memory, and the recent
conversation. They rarely all fit. Context assembly is the discipline of packing
the window deliberately:

  1. PRIORITIZE — when it won't all fit, drop the least important sections first
     (never the system prompt; usually keep recent turns over old documents).
  2. ORDER — position matters. Models attend most reliably to the START and END of
     a long context and can miss things buried in the MIDDLE (the "lost in the
     middle" effect). Put the most important material at the edges.

`assemble()` does the budgeting; `order_for_recall()` does the positioning. Both
are pure functions over (label, text, priority) sections — no model, no key.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import tokens


@dataclass
class Section:
    """One piece of candidate context. Higher `priority` = keep it first."""

    label: str
    text: str
    priority: int


@dataclass
class Assembled:
    kept: list[Section]
    dropped: list[Section]
    tokens_used: int
    budget: int

    def text(self) -> str:
        return "\n\n".join(f"## {s.label}\n{s.text}" for s in self.kept)


def assemble(sections: list[Section], budget_tokens: int) -> Assembled:
    """Greedily keep the highest-priority sections that fit `budget_tokens`.

    Ties keep the earlier-listed section. A section that doesn't fit is dropped and
    we keep trying lower-priority (smaller) ones — so a giant low-priority doc can't
    starve several small high-priority ones.
    """
    ordered = sorted(enumerate(sections), key=lambda p: (-p[1].priority, p[0]))
    kept: list[Section] = []
    dropped: list[Section] = []
    used = 0
    for _, section in ordered:
        cost = tokens.estimate(section.text)
        if used + cost <= budget_tokens:
            kept.append(section)
            used += cost
        else:
            dropped.append(section)
    return Assembled(kept=kept, dropped=dropped, tokens_used=used, budget=budget_tokens)


def order_for_recall(sections: list[Section]) -> list[Section]:
    """Reorder so the highest-priority sections sit at the edges, not the middle.

    Models recall the start and end of a long context most reliably. We place the
    top priorities first and last, and bury the lowest in the middle — the opposite
    of the naive "just concatenate" order.
    """
    by_priority = sorted(sections, key=lambda s: s.priority, reverse=True)
    front: list[Section] = []
    back: list[Section] = []
    for i, section in enumerate(by_priority):
        (front if i % 2 == 0 else back).append(section)
    back.reverse()
    return front + back
