"""
context/memory.py: three ways to remember a conversation under a budget.

The API is stateless: "memory" is just the message list you choose to resend each
turn. But you can't resend *everything* forever. Eventually the conversation
won't fit the window (or you don't want to pay to send 50 turns every time). So
"memory" is really a policy: **what do you keep, and what do you drop?** This file
implements the three policies that matter, behind one interface:

    mem.add(role, content)      # record a turn
    system, messages = mem.build()   # what to actually SEND this turn

  - FullMemory     keep every turn. Simple, correct, and unbounded, so it works
                     until it doesn't fit (or gets expensive).
  - WindowMemory   keep the most recent turns that fit a token budget; the
                     oldest fall off. Bounded, but it genuinely *forgets* old
                     facts (the failure the other dives' "simple trim" has).
  - SummaryMemory  when the budget is exceeded, replace the oldest turns with a
                     running **summary** and keep the recent turns verbatim. Bounded
                     *and* it preserves old facts. This is "compaction."

The summary lives in the system prompt (safe on both providers), so `build()`
always returns a (system, messages) pair you hand straight to `providers.generate`.
"""

from __future__ import annotations

from . import tokens
from .providers import summarize as _summarize


class FullMemory:
    """Keep the entire conversation. The baseline: correct but unbounded."""

    def __init__(self, system: str = ""):
        self.system = system
        self.turns: list[dict] = []

    def add(self, role: str, content: str) -> None:
        self.turns.append({"role": role, "content": content})

    def build(self) -> tuple[str, list[dict]]:
        return self.system, list(self.turns)

    def info(self) -> dict:
        system, msgs = self.build()
        return {"strategy": "full", "turns_sent": len(msgs), "tokens": tokens.estimate(system) + tokens.estimate_messages(msgs)}


class WindowMemory:
    """Keep the most recent turns that fit `budget_tokens`; drop the oldest.

    Bounded and cheap, but anything that scrolls off the window is *gone*. Use it
    when only recent context matters, or pair it with long-term memory for recall.
    """

    def __init__(self, budget_tokens: int, system: str = ""):
        self.budget = budget_tokens
        self.system = system
        self.turns: list[dict] = []

    def add(self, role: str, content: str) -> None:
        self.turns.append({"role": role, "content": content})

    def build(self) -> tuple[str, list[dict]]:
        kept: list[dict] = []
        total = tokens.estimate(self.system)
        for msg in reversed(self.turns):  # newest first
            cost = tokens.estimate_message(msg)
            if kept and total + cost > self.budget:
                break
            total += cost
            kept.append(msg)
        kept.reverse()
        return self.system, kept

    def info(self) -> dict:
        system, msgs = self.build()
        return {
            "strategy": "window",
            "turns_sent": len(msgs),
            "turns_dropped": len(self.turns) - len(msgs),
            "tokens": tokens.estimate(system) + tokens.estimate_messages(msgs),
            "budget": self.budget,
        }


class SummaryMemory:
    """Compaction: fold the oldest turns into a running summary when over budget.

    Keeps the most recent `keep_recent` turns verbatim and everything older as a
    compact summary in the system prompt. Bounded like a window, but old *facts*
    survive, because they're carried in the summary, not the raw turns.
    """

    def __init__(self, budget_tokens: int, system: str = "", *, keep_recent: int = 2, summarizer=_summarize):
        self.budget = budget_tokens
        self.system = system
        self.keep_recent = keep_recent
        self.summarizer = summarizer
        self.turns: list[dict] = []
        self.summary: str = ""
        self.compactions = 0

    def add(self, role: str, content: str) -> None:
        self.turns.append({"role": role, "content": content})
        self._maybe_compact()

    def _summary_block(self) -> str:
        return f"Earlier conversation summary:\n{self.summary}" if self.summary else ""

    def build(self) -> tuple[str, list[dict]]:
        system = self.system
        if self.summary:
            system = (system + "\n\n" + self._summary_block()).strip()
        return system, list(self.turns)

    def _maybe_compact(self) -> None:
        system, msgs = self.build()
        if tokens.fits(msgs, self.budget, system=system):
            return
        if len(self.turns) <= self.keep_recent:
            return  # already minimal; can't shrink the recent window further
        # Peel everything older than `keep_recent` and fold it into the summary.
        old = self.turns[: -self.keep_recent]
        self.turns = self.turns[-self.keep_recent :]
        to_summarize: list[dict] = []
        if self.summary:
            to_summarize.append({"role": "system", "content": self._summary_block()})
        to_summarize.extend(old)
        self.summary = self.summarizer(to_summarize)
        self.compactions += 1
        self._maybe_compact()  # may still be over budget after one pass

    def info(self) -> dict:
        system, msgs = self.build()
        return {
            "strategy": "summary",
            "turns_sent": len(msgs),
            "compactions": self.compactions,
            "has_summary": bool(self.summary),
            "tokens": tokens.estimate(system) + tokens.estimate_messages(msgs),
            "budget": self.budget,
        }
