"""
context/longterm.py — memory that outlives the window (and the session).
========================================================================

A sliding window or a summary keeps a conversation afloat *within* one session.
But some facts should survive forever — a user's name, their preferences, a
decision made last week. You can't hold those in the window indefinitely, and you
shouldn't try. The pattern is **long-term memory**: write durable facts to a store
outside the window, and *retrieve* the relevant few back into the window when a new
turn needs them.

That's RAG (from the RAG deep dive) pointed at the conversation instead of a
document corpus: store facts, embed/index them, retrieve the closest to the current
query. To stay from-scratch and dependency-free we use a tiny bag-of-words overlap
score instead of real embeddings — enough to show the *architecture*. Swap in real
embeddings from the RAG dive and the shape is identical.

The store persists to a JSON file, so it survives across runs — which is exactly
what lets the capstone greet you by name in a brand-new session.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field


def _tokenize(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2}


@dataclass
class LongTermMemory:
    """A persistent fact store with keyword-overlap retrieval."""

    path: str | None = None  # where to persist; None = in-memory only
    facts: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.path and os.path.exists(self.path):
            with open(self.path, encoding="utf-8") as f:
                self.facts = json.load(f)

    def remember(self, fact: str) -> bool:
        """Store a durable fact. Returns False if it's a duplicate."""
        fact = fact.strip()
        if not fact or fact.lower() in (f.lower() for f in self.facts):
            return False
        self.facts.append(fact)
        self._save()
        return True

    def recall(self, query: str, k: int = 3) -> list[str]:
        """Return the `k` stored facts most relevant to `query` (overlap score)."""
        q = _tokenize(query)
        if not q or not self.facts:
            return []
        scored = []
        for fact in self.facts:
            overlap = len(q & _tokenize(fact))
            if overlap:
                scored.append((overlap, fact))
        scored.sort(key=lambda s: s[0], reverse=True)
        return [fact for _, fact in scored[:k]]

    def clear(self) -> None:
        self.facts = []
        self._save()

    def _save(self) -> None:
        if not self.path:
            return
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.facts, f, indent=2)
