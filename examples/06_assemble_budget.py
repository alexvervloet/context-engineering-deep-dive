"""
Example 06 — assembling the window under a budget.
==================================================

A real request's context isn't just the chat history. It's the system prompt, tool
definitions, retrieved documents, long-term memory, AND the recent turns — all
competing for the same token budget, and rarely all fitting. Context assembly is
deciding, deliberately, what makes the cut.

The rule: **prioritize, then pack.** Give each candidate section a priority, then
keep the highest-priority sections that fit and drop the rest — so a giant
low-priority document can't crowd out the system prompt and the user's actual
question. `assemble()` does exactly that, offline, with no model.

Run:  python examples/06_assemble_budget.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from context import tokens
from context.assemble import Section, assemble

# The candidate context for one request. `text` is stand-in content sized to be
# realistic; `priority` encodes "how much do I need this if space is tight?"
CANDIDATES = [
    Section("system prompt", "You are Acme Cloud's support assistant. " * 6, 100),
    Section("user question", "How do I get a refund on an annual plan?", 90),
    Section("long-term memory", "The user is Dana, on the Pro plan, prefers metric units. " * 2, 70),
    Section("retrieved: refund policy", "Refunds within 30 days; annual plans are prorated. " * 6, 80),
    Section("retrieved: unrelated FAQ", "How to change your avatar, themes, and notifications. " * 20, 30),
    Section("recent turns", "user: hi\nassistant: hello\nuser: thanks " * 8, 60),
    Section("retrieved: marketing blog", "Ten reasons teams love Acme Cloud, a long post. " * 30, 10),
]


def main() -> None:
    budget = 220
    print(f"Token budget for the window: {budget}\n")

    print("Candidate sections (priority · est. tokens):")
    for s in sorted(CANDIDATES, key=lambda s: s.priority, reverse=True):
        print(f"  p{s.priority:<3} {tokens.estimate(s.text):>4} tok  {s.label}")
    total = sum(tokens.estimate(s.text) for s in CANDIDATES)
    print(f"  {'':4} {total:>4} tok  TOTAL (vs {budget} budget — won't all fit)\n")

    result = assemble(CANDIDATES, budget_tokens=budget)

    print(f"KEPT ({result.tokens_used}/{budget} tokens):")
    for s in result.kept:
        print(f"  ✓ p{s.priority:<3} {s.label}")
    print("DROPPED (didn't fit, lowest priority first):")
    for s in result.dropped:
        print(f"  ✗ p{s.priority:<3} {s.label}")

    print(
        "\nTakeaway: when context competes for space, the priority IS the design. "
        "The\nsystem prompt and the question are non-negotiable; a marketing blog is "
        "the first\nthing to cut. Assembling the window on purpose — instead of "
        "concatenating whatever\nyou have — is the difference between a focused "
        "request and a bloated, costly one."
    )


if __name__ == "__main__":
    main()
