"""
context/tokens.py — a budget you can do in your head.
=====================================================

Context engineering starts with arithmetic: *will this fit?* Every model has a
context window — a hard cap on how many tokens (input + output) it can consider at
once. Overflow it and the request fails; crowd it and quality drops and cost
climbs. So before you decide what to keep, you need a rough token count.

We estimate with a simple, provider-neutral rule — **~4 characters per token** for
English — plus a small per-message overhead the chat format adds. This is an
*approximation* on purpose: it needs no key, no network, and no tokenizer
download, and it's accurate enough to reason about budgets. For a real bill, trust
the `usage` field the API returns (the OpenAI/Claude dives show this). For
*deciding what to put in the window*, this is all you need.
"""

from __future__ import annotations

# Rough English heuristic: ~4 characters per token.
_CHARS_PER_TOKEN = 4
# The chat format wraps each message with role markers / delimiters; approximate
# that fixed cost so a 50-turn history isn't undercounted.
_PER_MESSAGE_OVERHEAD = 4

# A few representative context windows, for sizing demos. Real numbers move; see
# MODELS.md in the series. The point is the *ratio* to your conversation, not the
# exact figure.
CONTEXT_WINDOWS = {
    "gpt-4o-mini": 128_000,
    "claude-haiku-4-5": 200_000,
    "small-demo": 2_000,  # a deliberately tiny window so examples overflow quickly
}


def estimate(text: str) -> int:
    """Estimate the tokens in a string (~4 chars/token, minimum 1 for non-empty)."""
    if not text:
        return 0
    return max(1, len(text) // _CHARS_PER_TOKEN)


def estimate_message(message: dict) -> int:
    """Estimate one chat message's tokens, including the format's per-message overhead."""
    return estimate(message.get("content", "")) + _PER_MESSAGE_OVERHEAD


def estimate_messages(messages: list[dict]) -> int:
    """Estimate a whole message list's tokens."""
    return sum(estimate_message(m) for m in messages)


def fits(messages: list[dict], budget_tokens: int, *, system: str = "") -> bool:
    """Does system + messages fit inside `budget_tokens`?"""
    return estimate(system) + estimate_messages(messages) <= budget_tokens
