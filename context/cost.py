"""
context/cost.py: a tiny prompt-cache cost model (offline, deterministic).

Every other file here asks "what goes in the window?" This one asks the question
that decides your bill: **what does re-sending that window actually cost?**

Providers cache the prompt *prefix*. The rule that governs everything:

    Caching is a PREFIX match. Any change anywhere in the prefix invalidates
    everything after it.

The render order is `system` → `messages`, so the prompt is a list of segments
with the system prompt first. On each request the provider walks that list from
the front and reuses cached segments until the first one that differs; from there
on, every segment is processed fresh and written to the cache. Cached ("read")
tokens are cheap; freshly written tokens carry a premium. We model that with the
published multipliers:

    cache READ  ≈ 0.1×  base input price   (a hit, nearly free)
    cache WRITE ≈ 1.25× base input price   (a miss: you pay to store it)
    uncached    = 1.0×                     (no caching at all)

Everything here is deterministic and offline. We bill in "cost units" (tokens ×
multiplier), not dollars, because the *ratio* is the lesson, not the sticker
price. For a real bill, read `usage.cache_read_input_tokens` /
`cache_creation_input_tokens` from the API response.

The payoff (see examples/09): compaction and pruning *rewrite the prefix*. They
change the system prompt (the summary) and drop old turns, so they blow the cache
that an append-only history would have kept warm. "Cheaper context" can mean a
*bigger* bill.
"""

from __future__ import annotations

from . import tokens

CACHE_READ_MULTIPLIER = 0.1
CACHE_WRITE_MULTIPLIER = 1.25


def segments(system: str, messages: list[dict]) -> list[str]:
    """Render a request into ordered prefix segments: system first, then each
    message. Prefix caching matches these front-to-back."""
    return [system, *(m.get("content", "") for m in messages)]


def bill_turn(prev: list[str], current: list[str]) -> dict:
    """Bill one request against the previously cached prefix.

    Walk both segment lists from the front: segments that match are cache *reads*
    (0.1×); from the first mismatch onward, every segment is a cache *write*
    (1.25×). Returns the token split and the cost in units. `current` becomes the
    new cache entry for the next call."""
    matched = 0
    for a, b in zip(prev, current):
        if a == b:
            matched += 1
        else:
            break
    read_tokens = sum(tokens.estimate(current[i]) for i in range(matched))
    write_tokens = sum(tokens.estimate(current[i]) for i in range(matched, len(current)))
    cost = read_tokens * CACHE_READ_MULTIPLIER + write_tokens * CACHE_WRITE_MULTIPLIER
    return {
        "read_tokens": read_tokens,
        "write_tokens": write_tokens,
        "cost_units": cost,
        "cache_hit": matched > 0,
    }
