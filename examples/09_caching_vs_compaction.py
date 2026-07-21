"""
Example 09: the hidden cost of compaction: it blows your prompt cache.

Every technique in this dive so far, compaction (§4), pruning (§9), and reordering
(§6), makes the window *smaller*. This example shows the bill they can quietly
*raise*, because they all break the one thing that makes re-sending a long history
cheap: the **prompt cache**.

Providers cache the prompt *prefix*, and the rule is unforgiving: any change
anywhere in the prefix invalidates everything after it. An **append-only** history
(FullMemory) never changes its prefix (each turn just adds to the end) so every
prior token is a cache *read* (~0.1×) and only the new turn is written (~1.25×).
**Compaction** (SummaryMemory) rewrites the prefix: it changes the system prompt
(the summary) and drops old turns. Render order is system → messages, so changing
the system prompt invalidates the *entire* cache. The next turn pays full
cache-write price on the whole (smaller) context: a miss.

We run the SAME conversation both ways and bill each turn against the cache with
`context/cost.py`. Compaction keeps far fewer tokens in the window, and can still
cost MORE, because it keeps throwing the cache away. This is the honest tradeoff
the dive is built on: "cheaper context" is not automatically a cheaper bill.

Fully offline and deterministic. We bill in cost units (tokens × cache
multiplier), not dollars. Run:

    python examples/09_caching_vs_compaction.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from context import cost, tokens
from context.memory import FullMemory, SummaryMemory

SYSTEM = "You are a helpful assistant. Answer using only what's in this conversation."

# A modest support conversation: a dozen exchanges, each a couple of sentences.
TURNS = [
    ("user", "Hi, I'm Dana on the Pro plan. Remember our launch is Friday."),
    ("assistant", "Welcome, Dana! Noted — launch Friday, Pro plan. How can I help?"),
    ("user", "Walk me through how billing proration works when I upgrade mid-cycle."),
    ("assistant", "Proration splits the cycle: you're credited for unused days on the "
                  "old plan and charged the prorated remainder on the new one."),
    ("user", "And if I downgrade instead, does the credit roll over to next month?"),
    ("assistant", "Yes — a downgrade credit applies to the next invoice; it doesn't "
                  "expire at the cycle boundary the way a promo might."),
    ("user", "How do I export all my notes before the launch, just to be safe?"),
    ("assistant", "Settings → Data → Export produces a zip of every note as Markdown, "
                  "usually ready to download within a few minutes."),
    ("user", "Can teammates I invite this week see notes I created last month?"),
    ("assistant", "Only if the note is in a shared space; personal notes stay private "
                  "until you move or share them explicitly."),
    ("user", "What's my name and plan, and when did I say the launch was?"),
]


def run(mem, label: str) -> float:
    """Replay the conversation through `mem`, billing each USER turn against the
    prompt cache. Returns the total cost in units."""
    prev_segments: list[str] = []
    total = 0.0
    hits = 0
    user_turns = 0
    print(f"{label}")
    for role, content in TURNS:
        mem.add(role, content)
        if role != "user":
            continue  # you send a request (and pay) on each user turn
        user_turns += 1
        system, msgs = mem.build()
        segs = cost.segments(system, msgs)
        bill = cost.bill_turn(prev_segments, segs)
        prev_segments = segs
        total += bill["cost_units"]
        hits += 1 if bill["cache_hit"] else 0
        window = tokens.estimate(system) + tokens.estimate_messages(msgs)
        mark = "hit " if bill["cache_hit"] else "MISS"
        print(f"  turn {user_turns:>2}: window {window:>4}t  cache {mark}  "
              f"(read {bill['read_tokens']:>4}t, write {bill['write_tokens']:>4}t)  "
              f"+{bill['cost_units']:>6.1f} units")
    print(f"  → {hits}/{user_turns} cache hits, total {total:.1f} cost units\n")
    return total


def main() -> None:
    print("Same conversation, billed against the prompt cache two ways.\n")

    full_total = run(FullMemory(system=SYSTEM), "Append-only history (FullMemory), prefix never changes:")
    summary_total = run(
        SummaryMemory(budget_tokens=120, system=SYSTEM, keep_recent=2),
        "Compaction (SummaryMemory, budget=120), rewrites the prefix when it compacts:",
    )

    cheaper, dearer = ("compaction", "append-only") if summary_total < full_total else ("append-only", "compaction")
    ratio = max(full_total, summary_total) / max(1e-9, min(full_total, summary_total))
    print(f"Append-only total: {full_total:.1f} units | Compaction total: {summary_total:.1f} units")
    print(f"Here, {cheaper} is cheaper; {dearer} costs about {ratio:.1f}× as much.\n")

    print(
        "Takeaway: compaction kept the window small every turn, but each compaction\n"
        "changed the system prompt and dropped old turns: a prefix rewrite that\n"
        "invalidated the cache, so those turns paid full cache-WRITE price. Append-only\n"
        "sends more tokens, but they're almost all cheap cache READs. On a SHORT-to-\n"
        "medium chat the cache wins; the crossover comes on very long conversations,\n"
        "where the append-only window grows without bound and the small compacted\n"
        "context, even uncached, finally costs less. The lesson isn't 'never\n"
        "compact': it's that 'fewer tokens' and 'cheaper' are different axes. Measure\n"
        "the bill (usage.cache_read_input_tokens vs cache_creation_input_tokens), and\n"
        "when you must compact, do it rarely and in bulk so you pay the cache miss once,\n"
        "not every turn."
    )


if __name__ == "__main__":
    main()
