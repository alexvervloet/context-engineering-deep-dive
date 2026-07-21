"""
Example 08: pruning tool observations in an agent loop.

Agents (the Agents deep dive) bloat their own context faster than any chat: every
step appends a tool call AND its full result, and tool results are often huge: a
whole file, a 50-row query, a web page. Ten steps in, the window is mostly stale
observations the agent already used and will never need again. That's wasted budget
on every remaining step, and prime material for context rot (example 07).

The fix is **observation pruning**: keep the agent's reasoning and recent results
verbatim, but compact or drop the *old* tool outputs: replace a spent 2,000-token
result with a one-line note that it happened. The agent keeps its train of thought;
the dead weight goes.

This example builds a fake agent trace with big observations and compares the raw
window against a pruned one (keep the last N observations, stub the rest), offline.

Run:  python examples/08_pruning_observations.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from context import tokens

# A fake agent trace: each step is a short thought + a big tool observation.
STEPS = [
    ("Thought: I should look up the customer record.", "search_customers", "ROWS: " + ("id,name,plan,region,signup,seats; " * 40)),
    ("Thought: Now fetch their recent invoices.", "list_invoices", "INVOICES: " + ("date,amount,status,method; " * 40)),
    ("Thought: Check open support tickets.", "list_tickets", "TICKETS: " + ("id,subject,priority,age; " * 40)),
    ("Thought: Pull the refund policy doc.", "read_doc", "POLICY: " + ("Refunds within 30 days, prorated for annual. " * 12)),
    ("Thought: I have what I need to answer.", None, None),
]

KEEP_RECENT = 1  # how many of the most recent observations to keep verbatim


def build_raw() -> list[dict]:
    """The naive trace: every thought and every full observation, forever."""
    msgs: list[dict] = []
    for thought, tool, obs in STEPS:
        msgs.append({"role": "assistant", "content": thought + (f"\nAction: {tool}" if tool else "")})
        if obs is not None:
            msgs.append({"role": "user", "content": f"Observation: {obs}"})
    return msgs


def build_pruned() -> list[dict]:
    """Keep thoughts + the last KEEP_RECENT observations; stub older ones."""
    msgs = build_raw()
    obs_indices = [i for i, m in enumerate(msgs) if m["content"].startswith("Observation:")]
    to_stub = obs_indices[:-KEEP_RECENT] if KEEP_RECENT else obs_indices
    for i in to_stub:
        msgs[i] = {"role": "user", "content": "Observation: [pruned; result used in an earlier step]"}
    return msgs


def main() -> None:
    raw, pruned = build_raw(), build_pruned()
    raw_tokens = tokens.estimate_messages(raw)
    pruned_tokens = tokens.estimate_messages(pruned)

    print(f"Agent trace: {len(STEPS)} steps, {sum(1 for *_ , o in STEPS if o)} tool observations.\n")
    print(f"RAW window    (every observation kept):  {raw_tokens:>4} tokens")
    print(f"PRUNED window (keep last {KEEP_RECENT}, stub rest): {pruned_tokens:>4} tokens")
    print(f"-> {1 - pruned_tokens / raw_tokens:.0%} smaller, and the agent kept every Thought "
          f"(its reasoning)\n   plus the freshest result it's actually working from.\n")

    print("What the pruned window looks like:")
    for m in pruned:
        line = m["content"].replace("\n", " ")
        print(f"  {m['role']:>9}: {line[:64]}{'...' if len(line) > 64 else ''}")

    print(
        "\nTakeaway: in a loop, old tool results are the biggest, deadest weight in "
        "the\nwindow. Keep the reasoning and the most recent observations; compact "
        "or drop the\nrest. (Pair it with compaction for the thoughts and long-term "
        "memory for facts\nthat outlive the task: same toolkit, applied to an "
        "agent's context.)"
    )


if __name__ == "__main__":
    main()
