"""
Example 01: the context window is a budget.

Everything in this dive starts here: the context window is a fixed number of
tokens, and a conversation *grows*. Every turn you keep is re-sent (and re-billed)
on the next request, so the history marches toward the ceiling. When it crosses,
the request fails, or, on a big window, just gets slow and expensive.

This example needs no model and no key. It grows a conversation turn by turn,
estimates the running token total with `context.tokens` (the ~4-chars/token
heuristic), and shows it blow past a deliberately tiny 2,000-token "window." That
overflow is the problem every later lesson solves.

Run:  python examples/01_token_budget.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from context import tokens

WINDOW = tokens.CONTEXT_WINDOWS["small-demo"]  # 2,000 tokens, on purpose

SYSTEM = "You are a helpful assistant for Acme Cloud. Be concise and accurate."

# A realistic-ish support chat where each turn is a few sentences.
TURNS = [
    ("user", "Hi! My name is Dana and I just upgraded to the Pro plan. How do I invite my team?"),
    ("assistant", "Welcome, Dana! On Pro you can invite teammates under Settings → Members → Invite. "
                  "Paste their emails and pick a role; they'll get a join link that expires in 7 days."),
    ("user", "Great. Can I set different permission levels for different people?"),
    ("assistant", "Yes — each member gets a role: Admin (full access), Editor (create/edit), or "
                  "Viewer (read-only). You can change a role anytime from the same Members page."),
    ("user", "What happens to their access if I downgrade back to Free later?"),
    ("assistant", "On Free you keep one project and a single seat, so extra members are set to "
                  "inactive (not deleted). Re-upgrade and they're restored exactly as they were."),
    ("user", "Okay. Also, remember that our launch is on Friday — I'll need export working by then."),
    ("assistant", "Noted: launch on Friday, export must work. Export lives under Settings → Data → "
                  "Export; it builds an archive and emails you a link, usually within the hour."),
]


def main() -> None:
    print(f"Context window for this demo: {WINDOW:,} tokens\n")
    print(f"{'after turn':>10}  {'this turn':>9}  {'running total':>13}  status")
    print("-" * 56)

    history: list[dict] = []
    running = tokens.estimate(SYSTEM)  # the system prompt is always in the window
    print(f"{'(system)':>10}  {running:>9}  {running:>13}  ok")

    for i, (role, content) in enumerate(TURNS, 1):
        msg = {"role": role, "content": content}
        history.append(msg)
        this_turn = tokens.estimate_message(msg)
        running += this_turn
        status = "ok" if running <= WINDOW else "OVERFLOW: request would fail"
        print(f"{i:>10}  {this_turn:>9}  {running:>13,}  {status}")

    print("-" * 56)
    total = tokens.estimate(SYSTEM) + tokens.estimate_messages(history)
    print(f"\nFull history: {total:,} tokens across {len(history)} messages "
          f"(+ system) {total / WINDOW:.0%} of this tiny window.")
    print(
        "\nTakeaway: memory isn't free or infinite. It's a budget that fills up. The\n"
        "rest of this dive is about what to KEEP and what to DROP when it does. (This\n"
        "window is tiny to overflow fast; real windows are 128K–200K, but every\n"
        "conversation, tool result, and retrieved doc spends against the same budget.)"
    )


if __name__ == "__main__":
    main()
