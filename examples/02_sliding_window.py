"""
Example 02: the sliding window (and what it forgets).

The simplest way to stay under budget: keep the system prompt plus the most recent
turns that fit, and let the oldest scroll off. This is the "trim the history" move
the API and Agents dives gestured at. Here it's a real `WindowMemory` with a token
budget.

It's bounded and cheap. It's also genuinely forgetful: a fact stated early (your
name, a decision, something you asked the model to remember) is *gone* once it
falls off the back. This example proves it. Dana introduces herself, the
conversation runs long, and then we ask the model her name. On `PROVIDER=mock` the
answer is deterministic, so you can see the forgetting plainly (real models behave
the same: they can only answer from what's in the window).

Run:  python examples/02_sliding_window.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from context import describe, generate
from context.memory import WindowMemory

SYSTEM = "You are a helpful assistant. Answer using only what's in this conversation."


def main() -> None:
    print(f"Provider: {describe()}\n")

    # A small budget so the window overflows after a few long turns.
    mem = WindowMemory(budget_tokens=220, system=SYSTEM)

    # Turn 1: the fact we care about.
    mem.add("user", "Hi, my name is Dana and I'm on the Pro plan.")
    mem.add("assistant", "Welcome, Dana! How can I help with your Pro plan today?")

    # Then a long stretch of unrelated chatter that pushes turn 1 off the back.
    for i in range(6):
        mem.add("user", f"Walk me through feature number {i} in a fair amount of detail, please.")
        mem.add("assistant", f"Sure — feature {i} works like this, with several steps and caveats "
                             f"that take a couple of sentences to lay out properly for you.")

    # Now ask something only turn 1 can answer.
    mem.add("user", "What's my name?")
    system, sent = mem.build()
    answer = generate(system, sent)

    info = mem.info()
    print(f"Total turns recorded: {len(mem.turns)}")
    print(f"Turns actually sent:  {info['turns_sent']}  (dropped {info['turns_dropped']} oldest)")
    print(f"Window tokens:        {info['tokens']} / {info['budget']} budget\n")
    print('Asked: "What\'s my name?"')
    print("Answer:", answer)

    print(
        "\nTakeaway: the sliding window kept costs bounded but threw away the "
        "introduction,\nso the model can't recall it. For the model, the "
        "conversation IS the window.\nNext example keeps the same budget but stops "
        "forgetting, by summarizing what\nfalls off instead of deleting it."
    )


if __name__ == "__main__":
    main()
