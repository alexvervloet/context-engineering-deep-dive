"""
Example 04: long-term memory: remembering across sessions.

Compaction keeps a fact alive *within* a conversation. But close the session and
start fresh tomorrow, and even a perfect summary is gone; it lived in that
conversation's window. Some facts should outlive any single chat: your name, your
preferences, a decision from last week.

The pattern is **long-term memory**: write durable facts to a store *outside* the
window, and retrieve the relevant few back *into* the window when a new turn needs
them. That's RAG pointed at the conversation. `LongTermMemory` persists facts to a
JSON file and retrieves by keyword overlap (swap in real embeddings from the RAG
dive for production).

This example runs TWO sessions against a fresh, empty window each time. Session one
saves facts. Session two, a brand new conversation with nothing in the window, answers a
question correctly *only because* it recalled the relevant fact from the store.

Run:  python examples/04_long_term_memory.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from context import describe, generate
from context.longterm import LongTermMemory
from context.memory import WindowMemory

SYSTEM = "You are a helpful assistant. Use the recalled memory below if relevant.\n"


def session_two(store: LongTermMemory, question: str, *, use_memory: bool) -> str:
    """A fresh conversation (empty window). Optionally inject recalled facts."""
    system = SYSTEM
    if use_memory:
        recalled = store.recall(question, k=3)
        if recalled:
            system += "Recalled memory:\n" + "\n".join(f"- {f}" for f in recalled)
    mem = WindowMemory(budget_tokens=500, system=system)
    mem.add("user", question)
    sys_prompt, sent = mem.build()
    return generate(sys_prompt, sent)


def main() -> None:
    print(f"Provider: {describe()}\n")

    with tempfile.TemporaryDirectory() as tmp:
        store = LongTermMemory(path=os.path.join(tmp, "memory.json"))

        # --- Session one: learn durable facts and write them to the store. ---
        print("SESSION 1: the user tells us things worth keeping:")
        for fact in [
            "The user's name is Dana.",
            "The user is on the Pro plan.",
            "The user prefers answers in metric units.",
            "The user's launch is scheduled for Friday.",
        ]:
            store.remember(fact)
            print(f"  remembered: {fact}")
        print(f"  -> {len(store.facts)} facts persisted to disk.\n")

        # The session ends. A new LongTermMemory loads the SAME file from scratch,
        # proving the facts survived the session, not just this process's memory.
        reloaded = LongTermMemory(path=store.path)

        # --- Session two: a brand-new chat with an empty window. ---
        question = "When is my launch scheduled?"
        print("SESSION 2: a fresh conversation; the window starts empty.")
        print(f'Asked: "{question}"\n')

        without = session_two(reloaded, question, use_memory=False)
        print("WITHOUT long-term memory ->", without)

        with_mem = session_two(reloaded, question, use_memory=True)
        print("WITH long-term memory    ->", with_mem)
        print("\n  (recalled for this question:", reloaded.recall(question, k=3), ")")

    print(
        "\nTakeaway: window memory and compaction are per-conversation; long-term "
        "memory\nis the store that outlives them. Write durable facts out, retrieve "
        "the relevant\nfew back in. The skill is choosing WHAT to persist (lasting "
        "facts, not chit-chat)\nand retrieving only what THIS turn needs, putting "
        "the right text in the window."
    )


if __name__ == "__main__":
    main()
