"""
chat.py — the capstone: a chat that remembers, under a fixed budget.
====================================================================

Everything in this dive, assembled into one tool you'd actually use: a multi-turn
chat assistant that stays inside a token budget no matter how long you talk, AND
remembers durable facts across sessions. Three layers from the library, working
together on every turn:

  1. SummaryMemory   — the in-session window. When the budget fills, old turns are
                       compacted into a running summary; recent turns stay verbatim.
  2. LongTermMemory  — a persistent fact store. Durable things you say ("my name is
                       Dana", "remember the launch is Friday") are written to disk
                       and recalled in later turns — and later *sessions*.
  3. assembled system — each turn, the system prompt = your persona + the running
                       summary + the long-term facts relevant to what you just asked.

The payoff: chat for fifty turns and it never overflows; quit, come back tomorrow,
and it still greets you by name.

Run it (offline on PROVIDER=mock — no key, no cost):

    # one-shot:
    python hands_on/chat.py "Hi, my name is Dana. Remember our launch is Friday."
    python hands_on/chat.py "When is my launch?"        # a *new* run — still knows

    # interactive REPL (type 'quit' to exit; '/context' and '/memory' to inspect):
    python hands_on/chat.py

    # see exactly what's sent each turn, and the token budget:
    python hands_on/chat.py --show-context

    # smaller budget to watch compaction kick in sooner; wipe long-term memory:
    python hands_on/chat.py --budget 200
    python hands_on/chat.py --forget

Read the source: `respond()` is the whole turn — recall, assemble, generate,
persist. The library does the heavy lifting; this file just wires it to a CLI.
"""

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from context import describe, generate, tokens
from context.longterm import LongTermMemory
from context.memory import SummaryMemory

MEMORY_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".ctx_memory.json")

PERSONA = "You are a friendly, concise assistant with a good memory."


# --- Deciding what's worth remembering forever -----------------------------
# A real system might use a model to judge durability; here a few patterns catch
# the obvious durable facts so you can watch persistence work.
_DURABLE = [
    (r"\bmy name is ([A-Z][a-zA-Z]+)", "The user's name is {}."),
    (r"\bcall me ([A-Z][a-zA-Z]+)", "The user's name is {}."),
    (r"\b(?:on|using|have) the (\w+) plan", "The user is on the {} plan."),
    (r"\bi prefer (.+?)(?:\.|$)", "The user prefers {}."),
    (r"\bremember(?: that)?[:,]?\s+(.+?)(?:\.|$)", "{}"),
]


def durable_facts(text: str) -> list[str]:
    out = []
    for pattern, template in _DURABLE:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            out.append(template.format(m.group(1).strip()))
    return out


def build_system(store: LongTermMemory, mem: SummaryMemory, user_text: str) -> str:
    """Persona + the running summary (compaction) + relevant long-term memory."""
    system, _ = mem.build()  # persona + summary block, if any
    recalled = store.recall(user_text, k=3)
    if recalled:
        system += "\n\nKnown facts about this user:\n" + "\n".join(f"- {f}" for f in recalled)
    return system


def respond(store: LongTermMemory, mem: SummaryMemory, user_text: str, *, show: bool) -> str:
    # 1. Persist any durable facts the user just stated.
    for fact in durable_facts(user_text):
        if store.remember(fact):
            if show:
                print(f"  [long-term memory += {fact!r}]")

    # 2. Record the turn (compaction may trigger inside add()).
    mem.add("user", user_text)

    # 3. Assemble the system prompt and generate.
    system = build_system(store, mem, user_text)
    _, sent = mem.build()
    if show:
        used = tokens.estimate(system) + tokens.estimate_messages(sent)
        print(f"  [context: {used}/{mem.budget} tok · {len(sent)} turns sent · "
              f"{mem.info()['compactions']} compactions · {len(store.facts)} stored facts]")
        print("  [system sent]:\n    " + system.replace("\n", "\n    "))

    answer = generate(system, sent)

    # 4. Record the assistant turn so it's part of the next window.
    mem.add("assistant", answer)
    return answer


def main() -> int:
    parser = argparse.ArgumentParser(description="A chat that remembers, under a token budget.")
    parser.add_argument("question", nargs="*", help="ask once and exit; omit for an interactive REPL")
    parser.add_argument("--budget", type=int, default=400, help="window token budget (default 400)")
    parser.add_argument("--show-context", action="store_true", help="print what's sent each turn")
    parser.add_argument("--forget", action="store_true", help="wipe long-term memory and exit")
    args = parser.parse_args()

    store = LongTermMemory(path=MEMORY_FILE)

    if args.forget:
        n = len(store.facts)
        store.clear()
        print(f"Cleared {n} long-term fact(s) from {os.path.basename(MEMORY_FILE)}.")
        return 0

    mem = SummaryMemory(budget_tokens=args.budget, system=PERSONA, keep_recent=2)
    print(f"Provider: {describe()}   ·   window budget: {args.budget} tokens   ·   "
          f"{len(store.facts)} facts in long-term memory\n")

    # One-shot mode.
    if args.question:
        question = " ".join(args.question)
        print(f"you> {question}")
        print(f"bot> {respond(store, mem, question, show=args.show_context)}")
        return 0

    # Interactive REPL.
    print("Chat with me. I compact old turns and remember durable facts across runs.")
    print("Commands: 'quit' to exit, '/context' to toggle the context view, '/memory' to list stored facts.\n")
    show = args.show_context
    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit"}:
            break
        if user_input == "/context":
            show = not show
            print(f"  [context view {'on' if show else 'off'}]")
            continue
        if user_input == "/memory":
            if store.facts:
                print("  long-term memory:")
                for f in store.facts:
                    print(f"    - {f}")
            else:
                print("  long-term memory is empty.")
            continue
        print(f"bot> {respond(store, mem, user_input, show=show)}\n")

    print(f"\nGoodbye. {len(store.facts)} fact(s) saved for next time — try running me again.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
