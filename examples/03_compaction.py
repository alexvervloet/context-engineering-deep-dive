"""
Example 03: compaction: summarize what falls off instead of deleting it.

The sliding window (example 02) forgot Dana's name because it *deleted* old turns.
Compaction keeps the same token budget but, when the window fills, replaces the
oldest turns with a short running **summary** and keeps the recent turns verbatim.
The raw words are gone; the *facts* survive in the summary.

This is the single most important technique in the dive, and it's what real
assistants do under the hood when a long chat "remembers" things from way back.
`SummaryMemory` does it: each time the budget is exceeded it folds the oldest turns
into a summary (via `providers.summarize`) and carries that summary in the system
prompt.

We run the SAME long conversation as example 02, on the SAME budget, and this time
the model recalls both the name and the thing it was asked to remember.

Run:  python examples/03_compaction.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from context import describe, generate
from context.memory import SummaryMemory

SYSTEM = "You are a helpful assistant. Answer using only what's in this conversation."


def main() -> None:
    print(f"Provider: {describe()}\n")

    mem = SummaryMemory(budget_tokens=220, system=SYSTEM, keep_recent=2)

    mem.add("user", "Hi, my name is Dana and I'm on the Pro plan. Remember that our launch is Friday.")
    mem.add("assistant", "Welcome, Dana! Noted: launch on Friday. How can I help with Pro today?")

    for i in range(6):
        mem.add("user", f"Walk me through feature number {i} in a fair amount of detail, please.")
        mem.add("assistant", f"Sure — feature {i} works like this, with several steps and caveats "
                             f"that take a couple of sentences to lay out properly for you.")

    mem.add("user", "What's my name, and what did I ask you to remember?")
    system, sent = mem.build()
    answer = generate(system, sent)

    info = mem.info()
    print(f"Total turns recorded: {len(mem.turns) + info['compactions']}+  (recent kept verbatim: {info['turns_sent']})")
    print(f"Compactions so far:   {info['compactions']}")
    print(f"Window tokens:        {info['tokens']} / {info['budget']} budget\n")

    # Show the summary the model is actually carrying.
    print("The running summary now in the system prompt:")
    print("  " + (mem.summary or "(none yet)").replace("\n", "\n  "), "\n")

    print('Asked: "What\'s my name, and what did I ask you to remember?"')
    print("Answer:", answer)

    print(
        "\nTakeaway: same budget as the sliding window, but the facts survived, "
        "because\ncompaction summarized the old turns instead of deleting them. You "
        "trade exact\nwording (and one summarization call) for durable memory under a "
        "fixed budget.\nThe risk to watch: a bad summary drops a detail you needed, "
        "so summarize for\n*facts to preserve*, and keep the most recent turns "
        "verbatim."
    )


if __name__ == "__main__":
    main()
