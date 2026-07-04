"""
Example 07 — more context is not better (context rot).
======================================================

It's tempting to treat a big context window as a reason to stuff everything in:
all the docs, the whole history, every tool result, just in case. Don't. Beyond a
point, *more* context makes answers *worse* — the relevant signal gets diluted by
noise, the model latches onto a plausible-but-irrelevant passage, and you pay for
every wasted token on every turn. Practitioners call this "context rot": quality
degrades as junk accumulates, even well under the token limit.

This example shows BOTH halves of the cost. It answers the same question two ways —
a **lean** context with just the fact you need, and a **bloated** one padded with
irrelevant "retrieved" documents — one of which is a plausible distractor naming a
*different* person. You'll see the token blowup (the cheap, undeniable half) *and*
the bloated context return the WRONG name (the quality half).

A note on how the two providers show it, so the demo is honest:
  - On `PROVIDER=mock` the flip is deterministic: the mock naively latches onto the
    last "my name is …" it sees, so the buried distractor wins. That's a crude
    stand-in for what a real model does — its attention gets pulled toward a
    plausible, on-topic-looking passage.
  - On a real model (`PROVIDER=openai` / `claude`) you'll see a subtler version of
    the same rot on harder questions; the more junk competes with the signal, the
    more often it slips. Add a key and try it.

Run:  python examples/07_context_rot.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from context import describe, generate, tokens

SYSTEM = "You are a helpful assistant. Answer using only this conversation."

# The one fact that answers the question.
RELEVANT = "Hi, my name is Dana and I'm on the Pro plan."

# Irrelevant "retrieved documents" — just-in-case padding, plausible and
# on-topic-ish but useless for the question. One of them is a DISTRACTOR: a
# support-ticket doc that quotes a *different* customer's name. That's exactly the
# kind of passage a bloated context lets compete with the real answer.
NOISE = [
    "Acme Cloud was founded in 2019 and is headquartered in a mid-sized city. " * 4,
    "Our changelog for version 12 includes dozens of minor fixes and tweaks. " * 4,
    "Support ticket #4471: a customer wrote 'Hi, my name is Sam' asking about exports.",
    "The mobile app supports dark mode, widgets, and offline drafts. " * 4,
    "Our status page reports uptime across six global regions every minute. " * 4,
]


def ask(messages: list[dict]) -> tuple[str, int]:
    answer = generate(SYSTEM, messages)
    return answer, tokens.estimate(SYSTEM) + tokens.estimate_messages(messages)


def main() -> None:
    print(f"Provider: {describe()}\n")
    question = "What's my name?"

    lean = [{"role": "user", "content": RELEVANT}, {"role": "user", "content": question}]
    bloated = (
        [{"role": "user", "content": RELEVANT}]
        + [{"role": "user", "content": f"(FYI doc) {n}"} for n in NOISE]
        + [{"role": "user", "content": question}]
    )

    lean_answer, lean_tokens = ask(lean)
    bloat_answer, bloat_tokens = ask(bloated)

    print(f"LEAN context    — {lean_tokens:>4} tokens -> {lean_answer}")
    print(f"BLOATED context — {bloat_tokens:>4} tokens -> {bloat_answer}")
    verdict = "same answer" if lean_answer == bloat_answer else "and got the answer WRONG"
    print(f"\nThe bloated window costs {bloat_tokens / lean_tokens:.1f}x the tokens — {verdict} —"
          f"\nand you'd pay that multiplier on EVERY turn.")

    print(
        "\nTakeaway: a big window is a budget, not a goal. Padding it with "
        "just-in-case\ncontext burns tokens now AND buries the signal — here the "
        "distractor's name won.\nThat's context rot: more context making the answer "
        "worse, not just costlier.\nRetrieve and keep what THIS turn needs — relevance "
        "beats volume. The cheapest,\nfastest, most accurate token is the one you "
        "didn't send."
    )


if __name__ == "__main__":
    main()
