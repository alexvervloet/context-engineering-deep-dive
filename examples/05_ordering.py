"""
Example 05: order matters: don't bury the lede in the middle.

Fitting the right text in the window is half the job; *where* you put it is the
other half. Models attend most reliably to the **start** and **end** of a long
context and are measurably worse at using facts buried in the **middle**, the
well-documented "lost in the middle" effect. So the naive move (concatenate your
retrieved chunks in whatever order they came back) can hide the one chunk that
matters in the worst possible spot.

This example is offline; it's about *position*, not a model call. It takes a set of
context sections of varying importance and shows two layouts: the naive order
(important chunk lands in the middle) and `order_for_recall`, which places the
highest-priority sections at the edges and buries the filler in the middle.

Run:  python examples/05_ordering.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from context.assemble import Section, order_for_recall

# Five retrieved chunks. The KEY one (priority 10) is the answer to the user's
# question; the rest are plausible-but-irrelevant filler (lower priority).
SECTIONS = [
    Section("filler: pricing history", "Old pricing notes nobody asked about...", 2),
    Section("filler: changelog", "Release notes from three versions ago...", 3),
    Section("KEY: refund policy", "Refunds are available within 30 days of purchase.", 10),
    Section("filler: brand guidelines", "Our logo must have 24px of clear space...", 1),
    Section("filler: office hours", "Support is available 9-5 in three timezones...", 4),
]


def show(title: str, sections: list[Section]) -> None:
    print(title)
    for i, s in enumerate(sections):
        spot = "edge" if i == 0 or i == len(sections) - 1 else "middle"
        flag = "  <-- the chunk that answers the question" if s.priority == 10 else ""
        print(f"  [{i}] ({spot:^6}) p{s.priority:<2} {s.label}{flag}")
    print()


def main() -> None:
    print("A user asks about refunds. Five chunks were retrieved; one actually answers it.\n")

    show("NAIVE order (as retrieved): the key chunk sits in the middle:", SECTIONS)
    show("order_for_recall: highest priority at the edges, filler in the middle:",
         order_for_recall(SECTIONS))

    print(
        "Takeaway: same chunks, same token cost; only the order changed. Put the "
        "material\nmost likely to be needed where the model reads best: the start and "
        "the end. When\nyou over-retrieve (RAG dive), ranking is also a *placement* "
        "decision, not just a\nkeep/drop one. (And the shorter you keep the context, "
        "the less 'middle' there is\nto get lost in.)"
    )


if __name__ == "__main__":
    main()
