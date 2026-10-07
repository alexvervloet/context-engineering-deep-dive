"""
Example 11: append-only history stopped being only about cost.

Section 10 showed that rewriting the prompt prefix (compaction, a sliding window)
throws away the prompt cache. That was a bill. On the newest Claude models it can
also be an error.

Claude Fable 5.1, Opus 5.5 and Sonnet 5.5 sign every thinking block with the exact
prefix it was produced after: the system prompt, the tools, and every message
before it. Send that block back in a later request whose prefix differs, and the
block is invalid. On accounts created on or after 2026-08-31 the API rejects the
request with a 400 by default; on older accounts it says nothing unless you opt in.
So the same code can pass on your key and fail on your users' keys.

The edits that break thinking are the edits that restart the cache. One rule covers
both: freeze `system` and `tools` for the session and only ever append to
`messages`. When the history has to shrink, summarize ALL of it into one new user
message and send no earlier turns (what Anthropic calls simple compaction), so no
old thinking block is left to check.

Offline by default: it simulates the check with a hash so you can see which
strategy from this dive keeps which blocks. With --real it runs the real check
on claude-sonnet-5-5, about a cent: a valid append-only request, then the same
conversation with one sentence added to the system prompt, rejected and then
accepted with the stale block dropped.

Run it:

    python examples/11_append_only_contract.py
    secrun python examples/11_append_only_contract.py --real
"""

import hashlib
import json
import os
import sys

# --- The offline model of the check ------------------------------------------
# A real signature is an opaque, encrypted blob. Here it's a hash of the prefix,
# which is enough to show what the API compares: the system prompt, the tools, and
# every message before the block, exactly as they were sent.


def prefix_hash(system: str, tools: list, messages_before: list) -> str:
    blob = json.dumps([system, tools, messages_before], sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def thinking_block(system, tools, messages_before, text):
    return {"type": "thinking", "thinking": text, "sig": prefix_hash(system, tools, messages_before)}


def check(system, tools, messages):
    """Return (kept, failed): the thinking blocks whose prefix still matches, and those that don't."""
    kept, failed = 0, 0
    for i, msg in enumerate(messages):
        if msg["role"] != "assistant" or not isinstance(msg["content"], list):
            continue
        for block in msg["content"]:
            if block.get("type") != "thinking":
                continue
            if failed or block["sig"] != prefix_hash(system, tools, messages[:i]):
                failed += 1  # once one fails, every later block fails with it
            else:
                kept += 1
    return kept, failed


# Build a four-turn conversation the way an append-only client would.
SYSTEM = "You are a support assistant for Nimbus Notes."
TOOLS: list = [{"name": "search_notes"}]
questions = ["How long is the trash kept?", "Can I restore a deleted notebook?",
             "Does that work on the free plan?", "And on mobile?"]
history: list = []
for q in questions:
    history.append({"role": "user", "content": q})
    history.append({"role": "assistant", "content": [
        thinking_block(SYSTEM, TOOLS, list(history), f"reasoning about: {q}"),
        {"type": "text", "text": f"answer to: {q}"},
    ]})
NEXT = {"role": "user", "content": "Thanks. One more: is my data stored in the EU?"}


def strategies():
    """The next request, built four ways. Each returns (system, messages, cache_reuse)."""
    # 1. Append-only: everything as sent, plus the new turn.
    yield "append-only", SYSTEM, history + [NEXT], "whole history"
    # 2. Sliding window (section 3): drop the oldest exchange, keep the rest verbatim.
    yield "sliding window", SYSTEM, history[2:] + [NEXT], "none"
    # 3. Summary in the system prompt (sections 4 and 10): rewrite system, keep a tail.
    summary_system = SYSTEM + " Summary so far: trash kept 30 days; notebooks restorable."
    yield "summary in system", summary_system, history[4:] + [NEXT], "none"
    # 4. Simple compaction: one user message carrying the summary, no earlier turns.
    compacted = {"role": "user", "content": "Summary of our chat so far: trash is kept 30 days, "
                 "notebooks can be restored, on every plan and on mobile.\n\n" + NEXT["content"]}
    yield "simple compaction", SYSTEM, [compacted], "none, but nothing to check"


print("Which thinking blocks survive each way of shrinking the history?\n")
print(f"  {'strategy':<20}{'blocks sent':>12}{'valid':>7}{'invalid':>9}   new account sees   cache reused")
for name, system, messages, cache in strategies():
    kept, failed = check(system, TOOLS, messages)
    sent = kept + failed
    outcome = "400" if failed else "ok"
    print(f"  {name:<20}{sent:>12}{kept:>7}{failed:>9}   {outcome:<18} {cache}")

print(
    "\nAppend-only keeps every block, and the cache with it. The sliding window looks\n"
    "harmless, but dropping the first exchange changes the prefix of every block it\n"
    "kept. Rewriting the system prompt invalidates all of them. Simple compaction\n"
    "passes because it sends no old thinking at all: the model reasons afresh from\n"
    "the summary. On an account created before 2026-08-31 the middle two rows don't\n"
    "error and the model still reads the blocks, so a clean run on your own key proves\n"
    "nothing about a user whose newer account gets the 400."
)


# --- The real check ----------------------------------------------------------
def real() -> None:
    if not os.getenv("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY via secrun (see ../docs/SECRETS.md) for --real.")
    import anthropic

    client = anthropic.Anthropic()
    model = "claude-sonnet-5-5"
    betas = ["thinking-binding-controls-2026-08-01"]  # lets any account opt into the check
    system = "You are a careful assistant for a small bakery."
    first = [{"role": "user", "content": "A bakery sells muffins in boxes of 4, 6 and 9. What is "
              "the largest number of muffins that cannot be bought exactly? Reason carefully, then "
              "give the number and one line of justification."}]
    reply = client.messages.create(model=model, max_tokens=2000, system=system, messages=first)
    if not any(b.type == "thinking" for b in reply.content):
        sys.exit("The model answered without thinking this time, so there's no block to check. Run it again.")
    # Append the assistant turn EXACTLY as returned, thinking block and all.
    conversation = first + [
        {"role": "assistant", "content": [b.model_dump(exclude_none=True) for b in reply.content]},
        {"role": "user", "content": "Now with boxes of 4, 6 and 11?"},
    ]

    def send(label, system_prompt, behavior):
        try:
            r = client.beta.messages.create(
                model=model, max_tokens=2000, system=system_prompt, messages=conversation, betas=betas,
                thinking={"type": "adaptive", "block_binding": {"prefix_mismatch_behavior": behavior}},
            )
            dropped = [t.path for t in (getattr(r, "input_transformations", None) or [])]
            print(f"  {label:<40} accepted; dropped blocks: {dropped or 'none'}")
        except anthropic.BadRequestError as e:
            print(f"  {label:<40} 400: {str(e.message).split('Remove the block')[0][-90:]}")

    print(f"\nThe real check on {model}:")
    send("append-only, behavior=error", system, "error")
    edited = system + " Today is Tuesday."
    send("one sentence added to system, error", edited, "error")
    send("same edit, behavior=drop_block", edited, "drop_block")
    print(
        "\n'drop_block' hides the error but not the cost: the model answers without its\n"
        "earlier reasoning, and the cache restarts at the edit. To change instructions\n"
        "mid-session, append a role='system' message instead of editing `system`."
    )


if __name__ == "__main__":
    if "--real" in sys.argv:
        real()
