"""
Example 10: the API can now do this for you (server-side compaction & context editing).

Sections 4 and 9 built compaction by hand: summarize what falls off, keep the
window small, and pay for it in cache misses. That is still the right mental
model, and it is still what most stacks do. But the Anthropic API now ships two
of these jobs as server-side features, and you should know which is which,
because they are not the same tool.

  COMPACTION           `context_management={"edits": [{"type": "compact_20260112"}]}`
                       beta header: compact-2026-01-12
                       SUMMARIZES earlier turns when the conversation approaches
                       a threshold (150K tokens by default). This is §4, done
                       for you. Model-gated: Sonnet 4.6 / Opus 4.6 and newer.
                       Claude Haiku 4.5 returns a 400.

  CONTEXT EDITING      `context_management={"edits": [{"type": "clear_tool_uses_20250919"}]}`
                       beta header: context-management-2025-06-27
                       CLEARS old tool results outright. No summary, nothing
                       kept. This is §9 (pruning observations), done for you.
                       Works on Haiku 4.5.

SUMMARIZE VS CLEAR IS THE WHOLE DECISION
    A summary costs tokens to make and keeps a lossy trace of what happened.
    Clearing costs nothing and keeps nothing. For a chat transcript you usually
    want the summary, because the user will refer back to it. For an agent's
    tool results, the raw output of a `grep` from forty steps ago is almost
    always dead weight, and clearing it is strictly better than paying a model
    to write a paragraph about it.

THE GOTCHA THAT WILL BITE YOU
    When compaction is on, you MUST append `response.content` (the whole block
    list) back into your `messages`, not just the extracted text. The API
    returns a `compaction` block that carries the compacted state, and the next
    request needs it. Pull out `.text` the way every other example in this repo
    does and you silently throw that state away.

AND THE THING THAT HAS NOT CHANGED
    Server-side compaction is still a prefix rewrite. Everything §9 showed you
    about compaction destroying the prompt cache applies here exactly as it did
    to the hand-rolled version. Moving the work to the server makes it *easier*,
    not free. If you took one thing from this dive, it should be that "smaller
    context" and "cheaper bill" are different claims.

This example needs a real Anthropic key (these are server features, so the mock
cannot stand in). Run:

    secrun python examples/10_server_side_compaction.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

if not os.getenv("ANTHROPIC_API_KEY"):
    sys.exit(
        "This example calls Anthropic directly (server-side features have no mock).\n"
        "Set ANTHROPIC_API_KEY via secrun (see ../SECRETS.md) and try again.\n"
        "Every other example in this repo runs offline on PROVIDER=mock."
    )

import anthropic  # noqa: E402

client = anthropic.Anthropic()

COMPACTION_MODEL = "claude-sonnet-4-6"  # compaction needs 4.6+; Haiku 4.5 is a 400
EDITING_MODEL = "claude-haiku-4-5"      # context editing works on the cheap workhorse

# --- 1. Context editing: clear old tool results ------------------------------
# This is the agent case from §9. We declare a tool so the shape is realistic;
# the point is the `context_management` block, which tells the API it may drop
# tool results once they age out.
print("--- 1. context editing (clear tool results) ---")
edited = client.beta.messages.create(
    betas=["context-management-2025-06-27"],
    model=EDITING_MODEL,
    max_tokens=256,
    context_management={"edits": [{"type": "clear_tool_uses_20250919"}]},
    tools=[
        {
            "name": "read_file",
            "description": "Read a file from disk.",
            "input_schema": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        }
    ],
    messages=[{"role": "user", "content": "Say OK. Do not call any tool."}],
)
print("accepted; the API will clear aged tool results for us")
print("response blocks:", [b.type for b in edited.content])
print(
    "On a short conversation nothing is cleared yet. The config is a standing\n"
    "instruction, not a one-off command: it applies as the transcript grows."
)

# --- 2. Compaction: the round-trip your code has to get right ----------------
# Note what we append to `messages` below. Not the text. The whole content list.
print("\n--- 2. compaction, and the append-the-whole-content rule ---")
messages: list[dict] = [{"role": "user", "content": "In one sentence: what is a context window?"}]

response = client.beta.messages.create(
    betas=["compact-2026-01-12"],
    model=COMPACTION_MODEL,
    max_tokens=512,
    context_management={"edits": [{"type": "compact_20260112"}]},
    messages=messages,
)

# THIS is the line that matters. `response.content`, not a plucked-out string.
messages.append({"role": "assistant", "content": response.content})

text = next((b.text for b in response.content if b.type == "text"), "")
print(text)
print(f"\nblocks returned: {[b.type for b in response.content]}")
print(
    "No `compaction` block yet, because we are nowhere near the 150K trigger.\n"
    "It appears once the conversation is long enough to need one, which is\n"
    "exactly when a codebase that only ever kept `.text` discovers its bug."
)

# --- 3. The bill has not changed ---------------------------------------------
print("\n--- 3. what §9 said still holds ---")
print(
    "Compaction rewrites the prefix, so the cache is invalidated whether YOU\n"
    "write the summary or Anthropic does. Server-side compaction removes the\n"
    "code you have to maintain. It does not remove the tradeoff you have to\n"
    "reason about. Re-run examples/09_caching_vs_compaction.py if that has\n"
    "gone fuzzy: the arithmetic there is the same arithmetic here."
)
