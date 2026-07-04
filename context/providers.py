"""
context/providers.py — the ONLY file that talks to a model provider.
====================================================================

Same keystone idea as every sibling repo: hide the provider-specific calls behind
tiny functions so the rest of the code is provider-agnostic. This dive is about
*what you put in the context window*, which is provider-neutral — so we expose two
calls and nothing else:

  generate(system, messages) -> str   # answer a conversation
  summarize(messages)        -> str   # condense old turns (for compaction)

Pick your stack with `PROVIDER` in `.env`:

  PROVIDER=mock   ->  a deterministic, offline, in-process "model". No key, no
                      network, no cost. **The default**, so the whole repo runs
                      for free.
  PROVIDER=openai ->  OpenAI chat       (needs OPENAI_API_KEY)
  PROVIDER=claude ->  Claude messages   (needs ANTHROPIC_API_KEY)

Why a mock is perfect *here*. The thesis of this dive is "the model only knows
what's in the window right now." The mock proves it: it answers recall questions
("what's my name?") **only** from facts that are actually present in the messages
you pass it. So when a fact falls out of a sliding window, the mock genuinely
forgets it — and when compaction or long-term memory keeps it, the mock genuinely
remembers. The lesson is visible offline, deterministically, for $0. (Real models
behave the same way, just less predictably.)
"""

from __future__ import annotations

import os
import re
from functools import lru_cache

_OPENAI_CHAT = "gpt-4o-mini"
_CLAUDE_CHAT = "claude-haiku-4-5"
_MOCK_MODEL = "mock-1"

_KEYS = {
    "mock": [],  # the whole point: no key required
    "openai": ["OPENAI_API_KEY"],
    "claude": ["ANTHROPIC_API_KEY"],
}


def provider_name() -> str:
    """The active stack: 'mock' (default), 'openai', or 'claude'."""
    return os.getenv("PROVIDER", "mock").strip().lower()


def required_keys() -> list[str]:
    return _KEYS.get(provider_name(), [])


def describe() -> str:
    p = provider_name()
    if p == "mock":
        return f"mock  (offline, deterministic, model={_MOCK_MODEL}, no key)"
    if p == "openai":
        return f"openai  (chat={_OPENAI_CHAT})"
    if p == "claude":
        return f"claude  (chat={_CLAUDE_CHAT})"
    return f"unknown provider {p!r}"


def ensure_ready() -> None:
    """Fail fast with a friendly message if the stack isn't configured.

    For PROVIDER=mock this never fails — that's the point.
    """
    import sys

    p = provider_name()
    if p not in _KEYS:
        sys.exit(f"PROVIDER={p!r} is not recognized. Set PROVIDER=mock (default), openai, or claude in .env.")
    missing = [k for k in required_keys() if not os.getenv(k)]
    if missing:
        sys.exit(
            f"PROVIDER={p} needs {', '.join(missing)} in the environment. "
            f"Provide them via secrun (see SECRETS.md), or run `secrun python check_setup.py`. "
            f"(Tip: PROVIDER=mock needs no key and runs everything offline.)"
        )


# ---------------------------------------------------------------------------
# The mock "model" — deterministic, offline, and it only knows what you show it.
# ---------------------------------------------------------------------------
#
# `_extract_facts` scans every message it's given for a few simple facts. It
# recognizes both the *natural* way a user states them ("my name is Dana") and the
# *canonical* way `summarize()` writes them ("name: Dana") — so a fact survives
# compaction: it's still readable after the raw turns are replaced by a summary.

_FACT_PATTERNS = {
    "name": [r"\bmy name is ([A-Z][a-zA-Z]+)", r"\bcall me ([A-Z][a-zA-Z]+)", r"\bname:\s*([A-Za-z]+)"],
    "plan": [r"\b(?:on|using|have) the (\w+) plan", r"\bplan:\s*(\w+)"],
    "project": [r"\bmy project is(?: called)? (\w+)", r"\bproject:\s*(\w+)"],
}
# Free-form "remember that ..." notes, kept verbatim.
_REMEMBER_PATTERNS = [r"\bremember(?: that)?[:,]?\s+(.+)", r"\bremember:\s*(.+)"]


def _extract_facts(messages: list[dict]) -> dict:
    """Pull the handful of facts the mock can recall, from whatever is in `messages`."""
    facts: dict = {}
    notes: list[str] = []
    for msg in messages:
        text = msg.get("content", "")
        for key, patterns in _FACT_PATTERNS.items():
            for pat in patterns:
                m = re.search(pat, text, re.IGNORECASE)
                if m:
                    facts[key] = m.group(1).strip().rstrip(".")
        for pat in _REMEMBER_PATTERNS:
            for m in re.finditer(pat, text, re.IGNORECASE):
                note = m.group(1).strip().rstrip(".;]")
                if note and note.lower() not in (n.lower() for n in notes):
                    notes.append(note)
    if notes:
        facts["notes"] = notes
    return facts


def _last_user(messages: list[dict]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return msg.get("content", "")
    return ""


_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "do", "does", "did", "what", "when", "where",
    "who", "which", "how", "why", "my", "your", "you", "i", "me", "to", "of", "in", "on",
    "and", "or", "should", "use", "for", "that", "this", "it", "they", "them", "with",
}
_QUESTION_STARTS = {"what", "when", "where", "who", "which", "how", "why", "is", "are", "do", "does", "can", "could", "would", "will"}


def _looks_like_question(text: str) -> bool:
    words = text.lower().split()
    return "?" in text or (bool(words) and words[0] in _QUESTION_STARTS)


def _significant(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOPWORDS and len(w) > 2}


def _search_context(system: str, messages: list[dict], question: str) -> str | None:
    """Mock 'retrieval': return the context line that best overlaps the question.

    This is what lets injected long-term memory or retrieved docs actually get
    *used* by the mock — it answers from whatever text is in the window, exactly
    like a grounded real model would."""
    qtokens = _significant(question)
    if not qtokens:
        return None
    best, best_score = None, 0
    sources = [system] + [m.get("content", "") for m in messages if m.get("role") != "assistant"]
    for content in sources:
        for line in re.split(r"[\n.]", content):
            line = line.strip(" -")
            if not line or line.lower() == question.lower():
                continue
            score = len(qtokens & {w for w in re.findall(r"[a-z0-9]+", line.lower())})
            if score > best_score:
                best, best_score = line, score
    return best if best_score >= 1 else None


def _mock_generate(system: str, messages: list[dict]) -> str:
    # Scan the system prompt too: that's where compaction parks its summary, so a
    # fact carried in the summary must be recallable just like one in a raw turn.
    facts = _extract_facts([{"role": "system", "content": system}, *messages])
    q = _last_user(messages).lower()

    # Recall questions — answered ONLY from facts present in the window. A compound
    # question ("my name and what I asked you to remember") gets each part answered.
    parts: list[str] = []
    recall_intent = False
    if "my name" in q or "who am i" in q:
        recall_intent = True
        if "name" in facts:
            parts.append(f"Your name is {facts['name']}.")
    if "plan" in q and ("which" in q or "what" in q or "my" in q):
        recall_intent = True
        if "plan" in facts:
            parts.append(f"You're on the {facts['plan']} plan.")
    if "project" in q and ("which" in q or "what" in q or "my" in q or "name" in q):
        recall_intent = True
        if "project" in facts:
            parts.append(f"Your project is {facts['project']}.")
    if "remember" in q or "what did i ask" in q or "what should you" in q:
        recall_intent = True
        if facts.get("notes"):
            parts.append("You asked me to remember: " + "; ".join(facts["notes"]) + ".")
    if parts:
        return " ".join(parts)

    # No structured fact matched, but it's a recall question or any other question:
    # answer from whatever context is in the window (injected long-term memory,
    # retrieved docs, earlier turns) — grounded, like a real model would.
    if recall_intent or _looks_like_question(q):
        hit = _search_context(system, messages, _last_user(messages))
        if hit:
            return hit if hit.endswith((".", "!", "?")) else hit + "."
        return "I don't have that in our current context."

    # Otherwise: a deterministic, fact-aware acknowledgement so multi-turn chats
    # feel coherent offline without pretending to be a real model.
    known = []
    if "name" in facts:
        known.append(f"your name is {facts['name']}")
    if "plan" in facts:
        known.append(f"you're on the {facts['plan']} plan")
    if facts.get("notes"):
        known.append(f"I'm holding {len(facts['notes'])} note(s)")
    tail = f" (So far I know: {', '.join(known)}.)" if known else ""
    return f"Got it.{tail}"


def _mock_summarize(messages: list[dict]) -> str:
    """Condense turns into a canonical, re-readable fact summary (deterministic)."""
    facts = _extract_facts(messages)
    lines = []
    if "name" in facts:
        lines.append(f"name: {facts['name']}")
    if "plan" in facts:
        lines.append(f"plan: {facts['plan']}")
    if "project" in facts:
        lines.append(f"project: {facts['project']}")
    for note in facts.get("notes", []):
        lines.append(f"remember: {note}")
    body = "\n".join(lines) if lines else "(no durable facts stated)"
    return f"Summary of {len(messages)} earlier message(s):\n{body}"


# --- Real providers: created lazily so importing this module never forces an
#     SDK import or a network call. ---


@lru_cache(maxsize=1)
def _openai_client():
    from openai import OpenAI

    return OpenAI()


@lru_cache(maxsize=1)
def _anthropic_client():
    import anthropic

    return anthropic.Anthropic()


def generate(system: str, messages: list[dict], *, max_tokens: int = 512) -> str:
    """Answer a conversation: a system prompt + a list of {role, content} messages.

    The model only sees `messages` — which is the whole point of this dive. Trim
    them and it forgets; summarize them and it remembers the summary.
    """
    ensure_ready()
    p = provider_name()
    if p == "mock":
        return _mock_generate(system, messages)
    if p == "openai":
        resp = _openai_client().chat.completions.create(
            model=_OPENAI_CHAT,
            max_tokens=max_tokens,
            messages=[{"role": "system", "content": system}, *messages],  # type: ignore[arg-type]
        )
        return resp.choices[0].message.content or ""
    if p == "claude":
        resp = _anthropic_client().messages.create(
            model=_CLAUDE_CHAT,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": m["role"], "content": m["content"]} for m in messages],
        )
        return "".join(b.text for b in resp.content if b.type == "text")
    raise ValueError(f"Unknown PROVIDER={p!r} (expected 'mock', 'openai', or 'claude').")


_SUMMARY_SYSTEM = (
    "You compress a conversation into a short, factual summary that preserves every "
    "durable detail a later turn might need: names, preferences, decisions, IDs, and "
    "anything the user asked to remember. Be terse. Do not invent facts."
)


def summarize(messages: list[dict], *, max_tokens: int = 256) -> str:
    """Condense old turns into a compact summary string (the heart of compaction)."""
    ensure_ready()
    p = provider_name()
    if p == "mock":
        return _mock_summarize(messages)

    transcript = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
    summary = generate(
        _SUMMARY_SYSTEM,
        [{"role": "user", "content": f"Summarize this conversation:\n\n{transcript}"}],
        max_tokens=max_tokens,
    )
    return f"[Summary of {len(messages)} earlier message(s): {summary.strip()}]"
