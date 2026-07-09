# Context Engineering — A Guided Deep Dive

A hands-on playground for the skill the other dives keep bumping into: **managing
what's in the context window.** A model only knows what you put in front of it
*right now* — so as conversations get long, documents pile up, and agents loop, the
real work becomes deciding what to keep, what to drop, what to summarize, and in
what order. You'll build a token budgeter, three kinds of conversation memory, a
persistent long-term store, and a context assembler from scratch — and watch a chat
that *remembers* under a fixed budget.

The twist that makes this repo work: it runs **completely offline on a mock
provider**, with no API key. The mock is a deterministic "model" that answers recall
questions **only** from facts actually present in the window — so when a fact falls
off a sliding window the mock genuinely forgets it, and when compaction or long-term
memory keeps it, the mock genuinely remembers. The whole thesis is *visible*,
offline, for $0. Flip one env var and the same code runs against a real OpenAI or
Claude model.

This repo is **standalone**, but it's the missing half of "prompt engineering": if
[Prompt Engineering](https://github.com/alexvervloet/prompt-engineering-deep-dive) is
*how you ask*, this is *what the model can see when you ask*. It extends the "memory
is just the message list you resend" idea from the
[API](https://github.com/alexvervloet/openai-api-deep-dive) and
[Agents](https://github.com/alexvervloet/agents-deep-dive) dives, and its long-term memory
is the [RAG](https://github.com/alexvervloet/rag-deep-dive) pattern pointed at a
conversation — but its code depends on none of them.

Like its siblings, it's meant to be *walked through*. Each section ends with
something to run, and **every section runs offline and free** on the mock.
[EXERCISES.md](EXERCISES.md) has a predict-then-run prompt for each one.

---

## 0. The one big idea

> **The model only knows what's in its context window right now. Context
> engineering is deciding what goes in, in what order, and what to drop when it
> won't all fit.**

That's the whole repo. "Memory" isn't a model feature — it's a *policy* you
implement: resend the message list, but the list has a budget, so you choose what
survives. Compaction summarizes what won't fit; long-term memory stores what should
outlive the conversation; assembly decides which competing sources make the cut and
where they sit. Every section is a variation on that one sentence. Hold onto it and
none of this feels complicated.

---

## 1. Setup (5 minutes)

```bash
# 1. Create an isolated Python environment
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install dependencies (the default mock stack needs only python-dotenv)
pip install -r requirements.txt

# 3. Copy the env file — the default runs keyless (no API key needed)
cp .env.example .env
#    (Real provider instead of the mock? Its key goes in your OS keychain,
#     not .env — see ../SECRETS.md — then run scripts as `secrun python ...`.)

# 4. Confirm everything is wired up (makes no API call, costs nothing)
python check_setup.py
```

That's it — no key required. The default `PROVIDER=mock` is a deterministic,
in-process "model." Pick your stack with `PROVIDER` in `.env`:

| `PROVIDER` | What runs the model | Keys needed | Cost |
|------------|---------------------|-------------|------|
| `mock` (default) | a deterministic offline "model" | **none** | **$0** |
| `openai` | OpenAI `gpt-4o-mini` | `OPENAI_API_KEY` | tiny |
| `claude` | Claude `claude-haiku-4-5` | `ANTHROPIC_API_KEY` | tiny |

The only file that knows which you picked is [context/providers.py](context/providers.py).
Everything else is provider-neutral logic about *what goes in the window*.

> 💡 **Why a mock is the right call here.** The subject is the context window, not
> the model. The mock answers recall questions only from what's actually in the
> messages you pass it — so forgetting and remembering are *observable* offline and
> deterministically. Real models behave the same way, just less predictably.

---

## 2. The window is a budget

```bash
python examples/01_token_budget.py        # offline — no model needed
```

Everything starts with arithmetic: the context window is a fixed number of tokens,
and a conversation only grows. [context/tokens.py](context/tokens.py) estimates
tokens with a simple ~4-chars/token heuristic (no tokenizer, no key), and this
example grows a chat turn by turn until it overflows a deliberately tiny 2,000-token
window. That overflow is the problem the rest of the repo solves.

---

## 3. The sliding window — and what it forgets

```bash
python examples/02_sliding_window.py
```

The simplest fix for overflow: keep the system prompt plus the most recent turns
that fit, and let the oldest scroll off. `WindowMemory` does it. It's bounded and
cheap — and genuinely *forgetful*: Dana introduces herself, the chat runs long, and
when you ask her name it's gone. That's the failure the "simple trim" in the other
dives quietly has. **For the model, the conversation is the window.**

---

## 4. Compaction — summarize what falls off

```bash
python examples/03_compaction.py
```

Same budget as §3, but instead of *deleting* old turns, `SummaryMemory` folds them
into a running **summary** and keeps the recent turns verbatim. The exact words are
gone; the *facts* survive. Run the same long conversation and this time the model
recalls both the name and the thing it was asked to remember. This is the single
most important technique here — what real assistants do when a long chat
"remembers." You trade exact wording (and one summarization call) for durable memory
under a fixed budget.

---

## 5. Long-term memory — remembering across sessions

```bash
python examples/04_long_term_memory.py
```

Compaction keeps a fact alive *within* a conversation. Close the session and it's
gone. Long-term memory writes durable facts to a store *outside* the window and
retrieves the relevant few back *in* when a new turn needs them — RAG pointed at the
conversation. [context/longterm.py](context/longterm.py) persists facts to JSON and
retrieves by overlap (swap in real embeddings from the RAG dive for production). The
example runs two sessions against a fresh, empty window: session two answers
correctly *only because* it recalled a fact session one stored.

---

## 6. Order matters — don't bury the lede

```bash
python examples/05_ordering.py        # offline
```

Fitting the right text is half the job; *where* you put it is the other half. Models
attend most reliably to the **start** and **end** of a long context and can miss
what's buried in the **middle** (the "lost in the middle" effect). `order_for_recall`
places the highest-priority sections at the edges and the filler in the middle — same
tokens, better recall.

---

## 7. Assembling the window under a budget

```bash
python examples/06_assemble_budget.py        # offline
```

A real request's context competes for space: system prompt, tools, retrieved docs,
long-term memory, recent turns — and they rarely all fit. `assemble()` prioritizes,
then packs: keep the highest-priority sections that fit, drop the rest, so a
marketing blog can't crowd out the user's actual question. Assembling the window on
purpose is the difference between a focused request and a bloated one.

---

## 8. More context is not better (context rot)

```bash
python examples/07_context_rot.py
```

A big window is a budget, not a goal. Padding it "just in case" dilutes the signal,
invites the model to latch onto an irrelevant passage, and bills you for every
wasted token on every turn — "context rot." The example answers the same question
with a lean context and a bloated one where a plausible distractor names a
*different* person — and the bloated window both costs ~10× the tokens **and returns
the wrong name**. On the mock that flip is deterministic (it naively takes the last
"my name is …" it sees — a crude stand-in for a real model's attention wandering to
a distractor); add a key and a harder question to watch a subtler version on a real
model. Relevance beats volume; the cheapest, fastest, most accurate token is the one
you didn't send.

---

## 9. Pruning an agent's observations

```bash
python examples/08_pruning_observations.py        # offline
```

Agents bloat context fastest: every step appends a tool call *and* its full result,
and tool results are huge. Ten steps in, the window is mostly stale observations the
agent already used. Observation pruning keeps the reasoning and the most recent
results verbatim and stubs the old ones — here a 75% smaller window with the agent's
train of thought intact. It's compaction, long-term memory, and assembly applied to
an agent's context.

---

## 10. The hidden cost: compaction breaks your prompt cache

```bash
python examples/09_caching_vs_compaction.py        # offline
```

Every technique so far shrinks the window. This one shows the bill they can quietly
*raise*. Providers cache the prompt **prefix**, and the rule is unforgiving: any
change anywhere in the prefix invalidates everything after it. An **append-only**
history never touches its prefix — so every prior token is a cheap cache *read*
(~0.1×) and only the new turn is written (~1.25×). **Compaction rewrites the
prefix**: it changes the system prompt (the summary) and drops old turns, so the
next request is a cache *miss* that pays full write price on the whole context.
The example bills the same conversation both ways ([context/cost.py](context/cost.py))
and finds compaction costing **~1.5×** as much here — fewer tokens, bigger bill.
It's the honest tradeoff this series insists on: "cheaper context" and "cheaper
bill" are different axes. (The crossover: on *very* long chats the unbounded
append-only window finally loses — so when you must compact, do it rarely and in
bulk, paying the cache miss once, not every turn.)

---

## 11. The capstone: `chat.py`

Everything assembled into a chat you'd actually use: it stays inside a token budget
no matter how long you talk (compaction), **and** remembers durable facts across
sessions (long-term memory). Each turn, the system prompt is your persona + the
running summary + the long-term facts relevant to what you just asked.

```bash
# State some facts (offline on the mock — no key, no cost):
python hands_on/chat.py "Hi, my name is Dana. Remember our launch is Friday."

# A BRAND-NEW run — and it still knows, because the fact was persisted:
python hands_on/chat.py "When is my launch?"

# Interactive REPL ('/context' to watch the window, '/memory' to list stored facts):
python hands_on/chat.py

# See exactly what's sent each turn; shrink the budget to watch compaction kick in:
python hands_on/chat.py --show-context --budget 200

# Wipe the long-term store:
python hands_on/chat.py --forget
```

Read [hands_on/chat.py](hands_on/chat.py): `respond()` is the whole turn — recall
relevant facts, assemble the system prompt, generate, persist any new durable facts.
The library does the work; the capstone just wires it to a CLI. **Suggested
exercise:** chat for a dozen turns with `--show-context --budget 200` and watch the
compaction count climb while the window stays under budget — then quit, run again,
and notice it still greets you by name.

---

## Where to go next

You've built memory from scratch. The frontier is more of the same idea, at more
scale and rigor:

- **Real embeddings for long-term memory** — swap the keyword overlap for the vector
  store from the [RAG dive](https://github.com/alexvervloet/rag-deep-dive), so recall is by
  meaning, not shared words.
- **Smarter compaction** — summarize hierarchically, or keep structured state (a
  running JSON of facts/decisions) alongside the prose summary.
- **Memory-as-a-tool** — let an agent *decide* what to remember and recall by calling
  `remember()`/`recall()` tools, instead of doing it on every turn.
- **Eviction & freshness** — expire stale facts, resolve contradictions ("I moved to
  the Team plan"), and rank memory by recency *and* relevance.
- **Measure it** — score whether compaction preserved the facts that mattered with
  the [Evals dive](https://github.com/alexvervloet/evals-deep-dive): a memory bug is a
  silent quality regression.
- **Multi-agent context isolation** — give sub-agents their own focused windows so one
  agent's clutter never pollutes another's.

---

## From teaching code to production

The teaching shortcuts here are exactly what you'd harden once a memory layer is on a
live path:

| This repo's teaching shortcut | In production |
|-------------------------------|---------------|
| Token count is a ~4-chars/token estimate | The real tokenizer / the API's `usage`, and budgets enforced per request |
| Long-term memory is a local JSON file | A real **vector DB** with embeddings, per-user namespaces, and access control |
| Compaction summarizes inline, every overflow | Summarization wrapped in **retries + a cost budget** (it's an extra model call) |
| Recall is keyword overlap | Embedding similarity + reranking, and a relevance threshold so junk isn't injected |
| Facts are trusted and never expire | **Eviction, contradiction resolution, and provenance** on stored memory |
| A summary might silently drop a needed fact | An **eval gate** on memory quality, so a compaction regression fails the build |
| Compaction/pruning run on every overflow (§10) | **Cache-aware memory**: keep the prefix stable, compact rarely and in bulk, and watch `cache_read_input_tokens` vs `cache_creation_input_tokens` so shrinking the window doesn't grow the bill |
| Stored memory is trusted text | **Guardrails** — memory is untrusted input and a classic indirect-injection vector |

The general ops machinery — observability, cost, reliability, caching, guardrails,
prompt versioning, eval gates — is built from scratch and wired into one running app
in **[Production](https://github.com/alexvervloet/ai-in-production-deep-dive)** (#8 in the
series), which also runs offline on a mock provider.

---

## File map

```
check_setup.py              ← run first: Python, packages, provider
README.md                   ← this guide
EXERCISES.md                ← predict-then-run prompts, one per section
context/                    ← the from-scratch library (read it!)
  providers.py              ← the ONLY provider file: mock (default) + openai + claude
  tokens.py                 ← estimate the token budget (offline heuristic)
  cost.py                   ← a tiny prompt-cache cost model (read vs write vs miss)
  memory.py                 ← Full / Window / Summary (compaction) conversation memory
  longterm.py               ← persistent cross-session memory with retrieval
  assemble.py               ← fit & order sections under a token budget
hands_on/
  chat.py                   ← capstone: a chat that compacts + remembers across runs
examples/
  01_token_budget.py        ← the window is a budget (offline)
  02_sliding_window.py      ← keep recent, drop old — and what it forgets
  03_compaction.py          ← summarize what falls off instead of deleting it
  04_long_term_memory.py    ← remembering across sessions (RAG over the conversation)
  05_ordering.py            ← lost in the middle: put what matters at the edges (offline)
  06_assemble_budget.py     ← prioritize, then pack the window (offline)
  07_context_rot.py         ← more context is not better
  08_pruning_observations.py← trim stale tool results in an agent loop (offline)
  09_caching_vs_compaction.py← compaction blows the prompt cache — fewer tokens, bigger bill (offline)
```

(`.ctx_memory.json` is created by the capstone's long-term memory and is git-ignored.)

---

## Troubleshooting

Run `python check_setup.py` first — it catches most problems. Then, by symptom:

| What you see | What it means / the fix |
|--------------|-------------------------|
| `ModuleNotFoundError: dotenv` | Dependencies aren't installed or the venv isn't active. `source .venv/bin/activate` then `pip install -r requirements.txt`. |
| `PROVIDER=... needs ... in the environment` | You switched to a real provider without a key. Load it from your keychain with `secrun` (see [SECRETS.md](../SECRETS.md)), or go back to `PROVIDER=mock`. |
| The capstone "remembers" things from a previous run | That's long-term memory working — it persists to `.ctx_memory.json`. Run `python hands_on/chat.py --forget` to wipe it. |
| On a real provider, recall is fuzzier than the mock | The mock is deterministic; real models paraphrase and occasionally miss. That's why §8 (don't overload) and the Evals dive matter. |
| The summary dropped a fact I needed | Compaction is lossy — that's the tradeoff. Keep more recent turns verbatim (`keep_recent`) or store the fact in long-term memory. |
| `SyntaxError` / odd type errors on startup | You're likely on Python 3.9 or older; this repo needs 3.10+. `check_setup.py` confirms your version. |

Still stuck? Every file is small and self-contained — open it, read the docstring at
the top, and run it directly. [context/memory.py](context/memory.py) is the heart of
the dive.

---

## The series

This is one of sixteen standalone, hands-on deep dives into building with LLM APIs — eight core, plus eight bonus dives.
Each one stands on its own — its own setup, examples, and capstone — and they all
share the same house style: provider-agnostic, built from scratch (no frameworks),
offline-first examples, and a real capstone. Do them in any order; this sequence
builds naturally:

1. [OpenAI API](https://github.com/alexvervloet/openai-api-deep-dive) — the API from zero
2. [Claude API](https://github.com/alexvervloet/claude-api-deep-dive) — the same ideas, the Anthropic way
3. [Prompt Engineering](https://github.com/alexvervloet/prompt-engineering-deep-dive) — shape model behavior with better prompts
4. [RAG](https://github.com/alexvervloet/rag-deep-dive) — answer questions over your own documents
5. [Evals](https://github.com/alexvervloet/evals-deep-dive) — measure whether a change actually helps
6. [Agents](https://github.com/alexvervloet/agents-deep-dive) — give a model tools and a loop so it can act
7. [Prompt Injection & Guardrails](https://github.com/alexvervloet/prompt-injection-deep-dive) — attack and defend all of the above
8. [Production](https://github.com/alexvervloet/ai-in-production-deep-dive) — operate one app end to end

**Bonus dives** — standalone, slotting in where they're most useful:

- [Context Engineering](https://github.com/alexvervloet/context-engineering-deep-dive) — manage what's in the window: memory, compaction, assembly
- [Multimodal](https://github.com/alexvervloet/multimodal-deep-dive) — images & audio, not just text
- [Fine-tuning](https://github.com/alexvervloet/fine-tuning-deep-dive) — teach a model new behavior by example
- [MCP](https://github.com/alexvervloet/mcp-deep-dive) — serve tools, data & prompts to any LLM over a standard protocol
- [Local Models](https://github.com/alexvervloet/local-models-deep-dive) — run open-weight models on your own machine
- [Agent Harnesses](https://github.com/alexvervloet/agent-harness-deep-dive) — build on the loop: hooks, permissions, sandboxing, subagents
- [Realtime Voice](https://github.com/alexvervloet/realtime-voice-deep-dive) — low-latency speech-to-speech agents
- [Observability](https://github.com/alexvervloet/observability-deep-dive) — watch a running app over time: drift, quality, alerting, the flywheel

**Context Engineering is a bonus dive in the series.** It slots most naturally after
[Agents](https://github.com/alexvervloet/agents-deep-dive) (#6) — whose stateless "resend
the message list" memory it extends into compaction and long-term recall — and pairs
with [RAG](https://github.com/alexvervloet/rag-deep-dive) (#4), which its long-term memory
reuses.
