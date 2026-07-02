# Exercises — make the learning stick

Reading code teaches you less than *predicting* what it will do and then checking.
This file turns each section of the [README](README.md) into a few quick
active-recall prompts.

How to use it: work the section first, then come back. **Commit to an answer before
you run or reveal** — the prediction is where the learning happens, even
(especially) when you're wrong. Answers are hidden behind ▸ toggles.

> Everything runs offline on `PROVIDER=mock` — no key, no cost. The mock answers
> recall questions only from what's actually in the window, so "did it remember?"
> is a real, deterministic test.

---

## Section 2 — The window is a budget **(offline)**

**Recall.** Why is "memory" a budgeting problem and not a model feature? What two
costs grow with every turn you keep in the history?

<details><summary>▸ Answer</summary>

The API is stateless — "memory" is just the message list you resend, and that list
has a hard token ceiling (the context window). Every kept turn costs **tokens**
(you re-send and re-pay for it each request) and pushes you toward **overflow**
(the request fails, or quality drops). So you must choose what to keep.
</details>

---

## Section 3 — The sliding window **(mock)**

**Predict, then run.** In `examples/02_sliding_window.py`, Dana says her name in
turn 1, then the chat runs long on a small budget. When you finally ask "What's my
name?", what does the model answer, and *why*?

<details><summary>▸ Answer</summary>

It can't recall it — turn 1 scrolled off the back of the window, so it isn't in the
messages sent. For the model the conversation *is* the window; delete the
introduction and the name is genuinely gone. Bounded and cheap, but forgetful.
</details>

---

## Section 4 — Compaction **(mock)**

**Predict.** `examples/03_compaction.py` runs the SAME long chat on the SAME budget
as §3, but with `SummaryMemory`. Does the model recall the name now? What exactly
got kept, and what got thrown away?

<details><summary>▸ Answer</summary>

Yes — it recalls the name *and* the thing it was asked to remember. Compaction
replaced the oldest turns with a short **summary** (which preserves the *facts*) and
kept the recent turns **verbatim**. What's thrown away is the exact wording of old
turns — and you pay one summarization call. The risk: a bad summary drops a detail.
</details>

**Do.** Lower the budget (`SummaryMemory(budget_tokens=120, ...)`) and rerun. Does
the compaction count go up? Is the name still recalled?

<details><summary>▸ Answer</summary>

More compactions fire (the window fills sooner), but the name survives — because
each compaction folds the prior summary into the next, and the canonical fact lines
carry through. That's the point: bounded memory that doesn't forget the essentials.
</details>

---

## Section 5 — Long-term memory **(mock)**

**Recall.** Compaction kept a fact alive within one conversation. Why isn't that
enough, and what does long-term memory do differently?

<details><summary>▸ Answer</summary>

A summary lives in *that conversation's* window — close the session and it's gone.
Long-term memory writes durable facts to a store *outside* any window and retrieves
the relevant few back *in* when a new turn needs them. It's RAG pointed at the
conversation: persist, then retrieve by relevance.
</details>

**Predict, then run.** In `examples/04_long_term_memory.py`, session two starts with
an empty window. Why does it answer "When is my launch?" correctly *with* memory and
fail *without* it?

<details><summary>▸ Answer</summary>

The window holds nothing from session one, so without retrieval the fact simply
isn't present and the model says it doesn't have it. With long-term memory, the
relevant fact ("launch is Friday") is recalled from the store and injected into the
system prompt — back in the window, where the model can use it.
</details>

---

## Section 6 — Order matters **(offline)**

**Recall.** What is the "lost in the middle" effect, and what does `order_for_recall`
do about it?

<details><summary>▸ Answer</summary>

Models attend most reliably to the **start** and **end** of a long context and are
worse at using material buried in the **middle**. `order_for_recall` places the
highest-priority sections at the edges and the filler in the middle — same tokens,
better odds the key chunk is actually used.
</details>

---

## Section 7 — Assembling under a budget **(offline)**

**Predict.** In `examples/06_assemble_budget.py`, the candidate sections total more
than the budget. Which sections survive, and which is first to be cut?

<details><summary>▸ Answer</summary>

The highest-priority sections that fit survive — the system prompt and the user's
question are non-negotiable; long-term memory and the on-topic retrieved doc make
it; the marketing blog (lowest priority) is cut first. Prioritize, then pack: a
giant low-priority doc must never crowd out a small essential one.
</details>

---

## Section 8 — Context rot **(mock)**

**Predict, then run.** `examples/07_context_rot.py` answers the same question with a
lean context and a bloated one. The mock gets it right both times — so what's the
measured cost of the bloat, and what cost can't the mock show you?

<details><summary>▸ Answer</summary>

The bloated window costs ~10× the tokens (so ~10× the spend and more latency) for
the identical answer — on *every* turn. What the mock can't show is the quality
cost: on a real model, padding dilutes the signal and invites it to latch onto an
irrelevant passage. Relevance beats volume.
</details>

---

## Section 9 — Pruning observations **(offline)**

**Recall.** Why do agents bloat their context faster than chats, and what does
observation pruning keep vs. drop?

<details><summary>▸ Answer</summary>

Each agent step appends a tool call *and* its full (often huge) result, and most of
those results are spent after one step. Pruning keeps the agent's **reasoning** and
the most **recent** observations verbatim, and stubs/drops the old ones — a big
token cut with the train of thought intact.
</details>

---

## Section 10 — Caching vs. compaction **(offline)**

**Predict (`09`).** Compaction keeps the window far *smaller* every turn than an
append-only history. So it must cost less to run — right?

<details><summary>▸ Answer</summary>

Not necessarily — here it costs about **1.5× more**. Providers cache the prompt
*prefix*, and any change to the prefix invalidates everything after it. Append-only
never changes its prefix, so almost every token is a cheap cache *read* (~0.1×).
Compaction rewrites the system prompt (the summary) and drops old turns — a prefix
change — so those turns are cache *misses* billed at full write price (~1.25×) on the
whole context. Fewer tokens, bigger bill.
</details>

**Recall.** If compaction can raise the bill, why compact at all — and how do you get
its benefit without the cache penalty?

<details><summary>▸ Answer</summary>

You still need it eventually: an append-only window grows without bound, so on a
*very long* conversation its cheap-but-huge reads finally cost more than a small
uncached context. The two axes ("fewer tokens" vs "cheaper bill") cross over. The fix
is cadence: compact **rarely and in bulk** so you pay the cache miss once instead of
every turn, keep the stable part of the prefix byte-identical, and watch
`cache_read_input_tokens` vs `cache_creation_input_tokens` to see which regime you're
in.
</details>

---

## Capstone — `chat.py`

**Do.** Run `python hands_on/chat.py "Hi, my name is Dana. Remember our launch is
Friday."`, then run `python hands_on/chat.py "When is my launch?"` as a *separate*
command. How does the second run know, when it shares no memory in-process with the
first?

<details><summary>▸ Answer</summary>

The first run persisted the durable facts to `.ctx_memory.json`. The second run is a
fresh process with an empty window, but it loads that file and *recalls* the launch
fact into its system prompt. In-session memory (compaction) and cross-session memory
(the store) are different layers — the capstone uses both.
</details>

**Predict.** Run the REPL with `--show-context --budget 200` and chat past the
budget. What happens to the "compactions" counter and the token line as you go, and
why does the window never overflow?

<details><summary>▸ Answer</summary>

Compactions climb each time the budget is exceeded, and the token line stays at or
under 200 — because compaction folds old turns into the summary the moment they'd
push you over. Bounded memory: the window can't grow without bound no matter how long
you talk.
</details>

**Stretch.** Switch to `PROVIDER=openai` or `claude` (add a key) and repeat. Where
does the real model do *better* than the mock, and where might it do *worse*?

<details><summary>▸ Answer</summary>

Better: it summarizes prose far more intelligently and recalls paraphrased facts the
mock's keyword match would miss. Worse: it's nondeterministic — a summary can quietly
drop a detail, and recall can miss. That's exactly why memory quality is something you
**measure** (the Evals dive), not assume.
</details>

---

### Where to take it next

Invent your own. Take a long conversation you care about, pick a budget, and decide
your policy: what to summarize, what to persist forever, what to drop. The first time
your assistant recalls something from twenty turns (or twenty days) ago — under a
fixed budget — context engineering has clicked.
