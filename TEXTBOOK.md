# Chapter 10: The Window, or Memory as a Policy

*This is the textbook chapter for the Context Engineering deep dive, a bonus dive that slots after [Agents](../agents-deep-dive/TEXTBOOK.md) and pairs with [RAG](../rag-deep-dive/TEXTBOOK.md). The [README](README.md) is the lab manual; this is the lecture. It covers why "memory" in AI products is an illusion you construct, the techniques that construct it, and the two findings in this dive that most textbooks would not print: that more context makes answers worse, and that shrinking the window can raise the bill.*

---

## 10.1 The amnesiac at the desk

There is a useful way to picture a language model that makes this whole chapter fall into place. Imagine a brilliant analyst with total amnesia. Each morning they arrive knowing nothing about you, your project, or any previous conversation. On their desk is a stack of paper, whatever someone chose to leave there, and the desk has a fixed size. They read the stack, do excellent work on exactly what it contains, and at the end of the day everything is shredded. Tomorrow, a fresh stack.

The desk is the context window. The stack is the prompt. And the someone who decides what goes on the desk is you. Every impression you have ever had of an AI product "remembering" you (the chat that recalls your name from an hour ago, the assistant that knows your preferences from last month) is a fact about what got placed on the desk that morning, not about the analyst. You learned in Chapter 1 that the API is stateless; this chapter is about the consequences finally arriving in force, and about the discipline that emerged to manage them:

> **The model only knows what is in its context window right now. Context engineering is deciding what goes in, in what order, and what to drop when it won't all fit.**

The name is recent. Through 2023 and 2024, "prompt engineering" carried the load, but as applications grew from single requests into long conversations and looping agents, practitioners noticed the hard problem had moved: not how to phrase the request, but how to curate everything around it. By mid-2025 the term "context engineering" had stuck, with the sentiment, echoed by people like Andrej Karpathy, that it better described the actual industrial art: filling the window with just the right things, from competing sources, under a budget. If Chapter 3 was how you ask, this chapter is what the model can see when you ask.

This dive teaches it on a deliberately honest apparatus: a deterministic mock model that answers recall questions only from facts literally present in the messages you pass it. When a fact scrolls out of the window, the mock genuinely cannot answer; when a technique preserves the fact, the mock genuinely can. Forgetting and remembering become observable, offline, on cue, which no real model will do for you reliably enough to teach with.

## 10.2 The arithmetic, and the first honest failure

Everything starts with subtraction. The window is a fixed number of tokens; a conversation only grows; therefore, at some turn, it does not fit. The lab makes this concrete with a tiny 2,000-token window and a chat that walks into the wall turn by turn. Real windows are far larger, but "larger" only moves the wall; agents in particular (whose every step appends a tool call and its result) find any wall quickly.

The simplest response is the **sliding window**: keep the system prompt and the most recent turns that fit, and let the oldest scroll off. It is bounded, cheap, and the lab makes you watch its cost in the most human terms available: Dana introduces herself, the conversation runs long, and when you ask her name, it is simply gone. Not "the model is being forgetful," but the stronger, stranger fact this dive wants you to internalize: for the model, the conversation *is* the window. There is no other place the name could have lived. The earlier dives' "simple trim" of history has exactly this failure, silently, and most quick chatbot tutorials ship it.

Notice what kind of thing memory has turned out to be. It is not a feature the model has or lacks; it is a **policy you implement**: the list has a budget, and you choose what survives. Everything else in this chapter is a better policy.

## 10.3 Compaction: trading words for facts

The move that separates real assistants from demo chatbots is **compaction**: when old turns must go, do not delete them, summarize them. Fold what falls off into a running summary that rides along in the system prompt, keep recent turns verbatim, and the conversation stays under budget while its facts survive. The lab reruns the exact conversation that forgot Dana, and this time the model recalls both her name and the thing she asked it to remember. The exact wording of turn three is gone forever; what turn three established is not.

This is what commercial assistants actually do when a long chat "remembers" (Claude Code users have watched it happen at the `/compact` boundary), and it comes with the tradeoffs of any lossy compression. The summary is written by a model, so it can drop the one detail that turns out to matter, and it costs an extra model call each time it runs. Both facts have engineering consequences: the first means compaction quality is a real eval target (a memory bug is a silent quality regression, and Chapter 5's machinery applies), and the second joins a bigger surprise saved for section 10.7.

Human memory, incidentally, works more like compaction than like a transcript: you do not store conversations verbatim, you store gist and reconstruct. The analogy is imperfect in instructive ways (the model's summary is frozen text, not a living reconstruction), but it explains why compaction feels natural in products: it forgets the way people expect things to be forgotten, details first, substance last.

## 10.4 Long-term memory: RAG pointed at yourself

Compaction keeps facts alive within a session. Close the chat and they shred with the rest of the desk. Products that remember you across sessions (the assistant that knows your name next week) add a third layer: a durable store *outside* the window, with retrieval to bring the relevant few facts back *in* when a new turn needs them.

If that sounds familiar, it should: it is Chapter 4's retrieval-augmented generation, with the corpus being your own past conversations instead of a document folder. Write durable facts out; on each new turn, retrieve what is relevant to the question and place it in the context. The lab's version persists facts to a JSON file and retrieves by word overlap, an honest toy whose upgrade path is exactly the RAG dive's vector store; the architecture does not change, only the retriever's IQ. The demonstration is designed to be unambiguous: session two starts with a fresh, empty window and answers correctly *only because* it recalled a fact session one stored.

Two production realities deserve early mention because they change designs. First, memory needs lifecycle management: facts go stale ("I moved to the Team plan" should beat the six-month-old plan fact), contradictions need resolving, and rank should blend relevance with recency. Second, and easy to miss: stored memory is untrusted input. Text a model wrote into a store months ago, or that arrived from a user, re-enters the window later with all the authority of context, which makes memory a classic indirect-injection channel (Chapter 7's subject). Provenance on stored facts is not bureaucracy.

## 10.5 Order, assembly, and the case against a full desk

Fitting the right text is half the job. The other half is placement and restraint, and both rest on findings about how models actually read.

**Order matters.** Models attend most reliably to the start and the end of a long context and can miss what sits in the middle. This was documented carefully in a 2023 paper aptly titled "Lost in the Middle," which found accuracy on retrieval tasks forming a U-shape across position: strong at the edges, sagging in the center. The practical rule costs nothing to apply: put what matters most at the edges (instructions early, the question late) and let the filler take the middle. Same tokens, better recall. Anyone who has written for human skimmers will recognize the shape of the advice; do not bury the lede.

**Assembly is triage.** A real request's context is contested territory: system prompt, tool catalog, retrieved documents, long-term memories, recent turns, all competing for a budget they collectively exceed. The lab's assembler makes the policy explicit: assign priorities, pack the highest that fit, drop the rest, so a marketing page can never crowd out the user's actual question. The point is not this particular algorithm; it is that the window's contents should be a decision rather than an accumulation.

**And more is not better.** The intuition "give it everything, let the model sort it out" fails in a specific, demonstrable way the lab calls **context rot**. Padding the window dilutes the signal, invites the model to latch onto a plausible-but-irrelevant passage, and bills you for every wasted token on every turn (remember, the whole conversation is re-sent each time). The example is built to make the failure vivid: the same question answered from a lean context and from a bloated one containing a distractor that names a different person; the bloated window costs about ten times the tokens and returns the wrong name. On the mock, the flip is deterministic by construction; on real models the same effect arrives as attention wandering to distractors, less predictably but genuinely. The slogan earns its keep: relevance beats volume, and the cheapest, fastest, most accurate token is the one you did not send.

## 10.6 The agent case: pruning the observation pile

Agents deserve their own section because they are the pathological case. Every step of the loop appends a tool call and its full result, and tool results are enormous: file contents, query outputs, API responses. Ten steps in, the window is mostly stale observations the agent already extracted what it needed from, and the reasoning thread is drowning in its own exhaust.

**Observation pruning** applies the whole chapter to that specific shape: keep the agent's reasoning and the most recent tool results verbatim, and stub out the old ones ("[result used in step 3, 4,200 tokens, pruned]"). The lab's example reclaims about three quarters of the window while leaving the train of thought intact. Combined with the subagent isolation of Chapter 9 (delegate the noisy work to a child whose window is disposable), this is how long-horizon agents stay coherent past the point where a naive loop would have choked on its own history.

## 10.7 The plot twist: fewer tokens, bigger bill

The lab saves its most instructive result for last, and it is the kind this series exists to teach: a case where the two things you are optimizing quietly point in opposite directions.

Recall prompt caching from Chapters 1 and 2: providers cache the prompt *prefix*, and the rule is unforgiving, any change anywhere in the prefix invalidates everything after it. Now watch the two memory strategies through that lens. An append-only history never touches its prefix; every turn, all prior tokens are cheap cache reads (about a tenth of normal price) and only the new turn is written. Compaction, by contrast, *rewrites the prefix*: it edits the summary in the system prompt and drops old turns, so the next request is a cache miss that repays full freight on the whole context. The lab bills the same conversation both ways and compaction comes out around one and a half times more expensive, despite sending fewer tokens. Fewer tokens, bigger bill.

Sit with that, because the general lesson outranks the specific number. "Smaller context" and "smaller bill" are different axes, coupled through a caching mechanism two layers away, and an engineer who optimizes one metric without instrumenting the other ships regressions with a clean conscience. The synthesis is not "never compact"; unbounded histories lose eventually, on window limits if nothing else, and on very long chats the crossover arrives where compaction wins even on cost. The synthesis is cache-aware memory: keep the prefix stable as long as you can, compact rarely and in bulk (paying the miss once, not every turn), and watch the cache-read and cache-write numbers in your usage data rather than assuming. Honest tradeoffs, measured, over tidy rules: it is the house style of this series because it is how the systems actually behave.

## 10.8 Where this chapter leaves you

The capstone chat is the whole argument running in one place: it stays under a token budget no matter how long you talk (compaction), remembers durable facts across completely separate runs (long-term memory), and will show you its assembled window on demand (`--show-context`), which is the habit worth keeping. Engineers who regularly look at what was actually sent, rather than what they assume was sent, catch a class of bugs everyone else attributes to the model's mood.

You leave with a layered mental model that maps cleanly onto human terms without depending on the analogy: the window as working memory, expensive and small; compaction as gist; the long-term store as notes you keep and consult; assembly as deciding what belongs on the desk this morning. And you leave with this dive's two contrarian findings in your pocket (more context can be worse; less context can cost more), both of which you have watched happen rather than read about, which is the difference this series is built on.

The window question radiates into the neighboring dives from here. Multimodal (Chapter 11) raises the stakes on budgeting, because images are token-expensive tenants. Harnesses (Chapter 9) gave agents per-child windows; this chapter is why that mattered. And everything here is an eval target: whether the summary kept the facts that mattered is a measurable question, and Chapter 5 taught you to measure it.

---

*Lab manual: [README.md](README.md) · Exercises: [EXERCISES.md](EXERCISES.md) · Pairs with: [RAG](../rag-deep-dive/TEXTBOOK.md) · Builds on: [Agents](../agents-deep-dive/TEXTBOOK.md)*
