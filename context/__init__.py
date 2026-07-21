"""The from-scratch context-engineering library.

`providers.py` is the only file that talks to a model (mock / openai / claude).
Everything else is provider-neutral logic about *what goes in the window*:
  - tokens.py    estimate the budget
  - memory.py    full / window / summary (compaction) conversation memory
  - longterm.py  persistent cross-session memory with retrieval
  - assemble.py  fit & order sections under a token budget
"""

from . import assemble, cost, longterm, memory, tokens
from .providers import describe, ensure_ready, generate, provider_name, summarize

__all__ = [
    "tokens",
    "cost",
    "memory",
    "longterm",
    "assemble",
    "generate",
    "summarize",
    "provider_name",
    "describe",
    "ensure_ready",
]
