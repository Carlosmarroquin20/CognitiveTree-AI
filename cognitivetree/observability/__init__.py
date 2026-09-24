"""Measurement of reasoning runs, and the budgets derived from it.

:class:`RunMetrics` is post-hoc: it reads only the public
:class:`~cognitivetree.search.SearchResult` surface — the recorded phase
history and the final tree — so metrics impose no instrumentation on the
search core. :class:`AccountingLlmClient` is the one live component, a
transparent wrapper that tallies token usage as completions flow through it,
and :class:`TokenBudget` turns that same tally into a stop condition the
controller can act on.
"""

from cognitivetree.observability.accounting import AccountingLlmClient
from cognitivetree.observability.budget import TokenBudget
from cognitivetree.observability.logs import JsonLogFormatter, configure_logging
from cognitivetree.observability.metrics import RunMetrics, TokenUsage

__all__ = [
    "AccountingLlmClient",
    "JsonLogFormatter",
    "RunMetrics",
    "TokenBudget",
    "TokenUsage",
    "configure_logging",
]
