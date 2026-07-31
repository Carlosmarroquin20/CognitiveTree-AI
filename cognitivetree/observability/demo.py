"""Demonstration of run metrics, token accounting, and consumption budgets.

The offline LLM scenario is executed with its scripted backend wrapped in an
:class:`~cognitivetree.observability.accounting.AccountingLlmClient`, so the
printed report combines search-structural metrics (iterations, backtracks,
revisions, per-phase timing) with the token usage tallied across the run. The
same tally is then reused as a control signal: a second run of the identical
scenario under a deliberately tight :class:`TokenBudget` stops early with the
``budget_exhausted`` outcome.

Run with: ``python -m cognitivetree.observability.demo``
"""

from __future__ import annotations

from cognitivetree.llm.demo import TASK, build_offline_controller, clamp_responder
from cognitivetree.llm.scripted import ScriptedLlmClient
from cognitivetree.observability.accounting import AccountingLlmClient
from cognitivetree.observability.metrics import RunMetrics

# Deliberately below the cost of a single expansion, so the ceiling is crossed
# during the first iteration and detected at the next boundary. A ceiling
# derived from the unconstrained total would not reliably fire: the run only
# makes two calls, and the first may already sit under half the total.
DEMO_TOKEN_CEILING = 50


def main() -> None:
    """Reports an unconstrained run, then the same run under a token ceiling."""
    client = AccountingLlmClient(ScriptedLlmClient(clamp_responder))
    result = build_offline_controller(client=client).run(TASK)
    metrics = RunMetrics.from_result(result, token_usage=client.usage)

    print(metrics.format_report())

    print()
    print(f"--- same scenario under a {DEMO_TOKEN_CEILING}-token ceiling ---")

    capped_client = AccountingLlmClient(ScriptedLlmClient(clamp_responder))
    capped = build_offline_controller(
        client=capped_client,
        # The budget reads the very client the generator spends through.
        max_tokens=DEMO_TOKEN_CEILING,
    ).run(TASK)

    print(f"outcome          : {capped.outcome.value}")
    print(f"stopped because  : {capped.phase_history[-1].note}")
    print(f"iterations       : {capped.iterations} (unconstrained: {result.iterations})")
    print(
        f"tokens consumed  : {capped_client.usage.total_tokens} "
        f"(ceiling {DEMO_TOKEN_CEILING}; the boundary check permits "
        f"one iteration of overshoot)"
    )


if __name__ == "__main__":
    main()
