"""Demonstration of run metrics and token accounting over a full reasoning run.

The offline LLM scenario is executed with its scripted backend wrapped in an
:class:`~cognitivetree.observability.accounting.AccountingLlmClient`, so the
printed report combines search-structural metrics (iterations, backtracks,
revisions, per-phase timing) with the token usage tallied across the run.

Run with: ``python -m cognitivetree.observability.demo``
"""

from __future__ import annotations

from cognitivetree.llm.demo import TASK, build_offline_controller, clamp_responder
from cognitivetree.llm.scripted import ScriptedLlmClient
from cognitivetree.observability.accounting import AccountingLlmClient
from cognitivetree.observability.metrics import RunMetrics


def main() -> None:
    """Runs the offline scenario and prints its metrics report."""
    client = AccountingLlmClient(ScriptedLlmClient(clamp_responder))
    controller = build_offline_controller(client=client)

    result = controller.run(TASK)
    metrics = RunMetrics.from_result(result, token_usage=client.usage)

    print(metrics.format_report())


if __name__ == "__main__":
    main()
