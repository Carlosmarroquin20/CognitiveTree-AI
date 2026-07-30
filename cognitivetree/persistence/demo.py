"""Demonstration of archiving a run and diagnosing it in a later process.

The scenario is the one archives exist for: a search stops against its
wall-clock budget, leaving no solution and only a partial tree. The run is
written to disk, the in-memory objects are discarded, and the archive is
reopened — recovering the tree, the per-node execution records, the phase
history, and the metrics needed to work out where the budget went.

Run with: ``python -m cognitivetree.persistence.demo``
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from cognitivetree.config import SearchConfig
from cognitivetree.llm.demo import TASK, clamp_responder
from cognitivetree.llm.generator import LlmThoughtGenerator
from cognitivetree.llm.scripted import ScriptedLlmClient
from cognitivetree.observability import AccountingLlmClient, RunMetrics
from cognitivetree.persistence import load_run, save_run
from cognitivetree.sandbox.backends import select_executor
from cognitivetree.sandbox.demo import VALIDATION_HARNESS
from cognitivetree.sandbox.evaluation import CodeExecutionEvaluator
from cognitivetree.search import TreeSearchController


def main() -> None:
    """Archives a budget-limited run, then reopens and diagnoses it."""
    client = AccountingLlmClient(ScriptedLlmClient(clamp_responder))
    executor, backend = select_executor()

    # A deadline far below one sandboxed evaluation guarantees the run stops
    # against its budget rather than by solving the task.
    controller = TreeSearchController(
        config=SearchConfig(
            max_iterations=16,
            max_depth=2,
            branching_factor=3,
            seed=7,
            max_wall_seconds=0.01,
        ),
        generator=LlmThoughtGenerator(client),
        evaluator=CodeExecutionEvaluator(
            executor=executor, validation_harness=VALIDATION_HARNESS
        ),
    )

    print(f"execution backend: {backend}")
    result = controller.run(TASK)
    metrics = RunMetrics.from_result(result, token_usage=client.usage)
    print(f"live run outcome : {result.outcome.value}")

    destination = Path(tempfile.gettempdir()) / "cognitivetree-demo-run.json"
    save_run(result, destination, metrics=metrics.to_dict())
    print(f"archived to      : {destination} ({destination.stat().st_size} bytes)")

    # Everything from the live run is dropped; only the file survives.
    del controller, result, metrics, client

    print()
    print("--- reopened from disk ---")
    archive = load_run(destination)
    restored = archive.result

    print(f"saved at         : {archive.saved_at}")
    print(f"task             : {archive.task}")
    print(f"outcome          : {restored.outcome.value}")
    print(f"stopped because  : {restored.phase_history[-1].note}")
    print()
    print(restored.tree.render())
    print()

    for node in restored.tree.nodes():
        record = node.metadata.get("execution")
        if record:
            print(
                f"  node {node.id} exit={record.get('exit_code')} "
                f"status={record.get('status')}"
            )

    print()
    print(RunMetrics.from_result(restored).format_report())
    print()
    usage = (archive.metrics or {}).get("token_usage")
    if usage:
        print(
            f"token accounting preserved from save time: "
            f"{usage['total_tokens']} tokens across {usage['calls']} calls"
        )


if __name__ == "__main__":
    main()
