"""Measuring whether a configuration change actually helps.

The framework has many tunables — exploration weight, branching factor,
iteration and consumption budgets, worker counts — and no amount of reading
the code reveals which settings pay off on a given class of task. This
package runs a task suite under a configuration, aggregates the outcomes, and
compares two runs head to head.

Its headline metric is deliberately the one a test-time compute framework
lives or dies by: solve rate as a function of the compute budget.
"""

from cognitivetree.benchmark.llm_tasks import (
    CODING_PROBLEMS,
    CodingProblem,
    llm_suite,
    llm_task,
    offline_llm_suite,
)
from cognitivetree.benchmark.runner import (
    BenchmarkReport,
    TaskResult,
    compare_reports,
    run_benchmark,
    scaling_curve,
)
from cognitivetree.benchmark.suite import (
    BenchmarkTask,
    TaskSetup,
    TerminalOnlyEvaluator,
    default_suite,
)

__all__ = [
    "CODING_PROBLEMS",
    "BenchmarkReport",
    "BenchmarkTask",
    "CodingProblem",
    "TaskResult",
    "TaskSetup",
    "TerminalOnlyEvaluator",
    "compare_reports",
    "default_suite",
    "llm_suite",
    "llm_task",
    "offline_llm_suite",
    "run_benchmark",
    "scaling_curve",
]
