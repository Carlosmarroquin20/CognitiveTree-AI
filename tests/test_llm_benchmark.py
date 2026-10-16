"""Validates the LLM-backed benchmark tasks and their problem harnesses."""

import pytest

from cognitivetree.benchmark.llm_tasks import (
    CODING_PROBLEMS,
    CodingProblem,
    offline_llm_suite,
)
from cognitivetree.benchmark.runner import run_benchmark
from cognitivetree.config import SearchConfig
from cognitivetree.llm.caching import CompletionCache
from cognitivetree.sandbox.spec import ExecutionRequest
from cognitivetree.sandbox.subprocess_executor import SubprocessExecutor

EXECUTOR = SubprocessExecutor()
CONFIG = SearchConfig(max_iterations=10, seed=7)


def passes_harness(problem: CodingProblem, code: str) -> bool:
    program = f"{code.strip()}\n\n{problem.harness.strip()}\n"
    return EXECUTOR.execute(ExecutionRequest(code=program)).ok


@pytest.mark.parametrize("problem", CODING_PROBLEMS, ids=lambda p: p.name)
def test_reference_solution_passes_its_harness(problem: CodingProblem) -> None:
    assert passes_harness(problem, problem.reference)


@pytest.mark.parametrize(
    ("problem", "code"),
    [(p, code) for p in CODING_PROBLEMS for code in p.broken],
    ids=lambda value: value.name if isinstance(value, CodingProblem) else "",
)
def test_broken_variant_fails_its_harness(problem: CodingProblem, code: str) -> None:
    # A harness that accepts a broken variant would let a wrong model answer
    # count as a solve.
    assert not passes_harness(problem, code)


def test_offline_suite_solves_every_problem_through_revision() -> None:
    report = run_benchmark(CONFIG, tasks=offline_llm_suite(executor=EXECUTOR))

    assert report.solve_rate == 1.0
    assert {r.task for r in report.results} == {f"llm-{p.name}" for p in CODING_PROBLEMS}
    # One failing wave, then the revised candidate: the full critique cycle.
    assert all(r.iterations == 2 for r in report.results)
    assert all(r.tokens is not None and r.tokens > 0 for r in report.results)


def test_shared_cache_makes_repeated_runs_free() -> None:
    cache = CompletionCache()
    report = run_benchmark(
        CONFIG, tasks=offline_llm_suite(cache=cache, executor=EXECUTOR), repeats=2
    )

    first = [r.tokens for r in report.results if r.seed == 7]
    repeated = [r.tokens for r in report.results if r.seed == 8]
    assert all(tokens and tokens > 0 for tokens in first)
    assert repeated == [0] * len(CODING_PROBLEMS)
    assert report.solve_rate == 1.0


def test_without_a_cache_every_repeat_pays_again() -> None:
    report = run_benchmark(CONFIG, tasks=offline_llm_suite(executor=EXECUTOR), repeats=2)
    first = [r.tokens for r in report.results if r.seed == 7]
    repeated = [r.tokens for r in report.results if r.seed == 8]
    assert repeated == first


def test_sampled_requests_are_not_served_from_the_cache() -> None:
    cache = CompletionCache()
    run_benchmark(
        CONFIG,
        tasks=offline_llm_suite(cache=cache, temperature=0.7, executor=EXECUTOR),
        repeats=2,
    )
    assert cache.stats.hits == 0
