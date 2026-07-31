"""Validates consumption budgets and the stop-condition control path."""

import time

import pytest

from cognitivetree.config import SearchConfig
from cognitivetree.llm.client import (
    ChatMessage,
    CompletionRequest,
    CompletionResponse,
)
from cognitivetree.llm.demo import TASK, build_offline_controller, clamp_responder
from cognitivetree.llm.generator import LlmThoughtGenerator
from cognitivetree.llm.scripted import ScriptedLlmClient
from cognitivetree.node import ThoughtNode
from cognitivetree.observability import AccountingLlmClient, RunMetrics, TokenBudget
from cognitivetree.policies import Evaluation, StopCondition
from cognitivetree.sandbox.demo import VALIDATION_HARNESS
from cognitivetree.sandbox.evaluation import CodeExecutionEvaluator
from cognitivetree.sandbox.subprocess_executor import SubprocessExecutor
from cognitivetree.search import SearchOutcome, TreeSearchController
from cognitivetree.state import SearchPhase


class SpendingClient:
    """Reports a fixed token cost per completion, for exact budget arithmetic."""

    def __init__(self, tokens_per_call: int = 10) -> None:
        self._tokens = tokens_per_call

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        return CompletionResponse(
            text="### CANDIDATE\nthought\n",
            prompt_tokens=self._tokens,
            completion_tokens=0,
        )


def request() -> CompletionRequest:
    return CompletionRequest(messages=(ChatMessage(role="user", content="hi"),))


def spend(client: AccountingLlmClient, calls: int) -> None:
    for _ in range(calls):
        client.complete(request())


class TestTokenBudget:
    """Ceiling arithmetic and reason reporting."""

    def test_unbounded_budget_never_fires(self) -> None:
        client = AccountingLlmClient(SpendingClient())
        budget = TokenBudget(client)
        assert not budget.is_bounded
        spend(client, 100)
        assert budget.check() is None

    def test_token_ceiling_fires_only_once_crossed(self) -> None:
        client = AccountingLlmClient(SpendingClient(tokens_per_call=10))
        budget = TokenBudget(client, max_total_tokens=25)

        assert budget.check() is None
        spend(client, 2)  # 20 tokens
        assert budget.check() is None
        spend(client, 1)  # 30 tokens
        reason = budget.check()
        assert reason is not None
        assert "token budget of 25 exhausted" in reason
        assert "30 consumed" in reason

    def test_ceiling_fires_on_exact_equality(self) -> None:
        client = AccountingLlmClient(SpendingClient(tokens_per_call=10))
        budget = TokenBudget(client, max_total_tokens=20)
        spend(client, 2)
        assert budget.check() is not None

    def test_call_ceiling_is_independent(self) -> None:
        client = AccountingLlmClient(SpendingClient(tokens_per_call=1))
        budget = TokenBudget(client, max_calls=3)
        spend(client, 2)
        assert budget.check() is None
        spend(client, 1)
        reason = budget.check()
        assert reason is not None
        assert "call budget of 3 exhausted" in reason

    def test_token_ceiling_takes_precedence_when_both_cross(self) -> None:
        client = AccountingLlmClient(SpendingClient(tokens_per_call=10))
        budget = TokenBudget(client, max_total_tokens=5, max_calls=1)
        spend(client, 1)
        reason = budget.check()
        assert reason is not None and reason.startswith("token budget")

    def test_budget_satisfies_the_stop_condition_protocol(self) -> None:
        client = AccountingLlmClient(SpendingClient())
        assert isinstance(TokenBudget(client, max_total_tokens=1), StopCondition)

    @pytest.mark.parametrize(
        "kwargs", [{"max_total_tokens": 0}, {"max_calls": 0}, {"max_total_tokens": -5}]
    )
    def test_non_positive_ceilings_are_rejected(self, kwargs: dict) -> None:
        with pytest.raises(ValueError):
            TokenBudget(AccountingLlmClient(SpendingClient()), **kwargs)


class FixedStopCondition:
    """Fires after a set number of polls, isolating the controller's handling."""

    def __init__(self, fire_on_poll: int) -> None:
        self.polls = 0
        self._fire_on = fire_on_poll

    def check(self) -> str | None:
        self.polls += 1
        return "synthetic budget exhausted" if self.polls >= self._fire_on else None


class NeutralEvaluator:
    def evaluate(self, node: ThoughtNode) -> Evaluation:
        return Evaluation(score=0.5)


class EndlessGenerator:
    def generate(self, node: ThoughtNode, k: int) -> list[str]:
        return [f"{node.content} / step-{i}" for i in range(k)]


class SleepingGenerator(EndlessGenerator):
    """Overshoots a wall-clock deadline on its first call."""

    def __init__(self, delay_seconds: float) -> None:
        self._delay_seconds = delay_seconds
        self._called = False

    def generate(self, node: ThoughtNode, k: int) -> list[str]:
        if not self._called:
            self._called = True
            time.sleep(self._delay_seconds)
        return super().generate(node, k)


def build(stop_condition=None, **overrides) -> TreeSearchController:
    params = {"max_iterations": 20, "max_depth": 20, "branching_factor": 2, "seed": 1}
    params.update(overrides)
    return TreeSearchController(
        config=SearchConfig(**params),
        generator=EndlessGenerator(),
        evaluator=NeutralEvaluator(),
        stop_condition=stop_condition,
    )


class TestControllerIntegration:
    """How the controller reacts to a satisfied stop condition."""

    def test_stopping_yields_budget_exhausted_with_the_reason(self) -> None:
        result = build(FixedStopCondition(fire_on_poll=3)).run("task")

        assert result.outcome is SearchOutcome.BUDGET_EXHAUSTED
        assert result.phase_history[-1].target is SearchPhase.BUDGET_EXHAUSTED
        assert result.phase_history[-1].note == "synthetic budget exhausted"
        assert result.iterations == 2  # fired at the top of the third iteration

    def test_absent_stop_condition_leaves_behavior_unchanged(self) -> None:
        result = build(max_iterations=3).run("task")
        assert result.outcome is SearchOutcome.EXHAUSTED

    def test_condition_is_polled_once_per_iteration(self) -> None:
        condition = FixedStopCondition(fire_on_poll=999)
        build(condition, max_iterations=4).run("task")
        assert condition.polls == 4

    def test_partial_tree_survives_for_inspection(self) -> None:
        result = build(FixedStopCondition(fire_on_poll=3)).run("task")
        assert result.node_count > 1
        assert result.best_path
        assert result.solution is None

    def test_wall_clock_deadline_wins_when_both_would_fire(self) -> None:
        # The first iteration overshoots the deadline while the condition is
        # armed to fire at the next boundary, so both limits genuinely hold
        # when iteration two is considered; the deadline is tested first and
        # must therefore be the reported cause.
        condition = FixedStopCondition(fire_on_poll=2)
        controller = TreeSearchController(
            config=SearchConfig(
                max_iterations=20,
                max_depth=20,
                branching_factor=2,
                seed=1,
                max_wall_seconds=0.01,
            ),
            generator=SleepingGenerator(delay_seconds=0.15),
            evaluator=NeutralEvaluator(),
            stop_condition=condition,
        )
        result = controller.run("task")

        assert result.outcome is SearchOutcome.TIMED_OUT
        # The deadline short-circuited the boundary, so the armed condition
        # was never polled a second time.
        assert condition.polls == 1


class TestOfflineTokenBudget:
    """The budget driving a real LLM-adapter run without a model."""

    def test_tight_budget_stops_the_offline_run(self) -> None:
        result = build_offline_controller(max_tokens=50).run(TASK)

        assert result.outcome is SearchOutcome.BUDGET_EXHAUSTED
        assert "token budget of 50 exhausted" in result.phase_history[-1].note
        assert result.solution is None

    def test_generous_budget_lets_the_run_finish(self) -> None:
        result = build_offline_controller(max_tokens=1_000_000).run(TASK)
        assert result.outcome is SearchOutcome.SUCCEEDED

    def test_budget_absent_by_default(self) -> None:
        assert build_offline_controller().run(TASK).outcome is SearchOutcome.SUCCEEDED

    def test_overshoot_is_bounded_by_one_iteration(self) -> None:
        client = AccountingLlmClient(ScriptedLlmClient(clamp_responder))
        controller = TreeSearchController(
            config=SearchConfig(
                max_iterations=20, max_depth=4, branching_factor=3, seed=7
            ),
            generator=LlmThoughtGenerator(client),
            evaluator=CodeExecutionEvaluator(
                executor=SubprocessExecutor(), validation_harness=VALIDATION_HARNESS
            ),
            stop_condition=TokenBudget(client, max_total_tokens=100),
        )
        result = controller.run(TASK)

        assert result.outcome is SearchOutcome.BUDGET_EXHAUSTED
        # The ceiling is crossed inside an iteration and detected at the next
        # boundary, so exactly one iteration's spend may exceed it.
        assert client.usage.total_tokens > 100
        assert client.usage.calls == result.iterations

    def test_metrics_report_the_budget_outcome(self) -> None:
        result = build_offline_controller(max_tokens=50).run(TASK)
        metrics = RunMetrics.from_result(result)
        assert metrics.outcome == "budget_exhausted"
        assert metrics.solution_depth is None
        assert metrics.to_dict()["outcome"] == "budget_exhausted"
