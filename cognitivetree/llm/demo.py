"""Offline demonstration of the LLM-backed reasoning stack.

The scenario is the clamp-implementation task from the deterministic demos,
but here every candidate and diagnosis flows through the real LLM adapters —
:class:`~cognitivetree.llm.generator.LlmThoughtGenerator` and, optionally,
:class:`~cognitivetree.llm.critic.LlmCritic` — driven by a
:class:`~cognitivetree.llm.scripted.ScriptedLlmClient` instead of a live model.
The adapter code paths (prompt assembly, ``### CANDIDATE`` parsing, revision-
note injection) are exercised exactly as they would be against Llama or Qwen,
so the run proves the LLM layer end-to-end without any backend.

Run with: ``python -m cognitivetree.llm.demo``
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from cognitivetree.config import SearchConfig
from cognitivetree.feedback.demo import BROKEN_WAVE, REVISED_CANDIDATE
from cognitivetree.feedback.execution_critic import ExecutionTraceCritic
from cognitivetree.feedback.revision import (
    REVISION_ATTEMPTS_KEY,
    REVISION_NOTES_KEY,
    BoundedRevisionPolicy,
)
from cognitivetree.feedback.rewards import RewardShaper
from cognitivetree.llm.client import CompletionRequest, LlmClient
from cognitivetree.llm.critic import LlmCritic
from cognitivetree.llm.generator import LlmThoughtGenerator
from cognitivetree.llm.scripted import ScriptedLlmClient
from cognitivetree.observability.accounting import AccountingLlmClient
from cognitivetree.observability.budget import TokenBudget
from cognitivetree.policies import Critic
from cognitivetree.sandbox.backends import select_executor
from cognitivetree.sandbox.demo import VALIDATION_HARNESS
from cognitivetree.sandbox.evaluation import CodeExecutionEvaluator
from cognitivetree.search import SearchEvent, TreeSearchController

if TYPE_CHECKING:
    # Imported for typing only: cognitivetree.session imports this module's
    # siblings, so a runtime import here would close a cycle.
    from cognitivetree.session import ReasoningSession

TASK = "Implement clamp(value, low, high) correctly."

# Marker the generator prompt uses to surface critique-derived revision notes;
# its presence signals that a revised expansion is being requested.
REVISION_NOTES_HEADER = "REVISION NOTES"

# A JSON verdict for the critic prompt. The execution-trace critic resolves the
# clamp scenario's assertion failures on its own, so this path is reached only
# when the chained LLM critic is explicitly enabled and consulted.
_CRITIC_VERDICT = (
    '{"failure_class": "assertion", '
    '"summary": "bounds are not both respected", '
    '"guidance": "Return max(low, min(value, high)) so lower and upper bounds hold.", '
    '"severity": 0.6}'
)


def _as_candidate_block(candidates: tuple[str, ...]) -> str:
    """Formats candidates in the ``### CANDIDATE`` protocol the parser expects."""
    return "".join(f"### CANDIDATE\n{candidate}\n" for candidate in candidates)


def clamp_responder(request: CompletionRequest) -> str:
    """Routes a completion request to the scripted output for its role.

    The first wave of expansion yields the broken candidates; once the critic's
    revision notes appear in the prompt, expansion yields the corrected
    implementation. Critic prompts receive a structured JSON verdict.
    """
    system = request.messages[0].content
    user = request.messages[-1].content
    if "critique policy" in system:
        return _CRITIC_VERDICT
    if REVISION_NOTES_HEADER in user:
        return _as_candidate_block((REVISED_CANDIDATE,))
    return _as_candidate_block(BROKEN_WAVE)


def build_offline_controller(
    client: LlmClient | None = None,
    on_event: Callable[[SearchEvent], None] | None = None,
    use_llm_critic: bool = False,
    seed: int = 7,
    max_wall_seconds: float | None = None,
    max_tokens: int | None = None,
    evaluation_workers: int = 1,
) -> TreeSearchController:
    """Assembles the LLM-backed controller over a scripted client.

    ``client`` accepts any :class:`~cognitivetree.llm.client.LlmClient`, which
    lets an :class:`~cognitivetree.observability.accounting.AccountingLlmClient`
    wrap the scripted client to surface token usage for the run.
    ``max_wall_seconds`` threads through to the run's global time budget, and
    ``max_tokens`` installs a consumption ceiling — wrapping the client for
    accounting when the caller did not already do so.
    """
    client = client or ScriptedLlmClient(clamp_responder, model="scripted-llama")

    stop_condition = None
    if max_tokens is not None:
        if not isinstance(client, AccountingLlmClient):
            client = AccountingLlmClient(client)
        stop_condition = TokenBudget(client, max_total_tokens=max_tokens)

    executor, _ = select_executor()

    critic: Critic = ExecutionTraceCritic()
    if use_llm_critic:
        critic = _chained_critic(client)

    return TreeSearchController(
        config=SearchConfig(
            max_iterations=16,
            max_depth=1,
            branching_factor=len(BROKEN_WAVE),
            seed=seed,
            max_wall_seconds=max_wall_seconds,
            evaluation_workers=evaluation_workers,
        ),
        generator=LlmThoughtGenerator(client),
        evaluator=CodeExecutionEvaluator(
            executor=executor, validation_harness=VALIDATION_HARNESS
        ),
        critic=critic,
        revision_policy=BoundedRevisionPolicy(max_attempts=1),
        reward_model=RewardShaper(),
        on_event=on_event,
        stop_condition=stop_condition,
    )


def _chained_critic(client: LlmClient) -> Critic:
    from cognitivetree.feedback.composite import ChainedCritic

    return ChainedCritic([ExecutionTraceCritic(), LlmCritic(client)])


def build_offline_session(
    max_wall_seconds: float | None = None,
    max_tokens: int | None = None,
    evaluation_workers: int = 1,
    archive_dir: str | Path | None = None,
) -> ReasoningSession:
    """Builds a streaming session over the scripted LLM stack.

    The UI's ``llm-demo`` backend uses this to exercise the LLM path live
    without a model; each connection receives an independent scripted run.
    ``archive_dir`` saves every run there; see :class:`ReasoningSession`.
    """
    from cognitivetree.session import ReasoningSession

    client = AccountingLlmClient(ScriptedLlmClient(clamp_responder, model="scripted-llama"))
    return ReasoningSession(
        task=TASK,
        controller_factory=lambda sink: build_offline_controller(
            client=client,
            on_event=sink,
            max_wall_seconds=max_wall_seconds,
            max_tokens=max_tokens,
            evaluation_workers=evaluation_workers,
        ),
        archive_dir=archive_dir,
        usage=lambda: client.usage,
    )


def main() -> None:
    """Runs the offline LLM reasoning loop and prints its trace."""
    controller = build_offline_controller(on_event=_print_event)
    result = controller.run(TASK)
    root = result.tree.root

    print()
    print(result.tree.render())
    print()
    print("candidates proposed by the (scripted) model, graded by the sandbox:")
    for child in root.children:
        record = child.metadata.get("execution", {})
        print(f"- exit={record.get('exit_code')} | {child.rationale}")
    print()
    print("revision notes fed back into the generator prompt:")
    for line in str(root.metadata.get(REVISION_NOTES_KEY, "")).splitlines():
        print(f"  {line}")
    print()
    print(f"revision attempts : {root.metadata.get(REVISION_ATTEMPTS_KEY, 0)}")
    print(f"outcome           : {result.outcome.value}")
    print(f"iterations        : {result.iterations}")
    if result.solution:
        print()
        print(result.solution)


def _print_event(event: SearchEvent) -> None:
    node = event.node_id or "-"
    detail = f" | {event.detail}" if event.detail else ""
    print(f"[iter {event.iteration:03d}] {event.phase.value:<16} node={node}{detail}")


if __name__ == "__main__":
    main()
