"""Benchmark tasks that drive the search through a chat model.

The bundled suite is synthetic: its generator is a fixed vocabulary, so it
measures search effort but cannot compare anything on the model side, such
as completion caching, temperature, or the model itself. These tasks wire
the production LLM stack instead (prompting, ``### CANDIDATE`` parsing,
sandboxed validation, critique, and revision) around small coding problems,
and report the tokens every run consumed.

Each problem carries a reference solution and plausible broken variants.
The reference exists so tests can prove a harness accepts a correct answer;
it never reaches a live model. The broken variants feed the offline suite,
where a scripted client proposes them first and the reference once revision
notes arrive, exercising the full fail, critique, revise, succeed cycle
without a model.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from cognitivetree.benchmark.suite import BenchmarkTask, TaskSetup
from cognitivetree.config import SearchConfig
from cognitivetree.feedback.execution_critic import ExecutionTraceCritic
from cognitivetree.feedback.revision import BoundedRevisionPolicy
from cognitivetree.feedback.rewards import RewardShaper
from cognitivetree.llm.caching import CachingLlmClient, CompletionCache
from cognitivetree.llm.client import CompletionRequest, LlmClient
from cognitivetree.llm.generator import LlmThoughtGenerator
from cognitivetree.llm.scripted import ScriptedLlmClient
from cognitivetree.observability.accounting import AccountingLlmClient
from cognitivetree.sandbox.evaluation import CodeExecutionEvaluator
from cognitivetree.sandbox.executor import CodeExecutor
from cognitivetree.search import TreeSearchController

ClientFactory = Callable[[], LlmClient]

# Marker the generator prompt uses when revision notes are present.
_REVISION_MARKER = "REVISION NOTES"


@dataclass(frozen=True, slots=True)
class CodingProblem:
    """A self-checking programming task.

    Attributes:
        name: Identifier reported in results.
        statement: Task text placed at the tree root and shown to the model.
        harness: Assertions appended to every candidate; a candidate solves
            the problem exactly when the combined program exits cleanly.
        reference: A known-correct implementation, used only to validate the
            harness and by the scripted offline client.
        broken: Plausible wrong implementations the offline client proposes
            before revision; each must fail the harness.
    """

    name: str
    statement: str
    harness: str
    reference: str
    broken: tuple[str, ...]


CODING_PROBLEMS: tuple[CodingProblem, ...] = (
    CodingProblem(
        name="clamp",
        statement=(
            "Implement clamp(value, low, high), returning value limited to the "
            "closed range [low, high]."
        ),
        harness="""
assert clamp(5, 0, 10) == 5
assert clamp(-3, 0, 10) == 0
assert clamp(42, 0, 10) == 10
assert clamp(7, 7, 7) == 7
""",
        reference="""
def clamp(value, low, high):
    return max(low, min(value, high))
""",
        broken=(
            """
def clamp(value, low, high):
    return min(value, high)
""",
            """
def clamp(value, low, high):
    return max(low, value)
""",
        ),
    ),
    CodingProblem(
        name="run-length-encode",
        statement=(
            "Implement encode(text) returning the run-length encoding of text as "
            "each run's length followed by its character, e.g. "
            "encode('aaab') == '3a1b'."
        ),
        harness="""
assert encode("") == ""
assert encode("a") == "1a"
assert encode("aaab") == "3a1b"
assert encode("aabaa") == "2a1b2a"
assert encode("zzzzzzzzzzzz") == "12z"
""",
        reference="""
def encode(text):
    out = []
    i = 0
    while i < len(text):
        j = i
        while j < len(text) and text[j] == text[i]:
            j += 1
        out.append(f"{j - i}{text[i]}")
        i = j
    return "".join(out)
""",
        broken=(
            """
def encode(text):
    return "".join(f"{text.count(c)}{c}" for c in dict.fromkeys(text))
""",
            """
def encode(text):
    out, count = [], 1
    for prev, cur in zip(text, text[1:]):
        if cur == prev:
            count += 1
        else:
            out.append(f"{count}{prev}")
            count = 1
    return "".join(out)
""",
        ),
    ),
    CodingProblem(
        name="balanced-brackets",
        statement=(
            "Implement is_balanced(text) returning True when every (, [ and { "
            "in text is closed by the matching bracket in the right order; "
            "other characters are ignored."
        ),
        harness="""
assert is_balanced("")
assert is_balanced("(a[b]{c})")
assert not is_balanced("(]")
assert not is_balanced("([)]")
assert not is_balanced("((")
assert not is_balanced(")(")
""",
        reference="""
def is_balanced(text):
    pairs = {")": "(", "]": "[", "}": "{"}
    stack = []
    for char in text:
        if char in "([{":
            stack.append(char)
        elif char in pairs:
            if not stack or stack.pop() != pairs[char]:
                return False
    return not stack
""",
        broken=(
            """
def is_balanced(text):
    return all(text.count(o) == text.count(c) for o, c in ("()", "[]", "{}"))
""",
            """
def is_balanced(text):
    pairs = {")": "(", "]": "[", "}": "{"}
    stack = []
    for char in text:
        if char in "([{":
            stack.append(char)
        elif char in pairs:
            if not stack or stack.pop() != pairs[char]:
                return False
    return True
""",
        ),
    ),
    CodingProblem(
        name="merge-intervals",
        statement=(
            "Implement merge(intervals) taking a list of [start, end] pairs and "
            "returning the sorted list of merged, non-overlapping intervals; "
            "touching intervals such as [1, 2] and [2, 3] merge."
        ),
        harness="""
assert merge([]) == []
assert merge([[1, 3], [2, 6], [8, 10]]) == [[1, 6], [8, 10]]
assert merge([[8, 10], [1, 3]]) == [[1, 3], [8, 10]]
assert merge([[1, 2], [2, 3]]) == [[1, 3]]
assert merge([[1, 10], [2, 3]]) == [[1, 10]]
""",
        reference="""
def merge(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged
""",
        broken=(
            """
def merge(intervals):
    merged = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged
""",
            """
def merge(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return merged
""",
        ),
    ),
)


def as_candidate(description: str, code: str) -> str:
    """Formats one thought the way the generator prompt asks the model to."""
    return f"{description}\n```python\n{code.strip()}\n```"


def scripted_responder(problem: CodingProblem) -> Callable[[CompletionRequest], str]:
    """Returns a responder that fails first and fixes ``problem`` on revision."""
    first_wave = "".join(
        f"### CANDIDATE\n{as_candidate('Initial attempt.', code)}\n" for code in problem.broken
    )
    revised = f"### CANDIDATE\n{as_candidate('Revised after critique.', problem.reference)}\n"

    def respond(request: CompletionRequest) -> str:
        return revised if _REVISION_MARKER in request.messages[-1].content else first_wave

    return respond


def llm_task(
    problem: CodingProblem,
    client_factory: ClientFactory,
    temperature: float = 0.7,
    cache: CompletionCache | None = None,
    executor: CodeExecutor | None = None,
    revision_attempts: int = 1,
) -> BenchmarkTask:
    """Wraps ``problem`` as a benchmark task driven by a chat model.

    ``client_factory`` builds a fresh client per run, so every run is
    accounted separately. A shared ``cache`` sits outside that accounting,
    so a run served from the cache reports only the tokens it actually
    spent. The executor defaults to the strongest available backend.
    """

    def build(config: SearchConfig) -> TaskSetup:
        from cognitivetree.sandbox.backends import select_executor

        accounting = AccountingLlmClient(client_factory())
        client: LlmClient = (
            CachingLlmClient(accounting, cache) if cache is not None else accounting
        )
        controller = TreeSearchController(
            config=config,
            generator=LlmThoughtGenerator(client, temperature=temperature),
            evaluator=CodeExecutionEvaluator(
                executor=executor if executor is not None else select_executor()[0],
                validation_harness=problem.harness,
            ),
            critic=ExecutionTraceCritic(),
            revision_policy=BoundedRevisionPolicy(max_attempts=revision_attempts),
            reward_model=RewardShaper(),
        )
        return TaskSetup(controller=controller, usage=lambda: accounting.usage)

    return BenchmarkTask(
        name=f"llm-{problem.name}",
        statement=problem.statement,
        build=build,
        # Candidates are complete solutions, so the tree is one level deep and
        # depth comes from revision rather than from chaining thoughts.
        config_overrides={"max_depth": 1, "branching_factor": 3},
    )


def llm_suite(
    client_factory: ClientFactory,
    temperature: float = 0.7,
    cache: CompletionCache | None = None,
    problems: Sequence[CodingProblem] = CODING_PROBLEMS,
    executor: CodeExecutor | None = None,
) -> tuple[BenchmarkTask, ...]:
    """Returns every problem as a task against the model ``client_factory`` builds."""
    return tuple(
        llm_task(problem, client_factory, temperature, cache=cache, executor=executor)
        for problem in problems
    )


def offline_llm_suite(
    cache: CompletionCache | None = None,
    temperature: float = 0.0,
    problems: Sequence[CodingProblem] = CODING_PROBLEMS,
    executor: CodeExecutor | None = None,
) -> tuple[BenchmarkTask, ...]:
    """Returns the problems as tasks against a scripted client, with no model.

    Temperature defaults to ``0`` so requests are deterministic and a shared
    ``cache`` can serve repeated runs.
    """
    return tuple(
        llm_task(
            problem, _scripted_client(problem), temperature, cache=cache, executor=executor
        )
        for problem in problems
    )


def _scripted_client(problem: CodingProblem) -> ClientFactory:
    """Binds ``problem`` now; a lambda in the caller's loop would bind late."""
    return lambda: ScriptedLlmClient(scripted_responder(problem))
