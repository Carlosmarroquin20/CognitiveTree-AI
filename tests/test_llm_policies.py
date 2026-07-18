"""Validates the LLM-backed generator and critic against scripted clients."""

from cognitivetree.feedback.composite import ChainedCritic
from cognitivetree.llm.client import (
    CompletionRequest,
    CompletionResponse,
    LlmError,
)
from cognitivetree.llm.critic import LlmCritic
from cognitivetree.llm.generator import LlmThoughtGenerator
from cognitivetree.node import ThoughtNode
from cognitivetree.policies import Critique, FailureClass
from cognitivetree.sandbox.evaluation import METADATA_KEY


class ScriptedClient:
    """Returns a fixed completion while recording every request."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.requests: list[CompletionRequest] = []

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        self.requests.append(request)
        return CompletionResponse(text=self.text)


class FailingClient:
    def complete(self, request: CompletionRequest) -> CompletionResponse:
        raise LlmError("backend unreachable")


def tree_path() -> ThoughtNode:
    root = ThoughtNode(content="Implement the encoder.")
    step = root.attach_child("Decompose into scan and emit phases.")
    return step.attach_child("Write the scan loop first.")


def test_generator_prompt_carries_task_path_and_k() -> None:
    client = ScriptedClient("### CANDIDATE\nalpha\n### CANDIDATE\nbeta\n")
    node = tree_path()

    candidates = LlmThoughtGenerator(client).generate(node, k=2)

    assert candidates == ["alpha", "beta"]
    user_message = client.requests[0].messages[1].content
    assert "Implement the encoder." in user_message
    assert "1. Decompose into scan and emit phases." in user_message
    assert "2. Write the scan loop first." in user_message
    assert "Propose 2 candidate" in user_message
    assert "REVISION NOTES" not in user_message


def test_generator_injects_revision_notes_when_present() -> None:
    client = ScriptedClient("### CANDIDATE\nrevised idea\n")
    node = ThoughtNode(content="Implement the encoder.")
    node.metadata["revision_notes"] = "- [assertion] handle empty input"

    LlmThoughtGenerator(client).generate(node, k=3)

    user_message = client.requests[0].messages[1].content
    assert "REVISION NOTES" in user_message
    assert "handle empty input" in user_message


def test_generator_clips_to_k_candidates() -> None:
    client = ScriptedClient(
        "### CANDIDATE\none\n### CANDIDATE\ntwo\n### CANDIDATE\nthree\n"
    )
    assert LlmThoughtGenerator(client).generate(ThoughtNode(content="t"), k=2) == [
        "one",
        "two",
    ]


def test_critic_parses_json_verdict() -> None:
    client = ScriptedClient(
        'Diagnosis:\n{"failure_class": "assertion", "summary": "bounds wrong", '
        '"guidance": "respect the lower bound", "severity": 0.65}'
    )
    node = ThoughtNode(content="candidate")
    node.metadata[METADATA_KEY] = {
        "status": "completed",
        "exit_code": 1,
        "stderr": "AssertionError: bounds wrong",
    }

    verdict = LlmCritic(client).critique(node)

    assert verdict is not None
    assert verdict.failure_class is FailureClass.ASSERTION
    assert verdict.guidance == "respect the lower bound"
    assert verdict.severity == 0.65
    assert "AssertionError" in client.requests[0].messages[1].content


def test_critic_maps_unknown_class_and_clamps_severity() -> None:
    client = ScriptedClient(
        '{"failure_class": "cosmic_rays", "summary": "s", "guidance": "g", "severity": 7}'
    )
    verdict = LlmCritic(client).critique(ThoughtNode(content="candidate"))
    assert verdict is not None
    assert verdict.failure_class is FailureClass.EXCEPTION
    assert verdict.severity == 1.0


def test_critic_degrades_on_missing_guidance_or_junk() -> None:
    junk = LlmCritic(ScriptedClient("no json at all")).critique(
        ThoughtNode(content="candidate")
    )
    assert junk is None
    no_guidance = LlmCritic(
        ScriptedClient('{"failure_class": "exception", "summary": "s"}')
    ).critique(ThoughtNode(content="candidate"))
    assert no_guidance is None


def test_critic_degrades_on_backend_failure() -> None:
    assert LlmCritic(FailingClient()).critique(ThoughtNode(content="c")) is None


def test_chained_critic_returns_first_verdict() -> None:
    class SilentCritic:
        def critique(self, node: ThoughtNode) -> Critique | None:
            return None

    class FixedCritic:
        def critique(self, node: ThoughtNode) -> Critique | None:
            return Critique(
                failure_class=FailureClass.EXCEPTION,
                summary="fixed",
                guidance="do the fix",
                severity=0.5,
            )

    chained = ChainedCritic([SilentCritic(), FixedCritic()])
    verdict = chained.critique(ThoughtNode(content="c"))
    assert verdict is not None and verdict.summary == "fixed"

    silent = ChainedCritic([SilentCritic(), SilentCritic()])
    assert silent.critique(ThoughtNode(content="c")) is None
