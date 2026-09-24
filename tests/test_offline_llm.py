"""End-to-end validation of the LLM adapter stack driven offline."""

import pytest

from cognitivetree.config import SearchConfig
from cognitivetree.feedback.demo import BROKEN_WAVE, REVISED_CANDIDATE
from cognitivetree.llm.client import ChatMessage, CompletionRequest
from cognitivetree.llm.demo import (
    TASK,
    build_offline_controller,
    build_offline_session,
    clamp_responder,
)
from cognitivetree.llm.prompts import CRITIC_SYSTEM_PROMPT, GENERATOR_SYSTEM_PROMPT
from cognitivetree.llm.scripted import ScriptedLlmClient
from cognitivetree.observability import AccountingLlmClient
from cognitivetree.sandbox.demo import VALIDATION_HARNESS
from cognitivetree.search import SearchOutcome
from cognitivetree.session import LlmSessionSpec, build_llm_session


def _request(system: str, user: str) -> CompletionRequest:
    return CompletionRequest(
        messages=(
            ChatMessage(role="system", content=system),
            ChatMessage(role="user", content=user),
        )
    )


def test_responder_routes_by_role_and_revision_state() -> None:
    first = clamp_responder(_request(GENERATOR_SYSTEM_PROMPT, "propose candidates"))
    assert first.count("### CANDIDATE") == len(BROKEN_WAVE)

    revised = clamp_responder(
        _request(GENERATOR_SYSTEM_PROMPT, "REVISION NOTES\n- fix bounds")
    )
    assert "max(low, min(value, high))" in revised

    verdict = clamp_responder(_request(CRITIC_SYSTEM_PROMPT, "diagnose"))
    assert verdict.strip().startswith("{") and "guidance" in verdict


def test_offline_run_completes_through_llm_adapters() -> None:
    client = ScriptedLlmClient(clamp_responder)
    controller = build_offline_controller(client=client)
    result = controller.run(TASK)
    root = result.tree.root

    assert result.outcome is SearchOutcome.SUCCEEDED
    assert result.solution == REVISED_CANDIDATE
    assert result.iterations == 2
    assert root.metadata["revision_attempts"] == 1

    # The broken wave was proposed, executed, and critiqued before revision.
    first_wave = [c for c in root.children if c.content in BROKEN_WAVE]
    assert len(first_wave) == len(BROKEN_WAVE)
    assert all(c.metadata.get("critique") for c in first_wave)

    # Every completion travelled through the real generator prompt assembly.
    assert len(client.requests) >= 2
    assert all(
        "expansion policy" in r.messages[0].content for r in client.requests
    )
    assert any("REVISION NOTES" in r.messages[-1].content for r in client.requests)


def test_build_llm_session_accepts_injected_client() -> None:
    spec = LlmSessionSpec(
        task=TASK,
        base_url="unused://offline",
        model="scripted",
        validation_harness=VALIDATION_HARNESS,
        config=SearchConfig(max_iterations=16, max_depth=1, branching_factor=3, seed=7),
    )
    result = build_llm_session(spec, client=ScriptedLlmClient(clamp_responder)).run()
    assert result.outcome is SearchOutcome.SUCCEEDED
    assert result.solution == REVISED_CANDIDATE


def test_reused_session_applies_its_budget_per_run() -> None:
    # The clamp scenario solves in two generator calls, so a three-call
    # ceiling admits every run while a session-wide tally would stop the
    # second one after its first iteration.
    client = AccountingLlmClient(ScriptedLlmClient(clamp_responder))
    spec = LlmSessionSpec(
        task=TASK,
        base_url="unused://offline",
        model="scripted",
        validation_harness=VALIDATION_HARNESS,
        config=SearchConfig(max_iterations=16, max_depth=1, branching_factor=3, seed=7),
        max_llm_calls=3,
    )
    session = build_llm_session(spec, client=client)

    outcomes = [session.run().outcome for _ in range(3)]
    assert outcomes == [SearchOutcome.SUCCEEDED] * 3
    # The caller's handle still reports lifetime totals across every run.
    assert client.usage.calls == 6


def test_offline_session_streams_to_completion() -> None:
    envelopes = list(build_offline_session().stream())
    assert envelopes[0]["type"] == "phase"
    assert envelopes[-1]["type"] == "result"
    assert envelopes[-1]["outcome"] == "succeeded"
    assert "backtracking" in [e["phase"] for e in envelopes if e["type"] == "phase"]


def test_chained_llm_critic_path_still_succeeds() -> None:
    # The execution-trace critic resolves assertion failures, so the chained
    # LLM critic is present but not consulted; the run must still converge.
    result = build_offline_controller(use_llm_critic=True).run(TASK)
    assert result.outcome is SearchOutcome.SUCCEEDED


def _offline_spec(**overrides: object) -> LlmSessionSpec:
    params: dict[str, object] = {
        "task": TASK,
        "base_url": "unused://offline",
        "model": "scripted",
        "validation_harness": VALIDATION_HARNESS,
        "config": SearchConfig(max_iterations=16, max_depth=1, branching_factor=3, seed=7),
    }
    params.update(overrides)
    return LlmSessionSpec(**params)  # type: ignore[arg-type]


def test_temperatures_reach_the_generator_and_critic() -> None:
    client = ScriptedLlmClient(clamp_responder)
    spec = _offline_spec(temperature=0.0, critic_temperature=0.0, use_llm_critic=True)
    session = build_llm_session(spec, client=client)
    session.run()

    assert {r.temperature for r in client.requests} == {0.0}
    critics = session._factory(None)._critic._critics  # type: ignore[union-attr]
    assert critics[-1]._temperature == 0.0


@pytest.mark.parametrize("field", ["temperature", "critic_temperature"])
def test_spec_rejects_out_of_range_temperatures(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        _offline_spec(**{field: 2.5})
