"""Validates completion memoization and its interaction with accounting."""

import threading

import pytest

from cognitivetree.llm import CachingLlmClient, LlmClient, LlmError, ScriptedLlmClient
from cognitivetree.llm.client import ChatMessage, CompletionRequest
from cognitivetree.observability import AccountingLlmClient


def request(
    content: str = "prompt", temperature: float = 0.0, max_tokens: int = 64
) -> CompletionRequest:
    return CompletionRequest(
        messages=(ChatMessage(role="user", content=content),),
        temperature=temperature,
        max_tokens=max_tokens,
    )


def echo_backend() -> ScriptedLlmClient:
    return ScriptedLlmClient(lambda r: f"answer to {r.messages[-1].content}")


def test_identical_deterministic_request_is_served_from_cache() -> None:
    backend = echo_backend()
    client = CachingLlmClient(backend)

    first = client.complete(request())
    second = client.complete(request())

    assert len(backend.requests) == 1
    assert second.text == first.text == "answer to prompt"
    assert first.prompt_tokens > 0
    assert (second.prompt_tokens, second.completion_tokens) == (0, 0)
    assert client.stats.hits == 1
    assert client.stats.misses == 1


@pytest.mark.parametrize(
    "variant",
    [
        request(content="other"),
        request(max_tokens=128),
        CompletionRequest(
            messages=(
                ChatMessage(role="system", content="policy"),
                ChatMessage(role="user", content="prompt"),
            ),
            temperature=0.0,
            max_tokens=64,
        ),
    ],
    ids=["content", "max-tokens", "extra-message"],
)
def test_requests_differing_in_any_field_miss(variant: CompletionRequest) -> None:
    backend = echo_backend()
    client = CachingLlmClient(backend)
    client.complete(request())
    client.complete(variant)
    assert len(backend.requests) == 2


def test_sampled_requests_bypass_the_cache_by_default() -> None:
    backend = echo_backend()
    client = CachingLlmClient(backend)
    for _ in range(3):
        client.complete(request(temperature=0.7))
    assert len(backend.requests) == 3
    assert client.stats.hits == client.stats.misses == 0


def test_sampled_requests_are_cached_when_opted_in() -> None:
    backend = echo_backend()
    client = CachingLlmClient(backend, cache_sampled=True)
    for _ in range(3):
        client.complete(request(temperature=0.7))
    assert len(backend.requests) == 1
    assert client.stats.hits == 2


def test_least_recently_used_entry_is_evicted() -> None:
    backend = echo_backend()
    client = CachingLlmClient(backend, max_entries=2)
    client.complete(request("a"))
    client.complete(request("b"))
    client.complete(request("a"))  # refreshes "a", leaving "b" as the oldest
    client.complete(request("c"))  # evicts "b"

    assert client.stats.entries == 2
    client.complete(request("a"))
    assert len(backend.requests) == 3
    client.complete(request("b"))
    assert len(backend.requests) == 4


def test_failures_are_not_cached() -> None:
    outcomes: list[object] = [LlmError("backend down"), "recovered"]

    def responder(_: CompletionRequest) -> str:
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return str(outcome)

    client = CachingLlmClient(ScriptedLlmClient(responder))
    with pytest.raises(LlmError):
        client.complete(request())
    assert client.complete(request()).text == "recovered"


def test_cache_inside_accounting_charges_only_real_tokens() -> None:
    backend = echo_backend()
    accounting = AccountingLlmClient(CachingLlmClient(backend))
    first = accounting.complete(request())
    accounting.complete(request())

    usage = accounting.usage
    assert usage.calls == 2
    assert usage.total_tokens == first.prompt_tokens + first.completion_tokens


def test_cache_outside_accounting_counts_only_backend_calls() -> None:
    accounting = AccountingLlmClient(echo_backend())
    client = CachingLlmClient(accounting)
    for _ in range(4):
        client.complete(request())
    assert accounting.usage.calls == 1


def test_clear_drops_entries_but_keeps_counters() -> None:
    client = CachingLlmClient(echo_backend())
    client.complete(request())
    client.complete(request())
    client.clear()

    stats = client.stats
    assert stats.entries == 0
    assert (stats.hits, stats.misses) == (1, 1)
    assert stats.hit_rate == 0.5


def test_concurrent_lookups_keep_consistent_counters() -> None:
    client = CachingLlmClient(echo_backend())
    client.complete(request())

    def hammer() -> None:
        for _ in range(200):
            assert client.complete(request()).text == "answer to prompt"

    threads = [threading.Thread(target=hammer) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert client.stats.hits == 1600
    assert client.stats.misses == 1


def test_satisfies_the_client_protocol_and_validates_capacity() -> None:
    assert isinstance(CachingLlmClient(echo_backend()), LlmClient)
    with pytest.raises(ValueError):
        CachingLlmClient(echo_backend(), max_entries=0)
