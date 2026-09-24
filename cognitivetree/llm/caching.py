"""Transparent memoization of completions for repeated identical requests."""

from __future__ import annotations

import dataclasses
import threading
from collections import OrderedDict
from dataclasses import dataclass

from cognitivetree.llm.client import CompletionRequest, CompletionResponse, LlmClient


@dataclass(frozen=True, slots=True)
class CacheStats:
    """Point-in-time counters of a :class:`CachingLlmClient`."""

    hits: int
    misses: int
    entries: int

    @property
    def hit_rate(self) -> float:
        """Returns the fraction of eligible requests served from the cache."""
        lookups = self.hits + self.misses
        return self.hits / lookups if lookups else 0.0


class CachingLlmClient:
    """Wraps any ``LlmClient`` and replays completions for identical requests.

    The request itself is the key: it is immutable and hashable, so messages,
    temperature, and token limit all take part and distinct requests never
    collide. Only deterministic requests (temperature ``0``) are cached by
    default. A sampled request is repeated precisely to obtain a different
    answer, and replaying one would quietly remove the diversity that tree
    search relies on; ``cache_sampled`` opts into it for workloads, such as
    benchmark reruns, where reproducibility matters more.

    A cache hit reports zero tokens, because none were spent. Placing the
    cache inside an :class:`~cognitivetree.observability.AccountingLlmClient`
    therefore keeps token totals and budgets honest, while placing it outside
    also keeps the call count limited to real backend calls. Failures are
    never cached, and the backend call runs outside the lock, so concurrent
    callers are not serialized behind a slow completion.
    """

    def __init__(
        self,
        inner: LlmClient,
        max_entries: int = 1024,
        cache_sampled: bool = False,
    ) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be a positive integer")
        self._inner = inner
        self._max_entries = max_entries
        self._cache_sampled = cache_sampled
        self._entries: OrderedDict[CompletionRequest, CompletionResponse] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Returns a cached completion when one exists, else queries the backend."""
        if not self._is_cacheable(request):
            return self._inner.complete(request)
        with self._lock:
            cached = self._entries.get(request)
            if cached is not None:
                self._entries.move_to_end(request)
                self._hits += 1
                return dataclasses.replace(cached, prompt_tokens=0, completion_tokens=0)
            self._misses += 1
        response = self._inner.complete(request)
        with self._lock:
            self._entries[request] = response
            self._entries.move_to_end(request)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)
        return response

    @property
    def stats(self) -> CacheStats:
        """Returns the hit, miss, and occupancy counters so far."""
        with self._lock:
            return CacheStats(
                hits=self._hits, misses=self._misses, entries=len(self._entries)
            )

    def clear(self) -> None:
        """Drops every cached completion; the counters are kept."""
        with self._lock:
            self._entries.clear()

    def _is_cacheable(self, request: CompletionRequest) -> bool:
        return self._cache_sampled or request.temperature == 0.0
