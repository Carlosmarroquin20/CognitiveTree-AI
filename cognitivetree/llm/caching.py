"""Transparent memoization of completions for repeated identical requests.

The store and the client wrapper are separate so one store can outlive the
clients that use it: a server builds a fresh, individually accounted client
per session, and sharing only the store is what lets those sessions reuse
each other's completions without mixing their token tallies.
"""

from __future__ import annotations

import dataclasses
import threading
from collections import OrderedDict
from dataclasses import dataclass

from cognitivetree.llm.client import CompletionRequest, CompletionResponse, LlmClient


@dataclass(frozen=True, slots=True)
class CacheStats:
    """Point-in-time counters of a :class:`CompletionCache`."""

    hits: int
    misses: int
    entries: int

    @property
    def hit_rate(self) -> float:
        """Returns the fraction of lookups served from the cache."""
        lookups = self.hits + self.misses
        return self.hits / lookups if lookups else 0.0


class CompletionCache:
    """Thread-safe, bounded LRU store of completions keyed by request.

    The request itself is the key: it is immutable and hashable, so messages,
    temperature, and token limit all take part and distinct requests never
    collide.
    """

    def __init__(self, max_entries: int = 1024) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be a positive integer")
        self._max_entries = max_entries
        self._entries: OrderedDict[CompletionRequest, CompletionResponse] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, request: CompletionRequest) -> CompletionResponse | None:
        """Returns the stored completion for ``request``, recording the lookup."""
        with self._lock:
            cached = self._entries.get(request)
            if cached is None:
                self._misses += 1
                return None
            self._entries.move_to_end(request)
            self._hits += 1
            return cached

    def put(self, request: CompletionRequest, response: CompletionResponse) -> None:
        """Stores ``response``, evicting the least recently used entry if full."""
        with self._lock:
            self._entries[request] = response
            self._entries.move_to_end(request)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)

    @property
    def stats(self) -> CacheStats:
        """Returns the hit, miss, and occupancy counters so far."""
        with self._lock:
            return CacheStats(
                hits=self._hits, misses=self._misses, entries=len(self._entries)
            )

    def clear(self) -> None:
        """Drops every stored completion; the counters are kept."""
        with self._lock:
            self._entries.clear()


class CachingLlmClient:
    """Wraps any ``LlmClient`` and replays completions for identical requests.

    Only deterministic requests (temperature ``0``) are cached by default. A
    sampled request is repeated precisely to obtain a different answer, and
    replaying one would quietly remove the diversity that tree search relies
    on; ``cache_sampled`` opts into it for workloads, such as benchmark
    reruns, where reproducibility matters more.

    A cache hit reports zero tokens, because none were spent. Placing the
    cache inside an :class:`~cognitivetree.observability.AccountingLlmClient`
    therefore keeps token totals and budgets honest, while placing it outside
    also keeps the call count limited to real backend calls. Failures are
    never cached, and the backend call runs outside the store's lock, so
    concurrent callers are not serialized behind a slow completion.
    """

    def __init__(
        self,
        inner: LlmClient,
        cache: CompletionCache | None = None,
        cache_sampled: bool = False,
    ) -> None:
        self._inner = inner
        self._cache = cache if cache is not None else CompletionCache()
        self._cache_sampled = cache_sampled

    @property
    def cache(self) -> CompletionCache:
        """Returns the store this client reads and fills."""
        return self._cache

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Returns a cached completion when one exists, else queries the backend."""
        if not (self._cache_sampled or request.temperature == 0.0):
            return self._inner.complete(request)
        cached = self._cache.get(request)
        if cached is not None:
            return dataclasses.replace(cached, prompt_tokens=0, completion_tokens=0)
        response = self._inner.complete(request)
        self._cache.put(request, response)
        return response
