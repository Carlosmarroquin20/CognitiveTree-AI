"""Consumption budgets that halt a run before it overspends."""

from __future__ import annotations

from cognitivetree.observability.accounting import AccountingLlmClient


class TokenBudget:
    """Stops a run once its LLM consumption crosses a configured ceiling.

    The budget reads the live tally on an
    :class:`~cognitivetree.observability.accounting.AccountingLlmClient`, so
    it measures exactly what the backend reported rather than an estimate.
    Both ceilings are optional and independent; whichever is crossed first
    ends the run, and supplying neither leaves consumption unbounded.

    The controller polls the budget at an iteration boundary, so a run stops
    *after* the iteration that crossed the ceiling rather than mid-expansion:
    the final total can exceed ``max_total_tokens`` by roughly one iteration's
    consumption. Sizing the ceiling below a hard quota, rather than at it, is
    what turns this into a guarantee.
    """

    def __init__(
        self,
        client: AccountingLlmClient,
        max_total_tokens: int | None = None,
        max_calls: int | None = None,
    ) -> None:
        if max_total_tokens is not None and max_total_tokens < 1:
            raise ValueError("max_total_tokens must be a positive integer")
        if max_calls is not None and max_calls < 1:
            raise ValueError("max_calls must be a positive integer")
        self._client = client
        self._max_total_tokens = max_total_tokens
        self._max_calls = max_calls

    @property
    def is_bounded(self) -> bool:
        """Reports whether any ceiling is configured."""
        return self._max_total_tokens is not None or self._max_calls is not None

    def check(self) -> str | None:
        """Returns why the run must stop, or ``None`` to let it continue."""
        usage = self._client.usage
        if (
            self._max_total_tokens is not None
            and usage.total_tokens >= self._max_total_tokens
        ):
            return (
                f"token budget of {self._max_total_tokens} exhausted: "
                f"{usage.total_tokens} consumed across {usage.calls} calls"
            )
        if self._max_calls is not None and usage.calls >= self._max_calls:
            return (
                f"call budget of {self._max_calls} exhausted: "
                f"{usage.calls} calls consuming {usage.total_tokens} tokens"
            )
        return None
