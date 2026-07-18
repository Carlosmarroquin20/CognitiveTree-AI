"""Client for OpenAI-compatible chat-completion endpoints.

The dialect is served by Ollama (``http://localhost:11434/v1``), vLLM,
llama.cpp server, and LM Studio, which makes a single client sufficient for
every targeted open-source backend. Transport is injected behind a small
protocol so request shaping, retry behavior, and error mapping are testable
without a network.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Protocol, runtime_checkable

from cognitivetree.llm.client import (
    CompletionRequest,
    CompletionResponse,
    LlmError,
)

logger = logging.getLogger(__name__)

_RETRYABLE_STATUS_FLOOR = 500


@runtime_checkable
class HttpJsonTransport(Protocol):
    """Posts a JSON payload and returns the raw response."""

    def post(
        self, url: str, payload: dict, headers: dict[str, str], timeout: float
    ) -> tuple[int, bytes]:
        """Returns ``(status_code, body)``; raises :class:`LlmError` on
        connection-level failures."""
        ...


class UrllibTransport:
    """Standard-library transport used outside of tests."""

    def post(
        self, url: str, payload: dict, headers: dict[str, str], timeout: float
    ) -> tuple[int, bytes]:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json", **headers},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read()
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise LlmError(f"backend unreachable at {url}: {exc}") from exc


class OpenAICompatibleClient:
    """Implements :class:`~cognitivetree.llm.client.LlmClient` over HTTP.

    Retries are limited to connection failures and 5xx responses; client-side
    errors (4xx) surface immediately since retrying them cannot succeed.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout_seconds: float = 120.0,
        max_retries: int = 1,
        transport: HttpJsonTransport | None = None,
    ) -> None:
        if not base_url.strip():
            raise ValueError("base_url must be a non-empty URL")
        if not model.strip():
            raise ValueError("model must be a non-empty identifier")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        self._endpoint = f"{base_url.rstrip('/')}/chat/completions"
        self._model = model
        self._api_key = api_key
        self._timeout = timeout_seconds
        self._max_retries = max_retries
        self._transport = transport or UrllibTransport()

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Returns the backend completion, retrying transient failures."""
        payload = {
            "model": self._model,
            "messages": [
                {"role": m.role, "content": m.content} for m in request.messages
            ],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        headers: dict[str, str] = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        last_error: LlmError | None = None
        for attempt in range(self._max_retries + 1):
            try:
                status, body = self._transport.post(
                    self._endpoint, payload, headers, self._timeout
                )
            except LlmError as exc:
                last_error = exc
                logger.warning(
                    "completion attempt %d/%d failed: %s",
                    attempt + 1,
                    self._max_retries + 1,
                    exc,
                )
                continue
            if status >= _RETRYABLE_STATUS_FLOOR:
                last_error = LlmError(
                    f"backend error HTTP {status}: {body[:200]!r}"
                )
                logger.warning(
                    "completion attempt %d/%d failed: %s",
                    attempt + 1,
                    self._max_retries + 1,
                    last_error,
                )
                continue
            if status != 200:
                raise LlmError(f"request rejected with HTTP {status}: {body[:200]!r}")
            return self._parse_response(body)
        assert last_error is not None
        raise last_error

    def _parse_response(self, body: bytes) -> CompletionResponse:
        """Extracts the completion text and usage from a 200 response body."""
        try:
            document = json.loads(body)
            text = document["choices"][0]["message"]["content"]
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            raise LlmError(f"malformed completion response: {body[:200]!r}") from exc
        if not isinstance(text, str):
            raise LlmError("completion content is not a string")
        usage = document.get("usage") or {}
        return CompletionResponse(
            text=text,
            model=str(document.get("model", "")),
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
        )
