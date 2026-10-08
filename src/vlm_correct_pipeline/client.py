"""Typed Responses boundary; errors never contain credentials or request bodies."""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from time import monotonic
from typing import Protocol, cast

import httpx
from openai import APIConnectionError, APIError, APIStatusError, AsyncOpenAI
from openai.types.responses import ResponseInputParam, ToolChoiceFunctionParam, ToolParam
from openai.types.responses.response import Response
from openai.types.responses.response_output_item import ResponseOutputItem

from .config import APIConfig, Credentials
from .models import JSONObject, json_object


class APIProblem(Exception):
    """A sanitized transport/protocol error, classified for bounded retries."""

    def __init__(self, message: str, *, retryable: bool = False, fatal: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.fatal = fatal


@dataclass
class _StreamTrace:
    """Request-local transport diagnostics; never retain event contents or identifiers."""

    started: float = field(default_factory=monotonic)
    events: int = 0
    first_event_ms: int | None = None
    last_event: str = "none"

    def note(self, kind: str) -> None:
        self.events += 1
        if self.first_event_ms is None:
            self.first_event_ms = round((monotonic() - self.started) * 1000)
        # SDK model construction is permissive: do not log an arbitrary upstream type string.
        self.last_event = (
            kind
            if kind
            in {
                "response.created",
                "response.in_progress",
                "response.completed",
                "response.failed",
                "response.incomplete",
                "response.output_item.added",
                "response.output_item.done",
                "response.content_part.added",
                "response.content_part.done",
                "response.function_call_arguments.delta",
                "response.function_call_arguments.done",
                "response.output_text.delta",
                "response.output_text.done",
                "response.reasoning_summary_text.delta",
                "response.reasoning_summary_text.done",
                "response.reasoning_summary_part.added",
                "response.reasoning_summary_part.done",
                "error",
            }
            else "other"
        )

    def summary(self) -> str:
        first = str(self.first_event_ms) if self.first_event_ms is not None else "none"
        return (
            f" [stream_events={self.events}; first_event_ms={first}; last_event={self.last_event}]"
        )


@dataclass(frozen=True)
class FunctionCall:
    call_id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class Turn:
    output: list[JSONObject]
    calls: list[FunctionCall]
    input_tokens: int = 0
    output_tokens: int = 0
    response_id: str = ""


class Backend(Protocol):
    """Injectable network boundary for deterministic, offline loop tests."""

    def respond(
        self,
        history: list[JSONObject],
        tools: list[JSONObject],
        *,
        forced_tool: str | None = None,
    ) -> Turn: ...


def history_item(item: ResponseOutputItem) -> JSONObject:
    """Replay API fields, retaining reasoning/phase but excluding SDK-only parsed values."""
    value = json_object(item.model_dump(mode="json", exclude_unset=True, exclude_none=True))
    value.pop("parsed_arguments", None)
    content = value.get("content")
    if value.get("type") == "message" and isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "output_text":
                part.pop("parsed", None)
    return value


class ResponsesBackend:
    """Use stateless Responses history and disable the SDK's hidden retry budget."""

    def __init__(
        self,
        config: APIConfig,
        auth: Credentials,
        *,
        client_factory: Callable[[], AsyncOpenAI] | None = None,
    ) -> None:
        self.config = config
        self._auth = auth
        self._client_factory = client_factory or self._new_client

    def _new_client(self) -> AsyncOpenAI:
        """Each request owns its client so cancellation cannot affect another image."""
        return AsyncOpenAI(
            base_url=self._auth.base_url,
            api_key=self._auth.api_key.get_secret_value(),
            timeout=self.config.timeout_seconds,
            max_retries=0,
        )

    def close(self) -> None:
        """Request-scoped async contexts close all connections before respond returns."""

    async def _request(
        self,
        history: list[JSONObject],
        tools: list[JSONObject],
        forced_tool: str | None,
        trace: _StreamTrace,
    ) -> Response:
        """Bound total wall time, including heartbeat streams and partial-body reads."""
        choice: ToolChoiceFunctionParam = {"type": "function", "name": forced_tool or ""}
        async with asyncio.timeout(self.config.timeout_seconds), self._client_factory() as client:
            if self.config.stream:
                async with client.responses.stream(
                    model=self.config.model,
                    input=cast(ResponseInputParam, history),
                    tools=cast(list[ToolParam], tools),
                    tool_choice=choice if forced_tool else "required",
                    parallel_tool_calls=False,
                    store=False,
                    truncation="disabled",
                    include=["reasoning.encrypted_content"],
                    max_output_tokens=self.config.max_output_tokens,
                ) as stream:
                    async for event in stream:
                        trace.note(event.type)
                        if event.type == "response.completed":
                            # The SDK already parsed/accumulated this response. Waiting for
                            # get_final_response() also waits for EOF, which some gateways
                            # postpone with heartbeats after this authoritative terminal event.
                            return event.response
                        if event.type == "response.failed" or event.type == "response.incomplete":
                            # A contradictory nested status must not turn a failure event
                            # into an executable completed tool call.
                            raise APIProblem("Responses output was incomplete or failed")
                    raise RuntimeError("Responses stream ended before completion")
            return await client.responses.create(
                model=self.config.model,
                input=cast(ResponseInputParam, history),
                tools=cast(list[ToolParam], tools),
                tool_choice=choice if forced_tool else "required",
                parallel_tool_calls=False,
                store=False,
                truncation="disabled",
                include=["reasoning.encrypted_content"],
                max_output_tokens=self.config.max_output_tokens,
            )

    def respond(
        self,
        history: list[JSONObject],
        tools: list[JSONObject],
        *,
        forced_tool: str | None = None,
    ) -> Turn:
        """Provide a synchronous worker boundary backed by cancellable async HTTP."""
        trace = _StreamTrace()
        try:
            response = asyncio.run(self._request(history, tools, forced_tool, trace))
        except APIStatusError as error:
            status = error.status_code
            raise APIProblem(
                f"Responses HTTP {status}",
                retryable=status in {408, 409, 429} or status >= 500,
                fatal=status in {400, 401, 403, 404, 405, 422},
            ) from None
        except TimeoutError:
            diagnostics = trace.summary() if self.config.stream else ""
            raise APIProblem(
                "Responses wall-time deadline exceeded" + diagnostics, retryable=True
            ) from None
        except (APIConnectionError, httpx.TransportError):
            diagnostics = trace.summary() if self.config.stream else ""
            raise APIProblem(
                "Responses connection or timeout error" + diagnostics, retryable=True
            ) from None
        except APIError:
            raise APIProblem("Invalid Responses protocol response", fatal=True) from None
        except RuntimeError:
            raise APIProblem(
                "Responses stream ended before completion" + trace.summary(), retryable=True
            ) from None
        if not isinstance(response, Response):
            raise APIProblem("Endpoint returned a non-Responses payload", fatal=True)
        if response.status != "completed":
            raise APIProblem("Responses output was incomplete or failed")
        calls = [
            FunctionCall(item.call_id, item.name, item.arguments)
            for item in response.output
            if item.type == "function_call"
        ]
        usage = response.usage
        return Turn(
            output=[history_item(item) for item in response.output],
            calls=calls,
            input_tokens=usage.input_tokens if usage else 0,
            output_tokens=usage.output_tokens if usage else 0,
            response_id=response.id,
        )
