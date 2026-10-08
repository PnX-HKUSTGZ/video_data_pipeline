"""Typed Responses boundary; errors never contain credentials or request bodies."""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
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
                    return await stream.get_final_response()
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
        try:
            response = asyncio.run(self._request(history, tools, forced_tool))
        except APIStatusError as error:
            status = error.status_code
            raise APIProblem(
                f"Responses HTTP {status}",
                retryable=status in {408, 409, 429} or status >= 500,
                fatal=status in {400, 401, 403, 404, 405, 422},
            ) from None
        except TimeoutError:
            raise APIProblem("Responses wall-time deadline exceeded", retryable=True) from None
        except (APIConnectionError, httpx.TransportError):
            raise APIProblem("Responses connection or timeout error", retryable=True) from None
        except APIError:
            raise APIProblem("Invalid Responses protocol response", fatal=True) from None
        except RuntimeError:
            raise APIProblem("Responses stream ended before completion", retryable=True) from None
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
