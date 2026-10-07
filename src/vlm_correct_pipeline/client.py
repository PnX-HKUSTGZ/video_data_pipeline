"""Typed Responses boundary; errors never contain credentials or request bodies."""

from dataclasses import dataclass
from typing import Protocol, cast

from openai import APIConnectionError, APIError, APIStatusError, OpenAI
from openai.types.responses import ResponseInputParam, ToolChoiceFunctionParam, ToolParam
from openai.types.responses.response import Response

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


class ResponsesBackend:
    """Use stateless Responses history and disable the SDK's hidden retry budget."""

    def __init__(self, config: APIConfig, auth: Credentials) -> None:
        self.config = config
        self.client = OpenAI(
            base_url=auth.base_url,
            api_key=auth.api_key.get_secret_value(),
            timeout=config.timeout_seconds,
            max_retries=0,
        )

    def close(self) -> None:
        self.client.close()

    def respond(
        self,
        history: list[JSONObject],
        tools: list[JSONObject],
        *,
        forced_tool: str | None = None,
    ) -> Turn:
        choice: ToolChoiceFunctionParam = {"type": "function", "name": forced_tool or ""}
        response: Response
        try:
            if self.config.stream:
                with self.client.responses.stream(
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
                    response = stream.get_final_response()
            else:
                response = self.client.responses.create(
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
        except APIStatusError as error:
            status = error.status_code
            raise APIProblem(
                f"Responses HTTP {status}",
                retryable=status in {408, 409, 429} or status >= 500,
                fatal=status in {400, 401, 403, 404, 405, 422},
            ) from None
        except APIConnectionError:
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
            output=[json_object(item.model_dump(mode="json")) for item in response.output],
            calls=calls,
            input_tokens=usage.input_tokens if usage else 0,
            output_tokens=usage.output_tokens if usage else 0,
            response_id=response.id,
        )
