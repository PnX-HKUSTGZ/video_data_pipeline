"""Responses terminal events bound gateway streams without exposing partial tool calls."""

import asyncio
import json

import httpx
import pytest
from openai import AsyncOpenAI

from vlm_correct_pipeline.client import APIProblem, ResponsesBackend
from vlm_correct_pipeline.config import APIConfig, Credentials
from vlm_correct_pipeline.tools import tool_definitions


class EventStream(httpx.AsyncByteStream):
    """Keep the connection open after application events, like a heartbeat gateway."""

    def __init__(self, events):
        self.events = events
        self.closed = False
        self.heartbeats = 0

    async def __aiter__(self):
        for event in self.events:
            yield ("data: " + json.dumps(event) + "\n\n").encode()
        while True:
            self.heartbeats += 1
            await asyncio.sleep(0.005)
            yield b": heartbeat\n\n"

    async def aclose(self):
        self.closed = True


def response_payload(status="completed"):
    return {
        "id": "resp-test",
        "object": "response",
        "created_at": 0,
        "model": "test-model",
        "status": status,
        "output": [
            {
                "id": "reason-test",
                "type": "reasoning",
                "summary": [],
                "encrypted_content": "private-reasoning-ciphertext",
            },
            {
                "id": "fc-test",
                "type": "function_call",
                "call_id": "call-test",
                "name": "mark_empty",
                "arguments": '{"reason":"private target description"}',
                "status": "completed",
            },
        ],
        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    }


def created_event():
    return {
        "type": "response.created",
        "sequence_number": 0,
        "response": {**response_payload(), "status": "in_progress", "output": []},
    }


def backend(body, timeout=0.5):
    return ResponsesBackend(
        APIConfig(model="test-model", stream=True, timeout_seconds=timeout),
        Credentials(base_url="https://test.invalid/v1", api_key="private-api-key"),
        client_factory=lambda: AsyncOpenAI(
            api_key="private-api-key",
            base_url="https://test.invalid/v1",
            max_retries=0,
            http_client=httpx.AsyncClient(
                transport=httpx.MockTransport(
                    lambda _: httpx.Response(
                        200, headers={"content-type": "text/event-stream"}, stream=body
                    )
                )
            ),
        ),
    )


def test_completed_response_returns_without_waiting_for_stream_eof():
    body = EventStream(
        [
            created_event(),
            {"type": "response.completed", "sequence_number": 1, "response": response_payload()},
        ]
    )
    result = backend(body).respond([], tool_definitions())
    assert body.closed and body.heartbeats == 0
    assert result.response_id == "resp-test"
    assert result.input_tokens == 10 and result.output_tokens == 5
    assert result.calls[0].name == "mark_empty"
    assert result.calls[0].arguments == '{"reason":"private target description"}'
    assert result.output[0]["encrypted_content"] == "private-reasoning-ciphertext"
    assert "parsed_arguments" not in result.output[1]


@pytest.mark.parametrize("partial", [False, True])
def test_timeout_diagnostics_count_events_without_exposing_partial_output(partial):
    events = []
    if partial:
        events = [
            created_event(),
            {
                "type": "response.output_item.added",
                "sequence_number": 1,
                "output_index": 0,
                "item": {
                    **response_payload()["output"][1],
                    "arguments": "",
                    "status": "in_progress",
                },
            },
            {
                "type": "response.function_call_arguments.delta",
                "sequence_number": 2,
                "output_index": 0,
                "item_id": "fc-test",
                "delta": '{"reason":"private target description"}',
            },
        ]
    body = EventStream(events)
    with pytest.raises(APIProblem, match="wall-time deadline") as error:
        backend(body, timeout=0.08).respond([], tool_definitions())
    assert body.closed and error.value.retryable
    message = str(error.value)
    assert "private" not in message
    if partial:
        assert "stream_events=3" in message
        assert "last_event=response.function_call_arguments.delta" in message
        assert "first_event_ms=none" not in message
    else:
        assert "stream_events=0; first_event_ms=none; last_event=none" in message


@pytest.mark.parametrize("status", ["incomplete", "failed"])
@pytest.mark.parametrize("nested_status", ["matching", "completed"])
def test_terminal_failure_never_exposes_tools_or_waits_for_eof(status, nested_status):
    body = EventStream(
        [
            created_event(),
            {
                "type": f"response.{status}",
                "sequence_number": 1,
                "response": response_payload(
                    status if nested_status == "matching" else nested_status
                ),
            },
        ]
    )
    with pytest.raises(APIProblem, match="incomplete or failed"):
        backend(body).respond([], tool_definitions())
    assert body.closed and body.heartbeats == 0
