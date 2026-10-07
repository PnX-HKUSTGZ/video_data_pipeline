"""One-image interaction loop and a real, two-step visual capability probe."""

import json
import secrets
from collections.abc import Callable
from dataclasses import dataclass
from threading import Event

from PIL import Image, ImageDraw

from .client import APIProblem, Backend, FunctionCall, Turn
from .config import Config, RunConfig
from .models import JSONObject, SampleResult, Usage, json_object
from .render import image_item, message, observe, text_item
from .tools import AnnotationTools, ToolResult, tool_definitions

type Recorder = Callable[[JSONObject], None]


class CallLimit(Exception):
    """The per-image request budget, including retries, is exhausted."""


class RunCancelled(Exception):
    """Cooperative cancellation between network requests and state transitions."""


@dataclass
class Budget:
    """Bound real HTTP attempts independently from successful tool turns."""

    settings: RunConfig
    usage: Usage
    spent: int = 0

    def request(
        self,
        backend: Backend,
        history: list[JSONObject],
        tools: list[JSONObject],
        cancel: Event,
        record: Recorder,
        *,
        forced_tool: str | None = None,
    ) -> Turn:
        for retry in range(self.settings.max_retries + 1):
            if cancel.is_set():
                raise RunCancelled
            if self.spent >= self.settings.max_model_calls:
                raise CallLimit
            self.spent += 1
            self.usage.requests += 1
            record({"event": "request", "request_number": self.spent, "retry": retry})
            try:
                turn = backend.respond(history, tools, forced_tool=forced_tool)
            except APIProblem as error:
                record({"event": "api_error", "message": str(error), "fatal": error.fatal})
                if not error.retryable or retry == self.settings.max_retries:
                    raise
                if self.spent >= self.settings.max_model_calls:
                    raise CallLimit from None
                if cancel.wait(min(2**retry, 8)):
                    raise RunCancelled from None
                continue
            self.usage.input_tokens += turn.input_tokens
            self.usage.output_tokens += turn.output_tokens
            record(
                {
                    "event": "response",
                    "response_id": turn.response_id,
                    "input_tokens": turn.input_tokens,
                    "output_tokens": turn.output_tokens,
                }
            )
            return turn
        raise AssertionError("Unreachable request state")


def tool_output(
    call: FunctionCall,
    payload: JSONObject,
    observations: list[JSONObject] | None = None,
) -> JSONObject:
    """Attach image feedback to its tool call using native multimodal function outputs."""
    text = json.dumps(payload, ensure_ascii=False)
    return json_object(
        {
            "type": "function_call_output",
            "call_id": call.call_id,
            "output": [text_item(text), *observations] if observations else text,
        }
    )


def annotate(
    backend: Backend,
    config: Config,
    prefix: list[JSONObject],
    image: Image.Image,
    names: dict[int, str],
    state: SampleResult,
    record: Recorder,
    cancel: Event,
) -> SampleResult:
    """Only terminal tools create success; every edit must be followed by fresh vision input."""
    tools = AnnotationTools(state, names)
    state.status = "running"
    content, metadata = observe(
        image,
        state.boxes,
        names,
        config.image_max_edge,
        encoding=config.image_format,
        jpeg_quality=config.jpeg_quality,
    )
    initial = tools.snapshot(role="editable_target", **metadata)
    history = [*prefix, message([text_item(json.dumps(initial, ensure_ascii=False)), *content])]
    budget = Budget(config.run, state.usage)
    seen: dict[str, tuple[FunctionCall, ToolResult, JSONObject]] = {}
    try:
        while state.status == "running":
            turn = budget.request(backend, history, tool_definitions(), cancel, record)
            if cancel.is_set():
                raise RunCancelled
            history.extend(turn.output)
            if len(turn.calls) != 1:
                error = tools.error("one_tool_required", "Call exactly one tool per response")
                for call in turn.calls:
                    history.append(tool_output(call, error.payload))
                history.append(message([text_item("Use exactly one provided tool to continue.")]))
                record({"event": "protocol_error", "call_count": len(turn.calls)})
                continue
            call = turn.calls[0]
            duplicate = call.call_id in seen
            if duplicate:
                old_call, result, output = seen[call.call_id]
                if call != old_call:
                    raise APIProblem("Reused call_id with different arguments", fatal=True)
            else:
                result = tools.execute(call.name, call.arguments)
                content = []
                if state.status == "running" and (result.edited or result.region is not None):
                    content, metadata = observe(
                        image,
                        state.boxes,
                        names,
                        config.image_max_edge,
                        region=result.region,
                        encoding=config.image_format,
                        jpeg_quality=config.jpeg_quality,
                    )
                    result = ToolResult(
                        json_object({**result.payload, **metadata}),
                        result.edited,
                        result.region,
                    )
                output = tool_output(call, result.payload, content)
                seen[call.call_id] = (call, result, output)
            history.append(output)
            record(
                json_object(
                    {
                        "event": "tool",
                        "call_id": call.call_id,
                        "name": call.name,
                        "arguments": call.arguments,
                        "duplicate": duplicate,
                        "result": result.payload,
                    }
                )
            )
            if state.status != "running":
                break
    except CallLimit:
        state.status, state.reason = "max_calls", "Request budget exhausted before confirmation"
    except RunCancelled:
        state.status, state.reason = "interrupted", "Run cancelled before completion"
    except APIProblem as error:
        state.status, state.reason = "api_error", str(error)
        if error.fatal:
            cancel.set()
    record({"event": "terminal", "status": state.status, "reason": state.reason})
    return state


def probe(
    backend: Backend,
    *,
    cancel: Event | None = None,
    reference_context: list[JSONObject] | None = None,
    recorder: Recorder | None = None,
) -> JSONObject:
    """Verify two-image vision, strict functions, and a tool-result/new-image continuation."""
    stop = cancel or Event()
    usage = Usage()
    budget = Budget(RunConfig(max_model_calls=6, max_retries=1), usage)
    events: list[JSONObject] = []

    def record(value: JSONObject) -> None:
        events.append(value)
        if recorder:
            recorder(value)

    schema = json_object(
        {
            "type": "function",
            "name": "report_visual",
            "strict": True,
            "description": "Report the observed marker counts for this capability check.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {"first": {"type": "integer"}, "second": {"type": "integer"}},
                "required": ["first", "second"],
            },
        }
    )
    finish = json_object(
        {
            "type": "function",
            "name": "confirm_visual",
            "strict": True,
            "description": "Confirm receipt of the tool nonce and inspect the new image.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {"nonce": {"type": "string"}, "count": {"type": "integer"}},
                "required": ["nonce", "count"],
            },
        }
    )

    def synthetic(count: int) -> JSONObject:
        with Image.new("RGB", (320, 160), "white") as image:
            draw = ImageDraw.Draw(image)
            for index in range(count):
                x = 20 + 70 * index
                draw.rectangle((x, 40, x + 35, 90), fill="red")
            return image_item(image, 320)

    history = [
        *(reference_context or []),
        message(
            [
                text_item(
                    "Capability check: ignore prior reference images for this diagnostic. "
                    "Count red rectangles in the TWO images in THIS message, in order. "
                    "Call report_visual."
                ),
                synthetic(1),
                synthetic(2),
            ]
        ),
    ]
    first = budget.request(backend, history, [schema], stop, record, forced_tool="report_visual")
    record({"event": "probe_tools", "phase": "first", "call_count": len(first.calls)})
    if len(first.calls) != 1 or first.calls[0].name != "report_visual":
        raise APIProblem("Probe failed: expected exactly one report_visual call", fatal=True)
    args = json.loads(first.calls[0].arguments)
    if (
        args != {"first": 1, "second": 2}
        or type(args.get("first")) is not int
        or type(args.get("second")) is not int
    ):
        raise APIProblem("Probe failed: two-image visual counts differ", fatal=True)
    nonce = secrets.token_hex(12)
    history.extend(first.output)
    history.append(tool_output(first.calls[0], {"ok": True, "nonce": nonce}, [synthetic(3)]))
    history.append(
        message(
            [
                text_item(
                    "Return the tool nonce and count rectangles in the NEW image returned by the tool."
                ),
            ]
        )
    )
    last = budget.request(backend, history, [finish], stop, record, forced_tool="confirm_visual")
    record({"event": "probe_tools", "phase": "continuation", "call_count": len(last.calls)})
    if len(last.calls) != 1 or last.calls[0].name != "confirm_visual":
        raise APIProblem("Probe failed: missing continuation function call", fatal=True)
    if json.loads(last.calls[0].arguments) != {"nonce": nonce, "count": 3}:
        raise APIProblem("Probe failed: tool-result/new-image continuation differs", fatal=True)
    return json_object(
        {
            "status": "passed",
            "checks": [
                "multiple_images",
                "strict_functions",
                "single_tool",
                "tool_result_continuation",
                "new_image_continuation",
            ],
            "usage": usage.model_dump(mode="json"),
            "events": events,
        }
    )
