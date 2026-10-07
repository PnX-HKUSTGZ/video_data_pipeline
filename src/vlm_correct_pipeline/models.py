"""Validated annotation geometry, durable sample states, and JSON boundaries."""

import math
from typing import Annotated, Literal

from pydantic import AfterValidator, Field, JsonValue, TypeAdapter

from .config import StrictModel

type JSONObject = dict[str, JsonValue]
JSON_OBJECT = TypeAdapter(dict[str, JsonValue])


def json_object(value: object) -> JSONObject:
    """Constrain untyped SDK/JSON values at the integration boundary."""
    return JSON_OBJECT.validate_python(value)


def valid_bbox(value: list[float]) -> list[float]:
    """Validate full-image xyxy; never clip or reorder model coordinates."""
    if len(value) != 4 or not all(math.isfinite(v) and 0 <= v <= 1 for v in value):
        raise ValueError("bbox requires four finite coordinates in [0, 1]")
    x1, y1, x2, y2 = value
    if x1 >= x2 or y1 >= y2:
        raise ValueError("bbox must have strictly positive width and height")
    return value


type BBox = Annotated[list[float], Field(min_length=4, max_length=4), AfterValidator(valid_bbox)]
type Status = Literal[
    "pending",
    "running",
    "completed",
    "empty",
    "deferred",
    "discarded",
    "max_calls",
    "api_error",
    "invalid_input",
    "interrupted",
]
SUCCESS = {"completed", "empty"}
DONE = SUCCESS | {"discarded"}


class Box(StrictModel):
    box_id: str
    class_id: int = Field(ge=0)
    bbox: BBox


class Usage(StrictModel):
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


class SampleResult(StrictModel):
    """A checkpoint includes the draft even when no publishable annotation exists."""

    sample_id: str
    fingerprint: str
    attempt: int = 1
    status: Status = "pending"
    revision: int = 0
    boxes: list[Box] = Field(default_factory=list)
    reason: str = ""
    discard_category: Literal["poor_lighting"] | None = None
    usage: Usage = Field(default_factory=Usage)


def label_text(boxes: list[Box]) -> str:
    """Serialize normalized boxes with enough precision to preserve small objects."""
    lines: list[str] = []
    for box in boxes:
        x1, y1, x2, y2 = box.bbox
        values = ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)
        encoded = [format(v, ".15g") for v in values]
        cx, cy, width, height = map(float, encoded)
        if width <= 0 or height <= 0:
            raise ValueError("Serialization would produce a degenerate box")
        if min(cx - width / 2, cy - height / 2) < -1e-12:
            raise ValueError("Serialized box crosses the image boundary")
        if max(cx + width / 2, cy + height / 2) > 1 + 1e-12:
            raise ValueError("Serialized box crosses the image boundary")
        lines.append(f"{box.class_id} {' '.join(encoded)}\n")
    return "".join(lines)
