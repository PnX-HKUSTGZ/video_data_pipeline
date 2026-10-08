"""Validated annotation geometry, durable sample states, and JSON boundaries."""

import math
from collections.abc import Sequence
from typing import Annotated, Literal

from pydantic import AfterValidator, Field, JsonValue, TypeAdapter, field_validator

from .config import PoseConfig, StrictModel

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


class KeypointState(StrictModel):
    """A draft point keeps review status separate from YOLO visibility zero."""

    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    visibility: Literal[0, 1, 2]
    reviewed: bool

    @field_validator("visibility", mode="before")
    @classmethod
    def integer_visibility(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("visibility must be an integer")
        return value


class PoseBox(Box):
    """A stable instance with fixed-order full-image normalized keypoint slots."""

    keypoints: list[KeypointState]


def initial_keypoints(pose: PoseConfig, class_id: int) -> list[KeypointState]:
    """Create pending applicable slots; inactive padding is already confirmed absent."""
    indices = pose.class_keypoint_indices.get(class_id)
    if indices is None:
        raise ValueError("Unknown pose class_id")
    applicable = set(indices)
    return [
        KeypointState(x=0, y=0, visibility=0, reviewed=index not in applicable)
        for index in range(pose.kpt_shape[0])
    ]


def validate_pose_boxes(
    boxes: Sequence[Box], pose: PoseConfig, *, require_reviewed: bool = False
) -> None:
    """Validate fixed slots and, at completion, explicit visible-only point decisions."""
    for box in boxes:
        if not isinstance(box, PoseBox):
            raise ValueError("Pose mode requires keypoints for every instance")  # noqa: TRY004
        indices = pose.class_keypoint_indices.get(box.class_id)
        if indices is None or len(box.keypoints) != pose.kpt_shape[0]:
            raise ValueError("Pose instance class or keypoint count is invalid")
        applicable = set(indices)
        visible = 0
        for index, point in enumerate(box.keypoints):
            if index not in applicable:
                if point.x != 0 or point.y != 0 or point.visibility != 0:
                    raise ValueError("Inactive pose keypoint slots must be 0 0 0")
                continue
            if point.visibility == 0 and (point.x != 0 or point.y != 0):
                raise ValueError("Visibility-zero keypoints must use zero coordinates")
            if require_reviewed and not point.reviewed:
                raise ValueError(f"Unreviewed keypoint {index} in {box.box_id}")
            if require_reviewed and point.visibility == 1:
                raise ValueError(
                    "Final pose annotations cannot contain inferred visibility-one points"
                )
            visible += point.visibility == 2
        if require_reviewed and not visible:
            raise ValueError(
                f"Instance {box.box_id} requires at least one visible applicable point"
            )


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
    boxes: list[Box | PoseBox] = Field(default_factory=list)
    reason: str = ""
    discard_category: Literal["poor_lighting"] | None = None
    usage: Usage = Field(default_factory=Usage)


def label_text(boxes: Sequence[Box], pose: PoseConfig | None = None) -> str:
    """Serialize reviewed normalized annotations, preserving the detection wire format."""
    if pose is not None:
        validate_pose_boxes(boxes, pose, require_reviewed=True)
    elif any(isinstance(box, PoseBox) for box in boxes):
        raise ValueError("Pose serialization requires explicit pose settings")
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
        if isinstance(box, PoseBox):
            for point in box.keypoints:
                encoded.extend(
                    (format(point.x, ".15g"), format(point.y, ".15g"), str(point.visibility))
                )
        lines.append(f"{box.class_id} {' '.join(encoded)}\n")
    return "".join(lines)
