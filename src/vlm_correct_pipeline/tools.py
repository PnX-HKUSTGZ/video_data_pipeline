"""Whitelisted tools operate atomically on one image's in-memory annotation state."""

from dataclasses import dataclass
from typing import Literal

from pydantic import Field, ValidationError, field_validator

from .config import PoseConfig, StrictModel
from .models import (
    BBox,
    Box,
    JSONObject,
    KeypointState,
    PoseBox,
    SampleResult,
    initial_keypoints,
    json_object,
    validate_pose_boxes,
)


class AddArgs(StrictModel):
    class_id: int = Field(ge=0)
    bbox: BBox


class UpdateArgs(AddArgs):
    box_id: str


class DeleteArgs(StrictModel):
    box_id: str


class RegionArgs(StrictModel):
    bbox: BBox


class ReasonArgs(StrictModel):
    reason: str = Field(min_length=1, max_length=2000)


class DiscardArgs(ReasonArgs):
    category: Literal["poor_lighting"]


class KeypointUpdate(StrictModel):
    """An explicit visible or unlocatable point decision in the full-image frame."""

    index: int = Field(ge=0)
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    visibility: Literal[0, 2]

    @field_validator("visibility", mode="before")
    @classmethod
    def integer_visibility(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("visibility must be the integer 0 or 2")
        return value


class SetKeypointsArgs(StrictModel):
    box_id: str
    updates: list[KeypointUpdate] = Field(min_length=1)


ARGUMENTS: dict[str, type[StrictModel]] = {
    "add_box": AddArgs,
    "update_box": UpdateArgs,
    "delete_box": DeleteArgs,
    "inspect_region": RegionArgs,
    "finish_annotation": ReasonArgs,
    "mark_empty": ReasonArgs,
    "defer_annotation": ReasonArgs,
    "discard_image": DiscardArgs,
}
DESCRIPTIONS = {
    "add_box": "Add one box using FULL IMAGE normalized xyxy coordinates.",
    "update_box": "Replace the class and full-image xyxy coordinates of an existing box.",
    "delete_box": "Delete one existing box. This does not confirm a negative sample.",
    "inspect_region": "Inspect a crop. All subsequent edits still use FULL IMAGE coordinates.",
    "finish_annotation": "Confirm the latest reviewed nonempty annotation is complete.",
    "mark_empty": "Explicitly confirm no task objects exist; clear all boxes and finish.",
    "defer_annotation": "Stop for human review when evidence or annotation rules are unclear.",
    "discard_image": "Exclude an image whose poor lighting prevents reliable annotation. This is NOT a negative sample.",
    "set_keypoints": "Atomically review one or more applicable keypoints of one box. Use FULL IMAGE normalized x/y, visibility 2 for visibly located points, or visibility 0 with x=y=0 when not reliably visible. Never infer hidden coordinates.",
}


def tool_definitions(pose: PoseConfig | None = None) -> list[JSONObject]:
    """Expose only strict function schemas; the host also validates every argument."""
    arguments = dict(ARGUMENTS)
    if pose is not None:
        arguments["set_keypoints"] = SetKeypointsArgs
    return [
        json_object(
            {
                "type": "function",
                "name": name,
                "description": DESCRIPTIONS[name],
                "strict": True,
                "parameters": model.model_json_schema(),
            }
        )
        for name, model in arguments.items()
    ]


@dataclass(frozen=True)
class ToolResult:
    payload: JSONObject
    edited: bool = False
    region: list[float] | None = None
    focus_box_id: str | None = None


class AnnotationTools:
    """Own stable IDs and reject invalid operations before mutating the draft."""

    def __init__(
        self, state: SampleResult, names: dict[int, str], pose: PoseConfig | None = None
    ) -> None:
        self.state = state
        self.names = names
        self.pose = pose
        self.next_id = len(state.boxes) + 1

    def snapshot(self, **extra: object) -> JSONObject:
        if self.pose is not None:
            warnings: list[JSONObject] = []
            for box in self.state.boxes:
                if not isinstance(box, PoseBox):
                    continue
                x1, y1, x2, y2 = box.bbox
                for index, point in enumerate(box.keypoints):
                    if point.visibility == 2 and not (x1 <= point.x <= x2 and y1 <= point.y <= y2):
                        warnings.append(
                            {
                                "code": "keypoint_outside_bbox",
                                "box_id": box.box_id,
                                "index": index,
                            }
                        )
            extra = {"warnings": warnings, **extra}
        return json_object(
            {
                "revision": self.state.revision,
                "status": self.state.status,
                "boxes": [box.model_dump(mode="json") for box in self.state.boxes],
                **extra,
            }
        )

    def error(self, code: str, message: str) -> ToolResult:
        return ToolResult(self.snapshot(ok=False, error={"code": code, "message": message}))

    def execute(self, name: str, arguments: str) -> ToolResult:
        """Execute one validated operation; failures preserve both boxes and revision."""
        if self.state.status != "running":
            return self.error("closed", "This sample's annotation session is closed")
        model = SetKeypointsArgs if name == "set_keypoints" and self.pose else ARGUMENTS.get(name)
        if model is None:
            return self.error("unknown_tool", "Use one of the supplied tools")
        try:
            args = model.model_validate_json(arguments)
            if isinstance(args, AddArgs) and args.class_id not in self.names:
                raise ValueError("Unknown class_id")
            if isinstance(args, (UpdateArgs, DeleteArgs, SetKeypointsArgs)) and not any(
                box.box_id == args.box_id for box in self.state.boxes
            ):
                raise ValueError("Unknown box_id")
            if isinstance(args, ReasonArgs) and not args.reason.strip():
                raise ValueError("reason must not be blank")
            return self._apply(name, args)
        except ValidationError as error:
            message = "; ".join(str(e["msg"]) for e in error.errors(include_input=False))
            return self.error("invalid_arguments", message)
        except ValueError as error:
            return self.error("invalid_arguments", str(error))

    def _apply(self, name: str, args: StrictModel) -> ToolResult:
        state = self.state
        edited = False
        region: list[float] | None = None
        focus_box_id: str | None = None
        if isinstance(args, UpdateArgs):
            replacement: Box | PoseBox
            if self.pose is not None:
                current = next(b for b in state.boxes if b.box_id == args.box_id)
                if not isinstance(current, PoseBox):
                    raise ValueError("Pose mode requires pose instances")
                replacement = PoseBox(
                    box_id=args.box_id,
                    class_id=args.class_id,
                    bbox=args.bbox,
                    keypoints=(
                        current.keypoints
                        if current.class_id == args.class_id
                        else initial_keypoints(self.pose, args.class_id)
                    ),
                )
            else:
                replacement = Box(box_id=args.box_id, class_id=args.class_id, bbox=args.bbox)
            state.boxes = [replacement if b.box_id == args.box_id else b for b in state.boxes]
            edited = True
            focus_box_id = args.box_id if self.pose is not None else None
        elif isinstance(args, AddArgs):
            while any(b.box_id == f"box_{self.next_id:04d}" for b in state.boxes):
                self.next_id += 1
            box_id = f"box_{self.next_id:04d}"
            new_box: Box | PoseBox
            if self.pose is not None:
                new_box = PoseBox(
                    box_id=box_id,
                    class_id=args.class_id,
                    bbox=args.bbox,
                    keypoints=initial_keypoints(self.pose, args.class_id),
                )
            else:
                new_box = Box(box_id=box_id, class_id=args.class_id, bbox=args.bbox)
            state.boxes = [*state.boxes, new_box]
            self.next_id += 1
            edited = True
            focus_box_id = box_id if self.pose is not None else None
        elif isinstance(args, SetKeypointsArgs):
            if self.pose is None:
                raise ValueError("set_keypoints requires pose mode")
            current = next(b for b in state.boxes if b.box_id == args.box_id)
            if not isinstance(current, PoseBox):
                raise ValueError("Pose mode requires pose instances")  # noqa: TRY004
            applicable = set(self.pose.class_keypoint_indices[current.class_id])
            indices = [point.index for point in args.updates]
            if len(set(indices)) != len(indices):
                raise ValueError("Duplicate keypoint indices in one update")
            points = list(current.keypoints)
            for point in args.updates:
                if point.index not in applicable:
                    raise ValueError("Keypoint index is inactive for this class or out of range")
                if point.visibility == 0 and (point.x != 0 or point.y != 0):
                    raise ValueError("Visibility-zero keypoints must use zero coordinates")
                points[point.index] = KeypointState(
                    x=point.x, y=point.y, visibility=point.visibility, reviewed=True
                )
            changed = current.model_copy(update={"keypoints": points})
            validate_pose_boxes([changed], self.pose)
            state.boxes = [changed if b.box_id == args.box_id else b for b in state.boxes]
            edited = True
            focus_box_id = args.box_id
        elif isinstance(args, DeleteArgs):
            state.boxes = [b for b in state.boxes if b.box_id != args.box_id]
            edited = True
        elif isinstance(args, RegionArgs):
            region = args.bbox
        elif isinstance(args, DiscardArgs):
            state.status = "discarded"
            state.discard_category = args.category
            state.reason = args.reason.strip()
        elif isinstance(args, ReasonArgs):
            if name == "finish_annotation":
                if not state.boxes:
                    raise ValueError("Use mark_empty to explicitly confirm a negative sample")
                if self.pose is not None:
                    validate_pose_boxes(state.boxes, self.pose, require_reviewed=True)
                state.status = "completed"
            elif name == "mark_empty":
                state.boxes = []
                state.status = "empty"
                edited = True
            else:
                state.status = "deferred"
            state.reason = args.reason.strip()
        if edited:
            state.revision += 1
        return ToolResult(self.snapshot(ok=True), edited, region, focus_box_id)
