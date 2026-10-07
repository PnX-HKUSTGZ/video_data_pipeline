"""Whitelisted tools operate atomically on one image's in-memory annotation state."""

from dataclasses import dataclass
from typing import Literal

from pydantic import Field, ValidationError

from .config import StrictModel
from .models import BBox, Box, JSONObject, SampleResult, json_object


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
}


def tool_definitions() -> list[JSONObject]:
    """Expose only strict function schemas; the host also validates every argument."""
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
        for name, model in ARGUMENTS.items()
    ]


@dataclass(frozen=True)
class ToolResult:
    payload: JSONObject
    edited: bool = False
    region: list[float] | None = None


class AnnotationTools:
    """Own stable IDs and reject invalid operations before mutating the draft."""

    def __init__(self, state: SampleResult, names: dict[int, str]) -> None:
        self.state = state
        self.names = names
        self.next_id = len(state.boxes) + 1

    def snapshot(self, **extra: object) -> JSONObject:
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
        model = ARGUMENTS.get(name)
        if model is None:
            return self.error("unknown_tool", "Use one of the supplied tools")
        try:
            args = model.model_validate_json(arguments)
            if isinstance(args, AddArgs) and args.class_id not in self.names:
                raise ValueError("Unknown class_id")
            if isinstance(args, (UpdateArgs, DeleteArgs)) and not any(
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
        if isinstance(args, UpdateArgs):
            replacement = Box(box_id=args.box_id, class_id=args.class_id, bbox=args.bbox)
            state.boxes = [replacement if b.box_id == args.box_id else b for b in state.boxes]
            edited = True
        elif isinstance(args, AddArgs):
            state.boxes = [
                *state.boxes,
                Box(
                    box_id=f"box_{self.next_id:04d}",
                    class_id=args.class_id,
                    bbox=args.bbox,
                ),
            ]
            self.next_id += 1
            edited = True
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
        return ToolResult(self.snapshot(ok=True), edited, region)
