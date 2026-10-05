"""Typed API contracts shared by dataset parsing and the HTTP layer."""

from typing import Annotated, Literal

from pydantic import BaseModel, Field

Task = Literal["detect", "segment", "obb", "pose"]
LabelState = Literal["valid", "empty", "missing", "invalid"]


class OpenRequest(BaseModel):
    path: str = Field(min_length=1)
    task: Task = "detect"
    root: str | None = None
    kpt_shape: tuple[int, Literal[2, 3]] | None = None


class Point(BaseModel):
    x: float
    y: float


class Keypoint(Point):
    visibility: Literal[0, 1, 2] = 2


class Box(BaseModel):
    cx: float
    cy: float
    width: float
    height: float


class AnnotationBase(BaseModel):
    class_id: int
    line: int
    raw: str


class Detection(AnnotationBase):
    kind: Literal["detect"] = "detect"
    box: Box


class Segment(AnnotationBase):
    kind: Literal["segment"] = "segment"
    points: list[Point]


class OrientedBox(AnnotationBase):
    kind: Literal["obb"] = "obb"
    points: list[Point]


class Pose(AnnotationBase):
    kind: Literal["pose"] = "pose"
    box: Box
    keypoints: list[Keypoint]


Annotation = Annotated[Detection | Segment | OrientedBox | Pose, Field(discriminator="kind")]


class Issue(BaseModel):
    file: str
    line: int | None = None
    message: str


class Labels(BaseModel):
    state: LabelState
    annotations: list[Annotation] = Field(default_factory=list)
    issues: list[Issue] = Field(default_factory=list)


class SampleInfo(BaseModel):
    id: int
    path: str
    split: str
    state: LabelState
    instances: int
    class_ids: list[int]
    issue_count: int


class SampleDetail(SampleInfo):
    label_path: str
    width: int | None
    height: int | None
    annotations: list[Annotation]
    issues: list[Issue]
    image_error: str | None = None


class SamplePage(BaseModel):
    items: list[SampleInfo]
    total: int
    offset: int
    limit: int


class ClassCount(BaseModel):
    id: int
    name: str
    instances: int
    images: int


class Stats(BaseModel):
    images: int
    unique_images: int
    instances: int
    splits: dict[str, int]
    states: dict[str, int]
    classes: list[ClassCount]
    invalid_rows: int


class IndexStatus(BaseModel):
    id: str
    state: Literal["indexing", "ready", "error"]
    phase: str
    scanned: int = 0
    total: int = 0
    error: str | None = None
    candidates: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    task: Task
    names: dict[int, str] = Field(default_factory=dict)
    splits: list[str] = Field(default_factory=list)
