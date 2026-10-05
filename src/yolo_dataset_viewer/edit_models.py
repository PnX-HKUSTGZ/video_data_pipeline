"""Validated, explicit contracts for persistent dataset edits."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from .models import Box, Keypoint


class EditModel(BaseModel):
    """Reject misspelled operation fields instead of silently losing edits."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class KeypointDefinition(EditModel):
    id: str
    name: str


class KeypointInput(EditModel):
    id: str | None = None
    name: str


class ClassDefinition(EditModel):
    id: int
    name: str
    keypoints: list[KeypointDefinition] = Field(default_factory=list)


class DraftStatus(EditModel):
    session_id: str | None = None
    revision: int
    classes: list[ClassDefinition]
    task: Literal["detect", "pose"]
    dirty: bool
    can_undo: bool
    can_redo: bool
    conflicts: list[str] = Field(default_factory=list)
    modified_images: int = 0
    deleted_images: int = 0


class ClassAdd(EditModel):
    kind: Literal["class_add"] = "class_add"
    name: str


class ClassRename(EditModel):
    kind: Literal["class_rename"] = "class_rename"
    class_id: int
    name: str


class ClassDelete(EditModel):
    kind: Literal["class_delete"] = "class_delete"
    class_id: int


class KeypointsSet(EditModel):
    kind: Literal["keypoints_set"] = "keypoints_set"
    class_id: int
    keypoints: list[KeypointInput]


class SampleDelete(EditModel):
    kind: Literal["sample_delete"] = "sample_delete"
    sample_id: int
    deleted: bool = True


class AnnotationUpsert(EditModel):
    kind: Literal["annotation_upsert"] = "annotation_upsert"
    sample_id: int
    annotation_id: str | None = None
    class_id: int
    box: Box
    keypoints: list[Keypoint] | None = None
    issue_line: int | None = None


class AnnotationDelete(EditModel):
    kind: Literal["annotation_delete"] = "annotation_delete"
    sample_id: int
    annotation_id: str


class IssueDelete(EditModel):
    kind: Literal["issue_delete"] = "issue_delete"
    sample_id: int
    line: int


EditOperation = Annotated[
    ClassAdd
    | ClassRename
    | ClassDelete
    | KeypointsSet
    | SampleDelete
    | AnnotationUpsert
    | AnnotationDelete
    | IssueDelete,
    Field(discriminator="kind"),
]


class EditRequest(EditModel):
    expected_revision: int = Field(ge=0)
    operation: EditOperation


class RevisionRequest(EditModel):
    expected_revision: int = Field(ge=0)
