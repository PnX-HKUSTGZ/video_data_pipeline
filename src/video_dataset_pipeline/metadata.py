"""Typed dictionaries for existing metadata and report files."""

from typing import Literal, NotRequired, TypedDict

from .config import ConfigValues
from .records import Split


class VideoInfo(TypedDict):
    width: int
    height: int
    fps: float
    duration: float
    start_time: float
    codec: str


class DecodeStats(TypedDict):
    decoded_frames: int
    last_timestamp: float | None
    decode_complete: bool


class SplitInfo(TypedDict):
    basis: Literal["confidence_qualified_frames_in_time_order"]
    qualified_frames: int
    nominal_train_frames: int
    nominal_val_frames: int
    boundary_timestamp: float | None
    gap_seconds_each_side: float


class ConfidenceStats(TypedDict):
    minimum: float
    median: float
    maximum: float
    below_0_6: int


class DatasetSummary(TypedDict):
    candidate_frames: int
    confidence_qualified_frames: int
    included_frames: int
    no_detection_frames: int
    split_gap_frames: int
    no_detection_including_gap_frames: int
    splits: dict[Split, int]
    class_instances: dict[str, int]
    class_instances_by_split: dict[Split, dict[str, int]]
    preview_frames: int
    confidence: ConfidenceStats | None


class LoaderCount(TypedDict):
    images: int
    instances: int


class ValidationResult(DatasetSummary):
    valid: bool
    ultralytics_loader: dict[Split, LoaderCount]


class RunReport(DatasetSummary):
    split: SplitInfo
    pseudo_labels: bool
    evaluation_warning: str
    run_id: str
    decode: DecodeStats
    elapsed_seconds_this_attempt: float


class RunMetadata(TypedDict):
    schema_version: int
    run_id: str
    config: ConfigValues
    video: VideoInfo
    end: float
    names: dict[int, str]
    model: str
    model_yaml: str
    pseudo_labels: bool
    status: Literal["inference", "exporting", "validating", "complete", "smoke_complete", "failed"]
    decode: NotRequired[DecodeStats]
    candidate_frames: NotRequired[int]
    split: NotRequired[SplitInfo]
    elapsed_seconds_this_attempt: NotRequired[float]
    error: NotRequired[str]
