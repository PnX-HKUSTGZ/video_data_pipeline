"""Typed dictionaries for existing metadata and report files."""

from typing import Literal, NotRequired, TypedDict

from .config import ConfigValues
from .records import KeypointShape, Split, Task


class PoseMetadata(TypedDict):
    source_kpt_shape: KeypointShape
    kpt_names: NotRequired[dict[int, list[str]]]
    flip_idx: NotRequired[list[int]]


class KeypointStats(TypedDict):
    valid_by_index: list[int]
    missing_by_reason: dict[str, int]
    valid_points: int
    missing_points: int


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
    no_keypoints_frames: NotRequired[int]
    keypoints: NotRequired[KeypointStats]


class LoaderCount(TypedDict):
    images: int
    instances: int
    valid_keypoints: NotRequired[int]


class ValidationResult(DatasetSummary):
    valid: bool
    ultralytics_loader: dict[Split, LoaderCount]


SourceStatus = Literal["pending", "inference", "exporting", "complete", "failed"]


class SourceMetadata(TypedDict):
    """固定清单中的视频及其独立解码、划分和保存状态。"""

    source_id: str
    input: str
    video: VideoInfo
    start: float
    end: float
    decode: DecodeStats
    status: SourceStatus
    candidate_frames: int
    split: NotRequired[SplitInfo]
    inference_video: NotRequired[str]
    error: NotRequired[str]


class SourceReport(DatasetSummary):
    """单个视频的统计，零有效检测的视频也保留。"""

    source_id: str
    input: str
    decode: DecodeStats
    split: SplitInfo | None
    status: SourceStatus


class RunReport(DatasetSummary):
    split: dict[str, SplitInfo]
    videos: list[SourceReport]
    pseudo_labels: bool
    evaluation_warning: str
    run_id: str
    decode: DecodeStats
    elapsed_seconds_this_attempt: float


class RunMetadata(TypedDict):
    schema_version: int
    task: NotRequired[Task]
    pose: NotRequired[PoseMetadata]
    run_id: str
    config: ConfigValues
    videos: list[SourceMetadata]
    names: dict[int, str]
    model: str
    model_yaml: str
    pseudo_labels: bool
    status: Literal["inference", "exporting", "validating", "complete", "smoke_complete", "failed"]
    decode: NotRequired[DecodeStats]
    candidate_frames: NotRequired[int]
    split: NotRequired[dict[str, SplitInfo]]
    elapsed_seconds_this_attempt: NotRequired[float]
    error: NotRequired[str]
