"""YOLO pose 的元数据与关键点筛选；不推断骨架、类别点集合或左右对应关系。"""

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, cast

import yaml

from .metadata import PoseMetadata
from .records import Keypoint, KeypointRejectionReason, KeypointShape, RawKeypoint
from .util import as_int, as_list, as_mapping, as_str


def read_keypoint_shape(value: object) -> KeypointShape:
    """读取模型或 YAML 的 [点数, 维度]；只接受正点数及二维/三维坐标。"""
    values = list(cast(tuple[object, ...], value)) if isinstance(value, tuple) else as_list(value)
    if len(values) != 2:
        raise ValueError("kpt_shape must contain [count, dimensions]")
    count, dimensions = as_int(values[0]), as_int(values[1])
    if count <= 0 or dimensions not in (2, 3):
        raise ValueError("kpt_shape requires a positive count and dimensions 2 or 3")
    return count, cast(Literal[2, 3], dimensions)


def load_pose_metadata(path: str | Path, class_count: int) -> PoseMetadata:
    """读取显式定义的点数、点名和翻转映射；未定义的可选信息保持缺省。"""
    return parse_pose_metadata(yaml.safe_load(Path(path).read_text()), class_count)


def parse_pose_metadata(value: object, class_count: int) -> PoseMetadata:
    """收窄 YAML/JSON 元数据；运行记录用 source_kpt_shape 区分模型维度与标签维度。"""
    data = as_mapping(value)
    shape = read_keypoint_shape(data.get("source_kpt_shape", data.get("kpt_shape")))
    result: PoseMetadata = {"source_kpt_shape": shape}
    if "kpt_names" in data:
        names = data["kpt_names"]
        if isinstance(names, dict) and all(isinstance(v, str) for v in names.values()):
            indexed: dict[int, str] = {}
            for key, label in cast(dict[object, object], names).items():
                if isinstance(key, str) and key.isdecimal():
                    key = int(key)
                indexed[as_int(key)] = as_str(label)
            if sorted(indexed) != list(range(shape[0])):
                raise ValueError("Keypoint name indices must match kpt_shape")
            names = [indexed[i] for i in range(shape[0])]
        if isinstance(names, list):
            names = dict.fromkeys(range(class_count), names)
        if not isinstance(names, dict):
            raise ValueError("kpt_names must be a list or a class-to-names mapping")
        parsed: dict[int, list[str]] = {}
        for key, labels in cast(dict[object, object], names).items():
            if isinstance(key, str) and key.isdecimal():
                key = int(key)
            index = as_int(key)
            points = [as_str(p) for p in as_list(labels)]
            if not 0 <= index < class_count or len(points) != shape[0]:
                raise ValueError("kpt_names disagrees with classes or kpt_shape")
            parsed[index] = points
        if len(parsed) != class_count:
            raise ValueError("kpt_names must define every class")
        result["kpt_names"] = parsed
    if "flip_idx" in data:
        flip = [as_int(i) for i in as_list(data["flip_idx"])]
        if sorted(flip) != list(range(shape[0])) or any(flip[flip[i]] != i for i in flip):
            raise ValueError("flip_idx must be an involutive permutation of keypoint indices")
        result["flip_idx"] = flip
    return result


def filter_keypoints(
    points: Sequence[RawKeypoint], width: int, height: int, confidence: float
) -> tuple[Keypoint, ...]:
    """逐点保留原始预测，合法且达到阈值的点归一化，其余写为 (0, 0, 0)。

    不裁剪越界点以免制造边缘伪标签；无单点分数的二维模型仅检查坐标。
    visibility=2 仅表示接受的伪标签，不声称已确认物体可见或被遮挡。
    """
    result = []
    for point in points:
        x, y = point.xy
        score = point.confidence
        reason: KeypointRejectionReason | None = None
        if not math.isfinite(x) or not math.isfinite(y):
            reason = "nonfinite_coordinates"
        elif score is not None and (not math.isfinite(score) or not 0 <= score <= 1):
            reason = "invalid_confidence"
        elif score is not None and score < confidence:
            reason = "below_keypoint_confidence"
        elif not (0 <= x <= width and 0 <= y <= height):
            reason = "outside_image"
        result.append(
            Keypoint(point, (0.0, 0.0), 0, reason)
            if reason
            else Keypoint(point, (x / width, y / height), 2)
        )
    return tuple(result)


def check_keypoint_shape(points: Sequence[RawKeypoint] | None, shape: KeypointShape | None) -> None:
    """检查任务、点数和分数维度，结构错误立即报错，不默默丢弃整个目标。"""
    if shape is None:
        if points is not None:
            raise ValueError("Detection model returned unexpected keypoints")
        return
    if points is None or len(points) != shape[0]:
        raise ValueError("Keypoint count differs from kpt_shape")
    if any((p.confidence is None) != (shape[1] == 2) for p in points):
        raise ValueError("Keypoint confidence dimension differs from kpt_shape")
