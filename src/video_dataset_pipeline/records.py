"""Records shared by inference, filtering and export; JSON conversion stays at the edges."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

from .util import as_float, as_int, as_list, as_mapping, as_str

type BGRImage = NDArray[np.uint8]  # H x W x 3, BGR channel order.
type Box = tuple[float, float, float, float]
type Point = tuple[float, float]
type Task = Literal["detect", "pose"]
type KeypointShape = tuple[int, Literal[2, 3]]
type Split = Literal["train", "val"]
type FrameStatus = Literal["candidate", "no_detection", "no_keypoints", "split_gap", "included"]
type RejectionReason = Literal[
    "invalid_class",
    "nonfinite_coordinates_or_confidence",
    "invalid_confidence",
    "below_confidence",
    "empty_box",
    "unmapped_class",
]


type KeypointRejectionReason = Literal[
    "nonfinite_coordinates",
    "invalid_confidence",
    "below_keypoint_confidence",
    "outside_image",
    "not_applicable",
    "below_in_frame_confidence",
]


@dataclass(frozen=True)
class RawKeypoint:
    """模型返回的原图像素点；二维模型没有单点置信度，记为 None。"""

    xy: Point
    confidence: float | None

    def to_dict(self) -> dict[str, object]:
        """保存坐标及原始分数，不将分数解释为人工可见性。"""
        return {"xy": list(self.xy), "confidence": self.confidence}

    @classmethod
    def from_dict(cls, value: object) -> RawKeypoint:
        """读取原始点，保留非法数值以便诊断被拒绝的预测。"""
        data = as_mapping(value)
        return cls(
            _read_point(data["xy"]),
            None if data["confidence"] is None else as_float(data["confidence"]),
        )


@dataclass(frozen=True)
class Keypoint:
    """筛选后的点：xyn 为归一化坐标，visibility=2 表示接受的伪标签，0 表示缺失。"""

    source: RawKeypoint
    xyn: Point
    visibility: Literal[0, 2]
    reason: KeypointRejectionReason | None = None

    def to_dict(self) -> dict[str, object]:
        """显式保存原始预测、训练坐标及点筛选原因。"""
        return {
            "source": self.source.to_dict(),
            "xyn": list(self.xyn),
            "visibility": self.visibility,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, value: object) -> Keypoint:
        """解析点记录；记录行号由 JSONL 读取边界补充。"""
        data = as_mapping(value)
        visibility = as_int(data["visibility"])
        if visibility not in (0, 2):
            raise ValueError("Keypoint visibility must be 0 or 2")
        reason = data["reason"]
        if reason not in (
            None,
            "nonfinite_coordinates",
            "invalid_confidence",
            "below_keypoint_confidence",
            "outside_image",
            "not_applicable",
            "below_in_frame_confidence",
        ):
            raise ValueError("Invalid keypoint rejection reason")
        return cls(
            RawKeypoint.from_dict(data["source"]),
            _read_point(data["xyn"]),
            cast(Literal[0, 2], visibility),
            reason,
        )


@dataclass(frozen=True)
class NativeKeypoint:
    """LiteHRNet 原生点序：原图坐标、相对 ROI 坐标、热图分数及可选在框分数。"""

    name: str
    xy: Point
    roi_xy: Point
    score: float
    in_frame_score: float | None

    def to_dict(self) -> dict[str, object]:
        """分数保持原值；热图分数可以超过 1。"""
        return {
            "name": self.name,
            "xy": list(self.xy),
            "roi_xy": list(self.roi_xy),
            "score": self.score,
            "in_frame_score": self.in_frame_score,
        }

    @classmethod
    def from_dict(cls, value: object) -> NativeKeypoint:
        """显式解析 JSON 边界，保留非法预测供筛选诊断。"""
        data = as_mapping(value)
        return cls(
            as_str(data["name"]),
            _read_point(data["xy"]),
            _read_point(data["roi_xy"]),
            as_float(data["score"]),
            None if data["in_frame_score"] is None else as_float(data["in_frame_score"]),
        )


@dataclass(frozen=True)
class ROIResult:
    """两阶段的实例对应关系；xyxy 是传给原生仿射变换的扩展、裁剪后的 ROI。"""

    detection_index: int
    xyxy: Box
    keypoints: tuple[NativeKeypoint, ...]

    def to_dict(self) -> dict[str, object]:
        """只保存数值，不重复保存 ROI 图像。"""
        return {
            "detection_index": self.detection_index,
            "xyxy": list(self.xyxy),
            "keypoints": [p.to_dict() for p in self.keypoints],
        }

    @classmethod
    def from_dict(cls, value: object) -> ROIResult:
        """恢复 schema 7 的原生推理信息。"""
        data = as_mapping(value)
        index = as_int(data["detection_index"])
        if index < 0:
            raise ValueError("ROI detection_index must be nonnegative")
        return cls(
            index,
            _read_box(data["xyxy"]),
            tuple(NativeKeypoint.from_dict(p) for p in as_list(data["keypoints"])),
        )


@dataclass(frozen=True)
class RawDetection:
    """模型输出的检测结果；xyxy 是原图像素坐标中的 (左, 上, 右, 下)。"""

    class_id: int
    confidence: float
    xyxy: Box
    keypoints: tuple[RawKeypoint, ...] | None = None
    roi: ROIResult | None = None


@dataclass(frozen=True)
class Detection:
    """有效检测：xyxy 为裁剪框，source_xyxy 为原始框，xywhn 为归一化中心和宽高。"""

    class_id: int
    confidence: float
    xyxy: Box
    source_xyxy: Box
    xywhn: Box
    keypoints: tuple[Keypoint, ...] | None = None
    roi: ROIResult | None = None


@dataclass(frozen=True)
class RejectedDetection:
    """未通过筛选的检测，保留原始像素框、置信度及拒绝原因供诊断。"""

    class_id: int
    confidence: float
    xyxy: Box
    reason: RejectionReason
    keypoints: tuple[RawKeypoint, ...] | None = None
    roi: ROIResult | None = None


@dataclass
class FrameRecord:
    """单个处理帧的记录；timestamp 为相对秒数，产物路径相对于数据集根目录。"""

    frame_index: int
    timestamp: float
    width: int
    height: int
    split: Split | None
    status: FrameStatus
    detections: list[Detection] = field(default_factory=list)
    rejected_detections: list[RejectedDetection] = field(default_factory=list)
    candidate_image: str | None = None
    image: str | None = None
    label: str | None = None
    preview: str | None = None
    preview_reasons: list[str] = field(default_factory=list)
    source_id: str | None = None

    def to_dict(self) -> dict[str, object]:
        """转换为兼容 schema 4/5/6/7 的字段，框写为数组，省略未设置的产物路径。

        split 即使为 None 也保留；preview_reasons 仅在存在预览时写入，
        使 JSON/JSONL 结构与已有数据集兼容。
        """
        record: dict[str, object] = {
            "frame_index": self.frame_index,
            "timestamp": self.timestamp,
            "width": self.width,
            "height": self.height,
            "split": self.split,
            "status": self.status,
            "detections": [
                {
                    "class_id": d.class_id,
                    **({"roi": d.roi.to_dict()} if d.roi is not None else {}),
                    "confidence": d.confidence,
                    "xyxy": list(d.xyxy),
                    "source_xyxy": list(d.source_xyxy),
                    "xywhn": list(d.xywhn),
                    **(
                        {"keypoints": [p.to_dict() for p in d.keypoints]}
                        if d.keypoints is not None
                        else {}
                    ),
                }
                for d in self.detections
            ],
            "rejected_detections": [
                {
                    "class_id": d.class_id,
                    **({"roi": d.roi.to_dict()} if d.roi is not None else {}),
                    "confidence": d.confidence,
                    "xyxy": list(d.xyxy),
                    "reason": d.reason,
                    **(
                        {"keypoints": [p.to_dict() for p in d.keypoints]}
                        if d.keypoints is not None
                        else {}
                    ),
                }
                for d in self.rejected_detections
            ],
        }
        for key, path in (
            ("candidate_image", self.candidate_image),
            ("image", self.image),
            ("label", self.label),
            ("preview", self.preview),
        ):
            if path is not None:
                record[key] = path
        if self.preview is not None:
            record["preview_reasons"] = self.preview_reasons.copy()
        if self.source_id is not None:
            record["source_id"] = self.source_id
        return record

    @classmethod
    def from_dict(cls, value: object) -> FrameRecord:
        """从 schema 4/5/6/7 对象恢复记录，并检查必要字段、值类型及框的长度。

        这里只校验记录结构；图像配对和划分等规则由数据集校验负责。
        非有限数值仍可读取，以保留被拒绝检测的诊断信息；文件行号由调用方补充。
        """
        data = as_mapping(value)
        split = data["split"]
        if split not in (None, "train", "val"):
            raise ValueError("Invalid split")
        status = data["status"]
        if status not in ("candidate", "no_detection", "no_keypoints", "split_gap", "included"):
            raise ValueError("Invalid frame status")
        detections = []
        for item in as_list(data["detections"]):
            detection = as_mapping(item)
            detections.append(
                Detection(
                    class_id=as_int(detection["class_id"]),
                    roi=ROIResult.from_dict(detection["roi"]) if "roi" in detection else None,
                    confidence=as_float(detection["confidence"]),
                    xyxy=_read_box(detection["xyxy"]),
                    source_xyxy=_read_box(detection["source_xyxy"]),
                    xywhn=_read_box(detection["xywhn"]),
                    keypoints=tuple(Keypoint.from_dict(p) for p in as_list(detection["keypoints"]))
                    if "keypoints" in detection
                    else None,
                )
            )
        rejected = []
        for item in as_list(data["rejected_detections"]):
            detection = as_mapping(item)
            reason = detection["reason"]
            if reason not in (
                "invalid_class",
                "nonfinite_coordinates_or_confidence",
                "invalid_confidence",
                "below_confidence",
                "empty_box",
                "unmapped_class",
            ):
                raise ValueError("Invalid rejection reason")
            rejected.append(
                RejectedDetection(
                    class_id=as_int(detection["class_id"]),
                    roi=ROIResult.from_dict(detection["roi"]) if "roi" in detection else None,
                    confidence=as_float(detection["confidence"]),
                    xyxy=_read_box(detection["xyxy"]),
                    reason=reason,
                    keypoints=tuple(
                        RawKeypoint.from_dict(p) for p in as_list(detection["keypoints"])
                    )
                    if "keypoints" in detection
                    else None,
                )
            )
        source_id = as_str(data["source_id"]) if "source_id" in data else None
        if source_id == "":
            raise ValueError("source_id must not be empty")
        return cls(
            source_id=source_id,
            frame_index=as_int(data["frame_index"]),
            timestamp=as_float(data["timestamp"]),
            width=as_int(data["width"]),
            height=as_int(data["height"]),
            split=split,
            status=status,
            detections=detections,
            rejected_detections=rejected,
            candidate_image=as_str(data["candidate_image"]) if "candidate_image" in data else None,
            image=as_str(data["image"]) if "image" in data else None,
            label=as_str(data["label"]) if "label" in data else None,
            preview=as_str(data["preview"]) if "preview" in data else None,
            preview_reasons=[as_str(reason) for reason in as_list(data["preview_reasons"])]
            if "preview_reasons" in data
            else [],
        )


def _read_box(value: object) -> Box:
    """将四元素 JSON 数组转换为浮点元组，拒绝字符串或布尔值坐标。"""
    values = as_list(value)
    if len(values) != 4:
        raise ValueError("Expected four box coordinates")
    return as_float(values[0]), as_float(values[1]), as_float(values[2]), as_float(values[3])


def _read_point(value: object) -> Point:
    """将二元素 JSON 数组转换为坐标元组。"""
    values = as_list(value)
    if len(values) != 2:
        raise ValueError("Expected two point coordinates")
    return as_float(values[0]), as_float(values[1])
