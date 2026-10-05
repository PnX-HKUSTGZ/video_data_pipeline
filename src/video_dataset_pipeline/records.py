"""Records shared by inference, filtering and export; JSON conversion stays at the edges."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .util import as_float, as_int, as_list, as_mapping, as_str

type BGRImage = NDArray[np.uint8]  # H x W x 3, BGR channel order.
type Box = tuple[float, float, float, float]
type Split = Literal["train", "val"]
type FrameStatus = Literal["candidate", "no_detection", "split_gap", "included"]
type RejectionReason = Literal[
    "invalid_class",
    "nonfinite_coordinates_or_confidence",
    "invalid_confidence",
    "below_confidence",
    "empty_box",
]


@dataclass(frozen=True)
class RawDetection:
    """模型输出的检测结果；xyxy 是原图像素坐标中的 (左, 上, 右, 下)。"""

    class_id: int
    confidence: float
    xyxy: Box


@dataclass(frozen=True)
class Detection:
    """有效检测：xyxy 为裁剪框，source_xyxy 为原始框，xywhn 为归一化中心和宽高。"""

    class_id: int
    confidence: float
    xyxy: Box
    source_xyxy: Box
    xywhn: Box


@dataclass(frozen=True)
class RejectedDetection:
    """未通过筛选的检测，保留原始像素框、置信度及拒绝原因供诊断。"""

    class_id: int
    confidence: float
    xyxy: Box
    reason: RejectionReason


@dataclass
class FrameRecord:
    """单个采样帧的记录；timestamp 为相对秒数，产物路径相对于数据集根目录。"""

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

    def to_dict(self) -> dict[str, object]:
        """转换为 schema 4 字段，框写为数组，省略未设置的产物路径。

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
                    "confidence": d.confidence,
                    "xyxy": list(d.xyxy),
                    "source_xyxy": list(d.source_xyxy),
                    "xywhn": list(d.xywhn),
                }
                for d in self.detections
            ],
            "rejected_detections": [
                {
                    "class_id": d.class_id,
                    "confidence": d.confidence,
                    "xyxy": list(d.xyxy),
                    "reason": d.reason,
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
        return record

    @classmethod
    def from_dict(cls, value: object) -> FrameRecord:
        """从 schema 4 对象恢复记录，并检查必要字段、值类型及框的长度。

        这里只校验记录结构；图像配对和划分等规则由数据集校验负责。
        非有限数值仍可读取，以保留被拒绝检测的诊断信息；文件行号由调用方补充。
        """
        data = as_mapping(value)
        split = data["split"]
        if split not in (None, "train", "val"):
            raise ValueError("Invalid split")
        status = data["status"]
        if status not in ("candidate", "no_detection", "split_gap", "included"):
            raise ValueError("Invalid frame status")
        detections = []
        for item in as_list(data["detections"]):
            detection = as_mapping(item)
            detections.append(
                Detection(
                    class_id=as_int(detection["class_id"]),
                    confidence=as_float(detection["confidence"]),
                    xyxy=_read_box(detection["xyxy"]),
                    source_xyxy=_read_box(detection["source_xyxy"]),
                    xywhn=_read_box(detection["xywhn"]),
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
            ):
                raise ValueError("Invalid rejection reason")
            rejected.append(
                RejectedDetection(
                    class_id=as_int(detection["class_id"]),
                    confidence=as_float(detection["confidence"]),
                    xyxy=_read_box(detection["xyxy"]),
                    reason=reason,
                )
            )
        return cls(
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
