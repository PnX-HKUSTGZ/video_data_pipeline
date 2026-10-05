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
    class_id: int
    confidence: float
    xyxy: Box


@dataclass(frozen=True)
class Detection:
    class_id: int
    confidence: float
    xyxy: Box
    source_xyxy: Box
    xywhn: Box


@dataclass(frozen=True)
class RejectedDetection:
    class_id: int
    confidence: float
    xyxy: Box
    reason: RejectionReason


@dataclass
class FrameRecord:
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
    values = as_list(value)
    if len(values) != 4:
        raise ValueError("Expected four box coordinates")
    return as_float(values[0]), as_float(values[1]), as_float(values[2]), as_float(values[3])
