import math
from collections.abc import Sequence

from .metadata import SplitInfo
from .records import Detection, FrameRecord, RawDetection, RejectedDetection, RejectionReason, Split


def filter_detections(
    detections: Sequence[RawDetection],
    width: int,
    height: int,
    class_count: int,
    confidence: float,
) -> tuple[list[Detection], list[RejectedDetection]]:
    accepted: list[Detection] = []
    rejected: list[RejectedDetection] = []
    for detection in detections:
        reason: RejectionReason | None = None
        class_id = detection.class_id
        score = detection.confidence
        box = detection.xyxy
        if not isinstance(class_id, int) or not 0 <= class_id < class_count:
            reason = "invalid_class"
        elif len(box) != 4 or not all(math.isfinite(v) for v in [*box, score]):
            reason = "nonfinite_coordinates_or_confidence"
        elif not 0 <= score <= 1:
            reason = "invalid_confidence"
        elif score < confidence:
            reason = "below_confidence"
        if reason is None:
            x1, y1, x2, y2 = box
            x1, x2 = max(0, min(width, x1)), max(0, min(width, x2))
            y1, y2 = max(0, min(height, y1)), max(0, min(height, y2))
            if x2 <= x1 or y2 <= y1:
                reason = "empty_box"
        if reason:
            rejected.append(RejectedDetection(class_id, score, box, reason))
        else:
            accepted.append(
                Detection(
                    class_id=class_id,
                    confidence=score,
                    xyxy=(x1, y1, x2, y2),
                    source_xyxy=box,
                    xywhn=(
                        (x1 + x2) / (2 * width),
                        (y1 + y2) / (2 * height),
                        (x2 - x1) / width,
                        (y2 - y1) / height,
                    ),
                )
            )
    return accepted, rejected


def split_selected_frames(
    records: Sequence[FrameRecord], train_fraction: float, gap: float
) -> tuple[dict[int, Split | None], SplitInfo]:
    """Partition confidence-qualified frames in temporal order, then embargo the boundary."""
    selected = [record for record in records if record.detections]
    split_index = min(max(1, math.floor(len(selected) * train_fraction)), len(selected))
    boundary = (
        (selected[split_index - 1].timestamp + selected[split_index].timestamp) / 2
        if 0 < split_index < len(selected)
        else None
    )
    assignments: dict[int, Split | None] = {}
    for index, record in enumerate(selected):
        split: Split | None = "train" if index < split_index else "val"
        if boundary is not None and boundary - gap <= record.timestamp < boundary + gap:
            split = None
        assignments[record.frame_index] = split
    return assignments, {
        "basis": "confidence_qualified_frames_in_time_order",
        "qualified_frames": len(selected),
        "nominal_train_frames": split_index,
        "nominal_val_frames": len(selected) - split_index,
        "boundary_timestamp": boundary,
        "gap_seconds_each_side": gap,
    }


def label_text(detections: Sequence[Detection]) -> str:
    return "".join(
        f"{d.class_id} " + " ".join(f"{x:.9f}" for x in d.xywhn) + "\n" for d in detections
    )
