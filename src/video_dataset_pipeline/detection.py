import math
from collections.abc import Sequence
from typing import Literal

from .metadata import SplitInfo
from .pose import check_keypoint_shape, filter_keypoints
from .records import (
    Detection,
    FrameRecord,
    KeypointShape,
    RawDetection,
    RejectedDetection,
    RejectionReason,
    Split,
)


def filter_detections(
    detections: Sequence[RawDetection],
    width: int,
    height: int,
    class_count: int,
    confidence: float,
    keypoint_shape: KeypointShape | None = None,
    keypoint_confidence: float = 0.5,
) -> tuple[list[Detection], list[RejectedDetection]]:
    """筛选原图像素坐标中的检测框，返回有效检测与带原因的拒绝检测。

    width/height 为原图尺寸；置信度达到 confidence 才保留。有效框先裁剪到
    图像边界，再计算归一化中心坐标与宽高，同时保留裁剪前的 source_xyxy。
    空框、越界类别和非法数值会被拒绝，输入检测对象保持原样。pose 点数/维度
    错误直接报错；单点筛选使用独立阈值，同帧是否有有效点由 candidate_status 判定。
    """
    accepted: list[Detection] = []
    rejected: list[RejectedDetection] = []
    for detection in detections:
        check_keypoint_shape(detection.keypoints, keypoint_shape)
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
            rejected.append(RejectedDetection(class_id, score, box, reason, detection.keypoints))
        else:
            accepted.append(
                Detection(
                    class_id=class_id,
                    confidence=score,
                    xyxy=(x1, y1, x2, y2),
                    source_xyxy=box,
                    keypoints=filter_keypoints(
                        detection.keypoints, width, height, keypoint_confidence
                    )
                    if detection.keypoints is not None
                    else None,
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
    """按有效帧数量划分训练/验证集，再排除时间分界附近的帧。

    records 须按时间递增排列，仅可入选的帧参与比例计算，pose 至少有一个有效点。
    分界取两组相邻帧
    时间的中点，排除区间为 [分界 - gap, 分界 + gap)，gap 的单位为秒。
    返回帧号到划分的映射及统计：排除帧映射为 None，无检测帧不进入映射。
    样本不足时可能有空划分，由后续数据集校验拒绝。
    """
    selected = [record for record in records if candidate_status(record.detections) == "candidate"]
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


def candidate_status(
    detections: Sequence[Detection],
) -> Literal["candidate", "no_detection", "no_keypoints"]:
    """判定帧是否可入选；pose 至少需要一个有效点，同帧其余合格框仍保留。"""
    if not detections:
        return "no_detection"
    if all(d.keypoints is not None for d in detections) and not any(
        p.visibility == 2 for d in detections for p in (d.keypoints or ())
    ):
        return "no_keypoints"
    return "candidate"


def label_text(detections: Sequence[Detection]) -> str:
    """编码 YOLO 标签；检测五列，pose 追加 K 组 x/y/v，均不写入连续置信度。"""
    lines = []
    for detection in detections:
        line = f"{detection.class_id} " + " ".join(f"{x:.9f}" for x in detection.xywhn)
        if detection.keypoints is not None:
            line += "".join(
                f" {p.xyn[0]:.9f} {p.xyn[1]:.9f} {p.visibility}" for p in detection.keypoints
            )
        lines.append(line + "\n")
    return "".join(lines)
