"""两阶段共享的 ROI 几何、槽位映射及热图筛选，无模型库依赖。"""

import math
from collections.abc import Sequence

from .config import KeypointModelConfig
from .records import Box, Keypoint, KeypointRejectionReason, RawKeypoint, ROIResult


def expand_roi(box: Box, width: int, height: int, margin: float) -> Box:
    """按框宽高向每侧扩展 margin 比例并裁剪；原生 MMPose padding 另行保留。"""
    x1, y1, x2, y2 = box
    dx, dy = (x2 - x1) * margin, (y2 - y1) * margin
    return (
        max(0.0, x1 - dx),
        max(0.0, y1 - dy),
        min(float(width), x2 + dx),
        min(float(height), y2 + dy),
    )


def map_keypoints(roi: ROIResult, head: KeypointModelConfig, count: int) -> tuple[RawKeypoint, ...]:
    """将任意模型的原生点序映射到固定输出槽位，其余槽位显式补零。"""
    if len(roi.keypoints) != len(head.output_indices):
        raise ValueError("Native keypoint count differs from output_indices")
    points = [RawKeypoint((0.0, 0.0), 0.0) for _ in range(count)]
    for index, point in zip(head.output_indices, roi.keypoints, strict=True):
        points[index] = RawKeypoint(point.xy, point.score)
    return tuple(points)


def filter_roi_keypoints(
    points: Sequence[RawKeypoint],
    roi: ROIResult | None,
    head: KeypointModelConfig,
    width: int,
    height: int,
) -> tuple[Keypoint, ...]:
    """检查映射与 ROI 坐标并筛选原生热图；分数没有概率上界，不裁剪越界点。"""
    if roi is None or len(roi.keypoints) != len(head.output_indices):
        raise ValueError("Two-stage detection requires complete ROI predictions")
    native = dict(zip(head.output_indices, roi.keypoints, strict=True))
    result = []
    for index, point in enumerate(points):
        reason: KeypointRejectionReason | None = None
        x, y = point.xy
        score = point.confidence
        original = native.get(index)
        if original is None:
            if point != RawKeypoint((0.0, 0.0), 0.0):
                raise ValueError("Inactive output slot must be zero")
            reason = "not_applicable"
        else:
            # NaN 预测也需保存；比较对应字段时允许两边都是 NaN。
            pairs = [
                (x, original.xy[0]),
                (y, original.xy[1]),
                (score if score is not None else math.nan, original.score),
                (original.roi_xy[0], x - roi.xyxy[0]),
                (original.roi_xy[1], y - roi.xyxy[1]),
            ]
            if any(a != b and not (math.isnan(a) and math.isnan(b)) for a, b in pairs):
                raise ValueError("ROI coordinates or output mapping disagree")
            if not math.isfinite(x) or not math.isfinite(y):
                reason = "nonfinite_coordinates"
            elif score is None or not math.isfinite(score):
                reason = "invalid_confidence"
            elif score < head.confidence:
                reason = "below_keypoint_confidence"
            elif head.in_frame_confidence is not None:
                visible = original.in_frame_score
                if visible is None:
                    raise ValueError("in_frame_confidence requires model in-frame scores")
                if not math.isfinite(visible) or not 0 <= visible <= 1:
                    reason = "invalid_confidence"
                elif visible < head.in_frame_confidence:
                    reason = "below_in_frame_confidence"
            if reason is None and not (0 <= x <= width and 0 <= y <= height):
                reason = "outside_image"
        result.append(
            Keypoint(point, (0.0, 0.0), 0, reason)
            if reason
            else Keypoint(point, (x / width, y / height), 2)
        )
    return tuple(result)
