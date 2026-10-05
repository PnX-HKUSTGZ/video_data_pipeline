"""Parse standard YOLO text labels without training libraries or sidecar metadata."""

import math
from pathlib import Path

from .models import (
    Annotation,
    Box,
    Detection,
    Issue,
    Keypoint,
    Labels,
    OrientedBox,
    Point,
    Pose,
    Segment,
    Task,
)


def _unit(value: float) -> float:
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("坐标必须是 [0, 1] 范围内的有限数值")
    return value


def _box(values: list[float]) -> Box:
    cx, cy, width, height = (_unit(v) for v in values)
    if width <= 0 or height <= 0:
        raise ValueError("框的宽高必须大于 0")
    # Allow rounding at image boundaries, but never silently clip source labels.
    if (
        min(cx - width / 2, cy - height / 2) < -1e-6
        or max(cx + width / 2, cy + height / 2) > 1 + 1e-6
    ):
        raise ValueError("框边界超出图片")
    return Box(cx=cx, cy=cy, width=width, height=height)


def _annotation(
    raw: str, line: int, task: Task, names: dict[int, str], kpt_shape: tuple[int, int] | None
) -> Annotation:
    fields = raw.split()
    try:
        class_id = int(fields[0])
    except ValueError as error:
        raise ValueError("类别编号必须为非负整数") from error
    if class_id < 0 or (names and class_id not in names):
        raise ValueError(f"未知或非法类别编号 {class_id}")
    values = [float(v) for v in fields[1:]]
    if not all(math.isfinite(v) for v in values):
        raise ValueError("标注包含非有限数值")
    if task == "detect":
        if len(values) != 4:
            raise ValueError("检测标注需要 5 列：class cx cy w h")
        return Detection(class_id=class_id, line=line, raw=raw, box=_box(values))
    if task in ("segment", "obb"):
        if task == "obb" and len(values) != 8:
            raise ValueError("OBB 标注需要 9 列：class 和四组坐标")
        if len(values) < 6 or len(values) % 2:
            raise ValueError("分割标注需要至少三组完整的点坐标")
        points = [Point(x=_unit(x), y=_unit(y)) for x, y in zip(values[::2], values[1::2])]
        if task == "obb":
            return OrientedBox(class_id=class_id, line=line, raw=raw, points=points)
        return Segment(class_id=class_id, line=line, raw=raw, points=points)
    if kpt_shape is None:
        raise ValueError("姿态任务需要 kpt_shape")
    count, dims = kpt_shape
    if len(values) != 4 + count * dims:
        raise ValueError(f"姿态标注需要 {5 + count * dims} 列，kpt_shape={kpt_shape}")
    keypoints: list[Keypoint] = []
    for offset in range(4, len(values), dims):
        x, y = values[offset : offset + 2]
        visibility = values[offset + 2] if dims == 3 else 2
        if visibility not in (0, 1, 2):
            raise ValueError("关键点可见性必须为 0、1 或 2")
        # Negative coordinates are the conventional missing-point sentinel for 2D labels.
        if dims == 2 and (x < 0 or y < 0):
            keypoints.append(Keypoint(x=x, y=y, visibility=0))
        else:
            keypoints.append(
                Keypoint(
                    x=_unit(x),
                    y=_unit(y),
                    visibility=0 if visibility == 0 else 1 if visibility == 1 else 2,
                )
            )
    return Pose(class_id=class_id, line=line, raw=raw, box=_box(values[:4]), keypoints=keypoints)


def read_labels(
    path: Path, task: Task, names: dict[int, str], kpt_shape: tuple[int, int] | None
) -> Labels:
    """Keep valid objects when another row is malformed; missing labels are legitimate."""
    result = Labels(state="valid")
    try:
        content = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return Labels(state="missing")
    except (OSError, UnicodeError) as error:
        return Labels(state="invalid", issues=[Issue(file=str(path), message=str(error))])
    for line, raw in enumerate(content.splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            result.annotations.append(_annotation(raw, line, task, names, kpt_shape))
        except (ValueError, OverflowError) as error:
            result.issues.append(Issue(file=str(path), line=line, message=str(error)))
    result.state = "invalid" if result.issues else "valid" if result.annotations else "empty"
    return result
