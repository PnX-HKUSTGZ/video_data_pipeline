import html
import json
import math
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import TextIO

import cv2
import numpy as np
import yaml

from .detection import candidate_status, label_text, split_selected_frames
from .inference_video import validate_inference_video
from .metadata import (
    DatasetSummary,
    LoaderCount,
    PoseMetadata,
    RunReport,
    SplitInfo,
    ValidationResult,
)
from .pose import check_keypoint_shape, filter_keypoints, parse_pose_metadata
from .records import BGRImage, FrameRecord, Split
from .util import as_float, as_int, as_list, as_mapping, as_str, atomic_bytes, read_json, write_json

SPLITS: tuple[Split, Split] = ("train", "val")


def read_records(path: str | Path) -> list[FrameRecord]:
    """读取 JSONL 帧清单，要求同一来源内帧号唯一且递增，每行具有完整记录和末尾换行。

    文件不存在时返回空列表。解析或字段类型错误会报告从 1 开始的行号，
    不修改源文件，也不截断或跳过坏记录。
    """
    path = Path(path)
    if not path.exists():
        return []
    data = path.read_bytes()
    records = []
    lines = data.splitlines(keepends=True)
    for i, line in enumerate(lines):
        try:
            record = FrameRecord.from_dict(json.loads(line))
            if not line.endswith(b"\n"):
                raise ValueError("Uncommitted trailing record")
        except (ValueError, KeyError, TypeError) as error:
            raise ValueError(f"Invalid frame record {i + 1}: {error}") from error
        records.append(record)
    previous: dict[str | None, int] = {}
    for line_number, record in enumerate(records, 1):
        if record.frame_index <= previous.get(record.source_id, -1):
            raise ValueError(
                f"Invalid frame record {line_number}: frame IDs must be unique and ordered per source"
            )
        previous[record.source_id] = record.frame_index
    return records


def append_record(stream: TextIO, record: FrameRecord) -> None:
    """向已打开的文件追加一条完整 JSONL 记录，并 flush/fsync 使其落盘。"""
    stream.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def save_jpeg(path: str | Path, image: BGRImage, quality: int) -> None:
    """将 BGR 图像编码后原子写入 JPEG，支持中文路径；quality 取值为 1..100。"""
    success, data = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not success:
        raise ValueError(f"JPEG encoding failed: {path}")
    atomic_bytes(path, data.tobytes())


def export_records(
    root: Path,
    inference: Sequence[FrameRecord],
    train_fraction: float,
    gap: float,
    append: bool = False,
) -> tuple[list[FrameRecord], SplitInfo]:
    """从按时间排序的推理记录导出图片、标签和最终帧清单。

    返回记录副本及划分统计，不修改传入记录。入选图片通过硬链接复用暂存
    图像字节，要求目标 images/labels 子目录已创建且与暂存目录位于同一文件系统。
    最终清单保留无检测及分界排除记录，但移除所有 candidate_image 暂存路径。
    append=True 时追加到合并清单，调用方须确保输入只属于一个视频。
    """
    if len({r.source_id for r in inference}) > 1:
        raise ValueError("Export requires records from a single source")
    assignments, split_info = split_selected_frames(inference, train_fraction, gap)
    records = []
    for source in inference:
        record = replace(source, candidate_image=None)
        split = assignments.get(source.frame_index)
        record.split = split
        status = candidate_status(record.detections)
        record.status = (
            status if status != "candidate" else "split_gap" if split is None else "included"
        )
        if record.status == "included":
            if split is None or source.candidate_image is None:
                raise ValueError("Included frame requires a split and candidate image")
            name = Path(source.candidate_image).stem
            image_path = Path("images") / split / f"{name}.jpg"
            label_path = Path("labels") / split / f"{name}.txt"
            # Both directories share a filesystem, so image bytes need not be duplicated.
            os.link(root / source.candidate_image, root / image_path)
            atomic_bytes(root / label_path, label_text(record.detections).encode())
            record.image = str(image_path)
            record.label = str(label_path)
        records.append(record)
    payload = "".join(json.dumps(r.to_dict(), ensure_ascii=False) + "\n" for r in records).encode()
    if append:
        with (root / "frames.jsonl").open("ab") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    else:
        atomic_bytes(root / "frames.jsonl", payload)
    return records, split_info


def check_record_files(root: Path, record: FrameRecord) -> None:
    """检查记录声明的各产物路径在 root 下存在；不负责解码图片或验证标签内容。"""
    for relative in (record.image, record.label, record.preview, record.candidate_image):
        if relative is not None:
            path = root / relative
            if not path.is_file():
                raise ValueError(f"Missing artifact: {path}")


def annotate_frame(
    image: BGRImage,
    record: FrameRecord,
    names: Mapping[int, str],
    point_names: Mapping[int, Sequence[str]] | None = None,
) -> BGRImage:
    """在图像副本上绘制框、有效关键点、名称及源时间，供预览和视频共用。"""
    detections = record.detections
    canvas = image.copy()
    for d in detections:
        x1, y1, x2, y2 = map(round, d.xyxy)
        color = (40, 220, 40) if d.class_id % 2 == 0 else (0, 170, 255)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            canvas,
            f"{names[d.class_id]} {d.confidence:.3f}",
            (x1, max(22, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )
        for index, point in enumerate(d.keypoints or ()):
            if point.visibility == 0:
                continue
            px = min(record.width - 1, round(point.xyn[0] * record.width))
            py = min(record.height - 1, round(point.xyn[1] * record.height))
            name = point_names[d.class_id][index] if point_names is not None else str(index)
            cv2.circle(canvas, (px, py), 4, (255, 0, 255), -1)
            cv2.putText(
                canvas,
                name,
                (px + 5, max(12, py - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 0, 255),
                1,
            )
    cv2.putText(
        canvas,
        f"PSEUDO LABEL | t={record.timestamp:.3f}s | {record.status}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )
    return canvas


class Previews:
    """按时间段、类别和检测置信度抽检，预览选择独立于训练集收录规则。"""

    def __init__(
        self,
        root: Path,
        names: Mapping[int, str],
        start: float,
        end: float,
        smoke: bool = False,
        point_names: Mapping[int, Sequence[str]] | None = None,
    ) -> None:
        """初始化抽检计数；start/end 为相对秒数，smoke 模式保存每个采样帧。"""
        self.root, self.names, self.start, self.end = root, names, start, end
        self.smoke = smoke
        self.point_names = point_names
        self.counts: Counter[str] = Counter()

    def save(self, name: str, image: BGRImage, record: FrameRecord) -> None:
        """命中抽检条件时保存预览，并更新 record 的预览路径和入选原因。

        覆盖 20 个时间桶、各类别、低置信度和无检测样例，因此未收录的数据帧
        也可能有预览。绘制发生在图像副本上；未命中条件时不修改记录或保存图片。
        """
        reasons = []
        bucket = min(19, int(20 * (record.timestamp - self.start) / (self.end - self.start)))
        if self.counts[f"time_{bucket}"] == 0:
            reasons.append(f"time_{bucket}")
        if self.smoke:
            reasons.append("smoke")
        detections = record.detections
        for class_id in sorted({d.class_id for d in detections}):
            if self.counts[f"class_{class_id}"] < 2:
                reasons.append(f"class_{class_id}")
        if (
            detections
            and min(d.confidence for d in detections) < 0.6
            and self.counts["low_confidence"] < 10
        ):
            reasons.append("low_confidence")
        if not detections and self.counts["no_detection"] < 5:
            reasons.append("no_detection")
        if record.status == "no_keypoints" and self.counts["no_keypoints"] < 5:
            reasons.append("no_keypoints")
        if (
            any(p.visibility == 0 for d in detections for p in (d.keypoints or ()))
            and self.counts["missing_keypoints"] < 5
        ):
            reasons.append("missing_keypoints")
        if not reasons:
            return
        canvas = annotate_frame(image, record, self.names, self.point_names)
        path = Path("previews") / f"{name}.jpg"
        save_jpeg(self.root / path, canvas, 90)
        self.counts.update(reasons)
        record.preview = str(path)
        record.preview_reasons = reasons


def summarize(
    records: Sequence[FrameRecord],
    names: Mapping[int, str],
    pose: PoseMetadata | None = None,
) -> DatasetSummary:
    """汇总帧状态和划分数量，类别目标数与置信度仅统计 included 帧。

    confidence_qualified_frames 包含随后被分界间隔排除的有效帧；无入选目标时，
    confidence 返回 None，便于失败运行也能输出诊断统计。pose 额外统计无有效点
    的帧数，以及 included 帧中按索引排列的有效点数量和缺失原因。
    """
    counts = Counter(r.status for r in records)
    splits = Counter(r.split for r in records if r.status == "included")
    classes = Counter(d.class_id for r in records if r.status == "included" for d in r.detections)
    scores = [d.confidence for r in records if r.status == "included" for d in r.detections]
    result: DatasetSummary = {
        "candidate_frames": len(records),
        "confidence_qualified_frames": sum(
            candidate_status(r.detections) == "candidate" for r in records
        ),
        "included_frames": counts["included"],
        "no_detection_frames": counts["no_detection"],
        "split_gap_frames": counts["split_gap"],
        "no_detection_including_gap_frames": sum(not r.detections for r in records),
        "splits": {s: splits[s] for s in SPLITS},
        "class_instances": {names[k]: classes[k] for k in names},
        "class_instances_by_split": {
            split: {
                names[k]: sum(
                    d.class_id == k
                    for r in records
                    if r.status == "included" and r.split == split
                    for d in r.detections
                )
                for k in names
            }
            for split in SPLITS
        },
        "preview_frames": sum(r.preview is not None for r in records),
        "confidence": (
            {
                "minimum": min(scores),
                "median": float(np.median(scores)),
                "maximum": max(scores),
                "below_0_6": sum(s < 0.6 for s in scores),
            }
            if scores
            else None
        ),
    }

    if pose is not None:
        points = [
            d.keypoints or () for r in records if r.status == "included" for d in r.detections
        ]
        valid = [
            sum(ps[i].visibility == 2 for ps in points) for i in range(pose["source_kpt_shape"][0])
        ]
        missing = Counter(p.reason for ps in points for p in ps if p.reason is not None)
        result["no_keypoints_frames"] = counts["no_keypoints"]
        result["keypoints"] = {
            "valid_by_index": valid,
            "valid_points": sum(valid),
            "missing_points": sum(missing.values()),
            "missing_by_reason": {str(k): v for k, v in missing.items()},
        }
    return result


def _validate_sources(
    root: Path,
    run: Mapping[str, object],
    config: Mapping[str, object],
    records: Sequence[FrameRecord],
) -> dict[tuple[str | None, int], Split | None]:
    """校验来源、解码及逐视频划分；旧版元数据在读取边界映射为单个来源。

    不重新扫描输入目录，也不要求原视频仍然存在，校验以本次固定清单为准。
    各视频只在自身时间轴内检查顺序、处理范围及保存视频的时间戳。
    """
    modern = run["schema_version"] == 6
    sources = (
        [as_mapping(v) for v in as_list(run["videos"])]
        if modern
        else [
            {
                "input": config["input"],
                "video": run["video"],
                "end": run["end"],
                "start": config["start"],
                "decode": run["decode"],
                "split": run["split"],
                "candidate_frames": run["candidate_frames"],
                "inference_video": "inference.mp4",
            }
        ]
    )
    if not sources:
        raise ValueError("Run requires at least one source video")
    grouped: dict[str | None, list[FrameRecord]] = {}
    for record in records:
        grouped.setdefault(record.source_id, []).append(record)
    source_ids: list[str | None] = []
    paths: set[str] = set()
    splits: dict[str, SplitInfo] = {}
    assignments: dict[tuple[str | None, int], Split | None] = {}
    decoded_frames = 0
    for index, source in enumerate(sources, 1):
        source_id = as_str(source["source_id"]) if modern else None
        if modern and source_id != f"v{index:04d}":
            raise ValueError("Source IDs must match the ordered input list")
        path = as_str(source["input"])
        if path in paths:
            raise ValueError("Duplicate source video path")
        paths.add(path)
        source_ids.append(source_id)
        group = grouped.get(source_id, [])
        if len(group) != as_int(source["candidate_frames"]):
            raise ValueError(f"{source_id}: source frame count mismatch")
        if modern and source["status"] != "complete":
            raise ValueError(f"{source_id}: source processing did not complete")
        decode = as_mapping(source["decode"])
        if decode.get("decode_complete") is not True:
            raise ValueError(f"{source_id}: Video decode did not complete")
        decoded_frames += as_int(decode["decoded_frames"])
        info = as_mapping(source["video"])
        start, end = as_float(source["start"]), as_float(source["end"])
        configured_end = config.get("end")
        expected_end = min(
            as_float(configured_end) if configured_end is not None else as_float(info["duration"]),
            as_float(info["duration"]),
        )
        if start != as_float(config["start"]) or end != expected_end or start >= end:
            raise ValueError(f"{source_id}: processing interval disagrees with config")
        previous = -1.0
        for record in group:
            if not math.isfinite(record.timestamp) or record.timestamp <= previous:
                raise ValueError(f"{source_id}: Manifest timestamps must be increasing")
            if not start - 1e-7 <= record.timestamp < end:
                raise ValueError(f"{source_id}: frame timestamp outside processing interval")
            if (record.width, record.height) != (as_int(info["width"]), as_int(info["height"])):
                raise ValueError(f"{source_id}: frame dimensions differ from source video")
            previous = record.timestamp
        selected, split_info = split_selected_frames(
            group, as_float(config["train_fraction"]), as_float(config["split_gap_seconds"])
        )
        if source.get("split") != split_info:
            raise ValueError(
                f"{source_id}: Split metadata does not match confidence-qualified frame selection"
            )
        if source_id is not None:
            splits[source_id] = split_info
        assignments.update(((source_id, frame), split) for frame, split in selected.items())
        if config.get("save_video", False):
            expected_video = (
                "inference.mp4"
                if len(sources) == 1
                else f"inference_videos/{source_id}_{Path(path).stem}.mp4"
            )
            if source.get("inference_video") != expected_video:
                raise ValueError(f"{source_id}: inference video path mismatch")
            validate_inference_video(root, group, end, expected_video)
    if set(grouped) != set(source_ids):
        raise ValueError("Frame records contain missing or unknown sources")
    ordered = [(r.source_id, r.frame_index) for sid in source_ids for r in grouped[sid]]
    if ordered != [(r.source_id, r.frame_index) for r in records]:
        raise ValueError("Frame records must follow source order")
    if modern:
        if run.get("split") != splits:
            raise ValueError("Aggregate split metadata does not match source splits")
        aggregate = as_mapping(run["decode"])
        if as_int(aggregate["decoded_frames"]) != decoded_frames:
            raise ValueError("Aggregate decode count does not match source counts")
    return assignments


def validate_dataset(root: str | Path, ultralytics_check: bool = True) -> ValidationResult:
    """核对解码完成状态、逐帧划分、图片标签及文件集合，成功后写 validation.json。

    任一划分无有效样本或产物不一致都会报错。默认还用 Ultralytics 实际加载
    两个划分并读取样本张量，随后清理它生成的标签缓存；关闭该项仅跳过加载检查。
    这里验证的是数据格式和内部一致性，不能衡量伪标签的真实识别精度。
    """
    root = Path(root).resolve()
    run = as_mapping(read_json(root / "run.json"))
    config = as_mapping(run["config"])
    records = read_records(root / "frames.jsonl")
    if run.get("candidate_frames") != len(records):
        raise ValueError("Frame count differs from completed decode record")
    names = {int(k): as_str(v) for k, v in as_mapping(run["names"]).items()}
    data = as_mapping(yaml.safe_load((root / "data.yaml").read_text()))
    task = as_str(run.get("task", "detect"))
    if (task, run.get("schema_version")) not in (
        ("detect", 4),
        ("pose", 5),
        ("detect", 6),
        ("pose", 6),
    ):
        raise ValueError("Unsupported task/schema_version combination")
    pose = parse_pose_metadata(run["pose"], len(names)) if task == "pose" else None
    shape = pose["source_kpt_shape"] if pose is not None else None
    if pose is not None:
        if data.get("kpt_shape") != [pose["source_kpt_shape"][0], 3]:
            raise ValueError("Dataset kpt_shape disagrees with run metadata")
        # JSON 会将类别键转成字符串；复用解析函数比较点名和翻转映射。
        if (
            parse_pose_metadata({**data, "kpt_shape": list(pose["source_kpt_shape"])}, len(names))
            != pose
        ):
            raise ValueError("Dataset pose metadata disagrees with run metadata")
    elif "kpt_shape" in data or "pose" in run:
        raise ValueError("Detection dataset contains unexpected pose metadata")
    if data["names"] != names or data["train"] != "images/train" or data["val"] != "images/val":
        raise ValueError("data.yaml disagrees with run metadata")
    if Path(as_str(data["path"])).resolve() != root:
        raise ValueError("data.yaml path does not point to this dataset")
    if not as_mapping(run.get("decode", {})).get("decode_complete"):
        raise ValueError("Video decode did not complete")
    expected_files: set[str] = set()
    assignments = _validate_sources(root, run, config, records)
    for record in records:
        check_record_files(root, record)
        expected_split = assignments.get((record.source_id, record.frame_index))
        status = candidate_status(record.detections)
        expected_status = (
            status
            if status != "candidate"
            else "split_gap"
            if expected_split is None
            else "included"
        )
        for detection in record.detections:
            raw_points = (
                tuple(p.source for p in detection.keypoints)
                if detection.keypoints is not None
                else None
            )
            check_keypoint_shape(raw_points, shape)
            if raw_points is not None:
                expected_points = filter_keypoints(
                    raw_points, record.width, record.height, as_float(config["keypoint_confidence"])
                )
                for actual, expected in zip(
                    detection.keypoints or (), expected_points, strict=True
                ):
                    if (actual.xyn, actual.visibility, actual.reason) != (
                        expected.xyn,
                        expected.visibility,
                        expected.reason,
                    ):
                        raise ValueError(
                            f"Keypoint filtering mismatch at frame {record.frame_index}"
                        )
        for rejected in record.rejected_detections:
            check_keypoint_shape(rejected.keypoints, shape)
        if record.split != expected_split or record.status != expected_status:
            raise ValueError("Manifest split or inclusion rule mismatch")
        if record.status != "included":
            if record.image is not None or record.label is not None:
                raise ValueError("Excluded frame has training artifacts")
            continue
        if record.image is None or record.label is None:
            raise ValueError("Included frame is missing image/label paths")
        if record.image in expected_files or record.label in expected_files:
            raise ValueError("Different frame records reference the same artifact")
        expected_files.update((record.image, record.label))
        image = cv2.imdecode(np.fromfile(root / record.image, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None or list(image.shape[:2]) != [record.height, record.width]:
            raise ValueError(f"Image decode/dimension failure: {record.image}")
        if Path(record.image).parts[1] != record.split:
            raise ValueError("Image stored in wrong split")
        if Path(record.image).stem != Path(record.label).stem:
            raise ValueError("Image/label basename mismatch")
        label = (root / record.label).read_text()
        if label != label_text(record.detections):
            raise ValueError("Label does not match manifest")
        lines = label.splitlines()
        if not lines:
            raise ValueError("Positive-only dataset contains an empty label")
        for line in lines:
            fields = line.split()
            if (
                len(fields) != (5 + 3 * shape[0] if shape else 5)
                or not fields[0].isdigit()
                or int(fields[0]) not in names
            ):
                raise ValueError(f"Invalid YOLO class/columns: {line}")
            x, y, w, h = map(float, fields[1:5])
            if not all(math.isfinite(v) and 0 <= v <= 1 for v in (x, y, w, h)) or min(w, h) <= 0:
                raise ValueError(f"Invalid normalized box: {line}")
            if min(x - w / 2, y - h / 2) < -1e-8 or max(x + w / 2, y + h / 2) > 1 + 1e-8:
                raise ValueError("YOLO box lies outside image")
            for offset in range(5, len(fields), 3):
                px, py, visibility = map(float, fields[offset : offset + 3])
                if not all(
                    math.isfinite(v) and 0 <= v <= 1 for v in (px, py)
                ) or visibility not in (0, 2):
                    raise ValueError("Invalid YOLO keypoint coordinates/visibility")
                if visibility == 0 and (px, py) != (0, 0):
                    raise ValueError("Missing YOLO keypoint must be (0, 0, 0)")
    actual_files = {
        str(p.relative_to(root))
        for folder in ("images", "labels")
        for p in (root / folder).rglob("*")
        if p.is_file()
    }
    if actual_files != expected_files:
        raise ValueError(f"Dataset artifact set mismatch: {actual_files ^ expected_files}")
    summary = summarize(records, names, pose)
    if min(summary["splits"].values()) == 0:
        raise ValueError(f"Dataset requires positive samples in both splits: {summary['splits']}")
    loader_counts: dict[Split, LoaderCount] = {}
    if ultralytics_check:
        from ultralytics.data.dataset import YOLODataset

        # Use the actual training dataset class without the unrelated online font download.
        checked = {**data, "nc": len(names)}
        for split in SPLITS:
            try:
                dataset = YOLODataset(
                    img_path=str(root / "images" / split),
                    data=checked,
                    task=task,
                    imgsz=as_int(config["imgsz"]),
                    augment=False,
                    cache=False,
                    batch_size=1,
                    prefix=f"{split}: ",
                )
                if len(dataset) != summary["splits"][split]:
                    raise ValueError("Ultralytics dropped dataset images")
                instance_count = sum(len(label["cls"]) for label in dataset.labels)
                if instance_count != sum(summary["class_instances_by_split"][split].values()):
                    raise ValueError("Ultralytics dropped or changed labels")
                item = dataset[0]
                if item["img"].numel() == 0:
                    raise ValueError("Ultralytics returned an empty image tensor")
                loader_counts[split] = {"images": len(dataset), "instances": instance_count}
                if pose is not None:
                    expected_records = {
                        str(root / r.image): r
                        for r in records
                        if r.split == split and r.image is not None
                    }
                    valid_points = 0
                    for loaded in dataset.labels:
                        source = expected_records[str(loaded["im_file"])]
                        expected_array = np.array(
                            [
                                [(*p.xyn, p.visibility) for p in (d.keypoints or ())]
                                for d in source.detections
                            ]
                        )
                        actual_array = np.asarray(loaded["keypoints"], dtype=np.float64)
                        if actual_array.shape != expected_array.shape or not np.allclose(
                            actual_array, expected_array, atol=1e-7, rtol=0
                        ):
                            raise ValueError(
                                "Ultralytics changed keypoint coordinates or visibility"
                            )
                        valid_points += int(np.count_nonzero(actual_array[..., 2]))
                    tensor = item["keypoints"]
                    first = expected_records[str(item["im_file"])]
                    expected_mask = np.array(
                        [[p.visibility != 0 for p in (d.keypoints or ())] for d in first.detections]
                    )
                    if tuple(tensor.shape) != (*expected_mask.shape, 3) or not np.array_equal(
                        tensor[..., 2].numpy() > 0, expected_mask
                    ):
                        raise ValueError("Ultralytics pose tensor shape or mask mismatch")
                    loader_counts[split]["valid_keypoints"] = valid_points
            finally:
                (root / "labels" / f"{split}.cache").unlink(missing_ok=True)
    result: ValidationResult = {"valid": True, **summary, "ultralytics_loader": loader_counts}
    write_json(root / "validation.json", result)
    return result


def write_inspection_report(root: Path, records: Sequence[FrameRecord], report: RunReport) -> None:
    """为已有预览生成 HTML 索引，以及最多 24 张预览组成的缩略图拼图。"""
    preview_records = [record for record in records if record.preview is not None]
    cards, tiles = [], []
    for record in preview_records:
        assert record.preview is not None
        filename = html.escape(Path(record.preview).name)
        caption = f"{record.source_id or ''} | {record.timestamp:.2f}s | {record.status} | {', '.join(record.preview_reasons)}"
        cards.append(
            f'<figure><a href="{filename}"><img loading="lazy" src="{filename}"></a>'
            f"<figcaption>{html.escape(caption)}</figcaption></figure>"
        )
    indices = np.linspace(0, len(preview_records) - 1, min(24, len(preview_records)), dtype=int)
    for index in indices:
        record = preview_records[index]
        assert record.preview is not None
        image = cv2.imdecode(np.fromfile(root / record.preview, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot decode preview: {record.preview}")
        tile = np.zeros((210, 320, 3), dtype=np.uint8)
        tile[:180] = cv2.resize(image, (320, 180))
        cv2.putText(
            tile,
            f"{record.source_id or ''} | {record.timestamp:.2f}s | {record.status}",
            (5, 201),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
        )
        tiles.append(tile)
    if tiles:
        sheet = np.zeros((210 * math.ceil(len(tiles) / 4), 1280, 3), dtype=np.uint8)
        for i, tile in enumerate(tiles):
            sheet[(i // 4) * 210 : (i // 4 + 1) * 210, (i % 4) * 320 : (i % 4 + 1) * 320] = tile
        save_jpeg(root / "previews" / "contact_sheet.jpg", sheet, 90)
    document = (
        '<!doctype html><meta charset="utf-8"><title>RM26 pseudo-label inspection</title>'
        "<style>body{font-family:sans-serif;margin:24px;background:#eee}"
        "main{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:16px}"
        "figure{margin:0;background:white;padding:12px}img{width:100%}pre{white-space:pre-wrap}</style>"
        "<h1>自动标注抽检 · Pseudo labels</h1><p>标签来自模型预测，未经人工确认；不能作为真实精度评测依据。</p>"
        "<pre>" + html.escape(json.dumps(report, ensure_ascii=False, indent=2)) + "</pre>"
        "<main>" + "".join(cards) + "</main>"
    )
    atomic_bytes(root / "previews" / "index.html", document.encode())
