import logging
import shutil
import time
from contextlib import nullcontext
from logging.handlers import RotatingFileHandler
from pathlib import Path
from uuid import uuid4

import yaml

from .config import Config
from .dataset import (
    Previews,
    annotate_frame,
    append_record,
    check_record_files,
    export_records,
    save_jpeg,
    summarize,
    validate_dataset,
    write_inspection_report,
)
from .detection import candidate_status, filter_detections
from .device import configure_device, gpu_snapshot
from .inference_video import InferenceVideoWriter
from .inputs import discover_videos
from .metadata import DecodeStats, RunMetadata, RunReport, SourceMetadata, SourceReport
from .model import Detector, load_model_names
from .records import FrameRecord
from .util import atomic_bytes, write_json
from .video import VideoFrames, probe_video

LOG = logging.getLogger(__name__)


def _infer(
    config: Config,
    root: Path,
    run: RunMetadata,
    source: SourceMetadata,
    detector: Detector,
    frames: VideoFrames,
    records: list[FrameRecord],
    completed_samples: int,
) -> None:
    """使用共享模型处理一个视频；只暂存有效帧，并向总推理清单追加记录。

    记录携带来源编号，帧号和时间戳仍属于源视频。预览计数在每个视频开始时重置；
    保存视频时对区间内的全部源帧推理，保持该视频自身时间间隔。
    """
    names, pose = detector.names, detector.pose
    point_names = pose.get("kpt_names") if pose is not None else None
    previews = Previews(
        root, names, config.start, frames.end, bool(config.smoke_frames), point_names
    )
    prefix = f"{source['source_id']}_{frames.path.stem}"
    video_context = (
        InferenceVideoWriter(
            root,
            frames.info["width"],
            frames.info["height"],
            frames.info["fps"],
            frames.end,
            source["inference_video"],
        )
        if config.save_video
        else nullcontext()
    )
    with (root / "inference.jsonl").open("a", encoding="utf-8") as manifest, video_context as video:
        for index, timestamp, image in frames:
            height, width = image.shape[:2]
            raw = detector.predict(image)
            if not records:
                gpu_snapshot(root, f"{source['source_id']}_startup")
            detections, rejected = filter_detections(
                raw,
                width,
                height,
                len(names),
                config.confidence,
                pose["source_kpt_shape"] if pose is not None else None,
                config.keypoint_confidence,
            )
            record = FrameRecord(
                frame_index=index,
                timestamp=timestamp,
                width=width,
                height=height,
                split=None,
                status=candidate_status(detections),
                detections=detections,
                rejected_detections=rejected,
                source_id=source["source_id"],
            )
            name = f"{prefix}_f{index:08d}"
            if record.status == "candidate":
                path = Path(".staging") / f"{name}.jpg"
                save_jpeg(root / path, image, config.jpeg_quality)
                record.candidate_image = str(path)
            previews.save(name, image, record)
            if video is not None:
                video.write(annotate_frame(image, record, names, point_names), timestamp)
            append_record(manifest, record)
            records.append(record)
            source["candidate_frames"] = len(records)
            if len(records) % 50 == 0:
                LOG.info(
                    "%s: processed %s samples; timestamp %.2fs",
                    source["source_id"],
                    len(records),
                    timestamp,
                )
                write_json(
                    root / "progress.json",
                    {
                        "run_id": run["run_id"],
                        "source_id": source["source_id"],
                        "source_samples": len(records),
                        "samples": completed_samples + len(records),
                        "timestamp": timestamp,
                        "status": "inference",
                    },
                )
            if len(records) % 200 == 0:
                gpu_snapshot(root, "execution")


def _decode_summary(videos: list[SourceMetadata]) -> DecodeStats:
    """汇总解码完成状态；多视频没有统一的末帧时间，使用 None。"""
    return {
        "decoded_frames": sum(v["decode"]["decoded_frames"] for v in videos),
        "last_timestamp": videos[0]["decode"]["last_timestamp"] if len(videos) == 1 else None,
        "decode_complete": all(v["decode"]["decode_complete"] for v in videos),
    }


def _report(run: RunMetadata, records: list[FrameRecord], started: float) -> RunReport:
    """从最终或部分帧记录生成整体及逐视频统计，包含零检测和未处理视频。"""
    grouped: dict[str | None, list[FrameRecord]] = {}
    for record in records:
        grouped.setdefault(record.source_id, []).append(record)
    sources: list[SourceReport] = []
    for video in run["videos"]:
        item: SourceReport = {
            **summarize(grouped.get(video["source_id"], []), run["names"], run.get("pose")),
            "source_id": video["source_id"],
            "input": video["input"],
            "decode": video["decode"],
            "status": video["status"],
            "split": video.get("split"),
        }
        sources.append(item)
    return {
        **summarize(records, run["names"], run.get("pose")),
        "split": {v["source_id"]: v["split"] for v in run["videos"] if "split" in v},
        "videos": sources,
        "pseudo_labels": True,
        "evaluation_warning": "Teacher predictions are not ground-truth accuracy measurements.",
        "run_id": run["run_id"],
        "decode": _decode_summary(run["videos"]),
        "elapsed_seconds_this_attempt": time.monotonic() - started,
    }


def run_pipeline(config: Config) -> RunReport:
    """固定输入清单，只加载一次模型，独立划分每个视频并合并到同一数据集。

    启动前检查所有视频的处理区间。任一处理失败则整次运行失败，保留诊断和已
    导出产物，清理暂存文件。只有全部视频处理及数据校验通过才写完成标记。
    """
    root = Path(config.output).resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError("Output directory must be empty; choose a new --output directory")
    if not Path(config.model).is_file():
        raise ValueError(f"Model weights missing: {config.model}")
    names = load_model_names(config.model_yaml)
    inputs = discover_videos(config.input, root, config.recursive)
    videos: list[SourceMetadata] = []
    decoders: list[VideoFrames] = []
    for source in inputs:
        try:
            info = probe_video(source.path)
            frames = VideoFrames(source.path, info, config)
        except Exception as error:
            raise ValueError(
                f"Cannot process {source.source_id} ({source.path}): {error}"
            ) from error
        metadata: SourceMetadata = {
            "source_id": source.source_id,
            "input": str(source.path),
            "video": info,
            "start": config.start,
            "end": frames.end,
            "decode": frames.stats,
            "status": "pending",
            "candidate_frames": 0,
        }
        if config.save_video:
            metadata["inference_video"] = (
                "inference.mp4"
                if len(inputs) == 1
                else f"inference_videos/{source.source_id}_{source.path.stem}.mp4"
            )
        videos.append(metadata)
        decoders.append(frames)
    root.mkdir(parents=True, exist_ok=True)
    run: RunMetadata = {
        "schema_version": 6,
        "run_id": uuid4().hex,
        "config": config.to_dict(),
        "videos": videos,
        "names": names,
        "model": config.model,
        "model_yaml": config.model_yaml,
        "pseudo_labels": True,
        "status": "inference",
    }
    run_path = root / "run.json"
    write_json(run_path, run)
    for split in ("train", "val"):
        for directory in ("images", "labels"):
            (root / directory / split).mkdir(parents=True, exist_ok=True)
    atomic_bytes(
        root / "config.yaml", yaml.safe_dump(config.to_dict(), allow_unicode=True).encode()
    )
    handler = RotatingFileHandler(root / "pipeline.log", maxBytes=10 * 1024 * 1024, backupCount=3)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(handler)
    started = time.monotonic()
    records: list[FrameRecord] = []
    current: list[FrameRecord] = []
    active: SourceMetadata | None = None
    device_configured = False
    try:
        device = configure_device(config.device)
        device_configured = True
        gpu_snapshot(root, "before")
        detector = Detector(config, device)
        if detector.names != names:
            raise ValueError("Loaded detector changed the class mapping")
        run["task"] = detector.task
        data: dict[str, object] = {
            "path": str(root),
            "train": "images/train",
            "val": "images/val",
            "names": names,
        }
        if detector.pose is not None:
            pose = detector.pose
            run["pose"] = pose
            data["kpt_shape"] = [pose["source_kpt_shape"][0], 3]
            if "kpt_names" in pose:
                data["kpt_names"] = pose["kpt_names"]
            if "flip_idx" in pose:
                data["flip_idx"] = pose["flip_idx"]
        atomic_bytes(root / "data.yaml", yaml.safe_dump(data, allow_unicode=True).encode())
        if config.save_video:
            LOG.info("Video saving enabled: infer every source frame; sample_fps ignored")
        for active, frames in zip(videos, decoders, strict=True):
            current = []
            active["status"] = "inference"
            write_json(run_path, run)
            LOG.info("Processing %s: %s", active["source_id"], active["input"])
            _infer(config, root, run, active, detector, frames, current, len(records))
            if config.smoke_frames and len(current) != config.smoke_frames:
                raise ValueError("Smoke sampling did not produce the requested number of frames")
            active["status"] = "exporting"
            exported, split_info = export_records(
                root, current, config.train_fraction, config.split_gap_seconds, True
            )
            active["split"] = split_info
            active["status"] = "complete"
            records.extend(exported)
            current = []
            _cleanup_staging(root)
            run["candidate_frames"] = len(records)
            write_json(run_path, run)
            write_json(
                root / "progress.json",
                {
                    "run_id": run["run_id"],
                    "source_id": active["source_id"],
                    "completed_videos": sum(v["status"] == "complete" for v in videos),
                    "total_videos": len(videos),
                    "samples": len(records),
                    "status": "inference",
                },
            )
        active = None
        report = _report(run, records, started)
        run.update(
            {
                "status": "validating",
                "split": report["split"],
                "decode": report["decode"],
                "elapsed_seconds_this_attempt": report["elapsed_seconds_this_attempt"],
            }
        )
        write_json(run_path, run)
        write_json(root / "report.json", report)
        write_inspection_report(root, records, report)
        if min(report["splits"].values()) == 0:
            raise ValueError("Dataset requires positive samples in both splits")
        if config.smoke_frames:
            for record in records:
                check_record_files(root, record)
            run["status"] = "smoke_complete"
            write_json(root / "SMOKE_COMPLETE.json", report)
        else:
            validation = validate_dataset(root)
            run["status"] = "complete"
            write_json(root / "COMPLETE.json", {"run_id": run["run_id"], "validation": validation})
        write_json(run_path, run)
        write_json(
            root / "progress.json",
            {
                "run_id": run["run_id"],
                "samples": len(records),
                "status": run["status"],
                "completed_videos": len(videos),
                "total_videos": len(videos),
            },
        )
        LOG.info("%s: %s", run["status"], report)
        return report
    except BaseException as error:
        (root / "COMPLETE.json").unlink(missing_ok=True)
        (root / "SMOKE_COMPLETE.json").unlink(missing_ok=True)
        context = f"{active['source_id']} ({active['input']}): " if active is not None else ""
        diagnostic = f"{context}{type(error).__name__}: {error}"
        if active is not None:
            active.update({"status": "failed", "error": diagnostic})
        report = _report(run, records + current, started)
        run.update({"status": "failed", "error": diagnostic, "decode": report["decode"]})
        write_json(run_path, run)
        write_json(root / "report.json", {**report, "status": "failed", "error": diagnostic})
        write_json(
            root / "progress.json",
            {"run_id": run["run_id"], "status": "failed", "error": diagnostic},
        )
        LOG.exception("Pipeline failed: %s", diagnostic)
        raise
    finally:
        _cleanup_staging(root)
        for path in root.rglob("*.tmp"):
            path.unlink()
        logging.getLogger().removeHandler(handler)
        handler.close()
        if device_configured:
            gpu_snapshot(root, "shutdown")


def _cleanup_staging(root: Path) -> None:
    """删除本次运行的暂存图像；清理失败只记录警告，避免覆盖原始处理异常。"""
    staging = root / ".staging"
    if staging.exists():
        try:
            shutil.rmtree(staging)
        except OSError:
            LOG.warning("Could not clean staging directory: %s", staging, exc_info=True)
