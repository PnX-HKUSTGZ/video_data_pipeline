"""每张 GPU 常驻一个模型进程；逐视频隔离写入，主进程按来源顺序合并。"""

from __future__ import annotations

import logging
import multiprocessing as mp
import shutil
import traceback
from dataclasses import dataclass, replace
from logging.handlers import RotatingFileHandler
from multiprocessing.queues import Queue
from pathlib import Path
from queue import Empty

from .config import Config
from .dataset import export_records, read_records
from .device import configure_device, gpu_snapshot
from .inference import InferencePipeline, build_pipeline
from .metadata import PoseMetadata, RunMetadata, SourceMetadata
from .pipeline import _cleanup_staging, _infer
from .records import FrameRecord, Task
from .util import write_json
from .video import VideoFrames

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class SourceResult:
    """只传回来源与模型信息；较大的帧记录通过该来源的独立清单读取。"""

    source: SourceMetadata
    task: Task
    pose: PoseMetadata | None


@dataclass(frozen=True)
class WorkerError:
    """保留初始化或具体视频失败的诊断，父进程据此停止本次运行。"""

    source_id: str | None
    error: str


def _process_source(
    config: Config,
    root: Path,
    run: RunMetadata,
    source: SourceMetadata,
    detector: InferencePipeline,
) -> SourceResult:
    """处理独立来源目录，允许零检测视频；全局训练/验证非空检查留给父进程。"""
    root.mkdir(parents=True)
    for split in ("train", "val"):
        for directory in ("images", "labels"):
            (root / directory / split).mkdir(parents=True)
    frames = VideoFrames(source["input"], source["video"], config)
    source["decode"] = frames.stats
    source["status"] = "inference"
    records: list[FrameRecord] = []
    try:
        _infer(config, root, run, source, detector, frames, records, 0)
        if config.smoke_frames and len(records) != config.smoke_frames:
            raise ValueError("Smoke sampling did not produce the requested number of frames")
        _, source["split"] = export_records(
            root, records, config.train_fraction, config.split_gap_seconds
        )
        source["status"] = "complete"
        return SourceResult(source, detector.task, detector.pose)
    except BaseException as error:
        source.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
        raise
    finally:
        write_json(root / "source.json", source)
        _cleanup_staging(root)
        for path in root.rglob("*.tmp"):
            path.unlink()


def _worker(
    config: Config,
    root: Path,
    run: RunMetadata,
    device: int,
    tasks: Queue[SourceMetadata | None],
    results: Queue[SourceResult | WorkerError],
) -> None:
    """spawn 子进程先隔离物理 GPU 再加载模型，循环领取视频，结束时释放设备。"""
    evidence = root / "workers" / f"gpu{device}"
    evidence.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        evidence / "pipeline.log", maxBytes=10 * 1024 * 1024, backupCount=3
    )
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.INFO)
    source: SourceMetadata | None = None
    try:
        config = replace(config, device=device)
        framework_device = configure_device(device)
        gpu_snapshot(evidence, "before")
        detector = build_pipeline(config, framework_device)
        if detector.names != run["names"]:
            raise ValueError("Loaded detector changed the class mapping")
        while (source := tasks.get()) is not None:
            LOG.info("GPU %s processing %s: %s", device, source["source_id"], source["input"])
            result = _process_source(
                config, root / "workers" / source["source_id"], run, source, detector
            )
            results.put(result)
    except BaseException:
        LOG.exception("GPU worker failed")
        results.put(WorkerError(source["source_id"] if source else None, traceback.format_exc()))
    finally:
        gpu_snapshot(evidence, "shutdown")
        logging.getLogger().removeHandler(handler)
        handler.close()


def _merge_source(root: Path, result: SourceResult) -> list[FrameRecord]:
    """移动已导出的唯一文件名，不复制图像；保留来源诊断，清单稍后按固定顺序合并。"""
    fragment = root / "workers" / result.source["source_id"]
    records = read_records(fragment / "frames.jsonl")
    for directory in ("images", "labels", "previews", "inference_videos"):
        source = fragment / directory
        if not source.exists():
            continue
        for path in source.rglob("*"):
            if path.is_file():
                target = root / path.relative_to(fragment)
                if target.exists():
                    raise ValueError(f"Parallel output collision: {target}")
                target.parent.mkdir(parents=True, exist_ok=True)
                path.rename(target)
        shutil.rmtree(source)
    if "inference_video" in result.source:
        video = fragment / result.source["inference_video"]
        if video.exists():  # 单视频输出位于根目录。
            video.rename(root / result.source["inference_video"])
    return records


def infer_parallel(
    config: Config, root: Path, run: RunMetadata, records: list[FrameRecord], devices: list[int]
) -> None:
    """动态分配完整视频；只由父进程更新总清单，失败时终止并回收自己的所有子进程。

    模型和图像不经进程队列传递。输入来源编号在启动前固定，即使完成顺序不同，
    合并清单仍与串行运行相同；单视频内部的批量、采样和划分逻辑复用原实现。
    """
    context = mp.get_context("spawn")
    devices = devices[: len(run["videos"])]
    tasks: Queue[SourceMetadata | None] = context.Queue()
    results: Queue[SourceResult | WorkerError] = context.Queue(maxsize=len(devices))
    processes = [
        context.Process(target=_worker, args=(config, root, run, device, tasks, results))
        for device in devices
    ]
    completed: dict[str, list[FrameRecord]] = {}
    sources = {v["source_id"]: v for v in run["videos"]}
    success = False
    try:
        for process in processes:
            process.start()
        write_json(
            root / "parallel.json",
            {
                "devices": devices,
                "batch_size": config.batch_size,
                "workers": [
                    {"device": d, "pid": p.pid} for d, p in zip(devices, processes, strict=True)
                ],
            },
        )
        # 长视频优先，缩短只剩一个进程仍在处理长视频的尾部等待。
        for source in sorted(run["videos"], key=lambda v: v["end"] - v["start"], reverse=True):
            tasks.put(source)
        for _ in processes:
            tasks.put(None)
        while len(completed) < len(sources):
            try:
                result = results.get(timeout=1)
            except Empty:
                if any(p.exitcode not in (None, 0) for p in processes) or all(
                    p.exitcode is not None for p in processes
                ):
                    raise RuntimeError("GPU workers exited before completing all videos")
                continue
            if isinstance(result, WorkerError):
                if result.source_id is not None:
                    sources[result.source_id].update({"status": "failed", "error": result.error})
                # WorkerError 表示远端执行失败，不是调用参数类型错误。
                raise RuntimeError(  # noqa: TRY004
                    f"GPU worker failed ({result.source_id}): {result.error}"
                )
            if "task" in run and (run["task"] != result.task or run.get("pose") != result.pose):
                raise ValueError("GPU workers disagree on model task or keypoint metadata")
            run["task"] = result.task
            if result.pose is not None:
                run["pose"] = result.pose
            source_id = result.source["source_id"]
            sources[source_id].update(result.source)
            completed[source_id] = _merge_source(root, result)
            run["candidate_frames"] = sum(len(group) for group in completed.values())
            write_json(root / "run.json", run)
            write_json(
                root / "progress.json",
                {
                    "run_id": run["run_id"],
                    "status": "inference",
                    "completed_videos": len(completed),
                    "total_videos": len(sources),
                    "samples": run["candidate_frames"],
                },
            )
            LOG.info("Completed %s/%s videos (%s)", len(completed), len(sources), source_id)
        success = True
    finally:
        for process in processes:
            if not success and process.is_alive():
                process.terminate()
        for process in processes:
            if process.pid is not None:
                process.join(timeout=10)
                if process.is_alive():
                    process.kill()
                    process.join()
        tasks.cancel_join_thread()
        tasks.close()
        results.close()
        results.join_thread()
        # 等写入进程退出后才清理，避免仍在写的临时文件被删除。
        for path in (root / "workers").glob("v*/.staging"):
            _cleanup_staging(path.parent)
        for path in (root / "workers").rglob("*.tmp"):
            path.unlink()
        with (
            (root / "frames.jsonl").open("wb") as frames,
            (root / "inference.jsonl").open("wb") as inference,
        ):
            for source in run["videos"]:
                source_id = source["source_id"]
                if source_id not in completed:
                    continue
                records.extend(completed[source_id])
                fragment = root / "workers" / source_id
                for name, destination in (("frames.jsonl", frames), ("inference.jsonl", inference)):
                    path = fragment / name
                    with path.open("rb") as stream:
                        shutil.copyfileobj(stream, destination)
                    path.unlink()
    if any(p.exitcode != 0 for p in processes):
        raise RuntimeError("A GPU worker did not exit cleanly")
