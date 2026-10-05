import logging
import shutil
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from uuid import uuid4

import yaml

from .config import Config
from .dataset import (
    Previews,
    append_record,
    check_record_files,
    export_records,
    save_jpeg,
    summarize,
    validate_dataset,
    write_inspection_report,
)
from .detection import filter_detections
from .device import configure_device, gpu_snapshot
from .metadata import RunMetadata, RunReport
from .model import Detector, load_model_names
from .records import FrameRecord
from .util import atomic_bytes, write_json
from .video import VideoFrames, probe_video

LOG = logging.getLogger(__name__)


def _infer(
    config: Config,
    root: Path,
    run: RunMetadata,
    names: dict[int, str],
    frames: VideoFrames,
    records: list[FrameRecord],
) -> None:
    device = configure_device(config.device)
    gpu_snapshot(root, "before")
    try:
        detector = Detector(config, device)
        if detector.names != names:
            raise ValueError("Loaded detector changed the class mapping")
        previews = Previews(root, names, config.start, frames.end, bool(config.smoke_frames))
        prefix = Path(config.input).stem
        first = True
        with (root / "inference.jsonl").open("w", encoding="utf-8") as manifest:
            for index, timestamp, image in frames:
                height, width = image.shape[:2]
                raw = detector.predict(image)
                if first:
                    gpu_snapshot(root, "startup")
                    first = False
                detections, rejected = filter_detections(
                    raw, width, height, len(names), config.confidence
                )
                record = FrameRecord(
                    frame_index=index,
                    timestamp=timestamp,
                    width=width,
                    height=height,
                    split=None,
                    status="candidate" if detections else "no_detection",
                    detections=detections,
                    rejected_detections=rejected,
                )
                name = f"{prefix}_f{index:08d}"
                if detections:
                    path = Path(".staging") / f"{name}.jpg"
                    save_jpeg(root / path, image, config.jpeg_quality)
                    record.candidate_image = str(path)
                previews.save(name, image, record)
                append_record(manifest, record)
                records.append(record)
                if len(records) % 50 == 0:
                    LOG.info("Processed %s samples; timestamp %.2fs", len(records), timestamp)
                    write_json(
                        root / "progress.json",
                        {
                            "run_id": run["run_id"],
                            "samples": len(records),
                            "timestamp": timestamp,
                            "status": "inference",
                        },
                    )
                if len(records) % 200 == 0:
                    gpu_snapshot(root, "execution")
        run.update(
            {"decode": frames.stats, "candidate_frames": len(records), "status": "exporting"}
        )
        write_json(root / "run.json", run)
    finally:
        gpu_snapshot(root, "shutdown")


def run_pipeline(config: Config) -> RunReport:
    root = Path(config.output)
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError("Output directory must be empty; choose a new --output directory")
    if not Path(config.model).is_file():
        raise ValueError(f"Model weights missing: {config.model}")
    names = load_model_names(config.model_yaml)
    info = probe_video(config.input)
    frames = VideoFrames(config.input, info, config)
    run_id = uuid4().hex
    records: list[FrameRecord] = []
    run: RunMetadata = {
        "schema_version": 4,
        "run_id": run_id,
        "config": config.to_dict(),
        "video": info,
        "end": frames.end,
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
    atomic_bytes(
        root / "data.yaml",
        yaml.safe_dump(
            {"path": str(root), "train": "images/train", "val": "images/val", "names": names},
            allow_unicode=True,
        ).encode(),
    )
    handler = RotatingFileHandler(root / "pipeline.log", maxBytes=10 * 1024 * 1024, backupCount=3)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(handler)
    started = time.monotonic()
    try:
        _infer(config, root, run, names, frames, records)
        records, split_info = export_records(
            root, records, config.train_fraction, config.split_gap_seconds
        )
        run.update(
            {
                "status": "validating",
                "split": split_info,
                "elapsed_seconds_this_attempt": time.monotonic() - started,
            }
        )
        write_json(run_path, run)
        report: RunReport = {
            **summarize(records, names),
            "split": split_info,
            "pseudo_labels": True,
            "evaluation_warning": "Teacher predictions are not ground-truth accuracy measurements.",
            "run_id": run_id,
            "decode": run["decode"],
            "elapsed_seconds_this_attempt": run["elapsed_seconds_this_attempt"],
        }
        write_json(root / "report.json", report)
        write_inspection_report(root, records, report)
        if config.smoke_frames:
            if len(records) != config.smoke_frames:
                raise ValueError("Smoke sampling did not produce the requested number of frames")
            for record in records:
                check_record_files(root, record)
            run["status"] = "smoke_complete"
            write_json(root / "SMOKE_COMPLETE.json", report)
        else:
            validation = validate_dataset(root)
            run["status"] = "complete"
            write_json(
                root / "COMPLETE.json",
                {
                    "run_id": run_id,
                    "validation": validation,
                },
            )
        write_json(run_path, run)
        write_json(
            root / "progress.json",
            {"run_id": run_id, "samples": len(records), "status": run["status"]},
        )
        LOG.info("%s: %s", run["status"], report)
        return report
    except BaseException as error:
        (root / "COMPLETE.json").unlink(missing_ok=True)
        run.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
        run.setdefault("decode", frames.stats)
        write_json(run_path, run)
        write_json(
            root / "report.json",
            {
                **summarize(records, names),
                "status": "failed",
                "error": run["error"],
                "decode": run["decode"],
            },
        )
        LOG.exception("Pipeline failed; diagnostic records retained")
        raise
    finally:
        _cleanup_staging(root)
        for path in root.rglob("*.tmp"):
            path.unlink()
        logging.getLogger().removeHandler(handler)
        handler.close()


def _cleanup_staging(root: Path) -> None:
    staging = root / ".staging"
    if staging.exists():
        try:
            shutil.rmtree(staging)
        except OSError:
            LOG.warning(
                "Could not clean staging directory: %s",
                staging,
                exc_info=True,
            )
