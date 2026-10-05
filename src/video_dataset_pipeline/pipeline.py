import fcntl
import logging
import shutil
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

import yaml

from .dataset import (
    Previews,
    append_record,
    check_record_files,
    export_records,
    read_records,
    save_jpeg,
    summarize,
    validate_dataset,
    write_inspection_report,
)
from .detection import filter_detections
from .device import configure_device, gpu_snapshot
from .model import Detector, verify_model
from .util import atomic_bytes, fingerprint, read_json, sha256, write_json
from .video import VideoFrames, probe_video

LOG = logging.getLogger(__name__)


def run_pipeline(config, resume=False):
    root = Path(config.output)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Another process owns output directory: {root}") from None
        return _run_locked(config, root, resume)


def _infer(config, root, run, names, frames, records):
    device = configure_device(config.device)
    gpu_snapshot(root, "before")
    try:
        detector = Detector(config.model, config, device)
        if detector.names != names:
            raise ValueError("Loaded detector changed the class mapping")
        previews = Previews(
            root, names, config.start, frames.end, records, bool(config.smoke_frames)
        )
        committed = {record["frame_index"]: record for record in records}
        seen = set()
        prefix = f"{Path(config.input).stem.split('_')[0]}_{run['video_sha256'][:10]}"
        first = True
        with (root / "inference.jsonl").open("a", encoding="utf-8") as manifest:
            for index, timestamp, image in frames:
                if index in committed:
                    if abs(committed[index]["timestamp"] - timestamp) > 1e-7:
                        raise ValueError("Resume timestamp differs from committed manifest")
                    seen.add(index)
                    continue
                if committed and index <= max(committed):
                    raise ValueError("Resume manifest omitted a sampled frame")
                height, width = image.shape[:2]
                raw = detector.predict(image)
                if first:
                    gpu_snapshot(root, "startup")
                    first = False
                detections, rejected = filter_detections(
                    raw, width, height, len(names), config.confidence
                )
                record = {
                    "frame_index": index,
                    "timestamp": timestamp,
                    "width": width,
                    "height": height,
                    "split": None,
                    "status": "candidate" if detections else "no_detection",
                    "detections": detections,
                    "rejected_detections": rejected,
                }
                name = f"{prefix}_f{index:08d}"
                if detections:
                    path = Path(".staging") / f"{name}.jpg"
                    save_jpeg(root / path, image, config.jpeg_quality)
                    record.update(
                        candidate_image=str(path), candidate_image_sha256=sha256(root / path)
                    )
                record.update(previews.save(name, image, record))
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
        if seen != set(committed):
            raise ValueError(
                "Resume manifest contains frames outside the current sampling schedule"
            )
        run.update(
            decode=frames.stats,
            inference_sha256=sha256(root / "inference.jsonl"),
            candidate_frames=len(records),
            status="exporting",
        )
        write_json(root / "run.json", run)
    finally:
        gpu_snapshot(root, "shutdown")


def _run_locked(config, root, resume):
    model_manifest = verify_model(config.model)
    info = probe_video(config.input)
    frames = VideoFrames(config.input, info, config)
    identity = {
        "schema_version": 2,
        "config": config.to_dict(),
        "video_sha256": sha256(config.input),
        "model_revision": model_manifest["revision"],
        "model_files": model_manifest["files"],
    }
    run_id = fingerprint(identity)
    names = yaml.safe_load((Path(config.model) / model_manifest["metadata"]).read_text())["names"]
    names = {int(k): str(v) for k, v in names.items()}
    run_path = root / "run.json"
    if run_path.exists():
        run = read_json(run_path)
        if run["run_id"] != run_id:
            raise ValueError(
                "Input, model or configuration changed; select a fresh output directory"
            )
        if not resume:
            raise ValueError("Output exists; use --resume with the same configuration")
        if (root / "COMPLETE.json").exists():
            configure_device("cpu")
            try:
                result = validate_dataset(root)
            except Exception:
                (root / "COMPLETE.json").unlink(missing_ok=True)
                raise
            _cleanup_staging(root)
            LOG.info("Dataset already complete and revalidated")
            return result
        if (root / "SMOKE_COMPLETE.json").exists():
            for record in read_records(root / "frames.jsonl"):
                check_record_files(root, record)
            _cleanup_staging(root)
            return read_json(root / "report.json")
        records = read_records(root / "inference.jsonl", repair_tail=True)
        for record in records:
            check_record_files(root, record)
        if run.get("decode", {}).get("decode_complete") and (
            run.get("inference_sha256") != sha256(root / "inference.jsonl")
            or len(records) != run["candidate_frames"]
        ):
            raise ValueError("Committed inference count/hash mismatch")
    else:
        if any(p.name != ".run.lock" for p in root.iterdir()):
            raise ValueError("Refusing to mix generated files into a nonempty output directory")
        records = []
        run = {
            "run_id": run_id,
            **identity,
            "video": info,
            "end": frames.end,
            "names": names,
            "model": model_manifest,
            "pseudo_labels": True,
            "status": "inference",
        }
        write_json(run_path, run)
    for path in root.rglob("*.tmp"):
        path.unlink()
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
        if not run.get("decode", {}).get("decode_complete"):
            _infer(config, root, run, names, frames, records)
        else:
            configure_device("cpu")
        records, split_info = export_records(
            root, records, config.train_fraction, config.split_gap_seconds
        )
        run.update(
            status="validating",
            split=split_info,
            manifest_sha256=sha256(root / "frames.jsonl"),
            elapsed_seconds_this_attempt=time.monotonic() - started,
        )
        run.pop("error", None)
        write_json(run_path, run)
        report = {
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
                    "manifest_sha256": run["manifest_sha256"],
                },
            )
        write_json(run_path, run)
        write_json(
            root / "progress.json",
            {"run_id": run_id, "samples": len(records), "status": run["status"]},
        )
        _cleanup_staging(root)
        LOG.info("%s: %s", run["status"], report)
        return report
    except BaseException as error:
        (root / "COMPLETE.json").unlink(missing_ok=True)
        run.update(status="failed", error=f"{type(error).__name__}: {error}")
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
        LOG.exception("Pipeline failed; committed records preserved for --resume")
        raise
    finally:
        for path in root.rglob("*.tmp"):
            path.unlink()
        logging.getLogger().removeHandler(handler)
        handler.close()


def _cleanup_staging(root):
    staging = root / ".staging"
    if staging.exists():
        try:
            shutil.rmtree(staging)
        except OSError:
            LOG.warning(
                "Could not clean staging directory; final dataset remains valid: %s",
                staging,
                exc_info=True,
            )
