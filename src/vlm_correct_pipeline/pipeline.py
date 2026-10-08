"""Run isolated image sessions, checkpoint every transition, and publish verified results."""

import json
import logging
import shutil
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from threading import Event

from PIL import UnidentifiedImageError

from .client import APIProblem, Backend
from .config import Config, identity_config, local_path
from .data import DatasetInput, Sample, discover, file_hash, load_boxes, object_hash, unchanged
from .engine import RunCancelled, annotate, probe
from .models import DONE, JSONObject, SampleResult, json_object
from .prompts import references, system_message
from .render import load_image
from .storage import RunStore, write_json


def guard_output(root: Path, data: DatasetInput, config: Config, base: Path) -> None:
    """Reject overlapping inputs/outputs before creating any run files."""
    protected = [data.root]
    protected.extend(sample.image for sample in data.samples)
    protected.extend(sample.label for sample in data.samples)
    for ref in config.references:
        protected.extend((local_path(ref.image, base).parent, local_path(ref.label, base).parent))
    for path in protected:
        if root.is_relative_to(path) or path.is_relative_to(root):
            raise ValueError("Output overlaps input data or reference data")


def process_sample(
    sample: Sample,
    previous: SampleResult | None,
    data: DatasetInput,
    config: Config,
    prefix: list[JSONObject],
    backend: Backend,
    store: RunStore,
    cancel: Event,
) -> SampleResult:
    """Restart unfinished samples from source labels and preserve prior attempt usage."""
    state = SampleResult(sample_id=sample.sample_id, fingerprint=sample.fingerprint)
    if previous:
        state.attempt = previous.attempt + 1
        state.usage = previous.usage.model_copy(deep=True)

    def record(value: JSONObject) -> None:
        store.event(sample.sample_id, state.attempt, value)
        store.save(state)

    state.status = "running"
    record(
        json_object(
            {
                "event": "start",
                "image": str(sample.image),
                "label": str(sample.label),
                "image_hash": sample.image_hash,
                "label_hash": sample.label_hash,
                "splits": sample.splits,
            }
        )
    )
    if cancel.is_set():
        state.status, state.reason = "interrupted", "Run cancelled before this image"
        store.save(state)
        return state
    try:
        if not unchanged(sample):
            raise ValueError("Source changed since discovery")
        adjustments: list[JSONObject] = []
        state.boxes = load_boxes(
            sample.label,
            data.names,
            task=data.task,
            kpt_shape=data.kpt_shape,
            rounding_tolerance=config.dataset.source_rounding_tolerance,
            adjustments=adjustments,
            pose=config.pose,
        )
        record(
            json_object(
                {
                    "event": "source_labels", "input_task": data.task, "rounding": adjustments,
                    **({"boxes": [box.model_dump(mode="json") for box in state.boxes]}
                       if config.pose else {}),
                }
            )
        )
        image = load_image(sample.image)
    except (OSError, ValueError, UnidentifiedImageError):
        state.status, state.reason = "invalid_input", "Image or source label could not be validated"
        record({"event": "terminal", "status": state.status, "reason": state.reason})
        return state
    try:
        annotate(backend, config, prefix, image, data.names, state, record, cancel)
        if not unchanged(sample):
            state.status, state.reason = "invalid_input", "Source changed during annotation"
            record({"event": "terminal", "status": state.status, "reason": state.reason})
        return state
    except BaseException:
        state.status, state.reason = "interrupted", "Annotation interrupted by a host error"
        store.save(state)
        cancel.set()
        raise
    finally:
        image.close()


def run(
    config: Config,
    base: Path,
    backend: Backend,
    *,
    endpoint_identity: str,
    cancel: Event | None = None,
) -> JSONObject:
    """Validate inputs and endpoint capabilities before scheduling any target images."""
    stop = cancel or Event()
    data = discover(config, base)
    root = local_path(config.output.directory, base)
    guard_output(root, data, config, base)
    with RunStore(root) as store:
        run_path = root / "run.json"
        previous_run = json_object(json.loads(run_path.read_text())) if run_path.exists() else None
        if previous_run is None:
            write_json(run_path, {"status": "preparing", "format_version": 2 if config.pose else 1})
        logger = logging.getLogger(f"vlm_correct.{root}")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        handler = RotatingFileHandler(
            root / "logs" / "pipeline.log",
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
        results: list[SampleResult] = []
        run_info: JSONObject = {}

        def progress(status: str, stage: str) -> None:
            """Expose cheap progress snapshots without scanning every sample checkpoint."""
            write_json(
                root / "progress.json",
                {
                    "status": status,
                    "stage": stage,
                    "updated_at": datetime.now(UTC).isoformat(),
                    "unique_images": len(data.samples),
                    "processed_images": len(results),
                    "remaining_images": len(data.samples) - len(results),
                },
            )

        try:
            reference_messages, ref_hashes = references(config, base, data.names)
            identity = {
                "format_version": 2 if config.pose else 1,
                "config": identity_config(config),
                "base": str(base),
                "endpoint": endpoint_identity,
                "names": {str(key): value for key, value in data.names.items()},
                "references": ref_hashes,
                "source_files": data.source_files,
                "samples": [[s.sample_id, s.fingerprint, s.splits] for s in data.samples],
            }
            fingerprint = object_hash(identity)
            if previous_run and previous_run.get("fingerprint") not in (None, fingerprint):
                raise ValueError("Inputs/configuration changed; choose a new output directory")
            run_info = json_object({"status": "running", "fingerprint": fingerprint, **identity})
            store.verify_publication()
            previous_results = {s.sample_id: store.load(s.sample_id) for s in data.samples}
            for sample in data.samples:
                old = previous_results[sample.sample_id]
                if old is not None and (
                    old.sample_id != sample.sample_id or old.fingerprint != sample.fingerprint
                ):
                    raise ValueError("Sample checkpoint does not match its source")
            pending = [
                s
                for s in data.samples
                if (old := previous_results[s.sample_id]) is None or old.status not in DONE
            ]
            if pending or not (root / "dataset").exists():
                required = sum(s.image.stat().st_size * len(s.splits) for s in data.samples)
                if shutil.disk_usage(root).free < required + max(required // 10, 16 * 1024 * 1024):
                    raise ValueError("Insufficient space for an independent dataset snapshot")
            results = [r for r in previous_results.values() if r and r.status in DONE]
            if previous_run:
                run_info["capability_probe"] = previous_run.get("capability_probe")
            write_json(run_path, run_info)
            progress("running", "preflight" if pending else "export")
            if pending:
                logger.info("Checking endpoint capabilities before %d target images", len(pending))
                probe_events: list[JSONObject] = []

                def record_probe(value: JSONObject) -> None:
                    probe_events.append(value)
                    write_json(
                        root / "capability_probe.json",
                        {"status": "running", "events": probe_events},
                    )

                try:
                    run_info["capability_probe"] = probe(
                        backend,
                        cancel=stop,
                        reference_context=reference_messages,
                        recorder=record_probe,
                        max_retries=config.run.max_retries,
                    )
                except BaseException:
                    write_json(
                        root / "capability_probe.json", {"status": "failed", "events": probe_events}
                    )
                    raise
                write_json(root / "capability_probe.json", run_info["capability_probe"])
                write_json(run_path, run_info)
                prefix = [system_message(config, data.names), *reference_messages]
                progress("running", "annotation")
                waiting = deque(pending)
                api_errors = 0
                with ThreadPoolExecutor(max_workers=config.run.image_concurrency) as pool:
                    futures: set[Future[SampleResult]] = set()

                    def submit() -> None:
                        sample = waiting.popleft()
                        futures.add(
                            pool.submit(
                                process_sample,
                                sample,
                                previous_results[sample.sample_id],
                                data,
                                config,
                                prefix,
                                backend,
                                store,
                                stop,
                            )
                        )

                    try:
                        while waiting or futures:
                            while (
                                waiting
                                and len(futures) < config.run.image_concurrency
                                and not stop.is_set()
                            ):
                                submit()
                            if not futures:
                                break
                            ready, futures = wait(futures, return_when=FIRST_COMPLETED)
                            for future in ready:
                                result = future.result()
                                results.append(result)
                                progress("running", "annotation")
                                logger.info(
                                    "%s %s requests=%d",
                                    result.sample_id,
                                    result.status,
                                    result.usage.requests,
                                )
                                api_errors = api_errors + 1 if result.status == "api_error" else 0
                                if api_errors >= config.run.max_consecutive_api_errors:
                                    logger.error("Stopping after repeated API failures")
                                    stop.set()
                    except BaseException:
                        stop.set()
                        raise
            for path, digest in {**ref_hashes, **data.source_files}.items():
                if file_hash(Path(path)) != digest:
                    raise ValueError("Dataset metadata or reference changed during the run")
            if stop.is_set():
                mapping = []
            elif not pending and (root / "dataset").exists():
                mapping = store.publication_mapping()
            else:
                mapping = store.publish(data, results)
            counts = Counter(r.status for r in results)
            status = (
                "interrupted"
                if stop.is_set()
                else ("completed" if all(r.status in DONE for r in results) else "partial")
            )
            if stop.is_set() and any(r.status == "api_error" for r in results):
                status = "failed"
            processed_ids = {r.sample_id for r in results}
            report = json_object(
                {
                    "status": status,
                    "fingerprint": fingerprint,
                    "counts": dict(counts),
                    "unique_images": len(data.samples),
                    "not_processed": [
                        s.sample_id for s in data.samples if s.sample_id not in processed_ids
                    ],
                    "input_task": config.dataset.input_task,
                    "output_task": data.output_task,
                    "exported_entries": len(mapping),
                    "mapping": mapping,
                    "samples": [
                        r.model_dump(mode="json")
                        for r in sorted(results, key=lambda r: r.sample_id)
                    ],
                    "usage": {
                        "requests": sum(r.usage.requests for r in results),
                        "input_tokens": sum(r.usage.input_tokens for r in results),
                        "output_tokens": sum(r.usage.output_tokens for r in results),
                    },
                    "capability_probe": run_info.get("capability_probe"),
                }
            )
            write_json(root / "report.json", report)
            write_json(run_path, {**run_info, "status": status})
            progress(status, "finished")
            return report
        except BaseException as error:
            # Preserve the original identity on failed resume; never bless incompatible inputs.
            failure = str(error) if isinstance(error, APIProblem) else type(error).__name__
            logger.error("Run failed: %s", failure)
            failure_status = (
                "interrupted" if isinstance(error, (RunCancelled, KeyboardInterrupt)) else "failed"
            )
            if not previous_run or run_info:
                write_json(
                    run_path,
                    {
                        **(run_info or previous_run or {}),
                        "status": failure_status,
                        "error": failure,
                    },
                )
                progress(failure_status, "failed")
                write_json(
                    root / "report.json",
                    {
                        "status": failure_status,
                        "error": failure,
                        "unique_images": len(data.samples),
                        "input_task": config.dataset.input_task,
                        "output_task": data.output_task,
                        "counts": dict(Counter(r.status for r in results)),
                        "published_dataset": (root / "dataset").is_dir(),
                    },
                )
            raise
        finally:
            logger.removeHandler(handler)
            handler.close()
