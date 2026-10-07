"""Read-only lighting audit for detection/pose datasets; metrics do not decide deletion."""

import json
import math
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

from PIL import Image, ImageOps
from pydantic import Field

from yolo_dataset_viewer.dataset import Sample, build_index, resolve_spec
from yolo_dataset_viewer.labels import read_labels
from yolo_dataset_viewer.models import OpenRequest, Task

from .config import StrictModel
from .data import object_hash
from .engine import RunCancelled
from .models import JSONObject, json_object
from .storage import atomic_write, write_json


class Luminance(StrictModel):
    """8-bit luminance distribution; dark/bright fractions are dimensionless."""

    mean: float
    p10: int
    p50: int
    p90: int
    dark_fraction: float
    bright_fraction: float


class RegionQuality(StrictModel):
    class_id: int
    bbox: list[float]
    luminance: Luminance


class ImageQuality(StrictModel):
    sample_id: str
    image: str
    label: str
    split: str
    label_state: str
    source_bytes: int
    source_mtime_ns: int
    global_luminance: Luminance | None = None
    regions: list[RegionQuality] = Field(default_factory=list)
    candidate: str = "normal"
    error: str | None = None


def luminance(image: Image.Image) -> Luminance:
    """Measure a bounded grayscale thumbnail without importing GPU libraries."""
    with image.convert("L") as gray:
        histogram = gray.histogram()
    count = sum(histogram)

    def percentile(fraction: float) -> int:
        accumulated = 0
        for value, frequency in enumerate(histogram):
            accumulated += frequency
            if accumulated >= max(1, math.ceil(count * fraction)):
                return value
        return 255

    return Luminance(
        mean=sum(value * frequency for value, frequency in enumerate(histogram)) / count,
        p10=percentile(0.1),
        p50=percentile(0.5),
        p90=percentile(0.9),
        dark_fraction=sum(histogram[:17]) / count,
        bright_fraction=sum(histogram[245:]) / count,
    )


def inspect_lighting(
    sample: Sample,
    task: Task,
    names: dict[int, str],
    shape: tuple[int, int] | None,
) -> ImageQuality:
    """Inspect full frame and target ROIs; preserve uncertain candidates for VLM review."""
    stamp = sample.image.stat()
    row = ImageQuality(
        sample_id=object_hash(str(sample.image)),
        image=str(sample.image),
        label=str(sample.label),
        split=sample.split,
        label_state=sample.state,
        source_bytes=stamp.st_size,
        source_mtime_ns=stamp.st_mtime_ns,
    )
    try:
        labels = read_labels(sample.label, task, names, shape)
        row.label_state = labels.state
        if labels.state == "invalid":
            row.error = "invalid_label"
            return row
        with Image.open(sample.image) as source:
            source.draft("RGB", (384, 216))
            image = ImageOps.exif_transpose(source)
        try:
            image.thumbnail((384, 384))
            row.global_luminance = luminance(image)
            width, height = image.size
            for item in labels.annotations:
                if item.kind != "detect" and item.kind != "pose":
                    continue
                box = item.box
                bbox = [
                    box.cx - box.width / 2,
                    box.cy - box.height / 2,
                    box.cx + box.width / 2,
                    box.cy + box.height / 2,
                ]
                extent = (
                    max(0, math.floor(bbox[0] * width)),
                    max(0, math.floor(bbox[1] * height)),
                    min(width, math.ceil(bbox[2] * width)),
                    min(height, math.ceil(bbox[3] * height)),
                )
                if extent[2] <= extent[0] or extent[3] <= extent[1]:
                    continue
                with image.crop(extent) as crop:
                    row.regions.append(
                        RegionQuality(class_id=item.class_id, bbox=bbox, luminance=luminance(crop))
                    )
            measurements = [region.luminance for region in row.regions] or [row.global_luminance]
            if any(item.dark_fraction >= 0.6 and item.p90 < 65 for item in measurements):
                row.candidate = "underexposed_candidate"
            elif any(item.bright_fraction >= 0.6 and item.p10 > 180 for item in measurements):
                row.candidate = "overexposed_candidate"
        finally:
            image.close()
        current = sample.image.stat()
        if (current.st_size, current.st_mtime_ns) != (stamp.st_size, stamp.st_mtime_ns):
            row.error = "source_changed"
    except (OSError, ValueError):
        row.error = "image_decode_or_geometry_error"
    return row


def audit_lighting(
    source: Path, output: Path, *, workers: int = 4, cancel: Event | None = None
) -> JSONObject:
    """Audit every registered image and publish metrics only after the scan completes."""
    stop = cancel or Event()
    if source.is_dir() and (source / "data.yaml").is_file():
        source = source / "data.yaml"
    spec = resolve_spec(OpenRequest(path=str(source)))
    request = OpenRequest(
        path=str(source), task="pose" if spec.config_data.get("kpt_shape") else "detect"
    )
    dataset = build_index(request, stop, lambda *_: None)
    if output.resolve().is_relative_to(dataset.spec.root):
        raise ValueError("Lighting audit output must be outside the input dataset")
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Choose an empty audit output directory")
    rows: list[ImageQuality] = []
    counts: Counter[str] = Counter()
    pending = output / ".records.jsonl.tmp"

    def inspect(sample: Sample) -> ImageQuality:
        if stop.is_set():
            raise RunCancelled
        return inspect_lighting(sample, request.task, dataset.spec.names, dataset.spec.kpt_shape)

    try:
        with (
            pending.open("w", encoding="utf-8") as stream,
            ThreadPoolExecutor(max_workers=workers) as pool,
        ):
            for row in pool.map(inspect, dataset.samples):
                rows.append(row)
                counts[row.candidate if row.error is None else row.error] += 1
                stream.write(row.model_dump_json() + "\n")
                if len(rows) % 1000 == 0:
                    stream.flush()
                    print(f"Lighting audit: {len(rows)}/{len(dataset.samples)}", flush=True)
        pending.replace(output / "records.jsonl")
        split_counts = Counter(row.split for row in rows)
        report = json_object(
            {
                "status": "completed",
                "source": str(source.resolve()),
                "input_task": request.task,
                "images": len(rows),
                "source_bytes": sum(row.source_bytes for row in rows),
                "splits": dict(split_counts),
                "candidate_counts": dict(counts),
                "definitions": {
                    "dark_pixel": "luminance <= 16",
                    "bright_pixel": "luminance >= 245",
                    "underexposed_candidate": "target ROI dark fraction >= 0.6 and p90 < 65",
                    "overexposed_candidate": "target ROI bright fraction >= 0.6 and p10 > 180",
                },
                "policy": "Candidates for visual review only; no images or labels were modified or discarded.",
            }
        )
        write_json(output / "summary.json", report)
        atomic_write(output / "COMPLETE.json", (json.dumps({"images": len(rows)}) + "\n").encode())
        return report
    finally:
        pending.unlink(missing_ok=True)
