"""Adapt the existing lightweight YOLO reader without importing inference libraries."""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event
from typing import Literal

from yolo_dataset_viewer.dataset import build_index
from yolo_dataset_viewer.labels import read_labels
from yolo_dataset_viewer.models import OpenRequest

from .config import Config, local_path
from .models import Box, JSONObject, json_object


def file_hash(path: Path, *, missing: bool = False) -> str:
    """Stream file contents; a missing label has a distinct identity from an empty one."""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except FileNotFoundError:
        if missing:
            return "missing"
        raise
    return digest.hexdigest()


def object_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


@dataclass
class Sample:
    """A unique source image may be a member of several dataset splits."""

    sample_id: str
    image: Path
    label: Path
    relative: Path
    image_hash: str
    label_hash: str
    splits: list[str] = field(default_factory=list)

    @property
    def fingerprint(self) -> str:
        return object_hash([str(self.image), self.image_hash, str(self.label), self.label_hash])


@dataclass(frozen=True)
class DatasetInput:
    root: Path
    names: dict[int, str]
    samples: list[Sample]
    source_files: dict[str, str]
    task: Literal["detect", "pose"] = "detect"
    kpt_shape: tuple[int, int] | None = None


def discover(config: Config, base: Path) -> DatasetInput:
    """Discover layouts with the viewer, then enforce explicit fixed class definitions."""
    source = local_path(config.dataset.source, base)
    task: Literal["detect", "pose"] = (
        "pose" if config.dataset.input_task == "pose_boxes" else "detect"
    )
    dataset = build_index(OpenRequest(path=str(source), task=task), Event(), lambda *_: None)
    if dataset.spec.config_data.get("kpt_shape") is not None and task != "pose":
        raise ValueError("Pose input requires explicit dataset.input_task: pose_boxes")
    selected: set[Path] | None = None
    selection_path: Path | None = None
    if config.dataset.selection_file:
        selection_path = local_path(config.dataset.selection_file, base)
        items = json.loads(selection_path.read_text(encoding="utf-8"))
        if not isinstance(items, list) or not items or not all(isinstance(p, str) for p in items):
            raise ValueError("selection_file must contain a nonempty JSON array of image paths")
        selected = {local_path(p, dataset.spec.root) for p in items}
        if len(selected) != len(items):
            raise ValueError("selection_file contains duplicate images")
    given = config.dataset.names
    declared = dataset.spec.names
    if given and declared and given != declared:
        raise ValueError("Config names disagree with dataset names")
    names = given if given is not None else declared
    if (
        not names
        or sorted(names) != list(range(len(names)))
        or any(not n.strip() for n in names.values())
    ):
        raise ValueError("An explicit nonempty, contiguous class table is required")
    unique: dict[Path, Sample] = {}
    for row in dataset.samples:
        image = row.image.resolve()
        if selected is not None and image not in selected:
            continue
        label = row.label.resolve()
        label_digest = file_hash(label, missing=True)
        split = "train" if row.split == "all" else row.split
        if image in unique:
            current = unique[image]
            if current.label_hash != label_digest:
                raise ValueError("The same image has conflicting labels across sources")
            if split not in current.splits:
                current.splits.append(split)
            continue
        try:
            relative = image.relative_to(dataset.spec.root)
        except ValueError:
            relative = Path(image.name)
        # YOLO resolves labels by replacing the LAST images component. Preserve only
        # the path inside that image tree, avoiding a second images component in export.
        parts = list(relative.parts)
        image_components = [i for i, part in enumerate(parts[:-1]) if part == "images"]
        if image_components:
            parts = parts[image_components[-1] + 1 :]
        if len(parts) > 1 and parts[0] == split:
            parts = parts[1:]
        relative = Path(*parts)
        unique[image] = Sample(
            sample_id=object_hash(str(image)),
            image=image,
            label=label,
            relative=relative,
            image_hash=file_hash(image),
            label_hash=label_digest,
            splits=[split],
        )
    if not unique:
        raise ValueError("No input images found")
    if selected is not None and selected != set(unique):
        raise ValueError("selection_file contains images outside the registered dataset")
    source_files: dict[str, str] = {}
    if selection_path:
        source_files[str(selection_path)] = file_hash(selection_path)
    if dataset.spec.config_path:
        source_files[str(dataset.spec.config_path)] = file_hash(dataset.spec.config_path)
    for sources in dataset.spec.sources.values():
        for path in sources:
            if path.is_file() and path.suffix.lower() == ".txt":
                source_files[str(path)] = file_hash(path)
    return DatasetInput(
        dataset.spec.root,
        dict(names),
        list(unique.values()),
        source_files,
        task,
        dataset.spec.kpt_shape,
    )


def load_boxes(
    path: Path,
    names: dict[int, str],
    *,
    reference: bool = False,
    task: Literal["detect", "pose"] = "detect",
    kpt_shape: tuple[int, int] | None = None,
    rounding_tolerance: float = 0,
    adjustments: list[JSONObject] | None = None,
) -> list[Box]:
    """Reject a whole invalid input; a reference must have an explicit label file."""
    labels = read_labels(path, task, names, kpt_shape)
    if labels.state == "invalid" or (reference and labels.state == "missing"):
        raise ValueError(
            "Invalid or missing reference label" if reference else "Invalid YOLO label"
        )
    result: list[Box] = []
    for index, annotation in enumerate(labels.annotations, 1):
        if annotation.kind != "detect" and annotation.kind != "pose":
            raise ValueError("Only detection annotations are supported")
        box = annotation.box
        bbox = [
            box.cx - box.width / 2,
            box.cy - box.height / 2,
            box.cx + box.width / 2,
            box.cy + box.height / 2,
        ]
        if min(bbox) < 0 or max(bbox) > 1:
            if min(bbox) < -rounding_tolerance or max(bbox) > 1 + rounding_tolerance:
                raise ValueError("Source box crosses image boundary beyond rounding tolerance")
            clipped = [max(0.0, min(1.0, v)) for v in bbox]
            if adjustments is not None:
                adjustments.append(json_object({"line": index, "before": bbox, "after": clipped}))
            bbox = clipped
        result.append(
            Box(
                box_id=f"box_{index:04d}",
                class_id=annotation.class_id,
                bbox=bbox,
            )
        )
    return result


def unchanged(sample: Sample) -> bool:
    """Guard against edits to source bytes during a long annotation run."""
    return (
        file_hash(sample.image) == sample.image_hash
        and file_hash(sample.label, missing=True) == sample.label_hash
    )
