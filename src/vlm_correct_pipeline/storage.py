"""Durable checkpoints and atomic publication of independent YOLO snapshots."""

import fcntl
import json
import os
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import IO, Self
from uuid import uuid4

import yaml

from .data import DatasetInput, file_hash, load_boxes, unchanged
from .models import SUCCESS, JSONObject, SampleResult, json_object, label_text


def atomic_write(path: Path, content: bytes) -> None:
    """Flush a complete file before replacing its visible name."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_json(path: Path, value: object) -> None:
    atomic_write(path, (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode())


class RunStore:
    """Own one output directory under a process lock; source datasets are never modified."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.lock: IO[bytes] | None = None

    def __enter__(self) -> Self:
        if (
            self.root.exists()
            and any(self.root.iterdir())
            and not (self.root / "run.json").is_file()
        ):
            raise ValueError("Output must be empty or a resumable VLM run directory")
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = (self.root / ".vlm.lock").open("a+b")
        try:
            fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            for name in (".tmp", ".exports", "samples", "events", "logs"):
                folder = self.root / name
                if folder.is_symlink():
                    raise ValueError("Run-owned directories must not be symlinks")
                folder.mkdir(exist_ok=True)
            self.clean_temporary()
        except BaseException:
            self.lock.close()
            self.lock = None
            raise
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        try:
            self.clean_temporary()
        finally:
            if self.lock:
                self.lock.close()
                self.lock = None

    def clean_temporary(self) -> None:
        """Only remove files owned by this run, never inputs or other runs' caches."""
        folder = self.root / ".tmp"
        if folder.is_dir() and not folder.is_symlink():
            for child in folder.iterdir():
                if child.is_dir() and not child.is_symlink():
                    shutil.rmtree(child)
                else:
                    child.unlink()
        for name in ("samples",):
            for temporary in (self.root / name).glob(".*.tmp"):
                temporary.unlink()

    def save(self, result: SampleResult) -> None:
        write_json(
            self.root / "samples" / f"{result.sample_id}.json", result.model_dump(mode="json")
        )

    def load(self, sample_id: str) -> SampleResult | None:
        path = self.root / "samples" / f"{sample_id}.json"
        return SampleResult.model_validate_json(path.read_text()) if path.exists() else None

    def event(self, sample_id: str, attempt: int, value: JSONObject) -> None:
        entry = {"time": datetime.now(UTC).isoformat(), "attempt": attempt, **value}
        with (self.root / "events" / f"{sample_id}.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def verify_publication(self) -> None:
        """Detect manual edits before replacing a previously published snapshot."""
        pointer = self.root / "dataset"
        if not pointer.exists() and not pointer.is_symlink():
            return
        if not pointer.is_symlink() or pointer.resolve().parent != self.root / ".exports":
            raise ValueError("Unexpected dataset publication path")
        manifest_path = pointer / ".vlm-export.json"
        if not manifest_path.is_file():
            raise ValueError("Published dataset is incomplete")
        manifest = json_object(json.loads(manifest_path.read_text()))
        files = json_object(manifest["files"])
        for relative, expected in files.items():
            path = pointer / relative
            if path.resolve().is_relative_to(pointer.resolve()) is False:
                raise ValueError("Unsafe published manifest path")
            if not path.is_file() or file_hash(path) != expected:
                raise ValueError("Published dataset was modified or is incomplete")

    def publication_mapping(self) -> list[JSONObject]:
        """Reuse a verified complete snapshot without copying all images on resume."""
        self.verify_publication()
        manifest = json_object(json.loads((self.root / "dataset/.vlm-export.json").read_text()))
        rows = manifest.get("mapping")
        if not isinstance(rows, list):
            raise TypeError("Invalid publication mapping")
        return [json_object(row) for row in rows]

    def publish(self, data: DatasetInput, results: list[SampleResult]) -> list[JSONObject]:
        """Build and validate a snapshot, then atomically replace a relative dataset symlink."""
        successful = {r.sample_id: r for r in results if r.status in SUCCESS}
        if not successful:
            return []
        self.verify_publication()
        stage = self.root / ".tmp" / "dataset"
        stage.mkdir()
        splits: dict[str, list[str]] = {}
        files: dict[str, str] = {}
        mapping: list[JSONObject] = []
        for sample in data.samples:
            result = successful.get(sample.sample_id)
            if result is None:
                continue
            if (result.status == "empty") != (not result.boxes):
                raise ValueError("Successful state disagrees with annotation emptiness")
            if not unchanged(sample):
                raise ValueError("Source changed before export")
            text = label_text(result.boxes)
            for split in sample.splits:
                relative = Path(split) / sample.sample_id / sample.relative
                image_relative = Path("images") / relative
                label_relative = (Path("labels") / relative).with_suffix(".txt")
                target = stage / image_relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(sample.image, target)
                with target.open("rb") as stream:
                    os.fsync(stream.fileno())
                if file_hash(target) != sample.image_hash:
                    raise ValueError("Image changed during export")
                atomic_write(stage / label_relative, text.encode())
                parsed = load_boxes(
                    stage / label_relative, data.names, reference=True, rounding_tolerance=1e-12
                )
                if len(parsed) != len(result.boxes):
                    raise ValueError("Exported label count differs")
                splits.setdefault(split, []).append("./" + image_relative.as_posix())
                files[image_relative.as_posix()] = sample.image_hash
                files[label_relative.as_posix()] = file_hash(stage / label_relative)
                mapping.append(
                    json_object(
                        {
                            "sample_id": sample.sample_id,
                            "source": str(sample.image),
                            "split": split,
                            "image": image_relative.as_posix(),
                            "label": label_relative.as_posix(),
                            "status": result.status,
                        }
                    )
                )
        configuration: dict[str, object] = {"path": ".", "names": data.names}
        for split, items in sorted(splits.items()):
            path = stage / f"{split}.txt"
            atomic_write(path, ("\n".join(items) + "\n").encode())
            configuration[split] = path.name
            files[path.name] = file_hash(path)
        atomic_write(
            stage / "data.yaml", yaml.safe_dump(configuration, allow_unicode=True).encode()
        )
        files["data.yaml"] = file_hash(stage / "data.yaml")
        write_json(stage / ".vlm-export.json", {"files": files, "mapping": mapping})
        snapshot = self.root / ".exports" / uuid4().hex
        os.replace(stage, snapshot)
        sync_directory(snapshot.parent)
        temporary_link = self.root / ".tmp" / "dataset-link"
        temporary_link.symlink_to(Path(".exports") / snapshot.name)
        os.replace(temporary_link, self.root / "dataset")
        sync_directory(self.root)
        for old in snapshot.parent.iterdir():
            if old != snapshot and old.is_dir() and not old.is_symlink():
                shutil.rmtree(old)
        return mapping
