"""Portable snapshot export and recoverable, explicitly requested source updates.

Exporting never changes the draft. The service freezes edits while ``run`` executes,
then retires only the active workspace after a successful in-place reindex.
"""

import hashlib
import os
import shutil
import sqlite3
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from glob import escape
from pathlib import Path
from threading import Event
from typing import Literal, Protocol
from urllib.parse import quote
from uuid import uuid4

import yaml
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from .dataset import SPLITS, Dataset, DatasetError, Sample, build_index, check_cancel, file_stamp
from .edit_models import DraftStatus
from .editing import dataset_lock, restore_workspace_from_backup
from .models import Detection, Labels, OpenRequest, Pose, SampleInfo


class ExportRequest(BaseModel):
    """Export the exact revision acknowledged by the editing client."""

    model_config = ConfigDict(extra="forbid")
    expected_revision: int = Field(ge=0)
    mode: Literal["new", "inplace"]
    destination: str | None = None


class ExportClass(BaseModel):
    old_id: int
    new_id: int
    name: str
    keypoints: list[str]


class ExportAction(BaseModel):
    path: str
    action: Literal["create", "overwrite", "delete"]


class ExportPreview(BaseModel):
    revision: int
    mode: Literal["new", "inplace"]
    output_path: str
    backup_path: str | None = None
    task: Literal["detect", "pose"]
    kpt_shape: tuple[int, Literal[3]] | None = None
    classes: list[ExportClass]
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    required_bytes: int = 0
    available_bytes: int = 0
    images: int = 0
    deleted_images: int = 0
    deleted_objects: int = 0
    actions: list[ExportAction] = Field(default_factory=list)


class ExportResult(BaseModel):
    output_path: str
    config_path: str
    backup_path: str | None = None
    task: Literal["detect", "pose"]
    images: int
    revision: int


class ExportStatus(BaseModel):
    id: str
    state: Literal["running", "complete", "error", "cancelled"] = "running"
    phase: str = "预检"
    completed: int = 0
    total: int = 0
    error: str | None = None
    result: ExportResult | None = None
    reopen: OpenRequest | None = None


class RecoveryInfo(BaseModel):
    path: str
    state: str
    created_at: str
    files: int


class _BackupEntry(BaseModel):
    path: str
    stored: str | None
    original_digest: str | None
    generated_digest: str | None


class _RecoveryManifest(BaseModel):
    version: Literal[1] = 1
    root: str
    config_path: str
    original_request: OpenRequest
    draft_workspace: str
    created_at: str
    state: Literal["prepared", "applying", "applied", "complete", "failed", "recovered"]
    entries: list[_BackupEntry]
    created_dirs: list[str] = Field(default_factory=list)
    completed: int = 0
    error: str | None = None


class DraftSource(Protocol):
    """The exporter only needs the merged read view, never the database internals."""

    path: Path
    workspace: str

    def status(self) -> DraftStatus: ...
    def check_sources(self) -> list[str]: ...
    def labels(self, sample: Sample) -> Labels: ...
    def info(self, sample: Sample) -> SampleInfo: ...
    def sample_key(self, sample: Sample) -> str: ...


@dataclass
class _Plan:
    preview: ExportPreview
    writes: dict[Path, bytes]
    copies: dict[Path, Path]
    deletes: set[Path]
    stamps: dict[Path, tuple[int, int] | None]
    config: Path
    instances: int = 0


def _inside(path: Path, root: Path) -> bool:
    return path.resolve().is_relative_to(root.resolve())


def _free_space(path: Path) -> int:
    while not path.exists():
        path = path.parent
    return shutil.disk_usage(path).free


def _file_digest(path: Path) -> str | None:
    """Stream originals, including large images, without retaining their contents."""
    try:
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()
    except FileNotFoundError:
        return None


def _sync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write(path: Path, content: bytes) -> None:
    """Flush each complete replacement before publication; never expose partial text."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            shutil.copymode(path, temporary)
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _save_manifest(folder: Path, manifest: _RecoveryManifest) -> None:
    _atomic_write(folder / "manifest.json", manifest.model_dump_json(indent=2).encode())


def _read_manifest(root: Path, folder: Path) -> _RecoveryManifest:
    folder = folder.resolve()
    root = root.resolve()
    if folder.parent != root.parent or not folder.name.startswith(f".{root.name}.viewer-backup-"):
        raise DatasetError("备份必须位于数据集相邻的 viewer 备份目录")
    manifest = _RecoveryManifest.model_validate_json((folder / "manifest.json").read_bytes())
    if Path(manifest.root).resolve() != root:
        raise DatasetError("备份所属数据集与当前数据集不一致")
    for entry in manifest.entries:
        if not _inside(Path(entry.path), root):
            raise DatasetError(f"备份记录包含根目录外路径：{entry.path}")
        if entry.stored is not None and not _inside(folder / entry.stored, folder):
            raise DatasetError("备份记录包含非法存储路径")
    if any(not _inside(Path(path), root) for path in manifest.created_dirs):
        raise DatasetError("备份记录包含非法目录")
    return manifest


def discover_recoveries(root: Path) -> list[RecoveryInfo]:
    """List interrupted applications; completed and rolled-back backups remain archived."""
    root = root.resolve()
    result: list[RecoveryInfo] = []
    for folder in sorted(root.parent.glob(f".{escape(root.name)}.viewer-backup-*")):
        if not folder.is_dir() or not (folder / "manifest.json").is_file():
            continue
        try:
            manifest = _read_manifest(root, folder)
        except (OSError, ValueError):
            continue
        if manifest.state in ("prepared", "applying", "applied", "failed"):
            result.append(
                RecoveryInfo(
                    path=str(folder),
                    state=manifest.state,
                    created_at=manifest.created_at,
                    files=len(manifest.entries),
                )
            )
    return result


def recover_backup(root: Path, backup_path: str | Path) -> str:
    """Restore every original, including unattempted writes, after interrupted application."""
    with dataset_lock(root, exclusive=True):
        return _recover_backup_locked(root, backup_path)


def _recover_backup_locked(root: Path, backup_path: str | Path) -> str:
    folder = Path(backup_path).resolve()
    manifest = _read_manifest(root, folder)
    if manifest.state not in ("prepared", "applying", "applied", "failed"):
        raise DatasetError("该备份没有待恢复的应用操作")
    # Verify all backups before removing or replacing a single destination.
    for entry in manifest.entries:
        if entry.stored is not None and not (folder / entry.stored).is_file():
            raise DatasetError(f"备份文件缺失：{folder / entry.stored}")
        if (
            entry.stored is not None
            and _file_digest(folder / entry.stored) != entry.original_digest
        ):
            raise DatasetError(f"备份文件内容变化：{folder / entry.stored}")
        if _file_digest(Path(entry.path)) not in {entry.original_digest, entry.generated_digest}:
            raise DatasetError(f"导出后文件被外部修改，停止恢复以免覆盖：{entry.path}")
    draft = folder / "draft.sqlite3"
    database = root / ".yolo-viewer-draft.sqlite3"
    if draft.exists() and manifest.state == "applied":
        restore_workspace_from_backup(database, draft, manifest.draft_workspace, check_only=True)
    for entry in reversed(manifest.entries):
        target = Path(entry.path)
        if entry.stored is None:
            target.unlink(missing_ok=True)
        else:
            saved = folder / entry.stored
            _atomic_write(target, saved.read_bytes())
            shutil.copystat(saved, target)
    # Restore only this workspace; other configurations may have their own newer drafts.
    if draft.exists() and manifest.state == "applied":
        restore_workspace_from_backup(database, draft, manifest.draft_workspace)
    for path in sorted(manifest.created_dirs, key=len, reverse=True):
        try:
            Path(path).rmdir()
        except OSError:
            pass
    manifest.state = "recovered"
    _save_manifest(folder, manifest)
    return manifest.config_path


def recovery_request(root: Path, backup_path: str | Path) -> OpenRequest:
    """Use the original task and root override when reopening restored source files."""
    return _read_manifest(root, Path(backup_path)).original_request


def finalize_backup(backup_path: str | Path) -> None:
    """Mark an in-place export complete only after its new baseline has been indexed."""
    folder = Path(backup_path)
    manifest = _RecoveryManifest.model_validate_json((folder / "manifest.json").read_bytes())
    if manifest.state != "applied":
        raise DatasetError("备份不处于等待索引完成的状态")
    manifest.state = "complete"
    _save_manifest(folder, manifest)


class Exporter:
    """Build and validate a fixed merged snapshot before creating output files."""

    def __init__(self, dataset: Dataset, editor: DraftSource, request: OpenRequest) -> None:
        self.dataset = dataset
        self.editor = editor
        self.request = request

    def preview(self, request: ExportRequest) -> ExportPreview:
        return self._plan(request, Event(), lambda *_: None).preview

    def _plan(
        self, request: ExportRequest, cancel: Event, progress: Callable[[str, int, int], None]
    ) -> _Plan:
        status = self.editor.status()
        root = self.dataset.spec.root.resolve()
        output = (
            root
            if request.mode == "inplace"
            else Path(request.destination or ".").expanduser().absolute()
        )
        classes = [
            ExportClass(
                old_id=cls.id,
                new_id=index,
                name=cls.name,
                keypoints=[point.name for point in cls.keypoints],
            )
            for index, cls in enumerate(status.classes)
        ]
        maximum = max((len(cls.keypoints) for cls in classes), default=0)
        preview = ExportPreview(
            revision=status.revision,
            mode=request.mode,
            output_path=str(output),
            task="pose" if maximum else "detect",
            classes=classes,
            kpt_shape=(maximum, 3) if maximum else None,
        )
        plan = _Plan(preview, {}, {}, set(), {}, output / "data.yaml")
        errors = preview.errors
        if status.revision != request.expected_revision:
            errors.append("草稿修订已变化，请刷新后重新预检")
        errors.extend(self.editor.check_sources())
        if not classes:
            errors.append("至少保留一个类别才能导出")
        if self.dataset.spec.task not in ("detect", "pose"):
            errors.append("仅检测和姿态数据支持导出编辑")
        config = self.dataset.spec.config_path
        if request.mode == "new":
            if not request.destination or not request.destination.strip():
                errors.append("请填写新数据集输出目录")
            if output.is_symlink() or _inside(output, root) or _inside(root, output):
                errors.append("输出目录不能位于源数据集内部、包含源数据集或为符号链接")
            if output.exists() and (not output.is_dir() or any(output.iterdir())):
                errors.append("输出目录已存在且非空，不能覆盖")
        else:
            if config and not _inside(config, root):
                errors.append(f"配置位于根目录外，只能导出新数据集：{config}")
            for sources in self.dataset.spec.sources.values():
                for source in sources:
                    if not _inside(source, root):
                        errors.append(f"划分来源位于根目录外，只能导出新数据集：{source}")
            plan.config = config.resolve() if config else root / "data.yaml"
            if config is None and plan.config.exists():
                errors.append(f"新配置路径已有未登记文件：{plan.config}")
            pending = discover_recoveries(root)
            if pending:
                errors.append(f"存在未完成的原地应用，请先恢复：{pending[0].path}")
        mapping = {cls.old_id: cls.new_id for cls in classes}
        points = {cls.old_id: len(cls.keypoints) for cls in classes}
        seen: set[str] = set()
        live_images: set[Path] = set()
        removed_images: set[Path] = set()
        splits: dict[str, list[Path]] = {
            ("train" if split == "all" else split): [] for split in self.dataset.spec.sources
        }
        exported: dict[str, Path | None] = {}
        source_stamps: dict[Path, tuple[int, int] | None] = {}
        for index, sample in enumerate(self.dataset.samples):
            check_cancel(cancel)
            if index % 100 == 0:
                progress("验证图片与标注", index, len(self.dataset.samples))
            key = self.editor.sample_key(sample)
            if key in seen:
                previous = exported[key]
                if previous is not None:
                    splits["train" if sample.split == "all" else sample.split].append(previous)
                continue
            seen.add(key)
            info = self.editor.info(sample)
            if request.mode == "inplace":
                for source in (sample.image, sample.label):
                    if not _inside(source, root):
                        errors.append(f"文件位于根目录外，只能导出新数据集：{source}")
            source_stamps[sample.image] = file_stamp(sample.image)
            source_stamps[sample.label] = file_stamp(sample.label)
            if (root / sample.display_path).resolve() != sample.image:
                errors.append(f"图片链接目标已变化，请重新扫描：{sample.display_path}")
            if info.deleted:
                exported[key] = None
                removed_images.add(sample.image)
                preview.deleted_objects += sum(sample.counts.values())
                if request.mode == "inplace":
                    plan.deletes.update((sample.image.resolve(), sample.label.resolve()))
                continue
            live_images.add(sample.image)
            labels = self.editor.labels(sample)
            plan.instances += len(labels.annotations)
            errors.extend(
                f"{issue.file}:{issue.line or '-'} {issue.message}" for issue in labels.issues
            )
            preview.deleted_objects += max(0, sum(sample.counts.values()) - len(labels.annotations))
            try:
                with Image.open(sample.image) as image:
                    image.load()
            except (OSError, ValueError, Image.DecompressionBombError) as error:
                errors.append(f"图片无法读取：{sample.image}：{error}")
            content = self._label_bytes(labels, mapping, points, maximum)
            if request.mode == "new":
                relative = Path(sample.display_path)
                if relative.is_absolute() or ".." in relative.parts:
                    relative = Path("external") / sample.image.name
                # YOLO replaces the last images component when deriving labels.
                # Omit source layout markers so the exported top-level pair is authoritative.
                relative = Path(
                    *(part for part in relative.parts[:-1] if part != "images"), relative.name
                )
                digest = hashlib.sha256(key.encode()).hexdigest()[:16]
                image_path = output / "images" / digest / relative
                label = output / "labels" / digest / relative.with_suffix(".txt")
                plan.copies[image_path] = sample.image
            else:
                # Keep the lexical image path to preserve symlink-based label pairing.
                image_path = root / sample.display_path
                label = sample.label.resolve()
            if label in plan.writes and plan.writes[label] != content:
                errors.append(f"多个目标共享同一标签文件但编辑不同，请导出新数据集：{label}")
            plan.writes[label] = content
            exported[key] = image_path
            splits["train" if sample.split == "all" else sample.split].append(image_path)
        preview.images = len(plan.copies) if request.mode == "new" else len(live_images)
        preview.deleted_images = len(removed_images - live_images)
        plan.stamps = source_stamps
        if removed_images & live_images:
            errors.append("同一实际图片对应多组标签，删除状态冲突；请统一删除状态后导出")
        if plan.deletes.intersection(plan.writes):
            errors.append("删除样本与保留样本共用标签，无法原地应用；请导出新数据集")
        if request.mode == "inplace":
            # Use fresh manifests so another YAML's shared TXT sources are not rewritten.
            manifest_dir = output / ".yolo-viewer-splits" / f"revision-{status.revision}"
            if manifest_dir.exists():
                manifest_dir = (
                    output / ".yolo-viewer-splits" / f"revision-{status.revision}-{uuid4().hex[:8]}"
                )
        else:
            manifest_dir = output
        original = self.dataset.spec.config_data
        replaced_fields = {
            *SPLITS,
            "path",
            "names",
            "nc",
            "task",
            "download",
            "kpt_shape",
            "kpt_names",
            "flip_idx",
            "skeleton",
            "keypoint_weights",
            "sigmas",
        }
        config_data: dict[str, object] = {
            key: value for key, value in original.items() if key not in replaced_fields
        }
        config_data.update(
            {
                "path": os.path.relpath(output, plan.config.parent),
                "names": {cls.new_id: cls.name for cls in classes},
                "nc": len(classes),
            }
        )
        if "task" in original:
            config_data["task"] = plan.preview.task
        for split, images in splits.items():
            manifest_path = manifest_dir / f"{split}.txt"
            text = "".join(
                f"./{os.path.relpath(path, manifest_path.parent)}\n"
                for path in dict.fromkeys(images)
            )
            plan.writes[manifest_path] = text.encode("utf-8")
            config_data[split] = os.path.relpath(manifest_path, output)
        if maximum:
            config_data["kpt_shape"] = [maximum, 3]
            config_data["kpt_names"] = {cls.new_id: cls.keypoints for cls in classes}
        original_names = original.get("kpt_names")
        original_count = self.dataset.spec.kpt_shape[0] if self.dataset.spec.kpt_shape else 0
        structure_changed = maximum != original_count
        for cls in status.classes:
            names = (
                original_names.get(cls.id, original_names.get(str(cls.id)))
                if isinstance(original_names, dict)
                else None
            )
            count = len(names) if isinstance(names, list) else original_count
            structure_changed |= [point.id for point in cls.keypoints] != [
                f"base-{cls.id}-{index}" for index in range(count)
            ]
        removed_fields = [
            field
            for field in ("flip_idx", "skeleton", "keypoint_weights", "sigmas")
            if field in original and structure_changed
        ]
        if not structure_changed:
            for field in ("flip_idx", "skeleton", "keypoint_weights", "sigmas"):
                if field in original:
                    config_data[field] = original[field]
        if removed_fields:
            preview.warnings.append(
                "导出移除位置相关配置，请根据新的关键点列表检查训练设置："
                + ", ".join(removed_fields)
            )
        if self.dataset.spec.kpt_shape and self.dataset.spec.kpt_shape[1] == 2 and maximum:
            preview.warnings.append("二维关键点转为三维记录：有效点可见，缺失点未标注")
        if any(len(cls.keypoints) != maximum for cls in classes) and maximum:
            preview.warnings.append("各类别关键点数量不同；不足槽位填充为 0 0 0")
        plan.writes[plan.config] = yaml.safe_dump(
            config_data, allow_unicode=True, sort_keys=False
        ).encode()
        for path in plan.writes.keys() | plan.copies.keys() | plan.deletes:
            if request.mode == "inplace" and not _inside(path, root):
                errors.append(f"输出文件链接到根目录外，不能原地应用：{path}")
            if path.exists() and not path.is_file():
                errors.append(f"输出文件与已有目录冲突：{path}")
            parent = path.parent
            while not parent.exists():
                parent = parent.parent
            if not parent.is_dir():
                errors.append(f"输出文件的父目录被已有文件占用：{parent}")
            action: Literal["create", "overwrite", "delete"] = (
                "delete" if path in plan.deletes else "overwrite" if path.exists() else "create"
            )
            preview.actions.append(ExportAction(path=str(path), action=action))
        preview.actions.sort(key=lambda action: action.path)
        written_bytes = sum(map(len, plan.writes.values()))
        if request.mode == "new":
            written_bytes += sum((file_stamp(path) or (0, 0))[1] for path in plan.copies.values())
        else:
            written_bytes += sum(
                (file_stamp(path) or (0, 0))[1] for path in plan.writes.keys() | plan.deletes
            )
            written_bytes += (file_stamp(self.editor.path) or (0, 0))[1]
            written_bytes += max(map(len, plan.writes.values()), default=0)
        preview.required_bytes = written_bytes + 1024 * 1024
        try:
            preview.available_bytes = _free_space(output.parent)
            if preview.available_bytes < preview.required_bytes:
                errors.append("可用磁盘空间不足以完成导出和备份")
        except OSError as error:
            errors.append(f"无法检查输出目录磁盘空间：{error}")
        preview.errors = list(dict.fromkeys(errors))
        return plan

    @staticmethod
    def _label_bytes(
        labels: Labels, mapping: dict[int, int], points: dict[int, int], maximum: int
    ) -> bytes:
        rows: list[str] = []
        for annotation in labels.annotations:
            if not isinstance(annotation, (Detection, Pose)) or annotation.class_id not in mapping:
                raise DatasetError("导出标注包含不支持的任务或已删除类别")
            box = annotation.box
            values = [
                str(mapping[annotation.class_id]),
                *(format(value, ".12g") for value in (box.cx, box.cy, box.width, box.height)),
            ]
            for index in range(maximum):
                if isinstance(annotation, Pose) and index < min(
                    points[annotation.class_id], len(annotation.keypoints)
                ):
                    point = annotation.keypoints[index]
                    values.extend(
                        (format(point.x, ".12g"), format(point.y, ".12g"), str(point.visibility))
                        if point.visibility
                        else ("0", "0", "0")
                    )
                else:
                    values.extend(("0", "0", "0"))
            rows.append(" ".join(values))
        return ("\n".join(rows) + ("\n" if rows else "")).encode()

    def run(
        self, request: ExportRequest, progress: Callable[[str, int, int], None], cancel: Event
    ) -> ExportResult:
        """Revalidate, stage, and publish; callers must prevent edits until this returns."""
        with dataset_lock(self.dataset.spec.root, exclusive=True):
            return self._run_locked(request, progress, cancel)

    def _run_locked(
        self, request: ExportRequest, progress: Callable[[str, int, int], None], cancel: Event
    ) -> ExportResult:
        plan = self._plan(request, cancel, progress)
        if plan.preview.errors:
            raise DatasetError("\n".join(plan.preview.errors))
        self._check_snapshot(plan)
        if request.mode == "new":
            self._new(plan, progress, cancel)
            backup: Path | None = None
        else:
            backup = self._inplace(plan, progress, cancel)
        return ExportResult(
            output_path=plan.preview.output_path,
            config_path=str(plan.config),
            backup_path=str(backup) if backup else None,
            task=plan.preview.task,
            images=plan.preview.images,
            revision=plan.preview.revision,
        )

    def _check_snapshot(self, plan: _Plan) -> None:
        if self.editor.status().revision != plan.preview.revision:
            raise DatasetError("草稿在导出期间发生变化")
        conflicts = self.editor.check_sources()
        conflicts.extend(
            f"源文件在导出期间发生变化：{path}"
            for path, stamp in plan.stamps.items()
            if file_stamp(path) != stamp
        )
        conflicts.extend(
            f"图片链接目标在导出期间变化：{sample.display_path}"
            for sample in self.dataset.samples
            if (self.dataset.spec.root / sample.display_path).resolve() != sample.image
        )
        if conflicts:
            raise DatasetError("\n".join(conflicts))

    def _new(self, plan: _Plan, progress: Callable[[str, int, int], None], cancel: Event) -> None:
        output = Path(plan.preview.output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.viewer-export-", dir=output.parent))
        try:
            total = len(plan.writes) + len(plan.copies)
            for index, (path, source) in enumerate(plan.copies.items()):
                check_cancel(cancel)
                destination = stage / path.relative_to(output)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                _sync_file(destination)
                progress("复制图片", index + 1, total)
            for index, (path, content) in enumerate(plan.writes.items(), len(plan.copies)):
                check_cancel(cancel)
                _atomic_write(stage / path.relative_to(output), content)
                progress("写入标签与配置", index + 1, total)
            self._validate(stage / plan.config.relative_to(output), plan, cancel)
            self._check_snapshot(plan)
            check_cancel(cancel)
            if output.exists() and (not output.is_dir() or any(output.iterdir())):
                raise DatasetError("输出目录在导出期间变为非空，停止发布")
            os.replace(stage, output)
            _sync_directory(output.parent)
            progress("导出完成", total, total)
        finally:
            if stage.exists():
                shutil.rmtree(stage)

    def _validate(self, config: Path, plan: _Plan, cancel: Event) -> None:
        result = build_index(
            OpenRequest(path=str(config), task=plan.preview.task), cancel, lambda *_: None
        )
        if any(sample.issue_count or sample.state == "missing" for sample in result.samples):
            raise DatasetError("导出后校验发现非法或缺失标签")
        if len({sample.image for sample in result.samples}) != plan.preview.images:
            raise DatasetError("导出后图片数量不一致")
        unique = {(sample.image, sample.label): sample for sample in result.samples}
        if sum(sum(sample.counts.values()) for sample in unique.values()) != plan.instances:
            raise DatasetError("导出后有效目标数量不一致")

    def _inplace(
        self, plan: _Plan, progress: Callable[[str, int, int], None], cancel: Event
    ) -> Path:
        root = self.dataset.spec.root.resolve()
        folder = root.parent / f".{root.name}.viewer-backup-{uuid4().hex}"
        folder.mkdir()
        paths = sorted(plan.writes.keys() | plan.deletes)
        manifest = _RecoveryManifest(
            root=str(root),
            config_path=str(self.dataset.spec.config_path or root),
            original_request=self.request,
            draft_workspace=self.editor.workspace,
            created_at=datetime.now(UTC).isoformat(),
            state="prepared",
            entries=[],
        )
        try:
            if self.editor.path.exists():
                with (
                    sqlite3.connect(
                        f"file:{quote(str(self.editor.path))}?mode=ro", uri=True
                    ) as source_db,
                    sqlite3.connect(folder / "draft.sqlite3") as backup_db,
                ):
                    source_db.backup(backup_db)
                _sync_file(folder / "draft.sqlite3")
            for index, path in enumerate(paths):
                check_cancel(cancel)
                stored: str | None = None
                if path.exists():
                    stored = f"files/{index}"
                    saved = folder / stored
                    saved.parent.mkdir(exist_ok=True)
                    shutil.copy2(path, saved)
                    _sync_file(saved)
                manifest.entries.append(
                    _BackupEntry(
                        path=str(path),
                        stored=stored,
                        original_digest=_file_digest(folder / stored) if stored else None,
                        generated_digest=hashlib.sha256(plan.writes[path]).hexdigest()
                        if path in plan.writes
                        else None,
                    )
                )
                progress("备份原文件", index + 1, len(paths))
            if (folder / "files").exists():
                _sync_directory(folder / "files")
            missing_dirs: set[Path] = set()
            for path in plan.writes:
                parent = path.parent
                while not parent.exists() and _inside(parent, root):
                    missing_dirs.add(parent)
                    parent = parent.parent
            manifest.created_dirs = [str(path) for path in missing_dirs]
            # Serialize staged data first; no registered source file has changed yet.
            stage = folder / "staged"
            stage.mkdir()
            staged: dict[Path, Path] = {}
            for index, (path, content) in enumerate(plan.writes.items()):
                check_cancel(cancel)
                staged[path] = stage / str(index)
                _atomic_write(staged[path], content)
            self._check_snapshot(plan)
            check_cancel(cancel)
            _save_manifest(folder, manifest)
            manifest.state = "applying"
            _save_manifest(folder, manifest)
            try:
                for index, path in enumerate(paths):
                    check_cancel(cancel)
                    if path in plan.deletes:
                        path.unlink(missing_ok=True)
                    else:
                        _atomic_write(path, staged[path].read_bytes())
                    manifest.completed = index + 1
                    _save_manifest(folder, manifest)
                    progress("应用修改", index + 1, len(paths))
                self._validate(plan.config, plan, cancel)
                manifest.state = "applied"
                _save_manifest(folder, manifest)
            except BaseException as error:
                manifest.state = "failed"
                manifest.error = str(error)
                try:
                    _save_manifest(folder, manifest)
                except OSError:
                    # The already durable applying manifest also permits safe rollback.
                    pass
                try:
                    recover_backup(root, folder)
                except Exception as rollback_error:
                    raise DatasetError(
                        f"应用失败：{error}；恢复未完成：{rollback_error}；请保留备份：{folder}"
                    ) from rollback_error
                raise
            finally:
                shutil.rmtree(stage, ignore_errors=True)
        except BaseException:
            # Incomplete backup preparation never changed the source and can be discarded.
            if not (folder / "manifest.json").exists():
                shutil.rmtree(folder)
            raise
        return folder
