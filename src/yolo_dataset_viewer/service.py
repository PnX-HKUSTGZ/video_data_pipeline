"""One cancellable dataset session and bounded, on-demand image decoding."""

import logging
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import ExitStack
from io import BytesIO
from pathlib import Path
from threading import BoundedSemaphore, Event, Lock, RLock
from typing import Literal
from uuid import uuid4

import yaml
from PIL import Image, ImageOps, UnidentifiedImageError

from .dataset import Cancelled, Dataset, DatasetError, Sample, build_index, file_stamp
from .edit_models import DraftStatus, EditRequest
from .editing import DraftEditor, DraftError, dataset_lock
from .exports import (
    Exporter,
    ExportPreview,
    ExportRequest,
    ExportStatus,
    RecoveryInfo,
    discover_recoveries,
    finalize_backup,
    recover_backup,
    recovery_request,
)
from .labels import read_labels
from .models import IndexStatus, Issue, LabelState, OpenRequest, SampleDetail, SamplePage, Stats

logger = logging.getLogger(__name__)


class SessionError(ValueError):
    """The requested dataset was replaced, is not ready, or changed on disk."""


class ViewerService:
    """Own worker lifetime and never publish results from an obsolete indexing task."""

    def __init__(self, cache_bytes: int = 32 * 1024 * 1024) -> None:
        self._lock = Lock()
        self._edit_lock = RLock()
        self._decoders = BoundedSemaphore(2)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="yolo-index")
        self._cancel = Event()
        self._future: Future[None] | None = None
        self._status: IndexStatus | None = None
        self._dataset: Dataset | None = None
        self._editor: DraftEditor | None = None
        self._request: OpenRequest | None = None
        self._recovery_root: Path | None = None
        self._export: ExportStatus | None = None
        self._export_session: str | None = None
        self._export_cancel = Event()
        self._export_worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="yolo-export")
        self._cache: OrderedDict[tuple[str, int, tuple[int, int] | None], bytes] = OrderedDict()
        self._cache_size = 0
        self._cache_limit = cache_bytes

    def open(self, request: OpenRequest) -> IndexStatus:
        self._ensure_not_exporting()
        with self._edit_lock:
            return self._open(request)

    def _ensure_not_exporting(self) -> None:
        with self._lock:
            if self._export and self._export.state == "running":
                raise SessionError("正在导出，请等待完成后再编辑或切换数据集")

    def _open(self, request: OpenRequest) -> IndexStatus:
        """Replace the index while the mutation lock excludes queued writes."""
        with self._lock:
            if self._export and self._export.state == "running":
                raise SessionError("正在导出，请等待完成后再切换数据集")
            self._cancel.set()
            if self._future:
                self._future.cancel()
            self._cancel = Event()
            self._dataset = None
            if self._editor:
                self._editor.close()
            self._editor = None
            self._request = request.model_copy(deep=True)
            self._recovery_root = None
            self._cache.clear()
            self._cache_size = 0
            self._status = IndexStatus(
                id=uuid4().hex,
                state="indexing",
                phase="解析数据集",
                task=request.task,
                request=request.model_copy(deep=True),
            )
            status = self._status.model_copy(deep=True)
            self._future = self._executor.submit(self._load, status.id, request, self._cancel)
            return status

    def _load(self, session_id: str, request: OpenRequest, cancel: Event) -> None:
        def progress(phase: str, scanned: int, total: int) -> None:
            with self._lock:
                if self._status and self._status.id == session_id:
                    self._status.phase, self._status.scanned, self._status.total = (
                        phase,
                        scanned,
                        total,
                    )

        try:
            recovery_root = self._locate_root(request)
            with self._lock:
                if self._status is None or self._status.id != session_id or cancel.is_set():
                    return
                self._recovery_root = recovery_root
                if recovery_root and discover_recoveries(recovery_root):
                    self._status.recovery_available = True
                    raise DatasetError("发现未完成的原地导出，请先恢复备份再打开数据集")
            dataset = build_index(request, cancel, progress)
            editor = DraftEditor(dataset, request) if request.task in ("detect", "pose") else None
            with self._edit_lock, self._lock:
                if cancel.is_set() or self._status is None or self._status.id != session_id:
                    if editor:
                        editor.close()
                    return
                self._dataset = dataset
                self._recovery_root = dataset.spec.root
                self._editor = editor
                self._status.state = "ready"
                self._status.phase = "索引完成"
                self._status.scanned = self._status.total = len(dataset.samples)
                self._status.names = dataset.names
                self._status.splits = list(dataset.spec.sources)
                self._status.warnings = dataset.warnings
            logger.info(
                "Indexed %s: %d samples (%s)", request.path, len(dataset.samples), request.task
            )
        except Cancelled:
            return
        except Exception as error:
            logger.exception("Cannot open %s", request.path)
            with self._lock:
                if self._status and self._status.id == session_id:
                    self._status.state = "error"
                    self._status.error = str(error)
                    if isinstance(error, DatasetError):
                        self._status.candidates = error.candidates

    @staticmethod
    def _locate_root(request: OpenRequest) -> Path | None:
        """Find recovery journals even when an interrupted export changed the task schema."""
        entry = Path(request.path).expanduser().resolve()
        if entry.is_dir():
            candidates = sorted(p for p in entry.iterdir() if p.suffix.lower() in (".yaml", ".yml"))
            if len(candidates) == 1:
                entry = candidates[0]
        base = entry.parent if entry.is_file() else entry
        root_value = request.root
        if root_value is None and entry.is_file():
            try:
                data = yaml.safe_load(entry.read_text(encoding="utf-8-sig"))
                if isinstance(data, dict) and isinstance(data.get("path"), str):
                    root_value = str(data["path"])
            except (OSError, UnicodeError, yaml.YAMLError):
                return base
        root = Path(root_value).expanduser() if root_value else base
        return (root if root.is_absolute() else base / root).resolve()

    def status(self, session_id: str) -> IndexStatus:
        with self._lock:
            if self._status is None or self._status.id != session_id:
                raise SessionError("数据集已切换，请重新打开")
            return self._status.model_copy(deep=True)

    def dataset(self, session_id: str) -> Dataset:
        with self._lock:
            if not self._status or self._status.id != session_id:
                raise SessionError("数据集已切换，请重新打开")
            if self._dataset is None:
                raise SessionError("数据集尚未完成索引")
            return self._dataset

    def sample(self, session_id: str, sample_id: int) -> tuple[Dataset, Sample]:
        dataset = self.dataset(session_id)
        if sample_id < 0 or sample_id >= len(dataset.samples):
            raise LookupError("图片编号不存在")
        return dataset, dataset.samples[sample_id]

    def editor(self, session_id: str) -> DraftEditor:
        """Only detection and pose sessions expose mutation endpoints."""
        self.dataset(session_id)
        if self._editor is None:
            raise SessionError("分割与 OBB 当前仅支持浏览")
        return self._editor

    def draft(self, session_id: str) -> DraftStatus:
        with self._edit_lock:
            return self.editor(session_id).status()

    def edit(self, session_id: str, body: EditRequest) -> DraftStatus:
        self._ensure_not_exporting()
        with self._edit_lock:
            self._ensure_not_exporting()
            if self.recoveries(session_id):
                raise SessionError("发现未完成的原地导出，请先恢复备份")
            return self.editor(session_id).apply(body)

    def history(
        self, session_id: str, action: Literal["undo", "redo", "discard"], revision: int
    ) -> DraftStatus:
        self._ensure_not_exporting()
        with self._edit_lock:
            self._ensure_not_exporting()
            if self.recoveries(session_id):
                raise SessionError("发现未完成的原地导出，请先恢复备份")
            editor = self.editor(session_id)
            if action == "undo":
                return editor.undo(revision)
            if action == "redo":
                return editor.redo(revision)
            if self._request is None:
                raise SessionError("数据集尚未打开")
            # Rebuild first so discarding a conflicted draft establishes a fresh baseline.
            dataset = build_index(self._request, Event(), lambda *_: None)
            replacement = DraftEditor(dataset, self._request)
            result = replacement.discard(revision)
            with self._lock:
                editor.close()
                self._dataset, self._editor = dataset, replacement
                if self._status:
                    self._status.id = uuid4().hex
                    result.session_id = self._status.id
                    self._status.names = dataset.names
                    self._status.splits = list(dataset.spec.sources)
                    self._status.scanned = self._status.total = len(dataset.samples)
            return result

    def page(
        self,
        session_id: str,
        split: str | None,
        class_id: int | None,
        state: LabelState | None,
        search: str,
        offset: int,
        limit: int,
        edit_state: str | None = None,
    ) -> SamplePage:
        """Read metadata from one effective draft revision, without loading images."""
        with self._edit_lock:
            dataset = self.dataset(session_id)
            if self._editor:
                editor = self._editor
                selected = editor.selected(split, class_id, state, search, edit_state)
                revision = editor.revision
            else:
                selected = [s.info() for s in dataset.select(split, class_id, state, search)]
                revision = 0
            return SamplePage(
                revision=revision,
                items=selected[offset : offset + limit],
                total=len(selected),
                offset=offset,
                limit=limit,
            )

    def stats(
        self,
        session_id: str,
        split: str | None,
        class_id: int | None,
        state: LabelState | None,
        search: str,
        edit_state: str | None = None,
    ) -> Stats:
        with self._edit_lock:
            dataset = self.dataset(session_id)
            if self._editor:
                editor = self._editor
                result = editor.stats(editor.selected(split, class_id, state, search, edit_state))
                return result
            return dataset.stats(dataset.select(split, class_id, state, search))

    def detail(self, session_id: str, sample_id: int) -> SampleDetail:
        with self._edit_lock:
            return self._detail(session_id, sample_id)

    def _detail(self, session_id: str, sample_id: int) -> SampleDetail:
        dataset, sample = self.sample(session_id, sample_id)
        revision = self._editor.status().revision if self._editor else 0
        if file_stamp(sample.label) != sample.label_stamp:
            raise SessionError("标签文件已变化，请重新扫描数据集以更新统计")
        labels = (
            self._editor.labels(sample)
            if self._editor
            else read_labels(
                sample.label, dataset.spec.task, dataset.spec.names, dataset.spec.kpt_shape
            )
        )
        info = self._editor.info(sample) if self._editor else sample.info()
        width: int | None = None
        height: int | None = None
        image_error: str | None = None
        try:
            self._check_image_path(sample)
            with self._decoders, Image.open(sample.image) as source:
                width, height = source.size
                if source.getexif().get(274, 1) in (5, 6, 7, 8):
                    width, height = height, width
                # Decode only this sample; truncated files become per-image diagnostics.
                source.load()
        except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError) as error:
            image_error = f"图片无法读取：{error}"
            labels.issues.append(Issue(file=str(sample.image), message=image_error))
        return SampleDetail(
            **info.model_dump(),
            revision=revision,
            label_path=str(sample.label),
            width=width,
            height=height,
            annotations=labels.annotations,
            issues=labels.issues,
            image_error=image_error,
        )

    @staticmethod
    def _check_image_path(sample: Sample) -> None:
        if sample.image.resolve() != sample.image:
            raise ValueError("图片链接目标已变化，请重新扫描")

    def image(self, session_id: str, sample_id: int, thumbnail: bool) -> bytes:
        _, sample = self.sample(session_id, sample_id)
        self._check_image_path(sample)
        key = (session_id, sample_id, file_stamp(sample.image))
        if thumbnail:
            with self._lock:
                cached = self._cache.get(key)
                if cached is not None:
                    self._cache.move_to_end(key)
                    return cached
        with self._decoders, Image.open(sample.image) as source:
            # Both resolutions use the same EXIF-corrected coordinate frame.
            oriented = ImageOps.exif_transpose(source)
            try:
                if thumbnail:
                    oriented.thumbnail((360, 240))
                with oriented.convert("RGB") as rgb:
                    output = BytesIO()
                    rgb.save(output, format="JPEG", quality=82 if thumbnail else 95)
                    data = output.getvalue()
            finally:
                oriented.close()
        if thumbnail and len(data) <= self._cache_limit:
            with self._lock:
                if self._status and self._status.id == session_id:
                    previous = self._cache.pop(key, None)
                    self._cache_size -= len(previous) if previous else 0
                    self._cache[key] = data
                    self._cache_size += len(data)
                    while self._cache_size > self._cache_limit:
                        _, evicted = self._cache.popitem(last=False)
                        self._cache_size -= len(evicted)
        return data

    def preview_export(self, session_id: str, body: ExportRequest) -> ExportPreview:
        self._ensure_not_exporting()
        with self._edit_lock:
            self._ensure_not_exporting()
            dataset, editor = self.dataset(session_id), self.editor(session_id)
            if self._request is None:
                raise SessionError("数据集尚未打开")
            return Exporter(dataset, editor, self._request).preview(body)

    def start_export(self, session_id: str, body: ExportRequest) -> ExportStatus:
        """Freeze one revision for a background export; status uses a separate lock."""
        self._ensure_not_exporting()
        with self._edit_lock:
            editor = self.editor(session_id)
            if editor.status().revision != body.expected_revision:
                raise DraftError("草稿版本已变化，请刷新后重新预检")
            with self._lock:
                if self._export and self._export.state == "running":
                    raise SessionError("已有导出任务正在运行")
                self._export_cancel = Event()
                self._export = ExportStatus(id=uuid4().hex, state="running", phase="导出预检")
                self._export_session = session_id
                result = self._export.model_copy(deep=True)
            self._export_worker.submit(self._run_export, session_id, body)
            return result

    def export_status(self, session_id: str, export_id: str) -> ExportStatus:
        with self._lock:
            if (
                self._export is None
                or self._export.id != export_id
                or self._export_session != session_id
            ):
                raise LookupError("导出任务不存在")
            return self._export.model_copy(deep=True)

    def _run_export(self, session_id: str, body: ExportRequest) -> None:
        def progress(phase: str, completed: int, total: int) -> None:
            with self._lock:
                if self._export:
                    self._export.phase = phase
                    self._export.completed, self._export.total = completed, total

        backup: Path | None = None
        original_request: OpenRequest | None = None
        with self._edit_lock, ExitStack() as export_context:
            try:
                dataset, editor = self.dataset(session_id), self.editor(session_id)
                export_context.enter_context(dataset_lock(dataset.spec.root, exclusive=True))
                original_request = self._request
                if original_request is None:
                    raise SessionError("数据集尚未打开")
                result = Exporter(dataset, editor, original_request).run(
                    body, progress, self._export_cancel
                )
                reopen: OpenRequest | None = None
                if body.mode == "inplace":
                    backup = Path(result.backup_path) if result.backup_path else None
                    editor.clear_after_export()
                    reopen = OpenRequest(path=result.config_path, task=result.task)
                    self._replace_after_export(session_id, reopen, progress)
                    if backup:
                        finalize_backup(backup)
                with self._lock:
                    if body.mode == "inplace" and self._status:
                        # Invalidate old integer sample IDs and zero-revision clients after rebase.
                        self._status.id = uuid4().hex
                    if self._export:
                        self._export.result = result
                        self._export.reopen = reopen
                        self._export.phase = "导出完成"
                        self._export.state = "complete"
            except Exception as error:
                logger.exception("Dataset export failed")
                recovery_error = ""
                if backup and original_request:
                    try:
                        recover_backup(self.dataset(session_id).spec.root, backup)
                        self._replace_after_export(session_id, original_request, progress)
                    except Exception as rollback_error:
                        logger.exception("Export backup recovery failed")
                        recovery_error = f"；请恢复备份 {backup}：{rollback_error}"
                with self._lock:
                    if self._export:
                        self._export.state = (
                            "cancelled" if isinstance(error, Cancelled) else "error"
                        )
                        self._export.phase = "导出未完成"
                        self._export.error = str(error) + recovery_error

    def _replace_after_export(
        self,
        session_id: str,
        request: OpenRequest,
        progress: "Callable[[str, int, int], None]",
    ) -> None:
        """Rebase only after files were validated; preserve the current HTTP session."""
        dataset = build_index(request, Event(), progress)
        editor = DraftEditor(dataset, request)
        with self._lock:
            if self._status is None or self._status.id != session_id:
                editor.close()
                raise SessionError("导出期间数据集会话发生变化")
            if self._editor:
                self._editor.close()
            self._dataset, self._editor, self._request = dataset, editor, request
            self._recovery_root = dataset.spec.root
            self._cache.clear()
            self._cache_size = 0
            self._status.names = dataset.names
            self._status.task = request.task
            self._status.request = request.model_copy(deep=True)
            self._status.splits = list(dataset.spec.sources)
            self._status.scanned = self._status.total = len(dataset.samples)

    def recoveries(self, session_id: str) -> list[RecoveryInfo]:
        self.status(session_id)
        with self._lock:
            root = self._recovery_root
        return discover_recoveries(root) if root else []

    def recover(self, session_id: str, revision: int, backup_path: str) -> IndexStatus:
        self._ensure_not_exporting()
        with self._edit_lock:
            self._ensure_not_exporting()
            self.status(session_id)
            if (self._editor.status().revision if self._editor else 0) != revision:
                raise DraftError("草稿版本已变化，请刷新后重试")
            root = self._recovery_root
            if root is None:
                raise SessionError("数据集尚未打开")
            with dataset_lock(root, exclusive=True):
                request = recovery_request(root, Path(backup_path))
                recover_backup(root, Path(backup_path))
                return self._open(request)

    def close(self) -> None:
        with self._lock:
            self._cancel.set()
            self._export_cancel.set()
        self._export_worker.shutdown(wait=True, cancel_futures=True)
        self._executor.shutdown(wait=True, cancel_futures=True)
        with self._lock:
            self._dataset = None
            if self._editor:
                self._editor.close()
            self._editor = None
            self._cache.clear()
            self._cache_size = 0
