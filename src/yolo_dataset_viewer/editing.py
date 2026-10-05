"""Transactional dataset overlays: source files stay untouched until an explicit export."""

import fcntl
import hashlib
import math
import os
import sqlite3
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import local
from uuid import uuid4

from pydantic import BaseModel, Field

from .dataset import Dataset, Sample, file_stamp
from .edit_models import (
    AnnotationDelete,
    AnnotationUpsert,
    ClassAdd,
    ClassDefinition,
    ClassDelete,
    ClassRename,
    DraftStatus,
    EditOperation,
    EditRequest,
    IssueDelete,
    KeypointDefinition,
    KeypointsSet,
    SampleDelete,
)
from .labels import read_labels
from .models import (
    Annotation,
    Box,
    ClassCount,
    Detection,
    Issue,
    Keypoint,
    Labels,
    LabelState,
    OpenRequest,
    Pose,
    SampleInfo,
    Stats,
)


class DraftError(ValueError):
    """An edit cannot safely be accepted; the caller can retain and retry it."""


class DraftConflict(DraftError):
    """The source dataset or draft revision changed underneath an editor."""


@dataclass
class _HeldLock:
    descriptor: int
    exclusive: bool
    depth: int = 1


class _ThreadLocks(local):
    def __init__(self) -> None:
        self.held: dict[str, _HeldLock] = {}


_thread_locks = _ThreadLocks()


@contextmanager
def dataset_lock(root: Path, *, exclusive: bool = False) -> Iterator[None]:
    """Nonblocking, reentrant process lock using the dataset directory itself.

    Draft transactions share the directory lock; export/recovery exclusively hold
    it across source mutation and reindexing. No lock file is created or retained.
    """
    key = str(root.resolve())
    held = _thread_locks.held.get(key)
    if held is not None:
        if exclusive and not held.exclusive:
            raise DraftConflict("不能在草稿事务内开始导出/恢复")
        held.depth += 1
        try:
            yield
        finally:
            held.depth -= 1
        return
    try:
        descriptor = os.open(key, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    except OSError as error:
        raise DraftError(f"无法打开数据集目录锁：{error}") from error
    try:
        try:
            fcntl.flock(descriptor, (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise DraftConflict("数据集正在导出/恢复，请稍后重试") from error
        except OSError as error:
            raise DraftError(f"无法锁定数据集目录：{error}") from error
        _thread_locks.held[key] = _HeldLock(descriptor, exclusive)
        try:
            yield
        finally:
            _thread_locks.held.pop(key, None)
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


class Fingerprint(BaseModel):
    stamp: tuple[int, int] | None
    digest: str | None = None


class StoredObject(BaseModel):
    uid: str
    class_id: int
    line: int
    raw: str
    box: Box
    points: dict[str, Keypoint] = Field(default_factory=dict)


class StoredSample(BaseModel):
    deleted: bool = False
    objects: list[StoredObject] = Field(default_factory=list)
    issues: list[Issue] = Field(default_factory=list)
    missing: bool = False


class Patch(BaseModel):
    """History stores just the schema or one changed image, never the whole dataset."""

    classes: list[ClassDefinition] | None = None
    key: str | None = None
    sample: StoredSample | None = None


class Baseline(BaseModel):
    files: dict[str, Fingerprint]
    classes: list[ClassDefinition]


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _point(value: Keypoint, *, source: bool = False) -> Keypoint:
    if not all(math.isfinite(v) for v in (value.x, value.y)):
        raise DraftError("关键点坐标必须为有限数值")
    if not source and not all(0 <= v <= 1 for v in (value.x, value.y)):
        raise DraftError("关键点坐标必须在 [0, 1] 范围内")
    if value.visibility == 0:
        return Keypoint(x=0, y=0, visibility=0)
    if not all(math.isfinite(v) and 0 <= v <= 1 for v in (value.x, value.y)):
        raise DraftError("关键点坐标必须是 [0, 1] 范围内的有限数值")
    return value.model_copy(deep=True)


def _validate_box(box: Box) -> None:
    values = (box.cx, box.cy, box.width, box.height)
    if not all(math.isfinite(v) and 0 <= v <= 1 for v in values):
        raise DraftError("框坐标必须是 [0, 1] 范围内的有限数值")
    if box.width <= 0 or box.height <= 0:
        raise DraftError("框的宽高必须大于 0")
    if (
        min(box.cx - box.width / 2, box.cy - box.height / 2) < -1e-9
        or max(box.cx + box.width / 2, box.cy + box.height / 2) > 1 + 1e-9
    ):
        raise DraftError("框边界超出图片")


def _name(value: str, existing: list[str]) -> str:
    result = value.strip()
    if not result:
        raise DraftError("名称不能为空")
    if result in existing:
        raise DraftError(f"名称不能重复：{result}")
    return result


class DraftEditor:
    """Combine immutable indexed metadata with small, durable per-image differences."""

    def __init__(self, dataset: Dataset, request: OpenRequest) -> None:
        if dataset.spec.task not in ("detect", "pose"):
            raise DraftError("只有检测和 pose 数据集支持编辑")
        self.dataset = dataset
        self.request = request
        self.path = dataset.spec.root / ".yolo-viewer-draft.sqlite3"
        entry = dataset.spec.config_path or dataset.spec.root.resolve()
        self.workspace = _digest(
            f"{entry}\0{dataset.spec.root.resolve()}\0{request.task}\0{request.kpt_shape}"
        )
        self._samples = {sample.id: sample for sample in dataset.samples}
        self._keys = {sample.id: self.sample_key(sample) for sample in dataset.samples}
        self._unique = {self._keys[sample.id]: sample for sample in dataset.samples}
        self._base_classes = self._initial_classes()
        self._observed: dict[str, Fingerprint] = {}
        paths: dict[Path, bool] = {}
        if dataset.spec.config_path is not None:
            paths[dataset.spec.config_path] = True
        for sources in dataset.spec.sources.values():
            for source in sources:
                if source.is_file() and source.suffix.lower() == ".txt":
                    paths[source] = True
        for sample in self._unique.values():
            paths[sample.image] = False
            paths[sample.label] = True
        self._hash_paths = {str(path) for path, content in paths.items() if content}
        for path, content in paths.items():
            self._observed[str(path)] = self._fingerprint(path, content)
        self._fresh_baseline = Baseline(files=self._observed.copy(), classes=self._base_classes)
        self._baseline = self._fresh_baseline.model_copy(deep=True)
        self._classes = self._base_classes
        self._overrides: dict[str, StoredSample] = {}
        self._revision = 0
        self._cursor = 0
        self._last = 0
        self._load()

    def _initial_classes(self) -> list[ClassDefinition]:
        names = self.dataset.spec.config_data.get("kpt_names")
        count = self.dataset.spec.kpt_shape[0] if self.dataset.spec.kpt_shape else 0
        classes: list[ClassDefinition] = []
        for class_id, name in sorted(self.dataset.names.items()):
            raw: object = (
                names.get(class_id, names.get(str(class_id))) if isinstance(names, dict) else None
            )
            if raw is None:
                point_names = [f"keypoint_{index}" for index in range(count)]
            elif isinstance(raw, list) and all(isinstance(item, str) for item in raw):
                point_names = [str(item) for item in raw]
                if (
                    len(point_names) > count
                    or any(not item.strip() for item in point_names)
                    or len(set(point_names)) != len(point_names)
                ):
                    raise DraftError(f"类别 {class_id} 的 kpt_names 无效")
            else:
                raise DraftError(f"类别 {class_id} 的 kpt_names 必须为名称列表")
            classes.append(
                ClassDefinition(
                    id=class_id,
                    name=name,
                    keypoints=[
                        KeypointDefinition(id=f"base-{class_id}-{index}", name=point_name)
                        for index, point_name in enumerate(point_names)
                    ],
                )
            )
        return classes

    @property
    def revision(self) -> int:
        """Revision of the current in-memory snapshot, without an implicit reload."""
        return self._revision

    @staticmethod
    def sample_key(sample: Sample) -> str:
        """A split-independent identifier survives rescanning and duplicate memberships."""
        return _digest(f"{sample.image.resolve()}\0{sample.label.resolve()}")

    @staticmethod
    def _fingerprint(path: Path, content: bool) -> Fingerprint:
        stamp = file_stamp(path)
        digest: str | None = None
        if content and stamp is not None:
            try:
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError as error:
                digest = f"error:{error}"
        return Fingerprint(stamp=stamp, digest=digest)

    @contextmanager
    def _connection(self, *, write: bool = False) -> Iterator[sqlite3.Connection]:
        if write:
            with (
                dataset_lock(self.dataset.spec.root),
                self._sqlite_connection(write=True) as connection,
            ):
                yield connection
        else:
            with self._sqlite_connection() as connection:
                yield connection

    @contextmanager
    def _sqlite_connection(self, *, write: bool = False) -> Iterator[sqlite3.Connection]:
        """Scope connections so rollback journals disappear before returning to the UI."""
        if self.path.is_symlink():
            raise DraftError("草稿文件不能是符号链接")
        connection: sqlite3.Connection | None = None
        try:
            uri = self.path.resolve().as_uri() + ("?mode=rwc" if write else "?mode=ro")
            connection = sqlite3.connect(uri, uri=True, timeout=10, isolation_level=None)
            if write:
                connection.execute("PRAGMA synchronous=FULL")
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS workspaces (id TEXT PRIMARY KEY, revision INTEGER NOT NULL, cursor INTEGER NOT NULL, classes TEXT NOT NULL, baseline TEXT NOT NULL)"
                )
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS samples (workspace TEXT NOT NULL, key TEXT NOT NULL, data TEXT NOT NULL, PRIMARY KEY(workspace,key))"
                )
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS history (workspace TEXT NOT NULL, seq INTEGER NOT NULL, before_patch TEXT NOT NULL, after_patch TEXT NOT NULL, PRIMARY KEY(workspace,seq))"
                )
            else:
                # One read snapshot must cover revision, schema, overrides and history.
                connection.execute("BEGIN")
            yield connection
            if write:
                connection.execute("COMMIT")
        except sqlite3.Error as error:
            raise DraftError(f"无法保存或读取草稿：{error}") from error
        finally:
            if connection is not None:
                if connection.in_transaction:
                    connection.execute("ROLLBACK")
                connection.close()

    def _load(self, connection: sqlite3.Connection | None = None) -> None:
        if connection is None:
            if not self.path.exists():
                return
            with self._connection() as opened:
                self._load(opened)
            return
        if not connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='workspaces'"
        ).fetchone():
            return
        row = connection.execute(
            "SELECT revision,cursor,classes,baseline FROM workspaces WHERE id=?", (self.workspace,)
        ).fetchone()
        if row is None:
            self._baseline = self._fresh_baseline.model_copy(deep=True)
            self._base_classes = self._baseline.classes
            self._revision = self._cursor = self._last = 0
            self._classes = [item.model_copy(deep=True) for item in self._base_classes]
            self._overrides = {}
            return
        self._revision, self._cursor = row[0], row[1]
        self._baseline = Baseline.model_validate_json(row[3])
        self._base_classes = self._baseline.classes
        self._classes = Patch.model_validate_json(row[2]).classes or []
        self._overrides = {
            str(key): StoredSample.model_validate_json(value)
            for key, value in connection.execute(
                "SELECT key,data FROM samples WHERE workspace=?", (self.workspace,)
            )
        }
        self._last = connection.execute(
            "SELECT COALESCE(MAX(seq),0) FROM history WHERE workspace=?", (self.workspace,)
        ).fetchone()[0]

    def _ensure(self, connection: sqlite3.Connection) -> None:
        connection.execute(
            "INSERT OR IGNORE INTO workspaces VALUES (?,?,?,?,?)",
            (
                self.workspace,
                0,
                0,
                Patch(classes=self._base_classes).model_dump_json(),
                self._baseline.model_dump_json(),
            ),
        )

    def check_sources(self, *, force: bool = True) -> list[str]:
        """Return actionable conflicts while keeping the saved overlay available."""
        conflicts: list[str] = []
        for name, expected in self._baseline.files.items():
            path = Path(name)
            previous = self._observed.get(name)
            current_stamp = file_stamp(path)
            if previous is not None and previous.stamp == current_stamp and not force:
                current = previous
            else:
                current = self._fingerprint(path, name in self._hash_paths)
                self._observed[name] = current
            if current != expected:
                conflicts.append(f"源文件已变化：{name}")
        return conflicts

    def _class(self, class_id: int) -> ClassDefinition:
        for item in self._classes:
            if item.id == class_id:
                return item
        raise DraftError(f"类别不存在：{class_id}")

    def _sample(self, sample_id: int) -> Sample:
        try:
            return self._samples[sample_id]
        except KeyError as error:
            raise DraftError(f"图片不存在：{sample_id}") from error

    def _source(self, sample: Sample) -> StoredSample:
        labels = read_labels(
            sample.label,
            self.dataset.spec.task,
            self.dataset.spec.names,
            self.dataset.spec.kpt_shape,
        )
        definitions = {item.id: item for item in self._base_classes}
        objects: list[StoredObject] = []
        key = self._keys[sample.id]
        for annotation in labels.annotations:
            if not isinstance(annotation, (Detection, Pose)):
                continue
            definition = definitions.get(annotation.class_id)
            points = annotation.keypoints if isinstance(annotation, Pose) else []
            objects.append(
                StoredObject(
                    uid=_digest(f"{key}\0{annotation.line}\0{annotation.raw}"),
                    class_id=annotation.class_id,
                    line=annotation.line,
                    raw=annotation.raw,
                    box=annotation.box,
                    points={
                        point.id: _point(value, source=True)
                        for point, value in zip(definition.keypoints if definition else [], points)
                    },
                )
            )
        return StoredSample(
            objects=objects, issues=labels.issues, missing=labels.state == "missing"
        )

    def _stored(self, sample: Sample) -> StoredSample:
        override = self._overrides.get(self._keys[sample.id])
        return override.model_copy(deep=True) if override else self._source(sample)

    def labels(self, sample: Sample) -> Labels:
        stored = self._stored(sample)
        definitions = {item.id: item for item in self._classes}
        annotations: list[Annotation] = []
        pose = any(item.keypoints for item in self._classes)
        for obj in stored.objects:
            definition = definitions.get(obj.class_id)
            if definition is None or stored.deleted:
                continue
            if pose:
                annotations.append(
                    Pose(
                        class_id=obj.class_id,
                        line=obj.line,
                        raw=obj.raw,
                        uid=obj.uid,
                        box=obj.box,
                        keypoints=[
                            obj.points.get(point.id, Keypoint(x=0, y=0, visibility=0))
                            for point in definition.keypoints
                        ],
                    )
                )
            else:
                annotations.append(
                    Detection(
                        class_id=obj.class_id, line=obj.line, raw=obj.raw, uid=obj.uid, box=obj.box
                    )
                )
        issues = [] if stored.deleted else stored.issues
        state: LabelState = (
            "invalid"
            if issues
            else "valid"
            if annotations
            else "missing"
            if stored.missing
            else "empty"
        )
        return Labels(state=state, annotations=annotations, issues=issues)

    def _counts(self, sample: Sample) -> tuple[dict[int, int], int, LabelState]:
        override = self._overrides.get(self._keys[sample.id])
        live = {item.id for item in self._classes}
        if override is not None:
            counts = (
                dict(Counter(obj.class_id for obj in override.objects if obj.class_id in live))
                if not override.deleted
                else {}
            )
            issues = len(override.issues) if not override.deleted else 0
            missing = override.missing
        else:
            counts = {key: value for key, value in sample.counts.items() if key in live}
            issues, missing = sample.issue_count, sample.state == "missing"
        state: LabelState = (
            "invalid" if issues else "valid" if counts else "missing" if missing else "empty"
        )
        return counts, issues, state

    def info(self, sample: Sample) -> SampleInfo:
        counts, issues, state = self._counts(sample)
        key = self._keys[sample.id]
        base = {item.id: item for item in self._base_classes}
        current = {item.id: item for item in self._classes}
        schema_changed = any(
            base.get(class_id) != current.get(class_id) for class_id in sample.counts
        )
        return SampleInfo(
            id=sample.id,
            path=sample.display_path,
            split=sample.split,
            state=state,
            instances=sum(counts.values()),
            class_ids=sorted(counts),
            issue_count=issues,
            key=key,
            modified=key in self._overrides or schema_changed,
            deleted=bool(key in self._overrides and self._overrides[key].deleted),
        )

    def selected(
        self,
        split: str | None,
        class_id: int | None,
        state: LabelState | None,
        search: str,
        edit_state: str | None = None,
    ) -> list[SampleInfo]:
        self._load()
        query = search.casefold()
        result: list[SampleInfo] = []
        for sample in self.dataset.samples:
            if (
                split is not None
                and sample.split != split
                or query
                and query not in sample.display_path.casefold()
            ):
                continue
            info = self.info(sample)
            if (
                edit_state == "deleted"
                and not info.deleted
                or edit_state != "deleted"
                and info.deleted
            ):
                continue
            if (
                edit_state == "modified"
                and not info.modified
                or edit_state == "unchanged"
                and info.modified
            ):
                continue
            if (
                class_id is not None
                and class_id not in info.class_ids
                or state is not None
                and state != info.state
            ):
                continue
            result.append(info)
        return result

    def stats(self, selected: list[SampleInfo]) -> Stats:
        counts: Counter[int] = Counter()
        images: Counter[int] = Counter()
        for info in selected:
            sample_counts, _, _ = self._counts(self._samples[info.id])
            counts.update(sample_counts)
            images.update(sample_counts.keys())
        split_counts = Counter(info.split for info in selected)
        states: Counter[str] = Counter(info.state for info in selected)
        return Stats(
            images=len(selected),
            unique_images=len({info.key for info in selected}),
            instances=sum(counts.values()),
            splits={split: split_counts[split] for split in self.dataset.spec.sources},
            states={state: states[state] for state in ("valid", "empty", "missing", "invalid")},
            classes=[
                ClassCount(
                    id=item.id, name=item.name, instances=counts[item.id], images=images[item.id]
                )
                for item in self._classes
            ],
            invalid_rows=sum(info.issue_count for info in selected),
            revision=self._revision,
        )

    def status(self, *, refresh: bool = True) -> DraftStatus:
        if refresh:
            self._load()
        infos = [self.info(sample) for sample in self._unique.values()]
        return DraftStatus(
            revision=self._revision,
            classes=self._classes,
            task="pose" if any(item.keypoints for item in self._classes) else "detect",
            dirty=bool(self._overrides or self._classes != self._base_classes),
            can_undo=self._cursor > 0,
            can_redo=self._cursor < self._last,
            conflicts=self.check_sources(force=False),
            modified_images=sum(info.modified for info in infos),
            deleted_images=sum(info.deleted for info in infos),
        )

    def _guard(self, expected_revision: int) -> None:
        if expected_revision != self._revision:
            raise DraftConflict(
                f"草稿版本已变化（当前 {self._revision}，请求 {expected_revision}），请刷新后重试"
            )
        conflicts = self.check_sources()
        if conflicts:
            raise DraftConflict("\n".join(conflicts))

    def _patch(self, connection: sqlite3.Connection, patch: Patch) -> None:
        if patch.classes is not None:
            self._classes = patch.classes
            connection.execute(
                "UPDATE workspaces SET classes=? WHERE id=?",
                (Patch(classes=self._classes).model_dump_json(), self.workspace),
            )
        if patch.key is not None:
            if patch.sample is None:
                self._overrides.pop(patch.key, None)
                connection.execute(
                    "DELETE FROM samples WHERE workspace=? AND key=?", (self.workspace, patch.key)
                )
            else:
                self._overrides[patch.key] = patch.sample
                connection.execute(
                    "INSERT OR REPLACE INTO samples VALUES (?,?,?)",
                    (self.workspace, patch.key, patch.sample.model_dump_json()),
                )

    def _schema_operation(
        self, op: ClassAdd | ClassRename | ClassDelete | KeypointsSet
    ) -> list[ClassDefinition]:
        classes = [item.model_copy(deep=True) for item in self._classes]
        if isinstance(op, ClassAdd):
            # UUID-backed keypoints and monotonic class IDs remain stable across renames.
            used = [item.id for item in self._base_classes]
            classes.append(
                ClassDefinition(
                    id=max(used, default=-1) + self._revision + 1,
                    name=_name(op.name, [item.name for item in classes]),
                )
            )
            return classes
        original = self._class(op.class_id)
        target = next(item for item in classes if item.id == original.id)
        if isinstance(op, ClassDelete):
            return [item for item in classes if item.id != op.class_id]
        if isinstance(op, ClassRename):
            target.name = _name(op.name, [item.name for item in classes if item.id != op.class_id])
        else:
            existing = {point.id for point in original.keypoints}
            points: list[KeypointDefinition] = []
            for point in op.keypoints:
                name = _name(point.name, [item.name for item in points])
                if point.id is not None and point.id not in existing:
                    raise DraftError("未知关键点标识；新增关键点应省略 id")
                point_id = point.id or uuid4().hex
                if any(item.id == point_id for item in points):
                    raise DraftError("关键点标识不能重复")
                points.append(KeypointDefinition(id=point_id, name=name))
            target.keypoints = points
        return classes

    def _sample_operation(
        self, op: SampleDelete | AnnotationUpsert | AnnotationDelete | IssueDelete
    ) -> tuple[str, StoredSample]:
        sample = self._sample(op.sample_id)
        stored = self._stored(sample)
        if isinstance(op, SampleDelete):
            stored.deleted = op.deleted
            return self._keys[sample.id], stored
        if stored.deleted:
            raise DraftError("请先恢复已删除的图片")
        if isinstance(op, IssueDelete):
            if not any(issue.line == op.line for issue in stored.issues):
                raise DraftError("异常行不存在")
            stored.issues = [issue for issue in stored.issues if issue.line != op.line]
        elif isinstance(op, AnnotationDelete):
            if not any(obj.uid == op.annotation_id for obj in stored.objects):
                raise DraftError("目标不存在")
            stored.objects = [obj for obj in stored.objects if obj.uid != op.annotation_id]
        else:
            definition = self._class(op.class_id)
            _validate_box(op.box)
            previous = next((obj for obj in stored.objects if obj.uid == op.annotation_id), None)
            if op.annotation_id is not None and previous is None:
                raise DraftError("目标不存在")
            if op.keypoints is not None:
                if len(op.keypoints) != len(definition.keypoints):
                    raise DraftError("关键点数量必须与类别定义一致")
                points = {
                    point.id: _point(value)
                    for point, value in zip(definition.keypoints, op.keypoints)
                }
            elif previous is None:
                points = {}
            elif previous.class_id == op.class_id:
                points = previous.points
            else:
                old_class = self._class(previous.class_id)
                by_name = {
                    point.name: previous.points[point.id]
                    for point in old_class.keypoints
                    if point.id in previous.points
                }
                points = {
                    point.id: by_name[point.name]
                    for point in definition.keypoints
                    if point.name in by_name
                }
            if op.issue_line is not None:
                if not any(issue.line == op.issue_line for issue in stored.issues):
                    raise DraftError("待替换的异常行不存在")
                stored.issues = [issue for issue in stored.issues if issue.line != op.issue_line]
            obj = StoredObject(
                uid=previous.uid if previous else uuid4().hex,
                class_id=op.class_id,
                line=previous.line if previous else op.issue_line or 0,
                raw=previous.raw if previous else "",
                box=op.box,
                points=points,
            )
            if previous is None:
                stored.objects.append(obj)
            else:
                stored.objects = [
                    obj if item.uid == previous.uid else item for item in stored.objects
                ]
        stored.missing = False
        return self._keys[sample.id], stored

    def apply(self, request: EditRequest) -> DraftStatus:
        """Compare revision and append one atomic operation with its inverse."""
        with self._connection(write=True) as connection:
            self._load(connection)
            self._guard(request.expected_revision)
            self._ensure(connection)
            op: EditOperation = request.operation
            if isinstance(op, (ClassAdd, ClassRename, ClassDelete, KeypointsSet)):
                before = Patch(classes=self._classes)
                after = Patch(classes=self._schema_operation(op))
            else:
                key, sample = self._sample_operation(op)
                before = Patch(key=key, sample=self._overrides.get(key))
                after = Patch(key=key, sample=sample)
            connection.execute(
                "DELETE FROM history WHERE workspace=? AND seq>?", (self.workspace, self._cursor)
            )
            connection.execute(
                "INSERT INTO history VALUES (?,?,?,?)",
                (
                    self.workspace,
                    self._cursor + 1,
                    before.model_dump_json(),
                    after.model_dump_json(),
                ),
            )
            self._patch(connection, after)
            connection.execute(
                "UPDATE workspaces SET revision=revision+1,cursor=cursor+1 WHERE id=?",
                (self.workspace,),
            )
        self._revision += 1
        self._cursor += 1
        self._last = self._cursor
        return self.status(refresh=False)

    def _history(self, expected_revision: int, redo: bool) -> DraftStatus:
        with self._connection(write=True) as connection:
            self._load(connection)
            self._guard(expected_revision)
            seq = self._cursor + 1 if redo else self._cursor
            row = connection.execute(
                "SELECT before_patch,after_patch FROM history WHERE workspace=? AND seq=?",
                (self.workspace, seq),
            ).fetchone()
            if row is None:
                raise DraftError("没有可重做的操作" if redo else "没有可撤销的操作")
            patch = Patch.model_validate_json(row[1 if redo else 0])
            self._patch(connection, patch)
            connection.execute(
                "UPDATE workspaces SET revision=revision+1,cursor=? WHERE id=?",
                (seq if redo else seq - 1, self.workspace),
            )
        self._revision += 1
        self._cursor = seq if redo else seq - 1
        return self.status(refresh=False)

    def undo(self, expected_revision: int) -> DraftStatus:
        return self._history(expected_revision, False)

    def redo(self, expected_revision: int) -> DraftStatus:
        return self._history(expected_revision, True)

    def discard(self, expected_revision: int) -> DraftStatus:
        """Rebase to this editor's freshly indexed source, keeping revision monotonic.

        The service constructs a new index/editor before discarding. Calling this
        on an old index safely keeps any source conflicts unresolved.
        """
        with self._connection(write=True) as connection:
            self._load(connection)
            if self._revision != expected_revision:
                raise DraftConflict("草稿版本已变化，请刷新后重试")
            self._ensure(connection)
            connection.execute("DELETE FROM history WHERE workspace=?", (self.workspace,))
            connection.execute("DELETE FROM samples WHERE workspace=?", (self.workspace,))
            connection.execute(
                "UPDATE workspaces SET revision=revision+1,cursor=0,classes=?,baseline=? WHERE id=?",
                (
                    Patch(classes=self._fresh_baseline.classes).model_dump_json(),
                    self._fresh_baseline.model_dump_json(),
                    self.workspace,
                ),
            )
        self._revision += 1
        self._cursor = self._last = 0
        self._overrides = {}
        self._baseline = self._fresh_baseline.model_copy(deep=True)
        self._base_classes = self._baseline.classes
        self._classes = [item.model_copy(deep=True) for item in self._base_classes]
        return self.status(refresh=False)

    def clear_after_export(self) -> None:
        if self.path.exists():
            with self._connection(write=True) as connection:
                for table in ("history", "samples"):
                    connection.execute(f"DELETE FROM {table} WHERE workspace=?", (self.workspace,))
                connection.execute("DELETE FROM workspaces WHERE id=?", (self.workspace,))
        self._overrides = {}

    def close(self) -> None:
        """Connections are transaction-scoped; no open handles require cleanup."""


def _restore_workspace_from_backup(
    database: Path, backup: Path, workspace: str, *, check_only: bool = False
) -> None:
    """Restore one archived workspace without overwriting edits to any other dataset.

    Recovery may replace an absent workspace or an identical archived copy. A
    different current copy signals edits made after the export and is preserved.
    """
    if database.is_symlink():
        raise DraftError("草稿文件不能是符号链接")
    layouts = {
        "workspaces": ("id", "id,revision,cursor,classes,baseline", "id"),
        "samples": ("workspace", "workspace,key,data", "key"),
        "history": ("workspace", "workspace,seq,before_patch,after_patch", "seq"),
    }

    def rows(connection: sqlite3.Connection) -> dict[str, list[tuple[object, ...]]]:
        result: dict[str, list[tuple[object, ...]]] = {}
        for table, (key, columns, order) in layouts.items():
            exists = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
            ).fetchone()
            result[table] = (
                connection.execute(
                    f"SELECT {columns} FROM {table} WHERE {key}=? ORDER BY {order}",
                    (workspace,),
                ).fetchall()
                if exists
                else []
            )
        return result

    archived: sqlite3.Connection | None = None
    current: sqlite3.Connection | None = None
    try:
        archived = sqlite3.connect(backup.resolve().as_uri() + "?mode=ro", uri=True)
        expected = rows(archived)
        archived.close()
        archived = None
        if not database.exists() and check_only:
            return
        mode = "ro" if check_only else "rwc"
        current = sqlite3.connect(
            database.resolve().as_uri() + f"?mode={mode}", uri=True, isolation_level=None
        )
        if not check_only:
            current.execute("PRAGMA synchronous=FULL")
            current.execute("BEGIN IMMEDIATE")
        actual = rows(current)
        if actual["workspaces"] and actual != expected:
            raise DraftConflict("当前配置的草稿在导出后已修改；不能覆盖，请先保存该草稿")
        if not check_only:
            current.execute(
                "CREATE TABLE IF NOT EXISTS workspaces (id TEXT PRIMARY KEY, "
                "revision INTEGER NOT NULL, cursor INTEGER NOT NULL, classes TEXT NOT NULL, "
                "baseline TEXT NOT NULL)"
            )
            current.execute(
                "CREATE TABLE IF NOT EXISTS samples (workspace TEXT NOT NULL, key TEXT NOT NULL, "
                "data TEXT NOT NULL, PRIMARY KEY(workspace,key))"
            )
            current.execute(
                "CREATE TABLE IF NOT EXISTS history (workspace TEXT NOT NULL, seq INTEGER NOT NULL, "
                "before_patch TEXT NOT NULL, after_patch TEXT NOT NULL, PRIMARY KEY(workspace,seq))"
            )
            for table, (key, columns, _) in layouts.items():
                current.execute(f"DELETE FROM {table} WHERE {key}=?", (workspace,))
                placeholders = ",".join("?" for _ in columns.split(","))
                current.executemany(
                    f"INSERT INTO {table} ({columns}) VALUES ({placeholders})", expected[table]
                )
            current.execute("COMMIT")
    except sqlite3.Error as error:
        raise DraftError(f"无法恢复草稿：{error}") from error
    finally:
        if archived is not None:
            archived.close()
        if current is not None:
            if current.in_transaction:
                current.execute("ROLLBACK")
            current.close()


def restore_workspace_from_backup(
    database: Path, backup: Path, workspace: str, *, check_only: bool = False
) -> None:
    """Restore the archived workspace while cooperating with dataset export locks."""
    with dataset_lock(database.parent):
        _restore_workspace_from_backup(database, backup, workspace, check_only=check_only)
