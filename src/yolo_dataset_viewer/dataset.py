"""Resolve portable dataset layouts and build a small, image-free annotation index."""

import os
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event

import yaml

from .labels import read_labels
from .models import ClassCount, LabelState, OpenRequest, SampleInfo, Stats, Task

IMAGE_SUFFIXES = {
    ".bmp",
    ".dng",
    ".jpeg",
    ".jpg",
    ".mpo",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
    ".pfm",
    ".heic",
    ".avif",
    ".gif",
    ".ppm",
    ".pgm",
    ".pnm",
}
SPLITS = ("train", "val", "test", "valid", "validation", "minival")


class DatasetError(ValueError):
    """A recoverable input error with optional YAML choices for the open dialog."""

    def __init__(self, message: str, candidates: list[str] | None = None) -> None:
        super().__init__(message)
        self.candidates = candidates or []


class Cancelled(Exception):
    """Cooperative cancellation of discovery or indexing."""


def check_cancel(event: Event) -> None:
    if event.is_set():
        raise Cancelled


@dataclass(frozen=True)
class DatasetSpec:
    root: Path
    task: Task
    names: dict[int, str]
    kpt_shape: tuple[int, int] | None
    sources: dict[str, list[Path]]
    config_path: Path | None = None
    config_data: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class Sample:
    id: int
    image: Path
    label: Path
    display_path: str
    split: str
    state: LabelState
    counts: dict[int, int]
    issue_count: int
    label_stamp: tuple[int, int] | None

    def info(self) -> SampleInfo:
        return SampleInfo(
            id=self.id,
            path=self.display_path,
            split=self.split,
            state=self.state,
            instances=sum(self.counts.values()),
            class_ids=sorted(self.counts),
            issue_count=self.issue_count,
        )


@dataclass
class Dataset:
    spec: DatasetSpec
    samples: list[Sample] = field(default_factory=list)
    names: dict[int, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def select(
        self, split: str | None, class_id: int | None, state: LabelState | None, search: str
    ) -> list[Sample]:
        query = search.casefold()
        return [
            s
            for s in self.samples
            if (split is None or s.split == split)
            and (class_id is None or class_id in s.counts)
            and (state is None or s.state == state)
            and (not query or query in s.display_path.casefold())
        ]

    def stats(self, samples: list[Sample]) -> Stats:
        instances: Counter[int] = Counter()
        images: Counter[int] = Counter()
        for sample in samples:
            instances.update(sample.counts)
            images.update(sample.counts.keys())
        split_counts = Counter(s.split for s in samples)
        state_counts: Counter[str] = Counter(s.state for s in samples)
        return Stats(
            images=len(samples),
            unique_images=len({s.image for s in samples}),
            instances=sum(instances.values()),
            splits={split: split_counts[split] for split in self.spec.sources},
            states={
                state: state_counts[state] for state in ("valid", "empty", "missing", "invalid")
            },
            classes=[
                ClassCount(id=c, name=name, instances=instances[c], images=images[c])
                for c, name in sorted(self.names.items())
            ],
            invalid_rows=sum(s.issue_count for s in samples),
        )


def _mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(k, str) for k in value):
        raise DatasetError("数据集 YAML 必须为字段映射")
    return {str(k): v for k, v in value.items()}


def _names(data: dict[str, object]) -> dict[int, str]:
    raw, nc = data.get("names"), data.get("nc")
    if nc is not None and (type(nc) is not int or nc <= 0):
        raise DatasetError("nc 必须为正整数")
    names: dict[int, str] = {}
    if isinstance(raw, list):
        if not all(isinstance(v, str) for v in raw):
            raise DatasetError("names 的每个名称必须为字符串")
        names = {i: str(v) for i, v in enumerate(raw)}
    elif isinstance(raw, dict):
        for key, value in raw.items():
            if (
                not isinstance(key, (int, str))
                or isinstance(key, bool)
                or not isinstance(value, str)
            ):
                raise DatasetError("names 必须映射整数类别编号到名称")
            try:
                number = int(key)
            except ValueError as error:
                raise DatasetError("names 类别编号必须为整数") from error
            if number in names:
                raise DatasetError("names 存在重复类别编号")
            names[number] = value
    elif raw is not None:
        raise DatasetError("names 必须为名称列表或整数映射")
    if raw is not None and (not names or sorted(names) != list(range(len(names)))):
        raise DatasetError("names 必须包含从 0 开始的连续类别编号")
    if nc is not None and names and nc != len(names):
        raise DatasetError("nc 与 names 数量不一致")
    if not names and isinstance(nc, int):
        names = {i: f"class_{i}" for i in range(nc)}
    return names


def _path(value: object, base: Path) -> Path:
    if not isinstance(value, str) or not value.strip() or "://" in value:
        raise DatasetError("数据集路径必须是非空的本地路径")
    path = Path(value).expanduser()
    # Preserve an images-directory symlink until its corresponding labels path is derived.
    return Path(os.path.abspath(path if path.is_absolute() else base / path))


def resolve_spec(request: OpenRequest) -> DatasetSpec:
    """Interpret YAML-relative paths without reading global training-library settings."""
    entry = Path(request.path).expanduser().resolve()
    if not entry.exists():
        raise DatasetError(f"路径不存在：{entry}")
    if entry.is_dir():
        candidates = sorted(
            p for p in entry.iterdir() if p.is_file() and p.suffix.lower() in {".yaml", ".yml"}
        )
        if len(candidates) > 1:
            raise DatasetError("目录内有多个 YAML，请选择数据集配置", [str(p) for p in candidates])
        if candidates:
            entry = candidates[0]
    data: dict[str, object] = {}
    if entry.is_file():
        if entry.suffix.lower() not in {".yaml", ".yml"}:
            raise DatasetError("请选择数据集目录或 YAML 文件")
        try:
            data = _mapping(yaml.safe_load(entry.read_text(encoding="utf-8-sig")))
        except (yaml.YAMLError, UnicodeError) as error:
            raise DatasetError(f"无法解析 YAML：{error}") from error
        root = _path(request.root or data.get("path") or ".", entry.parent)
    else:
        root = _path(request.root, entry) if request.root else entry
    if not root.is_dir():
        raise DatasetError(f"数据集根目录不存在：{root}；可使用根目录覆盖")
    names = _names(data)
    shape: tuple[int, int] | None = None
    if request.task == "pose":
        raw_shape: object = request.kpt_shape or data.get("kpt_shape")
        if not isinstance(raw_shape, (tuple, list)) or len(raw_shape) != 2:
            raise DatasetError("姿态任务需要 kpt_shape，例如 [17, 3]；请在网页补充")
        count, dims = raw_shape
        if type(count) is not int or count <= 0 or type(dims) is not int or dims not in (2, 3):
            raise DatasetError("kpt_shape 必须为 [正整数, 2 或 3]")
        shape = (count, dims)
    sources: dict[str, list[Path]] = {}
    for split in SPLITS:
        value = data.get(split)
        if value is not None:
            values = value if isinstance(value, list) else [value]
            sources[split] = [_path(v, root) for v in values]
    if not sources:
        if entry.is_file():
            raise DatasetError("YAML 未声明 train、val 或 test 等图片来源")
        for split in SPLITS:
            choices = [p for p in (root / "images" / split, root / split / "images") if p.is_dir()]
            if len(choices) > 1:
                raise DatasetError(f"{split} 同时存在两种目录布局，请用 YAML 显式指定")
            if choices:
                sources[split] = choices
            elif (root / split).is_dir():
                sources[split] = [root / split]
        if not sources:
            sources["all"] = [root / "images" if (root / "images").is_dir() else root]
    return DatasetSpec(
        root=root,
        task=request.task,
        names=names,
        kpt_shape=shape,
        sources=sources,
        config_path=entry if entry.is_file() else None,
        config_data=data,
    )


def label_path(image: Path) -> Path:
    """Replace the last images component, retaining nested names and split directories."""
    parts = list(image.parts)
    positions = [i for i, part in enumerate(parts[:-1]) if part == "images"]
    if positions:
        parts[positions[-1]] = "labels"
    return Path(*parts).with_suffix(".txt")


def file_stamp(path: Path) -> tuple[int, int] | None:
    try:
        stat = path.stat()
        return stat.st_mtime_ns, stat.st_size
    except OSError:
        return None


def _images(source: Path, root: Path, cancel: Event) -> Iterator[Path]:
    if source.is_dir():

        def walk_error(error: OSError) -> None:
            raise error

        for folder, dirs, files in os.walk(source, onerror=walk_error, followlinks=False):
            check_cancel(cancel)
            dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d != "labels")
            for filename in sorted(files):
                check_cancel(cancel)
                path = Path(folder) / filename
                if path.suffix.lower() in IMAGE_SUFFIXES:
                    yield path.absolute()
    elif source.is_file() and source.suffix.lower() == ".txt":
        with source.open(encoding="utf-8-sig") as stream:
            for number, text in enumerate(stream, 1):
                check_cancel(cancel)
                text = text.strip()
                if not text:
                    continue
                path = Path(text).expanduser()
                if path.is_absolute():
                    resolved = path
                elif text.startswith(("./", "../")):
                    resolved = (source.parent / path).absolute()
                else:
                    candidates = {
                        Path(os.path.abspath(base / path))
                        for base in (source.parent, root)
                        if (base / path).is_file()
                    }
                    if len(candidates) > 1:
                        raise DatasetError(f"{source}:{number} 图片路径存在多解：{text}")
                    resolved = next(iter(candidates), (source.parent / path).absolute())
                if not resolved.is_file():
                    raise DatasetError(f"{source}:{number} 图片不存在：{resolved}")
                if resolved.suffix.lower() not in IMAGE_SUFFIXES:
                    raise DatasetError(f"{source}:{number} 不支持的图片扩展名：{resolved.suffix}")
                yield resolved
    elif source.is_file() and source.suffix.lower() in IMAGE_SUFFIXES:
        yield source
    else:
        raise DatasetError(f"图片来源不存在或不是目录/图片清单：{source}")


def build_index(
    request: OpenRequest, cancel: Event, progress: Callable[[str, int, int], None]
) -> Dataset:
    """Scan paths and labels only; discard annotation geometry after counting each sample."""
    spec = resolve_spec(request)
    dataset = Dataset(spec=spec, names=dict(spec.names))
    paths: list[tuple[str, Path]] = []
    for split, sources in spec.sources.items():
        seen: set[Path] = set()
        for source in sources:
            for image in _images(source, spec.root, cancel):
                check_cancel(cancel)
                normalized = Path(os.path.normpath(image))
                if normalized not in seen:
                    seen.add(normalized)
                    paths.append((split, normalized))
                    if len(paths) % 100 == 0:
                        progress("发现图片", len(paths), 0)
        if not seen:
            dataset.warnings.append(f"划分 {split} 没有图片")
    progress("读取标签", 0, len(paths))
    for split, image in paths:
        check_cancel(cancel)
        label = label_path(image)
        stamp = file_stamp(label)
        labels = read_labels(label, spec.task, spec.names, spec.kpt_shape)
        if file_stamp(label) != stamp:
            raise DatasetError(f"标签在索引期间发生变化，请重新扫描：{label}")
        counts = dict(Counter(a.class_id for a in labels.annotations))
        for class_id in counts:
            dataset.names.setdefault(class_id, f"class_{class_id}")
        display = os.path.relpath(image, spec.root)
        dataset.samples.append(
            Sample(
                id=len(dataset.samples),
                image=image.resolve(),
                label=label,
                display_path=display,
                split=split,
                state=labels.state,
                counts=counts,
                issue_count=len(labels.issues),
                label_stamp=stamp,
            )
        )
        if len(dataset.samples) % 100 == 0:
            progress("读取标签", len(dataset.samples), len(paths))
    check_cancel(cancel)
    return dataset
