"""One cancellable dataset session and bounded, on-demand image decoding."""

import logging
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from io import BytesIO
from threading import BoundedSemaphore, Event, Lock
from uuid import uuid4

from PIL import Image, ImageOps, UnidentifiedImageError

from .dataset import Cancelled, Dataset, DatasetError, Sample, build_index, file_stamp
from .labels import read_labels
from .models import IndexStatus, Issue, OpenRequest, SampleDetail

logger = logging.getLogger(__name__)


class SessionError(ValueError):
    """The requested dataset was replaced, is not ready, or changed on disk."""


class ViewerService:
    """Own worker lifetime and never publish results from an obsolete indexing task."""

    def __init__(self, cache_bytes: int = 32 * 1024 * 1024) -> None:
        self._lock = Lock()
        self._decoders = BoundedSemaphore(2)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="yolo-index")
        self._cancel = Event()
        self._future: Future[None] | None = None
        self._status: IndexStatus | None = None
        self._dataset: Dataset | None = None
        self._cache: OrderedDict[tuple[str, int, tuple[int, int] | None], bytes] = OrderedDict()
        self._cache_size = 0
        self._cache_limit = cache_bytes

    def open(self, request: OpenRequest) -> IndexStatus:
        with self._lock:
            self._cancel.set()
            if self._future:
                self._future.cancel()
            self._cancel = Event()
            self._dataset = None
            self._cache.clear()
            self._cache_size = 0
            self._status = IndexStatus(
                id=uuid4().hex, state="indexing", phase="解析数据集", task=request.task
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
            dataset = build_index(request, cancel, progress)
            with self._lock:
                if cancel.is_set() or self._status is None or self._status.id != session_id:
                    return
                self._dataset = dataset
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

    def detail(self, session_id: str, sample_id: int) -> SampleDetail:
        dataset, sample = self.sample(session_id, sample_id)
        if file_stamp(sample.label) != sample.label_stamp:
            raise SessionError("标签文件已变化，请重新扫描数据集以更新统计")
        labels = read_labels(
            sample.label, dataset.spec.task, dataset.spec.names, dataset.spec.kpt_shape
        )
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
            **sample.info().model_dump(),
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

    def close(self) -> None:
        with self._lock:
            self._cancel.set()
            self._dataset = None
            self._cache.clear()
            self._cache_size = 0
        self._executor.shutdown(wait=True, cancel_futures=True)
