"""将推理采样帧流式编码为可播放的带标注视频，不保留音频。"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path
from types import TracebackType
from typing import Self

import av
import numpy as np
from av.container import OutputContainer
from av.video.stream import VideoStream

from .records import BGRImage, FrameRecord

INFERENCE_VIDEO = "inference.mp4"
TIME_BASE = Fraction(1, 90000)


class InferenceVideoWriter:
    """按原始时间间隔保存采样帧；仅在成功关闭编码器后发布 MP4 文件。"""

    def __init__(self, root: Path, width: int, height: int, fps: float, end: float) -> None:
        """设置原图尺寸、名义采样率和源视频中的结束秒数，暂不打开输出文件。"""
        self.path = root / INFERENCE_VIDEO
        self.temporary = root / f"{INFERENCE_VIDEO}.tmp"
        self.width, self.height, self.fps, self.end = width, height, fps, end
        self._container: OutputContainer | None = None
        self._stream: VideoStream | None = None
        self._pending: av.VideoFrame | None = None
        self._origin = 0.0

    def __enter__(self) -> Self:
        """用 CPU 上的 libx264 编码 H.264；奇数边长补一个黑像素以兼容 yuv420p。"""
        self._container = av.open(
            str(self.temporary), "w", format="mp4", options={"movflags": "+faststart"}
        )
        try:
            stream = self._container.add_stream(
                "libx264",
                rate=Fraction(str(self.fps)).limit_denominator(1_000_000),
                options={"preset": "veryfast", "crf": "23", "tune": "zerolatency"},
            )
            if not isinstance(stream, VideoStream):
                raise TypeError("Expected a video encoder")
            self._stream = stream
            stream.width = self.width + self.width % 2
            stream.height = self.height + self.height % 2
            stream.pix_fmt = "yuv420p"
            stream.time_base = stream.codec_context.time_base = TIME_BASE
            stream.codec_context.thread_count = 4
            stream.codec_context.max_b_frames = 0
        except BaseException:
            try:
                self._container.close()
            finally:
                self.temporary.unlink(missing_ok=True)
            raise
        return self

    def write(self, image: BGRImage, timestamp: float) -> None:
        """接收已绘制的 BGR 图像及源时间秒数；延迟一帧以确定前一帧的显示时长。"""
        if image.shape != (self.height, self.width, 3):
            raise ValueError("Inference video frame dimensions changed")
        if self._pending is None:
            self._origin = timestamp
        if self.width % 2 or self.height % 2:
            image = np.pad(image, ((0, self.height % 2), (0, self.width % 2), (0, 0)))
        frame = av.VideoFrame.from_ndarray(image, format="bgr24")
        frame.time_base = TIME_BASE
        frame.pts = round((timestamp - self._origin) / TIME_BASE)
        if self._pending is not None:
            assert self._pending.pts is not None
            if frame.pts <= self._pending.pts:
                raise ValueError("Inference video timestamps must increase")
            self._pending.duration = frame.pts - self._pending.pts
            self._encode(self._pending)
        self._pending = frame

    def _encode(self, frame: av.VideoFrame) -> None:
        """写入一帧并设置 packet 时长，保留不等间隔采样和末帧的正确播放时长。"""
        assert self._stream is not None and self._container is not None
        # 零延迟、无 B 帧的编码器即时输出当前帧，不会将时长错配给其他帧。
        for packet in self._stream.encode(frame):
            packet.duration = frame.duration
            self._container.mux(packet)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """正常退出时延长末帧至处理区间结尾并完成封装；异常退出删除半成品。"""
        assert self._container is not None and self._stream is not None
        try:
            try:
                if exc_type is None:
                    if self._pending is None or self._pending.pts is None:
                        raise ValueError("No frames to save in inference video")
                    self._pending.duration = (
                        round((self.end - self._origin) / TIME_BASE) - self._pending.pts
                    )
                    if self._pending.duration <= 0:
                        raise ValueError("Inference video end must follow the last frame")
                    self._encode(self._pending)
                    for packet in self._stream.encode():
                        self._container.mux(packet)
            finally:
                self._container.close()
            if exc_type is None:
                self.temporary.replace(self.path)
        finally:
            self.temporary.unlink(missing_ok=True)


def validate_inference_video(root: Path, records: Sequence[FrameRecord], end: float) -> None:
    """流式解码可选视频，核对帧数、尺寸、相对时间戳和结束时长。"""
    if not records:
        raise ValueError("Inference video requires sampled frames")
    count = 0
    with av.open(str(root / INFERENCE_VIDEO)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 4
        if stream.duration is None or stream.time_base is None:
            raise ValueError("Inference video is missing duration")
        duration = float(stream.duration * stream.time_base)
        if abs(duration - (end - records[0].timestamp)) > 0.001:
            raise ValueError("Inference video duration differs from sampled interval")
        for frame in container.decode(stream):
            if count >= len(records) or frame.pts is None or frame.time_base is None:
                raise ValueError("Inference video frame count or timestamp mismatch")
            record = records[count]
            if frame.is_corrupt or (frame.width, frame.height) != (
                record.width + record.width % 2,
                record.height + record.height % 2,
            ):
                raise ValueError("Inference video decode/dimension failure")
            timestamp = float(frame.pts * frame.time_base)
            if abs(timestamp - (record.timestamp - records[0].timestamp)) > 2 * TIME_BASE:
                raise ValueError("Inference video timestamps differ from sampled frames")
            count += 1
    if count != len(records):
        raise ValueError("Inference video frame count differs from sampled frames")
