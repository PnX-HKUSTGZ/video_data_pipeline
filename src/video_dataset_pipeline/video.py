import json
import math
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import av

from .config import Config
from .metadata import DecodeStats, VideoInfo
from .records import BGRImage
from .util import as_int, as_list, as_mapping, as_str


def probe_video(path: str | Path) -> VideoInfo:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    data = as_mapping(json.loads(result.stdout))
    streams = [as_mapping(s) for s in as_list(data["streams"])]
    stream = next(s for s in streams if s["codec_type"] == "video")
    numerator, denominator = map(int, as_str(stream["avg_frame_rate"]).split("/"))
    fps = numerator / denominator if denominator else 0
    start = float(as_str(stream.get("start_time", "0")))
    duration = (
        float(as_str(stream["duration"]))
        if "duration" in stream
        else float(as_str(as_mapping(data["format"])["duration"])) - start
    )
    if fps <= 0 or duration <= 0:
        raise ValueError("Video requires a positive frame rate and duration")
    return {
        "width": as_int(stream["width"]),
        "height": as_int(stream["height"]),
        "fps": fps,
        "duration": duration,
        "start_time": start,
        "codec": as_str(stream["codec_name"]),
    }


class TimestampSampler:
    def __init__(self, fps: float, start: float, end: float, smoke_frames: int = 0) -> None:
        self.fps, self.start, self.end = fps, start, end
        self.index = 0
        self.targets = (
            [start + (end - start) * i / smoke_frames for i in range(smoke_frames)]
            if smoke_frames
            else None
        )

    def accepts(self, timestamp: float) -> bool:
        if timestamp < self.start - 1e-7 or timestamp >= self.end - 1e-7:
            return False
        if self.targets is not None:
            if self.index == len(self.targets) or timestamp + 1e-7 < self.targets[self.index]:
                return False
            self.index += 1
        else:
            target = self.start + self.index / self.fps
            if timestamp + 1e-7 < target:
                return False
            self.index = math.floor((timestamp - self.start) * self.fps + 1e-7) + 1
        return True


class VideoFrames:
    def __init__(self, path: str | Path, info: VideoInfo, config: Config) -> None:
        self.path, self.info, self.config = Path(path), info, config
        self.end = min(config.end if config.end is not None else info["duration"], info["duration"])
        if config.start >= self.end or config.sample_fps > info["fps"] + 1e-6:
            raise ValueError("Empty time interval or sample_fps exceeds the source frame rate")
        self.stats: DecodeStats = {
            "decoded_frames": 0,
            "last_timestamp": None,
            "decode_complete": False,
        }

    def __iter__(self) -> Iterator[tuple[int, float, BGRImage]]:
        sampler = TimestampSampler(
            self.config.sample_fps, self.config.start, self.end, self.config.smoke_frames
        )
        previous = -float("inf")
        with av.open(str(self.path)) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            stream.codec_context.thread_count = 4
            if stream.time_base is None:
                raise ValueError("Video stream is missing its time base")
            origin = (
                float(stream.start_time * stream.time_base)
                if stream.start_time is not None
                else None
            )
            for index, frame in enumerate(container.decode(stream)):
                if frame.pts is None or frame.time_base is None or frame.is_corrupt:
                    raise ValueError(f"Corrupt frame or missing PTS at frame {index}")
                absolute = float(frame.pts * frame.time_base)
                if origin is None:
                    origin = absolute
                timestamp = absolute - origin
                if timestamp <= previous:
                    raise ValueError(f"Non-increasing PTS at frame {index}")
                previous = timestamp
                self.stats.update({"decoded_frames": index + 1, "last_timestamp": timestamp})
                if timestamp >= self.end - 1e-7:
                    break
                if sampler.accepts(timestamp):
                    yield index, timestamp, cast(BGRImage, frame.to_ndarray(format="bgr24"))
        if previous + 1 / self.info["fps"] < self.end - max(0.1, 2 / self.info["fps"]):
            raise ValueError(f"Decode ended early at {previous:.3f}s; expected {self.end:.3f}s")
        self.stats["decode_complete"] = True
