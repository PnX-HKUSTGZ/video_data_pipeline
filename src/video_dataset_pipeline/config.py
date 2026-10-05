import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import NotRequired, TypedDict, cast

import yaml

from .device import parse_device
from .util import as_float, as_int, as_mapping, as_str


class ConfigValues(TypedDict):
    input: str
    output: str
    model: str
    model_yaml: str
    sample_fps: NotRequired[float]
    start: NotRequired[float]
    end: NotRequired[float | None]
    confidence: NotRequired[float]
    iou: NotRequired[float]
    imgsz: NotRequired[int]
    train_fraction: NotRequired[float]
    split_gap_seconds: NotRequired[float]
    device: NotRequired[int | str]
    jpeg_quality: NotRequired[int]
    smoke_frames: NotRequired[int]


@dataclass(frozen=True)
class Config:
    """单次运行参数；时间以秒计，device 使用物理 GPU 编号或 'cpu'。"""

    input: str
    output: str
    model: str
    model_yaml: str
    sample_fps: float = 2.0
    start: float = 0.0
    end: float | None = None
    confidence: float = 0.5
    iou: float = 0.7
    imgsz: int = 640
    train_fraction: float = 0.8
    split_gap_seconds: float = 5.0
    device: int | str = 0
    jpeg_quality: int = 95
    smoke_frames: int = 0

    def __post_init__(self) -> None:
        """校验采样、阈值和图像参数的范围，并将设备编号字符串转换为整数。"""
        for key in (
            "sample_fps",
            "start",
            "confidence",
            "iou",
            "train_fraction",
            "split_gap_seconds",
        ):
            if not math.isfinite(getattr(self, key)):
                raise ValueError(f"{key} must be finite")
        if self.sample_fps <= 0 or self.start < 0 or self.split_gap_seconds < 0:
            raise ValueError("Invalid sampling interval or split gap")
        if self.end is not None and (not math.isfinite(self.end) or self.end <= self.start):
            raise ValueError("end must be finite and greater than start")
        if not 0 < self.confidence <= 1 or not 0 < self.iou <= 1:
            raise ValueError("confidence and iou must be in (0, 1]")
        if not 0 < self.train_fraction < 1:
            raise ValueError("train_fraction must be in (0, 1)")
        if self.imgsz <= 0 or self.imgsz % 32 or not 1 <= self.jpeg_quality <= 100:
            raise ValueError("imgsz must be a positive multiple of 32; JPEG quality is 1..100")
        if self.smoke_frames < 0:
            raise ValueError("smoke_frames must be nonnegative")
        object.__setattr__(self, "device", parse_device(self.device))

    def to_dict(self) -> ConfigValues:
        """返回包含默认值在内的生效配置，供运行配置和元数据写入使用。"""
        return cast(ConfigValues, asdict(self))


def load_config(path: str | Path, overrides: Mapping[str, object] | None = None) -> Config:
    """读取 YAML 并应用非 None 的命令行覆盖值，返回已校验的配置。

    相对路径以当前工作目录为基准，转换为绝对路径；同时检查输入视频、权重和
    类别 YAML 存在。未知字段或错误值会报错，输出目录是否为空由运行入口检查。
    """
    data = as_mapping(yaml.safe_load(Path(path).read_text()) or {})
    allowed = {f.name for f in fields(Config)}
    data.update({k: v for k, v in (overrides or {}).items() if v is not None})
    if unknown := set(data) - allowed:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    for key in ("input", "output", "model", "model_yaml"):
        if not data.get(key):
            raise ValueError(f"Missing required config field: {key}")
        data[key] = str(Path(as_str(data[key])).resolve())
    for key in ("sample_fps", "start", "confidence", "iou", "train_fraction", "split_gap_seconds"):
        if key in data:
            data[key] = as_float(data[key])
    if data.get("end") is not None:
        data["end"] = as_float(data["end"])
    for key in ("imgsz", "jpeg_quality", "smoke_frames"):
        if key in data:
            data[key] = as_int(data[key])
    if "device" in data:
        data["device"] = parse_device(data["device"])
    config = Config(**cast(ConfigValues, data))
    if not Path(config.input).is_file():
        raise ValueError(f"Input video missing: {config.input}")
    for key in ("model", "model_yaml"):
        if not Path(getattr(config, key)).is_file():
            raise ValueError(f"Configured {key} file missing: {getattr(config, key)}")
    if Path(config.output) in (Path(config.input).parent, Path(config.model).parent):
        raise ValueError("Output must be separate from source video and model directories")
    return config
