import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Literal, NotRequired, TypedDict, cast

import yaml

from .device import parse_device
from .util import as_float, as_int, as_list, as_mapping, as_str


class CommonConfigValues(TypedDict):
    input: str | list[str]
    output: str
    model: str
    model_yaml: str
    sample_fps: NotRequired[float]
    start: NotRequired[float]
    end: NotRequired[float | None]
    confidence: NotRequired[float]
    keypoint_confidence: NotRequired[float]
    iou: NotRequired[float]
    imgsz: NotRequired[int]
    batch_size: NotRequired[int]
    train_fraction: NotRequired[float]
    split_gap_seconds: NotRequired[float]
    device: NotRequired[int | str]
    jpeg_quality: NotRequired[int]
    smoke_frames: NotRequired[int]
    save_video: NotRequired[bool]
    recursive: NotRequired[bool]


class KeypointModelValues(TypedDict):
    config: str
    checkpoint: str
    output_indices: tuple[int, ...]
    confidence: float
    in_frame_confidence: float | None
    crop_margin: float
    batch_size: int


class ConfigValues(CommonConfigValues):
    pipeline: NotRequired[Literal["single_stage", "two_stage"]]
    output_keypoints: NotRequired[tuple[str, ...]]
    keypoint_models: NotRequired[dict[int, KeypointModelValues]]


@dataclass(frozen=True)
class KeypointModelConfig:
    """一个检测类别使用的 MMPose 模型及其原生点序到导出槽位的映射。"""

    config: str
    checkpoint: str
    output_indices: tuple[int, ...]
    confidence: float = 0.1
    in_frame_confidence: float | None = None
    crop_margin: float = 0.0
    batch_size: int = 1

    def __post_init__(self) -> None:
        """热图阈值不是概率上界；检查有限数、ROI 范围和批大小。"""
        if not self.config or not self.checkpoint:
            raise ValueError("Keypoint model requires config and checkpoint")
        if not math.isfinite(self.confidence) or self.confidence < 0:
            raise ValueError("Heatmap confidence must be finite and nonnegative")
        if self.in_frame_confidence is not None and (
            not math.isfinite(self.in_frame_confidence) or not 0 <= self.in_frame_confidence <= 1
        ):
            raise ValueError("in_frame_confidence must be in [0,1] or null")
        if not math.isfinite(self.crop_margin) or self.crop_margin < 0:
            raise ValueError("crop_margin must be finite and nonnegative")
        if as_int(self.batch_size) <= 0:
            raise ValueError("Keypoint batch_size must be positive")
        if not self.output_indices or len(set(self.output_indices)) != len(self.output_indices):
            raise ValueError("output_indices must be nonempty and unique")
        if any(as_int(i) < 0 for i in self.output_indices):
            raise ValueError("output_indices must be nonnegative integers")


def parse_keypoint_models(
    value: object, *, resolve_paths: bool = False
) -> dict[int, KeypointModelConfig]:
    """读取 YAML/JSON 类别映射；离线数据校验不要求原权重仍存在。"""
    if not isinstance(value, dict):
        raise TypeError("keypoint_models must be a class ID mapping")
    result: dict[int, KeypointModelConfig] = {}
    for key, raw in cast(dict[object, object], value).items():
        if isinstance(key, str) and key.isascii() and key.isdecimal():
            key = int(key)
        index = as_int(key)
        if index < 0 or index in result:
            raise ValueError("Keypoint class IDs must be unique nonnegative integers")
        data = as_mapping(raw)
        if unknown := set(data) - {f.name for f in fields(KeypointModelConfig)}:
            raise ValueError(f"Unknown keypoint model keys: {sorted(unknown)}")
        if missing := {"config", "checkpoint", "output_indices"} - set(data):
            raise ValueError(f"Missing keypoint model fields: {sorted(missing)}")
        paths = [as_str(data[name]) for name in ("config", "checkpoint")]
        if resolve_paths:
            paths = [str(Path(p).expanduser().resolve()) for p in paths]
            for path in paths:
                if not Path(path).is_file():
                    raise ValueError(f"Keypoint model file missing: {path}")
        indices = data["output_indices"]
        items = list(indices) if isinstance(indices, tuple) else as_list(indices)
        result[index] = KeypointModelConfig(
            config=paths[0],
            checkpoint=paths[1],
            output_indices=tuple(as_int(i) for i in items),
            confidence=as_float(data.get("confidence", 0.1)),
            in_frame_confidence=None
            if data.get("in_frame_confidence") is None
            else as_float(data["in_frame_confidence"]),
            crop_margin=as_float(data.get("crop_margin", 0.0)),
            batch_size=as_int(data.get("batch_size", 1)),
        )
    return result


@dataclass(frozen=True)
class Config:
    """单次运行参数；时间以秒计，device 使用物理 GPU 编号或 'cpu'。"""

    input: str | list[str]
    output: str
    model: str
    model_yaml: str
    sample_fps: float = 2.0
    start: float = 0.0
    end: float | None = None
    confidence: float = 0.5
    keypoint_confidence: float = 0.5
    iou: float = 0.7
    imgsz: int = 640
    batch_size: int = 1
    train_fraction: float = 0.8
    split_gap_seconds: float = 5.0
    device: int | str = 0
    jpeg_quality: int = 95
    smoke_frames: int = 0
    save_video: bool = False
    recursive: bool = False
    pipeline: Literal["single_stage", "two_stage"] = "single_stage"
    output_keypoints: tuple[str, ...] = ()
    keypoint_models: dict[int, KeypointModelConfig] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """校验采样、阈值和图像参数的范围，并将设备编号字符串转换为整数。"""
        values = [self.input] if isinstance(self.input, str) else as_list(self.input)
        if not values or any(not as_str(value).strip() for value in values):
            raise ValueError("input must contain nonempty file or directory paths")
        if not isinstance(self.recursive, bool):
            raise TypeError("recursive must be a boolean")
        for key in (
            "sample_fps",
            "start",
            "confidence",
            "keypoint_confidence",
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
        if self.keypoint_confidence < 0 or (
            self.pipeline == "single_stage" and self.keypoint_confidence > 1
        ):
            raise ValueError("keypoint_confidence must be in [0, 1]")
        if not 0 < self.train_fraction < 1:
            raise ValueError("train_fraction must be in (0, 1)")
        if self.imgsz <= 0 or self.imgsz % 32 or not 1 <= self.jpeg_quality <= 100:
            raise ValueError("imgsz must be a positive multiple of 32; JPEG quality is 1..100")
        if as_int(self.batch_size) <= 0:
            raise ValueError("batch_size must be a positive integer")
        if self.smoke_frames < 0:
            raise ValueError("smoke_frames must be nonnegative")
        if not isinstance(self.save_video, bool):
            raise TypeError("save_video must be a boolean")
        if self.save_video and self.smoke_frames:
            raise ValueError(
                "save_video cannot be combined with smoke_frames; use start/end instead"
            )
        object.__setattr__(self, "device", parse_device(self.device))
        if self.pipeline not in ("single_stage", "two_stage"):
            raise ValueError("pipeline must be single_stage or two_stage")
        if self.pipeline == "single_stage":
            if self.output_keypoints or self.keypoint_models:
                raise ValueError("Keypoint model mappings require pipeline: two_stage")
        else:
            if not self.keypoint_models or not self.output_keypoints:
                raise ValueError("two_stage requires keypoint_models and output_keypoints")
            if len(set(self.output_keypoints)) != len(self.output_keypoints) or any(
                not as_str(p).strip() for p in self.output_keypoints
            ):
                raise ValueError("output_keypoints must be unique nonempty names")
            for index, head in self.keypoint_models.items():
                if as_int(index) < 0 or max(head.output_indices) >= len(self.output_keypoints):
                    raise ValueError("Class ID or keypoint output index out of range")

    def to_dict(self) -> ConfigValues:
        """返回包含默认值在内的生效配置，供运行配置和元数据写入使用。"""
        return cast(ConfigValues, asdict(self))


def load_config(path: str | Path, overrides: Mapping[str, object] | None = None) -> Config:
    """读取 YAML 并应用非 None 的命令行覆盖值，返回已校验的配置。

    相对路径以当前工作目录为基准转换为绝对路径，并检查权重和类别 YAML。
    输入展开、视频检查及空输出目录检查由运行入口完成。未知字段或错误值报错。
    """
    data = as_mapping(yaml.safe_load(Path(path).read_text()) or {})
    allowed = {f.name for f in fields(Config)}
    data.update({k: v for k, v in (overrides or {}).items() if v is not None})
    if unknown := set(data) - allowed:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    raw_input = data.get("input")
    if raw_input is None:
        raise ValueError("Missing required config field: input")
    if isinstance(raw_input, str):
        if not raw_input.strip():
            raise ValueError("input contains an empty path")
        data["input"] = str(Path(raw_input).expanduser().resolve())
    else:
        inputs = [as_str(p) for p in as_list(raw_input)]
        if not inputs or any(not p.strip() for p in inputs):
            raise ValueError("input must contain nonempty paths")
        data["input"] = [str(Path(p).expanduser().resolve()) for p in inputs]
    for key in ("output", "model", "model_yaml"):
        if not data.get(key):
            raise ValueError(f"Missing required config field: {key}")
        data[key] = str(Path(as_str(data[key])).expanduser().resolve())
    for key in (
        "sample_fps",
        "start",
        "confidence",
        "keypoint_confidence",
        "iou",
        "train_fraction",
        "split_gap_seconds",
    ):
        if key in data:
            data[key] = as_float(data[key])
    if data.get("end") is not None:
        data["end"] = as_float(data["end"])
    for key in ("imgsz", "batch_size", "jpeg_quality", "smoke_frames"):
        if key in data:
            data[key] = as_int(data[key])
    if "device" in data:
        data["device"] = parse_device(data["device"])
    mode = data.pop("pipeline", "single_stage")
    if mode not in ("single_stage", "two_stage"):
        raise ValueError("pipeline must be single_stage or two_stage")
    points = tuple(as_str(p) for p in as_list(data.pop("output_keypoints", [])))
    heads = parse_keypoint_models(data.pop("keypoint_models", {}), resolve_paths=True)
    if overrides is not None and overrides.get("keypoint_confidence") is not None:
        heads = {
            i: replace(h, confidence=as_float(overrides["keypoint_confidence"]))
            for i, h in heads.items()
        }
    config = Config(
        **cast(CommonConfigValues, data),
        pipeline=mode,
        output_keypoints=points,
        keypoint_models=heads,
    )
    for key in ("model", "model_yaml"):
        if not Path(getattr(config, key)).is_file():
            raise ValueError(f"Configured {key} file missing: {getattr(config, key)}")
    if Path(config.output) == Path(config.model).parent:
        raise ValueError("Output must be separate from source video and model directories")
    return config
