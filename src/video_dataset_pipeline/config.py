import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Config:
    input: str
    output: str
    model: str = "models/RM26_engineer_exchange_model"
    sample_fps: float = 2.0
    start: float = 0.0
    end: float | None = None
    confidence: float = 0.5
    iou: float = 0.7
    imgsz: int = 640
    train_fraction: float = 0.8
    split_gap_seconds: float = 5.0
    device: str = "GPU-2d6b9be4-5a54-1137-aad1-64771b415a70"
    jpeg_quality: int = 95
    smoke_frames: int = 0

    def __post_init__(self):
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
        if self.device != "cpu" and not self.device.startswith("GPU-"):
            raise ValueError("device must be a physical GPU UUID or 'cpu'")

    def to_dict(self):
        return asdict(self)


def load_config(path, overrides=None):
    data = yaml.safe_load(Path(path).read_text()) or {}
    allowed = {f.name for f in fields(Config)}
    if unknown := set(data) - allowed:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    data.update({k: v for k, v in (overrides or {}).items() if v is not None})
    for key in ("input", "output", "model"):
        data[key] = str(Path(data.get(key, Config.__dataclass_fields__[key].default)).resolve())
    config = Config(**data)
    if not Path(config.input).is_file():
        raise ValueError(f"Input video missing: {config.input}")
    if Path(config.output) in (Path(config.input).parent, Path(config.model)):
        raise ValueError("Output must be separate from source video and model directories")
    return config
