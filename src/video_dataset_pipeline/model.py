import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Literal, TypedDict, cast

import yaml

from .config import Config
from .metadata import PoseMetadata
from .pose import load_pose_metadata, read_keypoint_shape
from .records import BGRImage, RawDetection, RawKeypoint, Task
from .util import as_mapping

REPO_ID = "hkustenterprize/RM26_engineer_exchange_model"
REVISION = "afab41a8b61c399bf71480f0fd52352ed660b994"
WEIGHTS = "yolo/v10.12/best.pt"
METADATA = "yolo/v10.12/openvino_best/metadata.yaml"
POSE_WEIGHTS = "yolopose/v11.02/best.pt"
POSE_METADATA = "yolopose/v11.02/openvino_best/metadata.yaml"
LOG = logging.getLogger(__name__)


class ModelPaths(TypedDict):
    model: str
    model_yaml: str


def download_model(
    destination: str | Path,
    revision: str = REVISION,
    endpoint: str | None = None,
    task: Literal["detect", "pose"] = "detect",
) -> ModelPaths:
    """下载指定 revision 的权重和类别 YAML，返回两个本地文件的绝对路径。

    已有文件直接复用；未指定 endpoint 时依次尝试官方站点与镜像。
    下载通过临时文件完成，失败时清理临时文件并报错。
    """
    if task not in ("detect", "pose"):
        raise ValueError("Download task must be detect or pose")
    weights, metadata = (POSE_WEIGHTS, POSE_METADATA) if task == "pose" else (WEIGHTS, METADATA)
    destination = Path(destination).resolve()
    endpoints = [endpoint] if endpoint else ["https://huggingface.co", "https://hf-mirror.com"]
    for filename in (weights, metadata):
        local = destination / filename
        if local.is_file():
            continue
        local.parent.mkdir(parents=True, exist_ok=True)
        errors = []
        for host in endpoints:
            with tempfile.NamedTemporaryFile(
                dir=local.parent, suffix=".tmp", delete=False
            ) as stream:
                temporary = Path(stream.name)
            try:
                LOG.info("Downloading %s via %s", filename, host)
                subprocess.run(
                    [
                        "curl",
                        "--fail",
                        "--location",
                        "--silent",
                        "--show-error",
                        "--connect-timeout",
                        "10",
                        "--max-time",
                        "900",
                        "--retry",
                        "2",
                        "--retry-all-errors",
                        "--output",
                        str(temporary),
                        f"{host}/{REPO_ID}/resolve/{revision}/{filename}?download=true",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                if not temporary.stat().st_size:
                    raise ValueError(f"Downloaded file is empty: {filename}")
                temporary.replace(local)
                break
            except (subprocess.CalledProcessError, OSError, ValueError) as error:
                errors.append(f"{host}: {error}")
                LOG.warning("Download failed via %s: %s", host, error)
            finally:
                temporary.unlink(missing_ok=True)
        else:
            raise RuntimeError("Model download failed: " + "; ".join(errors))
    return {"model": str(destination / weights), "model_yaml": str(destination / metadata)}


def load_model_names(path: str | Path) -> dict[int, str]:
    """读取 YAML 的 names 列表或映射，要求类别编号从 0 连续排列且非空。"""
    metadata: object = yaml.safe_load(Path(path).read_text())
    names: object = metadata.get("names") if isinstance(metadata, dict) else None
    if isinstance(names, list):
        names = dict(enumerate(names))
    if not isinstance(names, dict) or not names:
        raise ValueError("Model YAML must contain a nonempty 'names' mapping or list")
    result: dict[int, str] = {}
    for key, value in cast(dict[object, object], names).items():
        if not isinstance(key, (str, int)):
            raise TypeError("Model YAML class IDs must be integers")
        result[int(key)] = str(value)
    if sorted(result) != list(range(len(result))):
        raise ValueError("Model YAML class IDs must be contiguous and start at zero")
    return result


class Detector:
    """Ultralytics YOLO 检测/姿态适配器，返回类型明确的原图框和关键点。"""

    def __init__(self, config: Config, device: str) -> None:
        """加载权重并核对任务、类别顺序及 pose 点数；调用前必须完成设备配置。

        device 使用 configure_device 返回的框架内编号，例如隔离 GPU 后的 '0'。
        权重类别与配置 YAML 不一致时拒绝继续推理。
        """
        import torch
        from ultralytics.models.yolo.model import YOLO

        torch.set_num_threads(4)
        self.config, self.device = config, device
        self.model = YOLO(config.model)
        task = self.model.task
        if task not in ("detect", "pose"):
            raise ValueError(f"Expected detect or pose model, got {task}")
        if task == "pose" and Path(config.model).suffix.lower() != ".pt":
            raise ValueError("Only Ultralytics YOLO .pt pose weights are supported")
        self.task = cast(Task, task)
        self.pose: PoseMetadata | None = None
        self.names: dict[int, str] = {int(k): str(v) for k, v in self.model.names.items()}
        published = load_model_names(config.model_yaml)
        if self.names != published or sorted(self.names) != list(range(len(self.names))):
            raise ValueError("Checkpoint and configured YAML disagree about class ordering")
        metadata = as_mapping(yaml.safe_load(Path(config.model_yaml).read_text()))
        if metadata.get("task", self.task) != self.task:
            raise ValueError("Checkpoint and configured YAML disagree about task")
        if self.task == "pose":
            self.pose = load_pose_metadata(config.model_yaml, len(self.names))
            shape = read_keypoint_shape(getattr(self.model.model, "kpt_shape", None))
            if shape != self.pose["source_kpt_shape"]:
                raise ValueError("Checkpoint and configured YAML disagree about kpt_shape")
        elif "kpt_shape" in metadata:
            raise ValueError("Detection model YAML must not define kpt_shape")
        if config.imgsz != 640:
            LOG.info("Using configured image size %s (published model size: 640)", config.imgsz)

    def predict(self, image: BGRImage) -> list[RawDetection]:
        """对单张原分辨率 BGR 图像推理，返回像素坐标框、类别和置信度。

        Ultralytics 负责缩放及坐标还原；这里额外执行类别内 NMS，使配置的 IoU
        阈值对 YOLO26 端到端检测头同样生效；同一索引同时筛选框和点，保持对应。
        pose 点保留原分数，坐标由 Ultralytics 映射回原图，后续筛选负责归一化。
        """
        import torch

        # torchvision does not ship a py.typed marker.
        from torchvision.ops import batched_nms  # type: ignore[import-untyped]
        from ultralytics.engine.results import Results

        # Non-streaming detection returns Results, not embedding tensors or an iterator.
        result = cast(
            list[Results],
            self.model.predict(
                image,
                imgsz=self.config.imgsz,
                conf=self.config.confidence,
                iou=self.config.iou,
                device=self.device,
                verbose=False,
                save=False,
                quantize=32,
                max_det=300,
                stream=False,
            ),
        )[0]
        if result.orig_shape != image.shape[:2]:
            raise ValueError("Detector did not return coordinates in the source image space")
        if result.boxes is None or not isinstance(result.boxes.data, torch.Tensor):
            raise ValueError("Detector did not return tensor detection boxes")
        boxes = result.boxes.data.detach().cpu()
        if boxes.ndim != 2 or boxes.shape[1] != 6:
            raise ValueError("Expected N x 6 detection boxes")
        keypoints: torch.Tensor | None = None
        if self.pose is not None:
            if result.keypoints is None or not isinstance(result.keypoints.data, torch.Tensor):
                raise ValueError("Pose model did not return tensor keypoints")
            keypoints = result.keypoints.data.detach().cpu()
            if tuple(keypoints.shape) != (len(boxes), *self.pose["source_kpt_shape"]):
                raise ValueError("Pose result shape disagrees with boxes or kpt_shape")
        if boxes.numel():
            # YOLO26 end-to-end heads do not apply the configurable NMS threshold.
            # Explicit class-aware NMS gives this pipeline's iou option consistent semantics.
            keep = batched_nms(
                boxes[:, :4], boxes[:, 4], boxes[:, 5].to(torch.int64), self.config.iou
            )
            boxes = boxes[keep]
            if keypoints is not None:
                keypoints = keypoints[keep]
        return [
            RawDetection(
                class_id=int(row[5]),
                confidence=float(row[4]),
                xyxy=(float(row[0]), float(row[1]), float(row[2]), float(row[3])),
                keypoints=tuple(
                    RawKeypoint((float(p[0]), float(p[1])), float(p[2]) if len(p) == 3 else None)
                    for p in keypoints[index].tolist()
                )
                if keypoints is not None
                else None,
            )
            for index, row in enumerate(boxes.tolist())
        ]
