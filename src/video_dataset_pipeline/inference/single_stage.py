from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, cast

import yaml

from ..config import Config
from ..metadata import PoseMetadata
from ..model_metadata import load_model_names
from ..pose import load_pose_metadata, read_keypoint_shape
from ..records import BGRImage, RawDetection, RawKeypoint, Task
from ..util import as_mapping

if TYPE_CHECKING:
    from ultralytics.engine.results import Results

LOG = logging.getLogger(__name__)


class SingleStagePipeline:
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
        """单帧接口复用批量推理，保持原有调用和返回类型。"""
        return self.predict_batch([image])[0]

    def predict_batch(self, images: Sequence[BGRImage]) -> list[list[RawDetection]]:
        """一次推理同一视频的一组 BGR 图像，按输入顺序返回逐帧检测结果。

        图像尺寸必须相同，以保持与单帧相同的矩形缩放方式；调用方负责按配置
        限制批次大小。空批次不调用模型，末批无需补齐。输出数量不符时直接报错，
        不截断或错配帧。CUDA 显存不足沿用运行失败流程，不自动改变 batch 或精度。
        """
        if not images:
            return []
        if any(image.shape != images[0].shape for image in images):
            raise ValueError("Inference batch requires images with the same dimensions")
        results = cast(
            "list[Results]",
            self.model.predict(
                list(images),
                batch=len(images),
                imgsz=self.config.imgsz,
                conf=self.config.confidence,
                iou=self.config.iou,
                device=self.device,
                verbose=False,
                save=False,
                quantize=32,
                max_det=300,
                stream=False,
                rect=True,
            ),
        )
        if len(results) != len(images):
            raise ValueError("Detector result count differs from inference batch size")
        return [
            self._decode_result(result, image)
            for result, image in zip(results, images, strict=True)
        ]

    def _decode_result(self, result: Results, image: BGRImage) -> list[RawDetection]:
        """解析一帧的原图结果；类别内 NMS 同时筛选框和点，保持实例对应。

        每张图独立执行 NMS，避免不同帧的目标互相抑制；YOLO26 端到端检测头也
        遵守配置的 IoU 阈值。pose 点保留原分数，后续筛选负责归一化。
        """
        import torch

        # torchvision does not ship a py.typed marker.
        from torchvision.ops import batched_nms  # type: ignore[import-untyped]

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
