"""MMPose 适配边界，原生预处理/解码负责将点直接还原到原图。"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np

from ..config import KeypointModelConfig
from ..records import BGRImage, Box, NativeKeypoint
from ..util import as_int, as_mapping, as_str

if TYPE_CHECKING:
    from torch import Tensor, nn


class PoseModel(Protocol):
    """MMPose 模型的局部适配接口，动态库类型不扩散到数据生成流程。"""

    backbone: nn.Module
    dataset_meta: dict[str, object]

    def load_state_dict(self, state_dict: Mapping[str, Tensor], strict: bool) -> object: ...
    def eval(self) -> object: ...


class LiteHRNet:
    """一个配置与 checkpoint 对应一个常驻模型，不绑定检测类别或目标名称。"""

    def __init__(self, config: KeypointModelConfig, device: str) -> None:
        """加载可信本地 checkpoint 并严格核对参数；配置与 checkpoint 点序必须相符。"""
        os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
        import torch
        from mmpose.apis import init_model  # type: ignore[import-untyped]

        self.model = cast(
            PoseModel,
            init_model(
                config.config,
                checkpoint=None,
                device="cpu" if device == "cpu" else f"cuda:{device}",
            ),
        )
        if type(self.model.backbone).__name__ != "LiteHRNet":
            raise ValueError("Two-stage keypoint models currently require LiteHRNet")
        meta = as_mapping(self.model.dataset_meta)
        names = meta.get("keypoint_id2name")
        if not isinstance(names, dict):
            raise TypeError("MMPose config must define native keypoint metadata")
        indexed = {as_int(k): as_str(v) for k, v in cast(dict[object, object], names).items()}
        if sorted(indexed) != list(range(len(indexed))) or not indexed:
            raise ValueError("Native keypoint indices must be contiguous")
        self.names = tuple(indexed[i] for i in range(len(indexed)))
        # 本地权重包含训练元数据，因此显式允许完整 checkpoint；不改全局 Torch 环境。
        checkpoint = as_mapping(
            torch.load(config.checkpoint, map_location="cpu", weights_only=False)
        )
        state = as_mapping(checkpoint["state_dict"])
        if not all(isinstance(v, torch.Tensor) for v in state.values()):
            raise ValueError("Checkpoint state_dict must contain tensors")
        self.model.load_state_dict(cast("dict[str, torch.Tensor]", state), strict=True)
        ckpt_meta = as_mapping(checkpoint.get("meta", {})).get("dataset_meta")
        if ckpt_meta is not None:
            original = as_mapping(ckpt_meta)
            for field in ("keypoint_id2name", "flip_indices"):
                if field in original and original[field] != meta[field]:
                    raise ValueError(f"Checkpoint and config disagree about {field}")
        self.model.eval()

    def predict(self, image: BGRImage, boxes: Sequence[Box]) -> list[tuple[NativeKeypoint, ...]]:
        """对同一帧的 ROI 批量调用原生 API；结果已是原图坐标，不能再次加 ROI 偏移。"""
        from mmpose.apis import inference_topdown

        if not boxes:
            return []
        samples = inference_topdown(
            self.model, image, bboxes=np.asarray(boxes, dtype=np.float32), bbox_format="xyxy"
        )
        if len(samples) != len(boxes):
            raise ValueError("MMPose result count differs from ROI batch size")
        result: list[tuple[NativeKeypoint, ...]] = []
        for sample, box in zip(samples, boxes, strict=True):
            pred = sample.pred_instances
            xy = np.asarray(pred.keypoints, dtype=np.float64)
            scores = np.asarray(pred.keypoint_scores, dtype=np.float64)
            visible_value = getattr(pred, "keypoints_in_frame", None)
            visible = None if visible_value is None else np.asarray(visible_value, dtype=np.float64)
            if xy.shape != (1, len(self.names), 2) or scores.shape != (1, len(self.names)):
                raise ValueError("Native keypoint shape differs from config metadata")
            if visible is not None and visible.shape != scores.shape:
                raise ValueError("In-frame score shape differs from keypoint scores")
            result.append(
                tuple(
                    NativeKeypoint(
                        name,
                        (float(xy[0, i, 0]), float(xy[0, i, 1])),
                        (float(xy[0, i, 0]) - box[0], float(xy[0, i, 1]) - box[1]),
                        float(scores[0, i]),
                        None if visible is None else float(visible[0, i]),
                    )
                    for i, name in enumerate(self.names)
                )
            )
        return result
