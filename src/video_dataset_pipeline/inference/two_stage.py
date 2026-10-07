"""YOLO ROI 检测及按类别配置的 LiteHRNet 关键点推理。"""

from collections.abc import Sequence
from dataclasses import replace
from itertools import batched

from ..config import Config
from ..detection import filter_detections
from ..metadata import PoseMetadata
from ..records import BGRImage, RawDetection, ROIResult, Task
from ..roi import expand_roi, map_keypoints
from .litehrnet import LiteHRNet
from .single_stage import SingleStagePipeline


class TwoStagePipeline:
    """复用检测器，每个唯一配置/权重只加载一次；类别和点序完全来自 YAML。"""

    def __init__(self, config: Config, device: str) -> None:
        """检查检测任务、类别映射和原生点名，任何不一致在处理视频前失败。"""
        self.config = config
        self.detector = SingleStagePipeline(config, device)
        if self.detector.task != "detect":
            raise ValueError("two_stage requires a YOLO detection model for stage one")
        self.names = self.detector.names
        self.task: Task = "pose"
        self.pose: PoseMetadata | None = {
            "source_kpt_shape": (len(config.output_keypoints), 3),
            "kpt_names": {i: list(config.output_keypoints) for i in self.names},
        }
        self.heads: dict[int, LiteHRNet] = {}
        cache: dict[tuple[str, str], LiteHRNet] = {}
        for class_id, head in config.keypoint_models.items():
            if class_id not in self.names:
                raise ValueError(f"Configured keypoint class is absent from detector: {class_id}")
            key = head.config, head.checkpoint
            if key not in cache:
                cache[key] = LiteHRNet(head, device)
            model = cache[key]
            expected = tuple(config.output_keypoints[i] for i in head.output_indices)
            if model.names != expected:
                raise ValueError(
                    f"Native keypoint names/order disagree with output_indices for class {class_id}: {model.names} != {expected}"
                )
            self.heads[class_id] = model

    def predict_batch(self, images: Sequence[BGRImage]) -> list[list[RawDetection]]:
        """先批量检测，再逐帧按模型分组批量处理 ROI，保持检测原顺序。"""
        predictions = self.detector.predict_batch(images)
        if len(predictions) != len(images):
            raise ValueError("Detector result count differs from inference batch size")
        for image, detections in zip(images, predictions, strict=True):
            height, width = image.shape[:2]
            for class_id, head in self.config.keypoint_models.items():
                indices = []
                for index, detection in enumerate(detections):
                    if detection.class_id != class_id:
                        continue
                    accepted, _ = filter_detections(
                        [detection], width, height, len(self.names), self.config.confidence
                    )
                    if accepted:
                        indices.append(
                            (index, expand_roi(accepted[0].xyxy, width, height, head.crop_margin))
                        )
                for group in batched(indices, head.batch_size):
                    points = self.heads[class_id].predict(image, [box for _, box in group])
                    if len(points) != len(group):
                        raise ValueError("Keypoint result count differs from ROI batch size")
                    for (index, box), native in zip(group, points, strict=True):
                        roi = ROIResult(index, box, native)
                        detections[index] = replace(
                            detections[index],
                            roi=roi,
                            keypoints=map_keypoints(roi, head, len(self.config.output_keypoints)),
                        )
        return predictions
