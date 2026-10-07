"""模型推理的唯一选择入口，视频、导出和并行调度不依赖具体实现。"""

from collections.abc import Sequence
from typing import Protocol

from ..config import Config
from ..metadata import PoseMetadata
from ..records import BGRImage, RawDetection, Task


class InferencePipeline(Protocol):
    """两种推理实现共享的最小接口，坐标均属于原始图像。"""

    names: dict[int, str]
    task: Task
    pose: PoseMetadata | None

    def predict_batch(self, images: Sequence[BGRImage]) -> list[list[RawDetection]]:
        """按输入顺序返回逐帧实例，禁止跨帧错配。"""
        ...


def build_pipeline(config: Config, device: str) -> InferencePipeline:
    """在配置设备之后延迟导入模型库，只加载选中的推理实现。"""
    if config.pipeline == "two_stage":
        from .two_stage import TwoStagePipeline

        return TwoStagePipeline(config, device)
    from .single_stage import SingleStagePipeline

    return SingleStagePipeline(config, device)
