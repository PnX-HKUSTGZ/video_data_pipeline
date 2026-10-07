"""兼容已有 LiteHRNet 权重的在框分数头；这里只实现数据生成所需的推理。"""

from typing import Protocol, cast

import numpy as np
from mmpose.models.heads import HeatmapHead  # type: ignore[import-untyped]
from mmpose.registry import MODELS  # type: ignore[import-untyped]
from numpy.typing import NDArray
from torch import Tensor, nn

from ..util import as_int, as_list


class PoseInstance(Protocol):
    """第三方实例对象的最小写入接口。"""

    keypoints_visible: NDArray[np.float32]
    keypoints_in_frame: NDArray[np.float32]


class PoseSample(Protocol):
    """翻转测试只读取原生元数据中的点序。"""

    metainfo: dict[str, object]


type Features = tuple[Tensor, ...] | list[tuple[Tensor, ...]]
type Predictions = list[PoseInstance] | tuple[list[PoseInstance], object]


class HeatmapHeadWithVisibility(HeatmapHead):  # type: ignore[misc]
    """保持源项目 vis_head 参数名和翻转平均逻辑，与严格权重加载兼容。"""

    def __init__(self, vis_loss: dict[str, object] | None = None, **kwargs: object) -> None:
        """复用原生热图头及解码器，添加每个点的在框概率输出。"""
        super().__init__(**kwargs)
        loss = vis_loss or {"type": "BCELoss", "use_target_weight": False, "use_sigmoid": True}
        self.vis_loss_module = MODELS.build(loss)
        self.use_sigmoid = bool(loss.get("use_sigmoid", True))
        layers: list[nn.Module] = [
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(int(self.in_channels), int(self.out_channels)),
        ]
        if self.use_sigmoid:
            layers.append(nn.Sigmoid())
        self.vis_head = nn.Sequential(*layers)

    def vis_forward(self, feats: tuple[Tensor, ...]) -> Tensor:
        """最后一层骨干特征经池化得到逐点在框分数。"""
        return cast(Tensor, self.vis_head(feats[-1])).reshape(-1, int(self.out_channels))

    def predict(
        self,
        feats: Features,
        batch_data_samples: list[PoseSample],
        test_cfg: dict[str, object] | None = None,
    ) -> Predictions:
        """保留 MMPose 原生热图预测，只附加在框分数；翻转时使用原生 flip_indices。"""
        cfg = test_cfg or {}
        preds = cast(Predictions, super().predict(feats, batch_data_samples, cfg))
        if cfg.get("flip_test", False):
            if not isinstance(feats, list) or len(feats) != 2:
                raise ValueError("Flip test requires original and flipped features")
            indices = [as_int(i) for i in as_list(batch_data_samples[0].metainfo["flip_indices"])]
            vis = (self.vis_forward(feats[0]) + self.vis_forward(feats[1])[:, indices]) * 0.5
        else:
            if not isinstance(feats, tuple):
                raise ValueError("Expected a feature tuple without flip test")
            vis = self.vis_forward(feats)
        instances = preds[0] if isinstance(preds, tuple) else preds
        scores = vis.detach().cpu().numpy()
        if len(instances) != len(scores):
            raise ValueError("Visibility result count differs from heatmap predictions")
        for instance, score in zip(instances, scores, strict=True):
            instance.keypoints_visible = score[None, :]
            instance.keypoints_in_frame = score[None, :]
        return preds


MODELS.register_module()(HeatmapHeadWithVisibility)
