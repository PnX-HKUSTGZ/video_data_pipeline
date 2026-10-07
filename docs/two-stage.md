# YOLO ROI → LiteHRNet 关键点数据集

同一个 `video-dataset run --config ...` 入口使用 `pipeline` 选择实现。省略或设置
`single_stage` 保持 YOLO detect/pose 原行为；`two_stage` 使用 YOLO 检测 ROI，再按类别调用
YAML 指定的 LiteHRNet。核心实现不绑定 pillar、exchange、类别数或两个指定 checkpoint。

## 环境与权重准备

继续使用项目的 Python 3.12 uv 环境。锁定 MMPose 1.3.2、MMEngine 0.10.7、MMCV 2.1.0、
MMDetection 3.2.0，保持 Torch 2.7.1/cu128 和 NumPy 2。首次安装需要 C/C++ 编译器：

```bash
bash scripts/setup_env.sh
# 如需同时安装浏览器及其测试依赖：
# bash scripts/setup_env.sh --extra viewer --group browser
uv sync --locked
```

脚本先安装构建依赖，再通过 uv 构建 MMCV、xtcocotools 和 chumpy。MMCV 使用 CPU 扩展构建，
这里的 LiteHRNet CUDA 推理由 PyTorch 实现，不使用 MMCV CUDA 自定义算子。
xtcocotools 从源码构建以匹配 NumPy 2；不修改 site-packages，不使用外部 Python 环境。
同一环境之后直接 `uv sync --locked` 即可。Torch/NumPy 或平台改变时应重建环境和扩展。

从用户指定的本地项目复制三个权重，提取独立的推理配置：

```bash
uv run --locked python scripts/prepare_hkust_models.py \
  --source ~/code/tech_core_hkust --output models/hkust
```

目标目录必须为空，源文件保持原样。该脚本是此组模型的可选准备工具；运行 Pipeline
只依赖 YAML 中指定的文件，不依赖源项目位置或准备脚本。

| 模型 | 源项目相对路径 | 本项目路径 |
| --- | --- | --- |
| YOLO26s | `checkpoints/yolo_v2/weights/best.pt` | `models/hkust/detector/best.pt` |
| marker LiteHRNet-30 | `reports/precision_pillar_noflip_selection_v1/result/validation/pillar5/epoch_19/checkpoint_snapshot.pth` | `models/hkust/marker/checkpoint.pth` |
| station LiteHRNet-30 | `reports/station7_sampling2x_v1/result/train_station7/iter_4000.pth` | `models/hkust/station/checkpoint.pth` |

marker 配置来自同目录 `config_snapshot.py`；station 来自训练目录 `runtime_config.py`。
提取模型图、metainfo 和 test_pipeline，生成各自的 `inference.py`，保留原生归一化、
GetBBoxCenterScale 的 1.25 padding、仿射变换和热图解码。marker 输入为 256²、不开 flip_test；
station 输入为 768²、开启 flip_test。仅移除训练数据路径和训练专用扩展导入。
检测器 `metadata.yaml` 从 checkpoint 的类别信息生成，类别为 `0: pillar, 1: exchange`。

本项目的 `HeatmapHeadWithVisibility` 保留原头的参数名、在框分数和翻转平均逻辑，仅用于推理。
checkpoint 使用严格参数加载，配置与 checkpoint 的原生点名/翻转映射不一致时失败。

## 配置和执行

完整例子见 `configs/rm26_two_stage.yaml`。从项目根目录运行，相对路径基于当前工作目录。

```bash
uv run --locked video-dataset run --config configs/rm26_two_stage.yaml
# 重跑时指定空的新输出目录：
uv run --locked video-dataset run --config configs/rm26_two_stage.yaml \
  --output datasets/generated/rm26_two_stage/new_run
uv run --locked video-dataset validate datasets/generated/rm26_two_stage/new_run
```

关键字段：

```yaml
pipeline: two_stage
model: models/hkust/detector/best.pt
model_yaml: models/hkust/detector/metadata.yaml
batch_size: 8  # 每批视频帧数，用于第一阶段
output_keypoints: [TL, TR, BL, BR, ring, light_BR, light_TR, shell_R, shell_M, shell_L, light_TL, light_BL]
keypoint_models:
  0:  # 检测器类别 ID
    config: models/hkust/marker/inference.py
    checkpoint: models/hkust/marker/checkpoint.pth
    output_indices: [0, 1, 2, 3, 4]  # 原生点序依次对应的导出槽位
    confidence: 0.1                # 热图分数阈值
    in_frame_confidence: null      # 可选在框概率阈值，默认不筛选
    crop_margin: 0.0               # 每侧额外扩展原框宽/高的比例
    batch_size: 1                 # 同一帧、同一类别的 ROI 批大小
  1:
    config: models/hkust/station/inference.py
    checkpoint: models/hkust/station/checkpoint.pth
    output_indices: [5, 6, 7, 8, 9, 10, 11]
    confidence: 0.1
```

模型可由多个类别复用；相同配置和 checkpoint 在每个 GPU 进程中只加载一次。
`output_indices` 必须完整覆盖该模型原生点数，且映射后的点名必须与模型 metainfo 顺序一致。
未配置关键点模型的检测类别保留为 `unmapped_class` 拒绝记录，不进入姿态训练标签。
原生热图分数可能大于 1，只检查有限值和下限；`in_frame_confidence` 是独立的 [0,1] 概率筛选。
CLI 的 `--keypoint-confidence` 在两阶段模式下覆盖所有类别的热图阈值。

第一阶段遵循 `confidence`、`iou`、`imgsz` 等原参数。保留检测框用于训练标签；
扩展并裁剪的 ROI 仅供第二阶段推理，原生 padding 仍由 MMPose 执行。MMPose 输出已经是
原图坐标，不再添加 ROI 偏移；记录中的 `roi_xy` 是原图点减去所提供 ROI 的左上角。
它不是 256²/768² 输入张量坐标，也不表示额外 padding 后的边界。

## 输出与兼容性

两阶段输出 schema 7，一阶段保持 schema 6；旧 schema 4/5/6 仍可读取和校验。
视频解码、有效帧筛选、时间划分、预览、视频编码、多视频及多 GPU 调度共用现有流程。

- 标准 YOLO Pose：`kpt_shape: [12, 3]`，每目标 41 列（5 列框 + 12 组 x/y/v）。
- 当前类别不适用的槽位写 `0 0 0`，原因 `not_applicable`；其他被筛除点同样补零并保留原因。
- 热图分数和在框分数保存在 JSONL，不写入训练标签；`v=2` 表示被接受的伪标签。
- `inference.jsonl` 保留推理阶段状态，`frames.jsonl` 保存独立的最终导出记录。
- 每个有效两阶段目标的 `roi` 包含 `detection_index`、ROI `xyxy`，以及原生点名、原图 `xy`、
  相对 ROI 的 `roi_xy`、热图 `score` 和可选 `in_frame_score`。未配置类别/非法框不执行关键点推理。
- `validate` 不要求源权重存在，会复核 ROI 几何、实例索引、点序映射、筛选、划分和训练文件，
  并实际用 Ultralytics 加载两组数据。
- 只有整个任务成功校验才产生 `COMPLETE.json`。无有效样本、空划分、模型错误或解码失败会保留诊断。
- 不包含 PnP、3D、跟踪或目标配对。预览和验证集仍是伪标签抽检材料。

未定义训练用 `flip_idx`，因此不要自行对不同类别的统一槽位进行未配置的左右交换。

## 多视频、GPU 和带标注视频

仍使用 `input` 的文件/目录列表以及 `--input`、`--recursive`。每个视频独立时间划分。
单卡 `device: 5`；多卡使用 `--devices 5 6`，每进程持有检测器及其关键点模型。
请依据运行时空闲情况选择卡号。

`save_video: true` 禁止抽帧，对整个指定区间逐帧推理，保留原始时间戳。短片段示例：

```bash
uv run --locked video-dataset run --config configs/rm26_two_stage.yaml \
  --save-video --start 190 --end 200 --split-gap-seconds 0 \
  --output datasets/generated/rm26_two_stage/new_video_test
```

## 2026-10-07 实测

当前验证使用空闲 GPU5、6。完整视频以样例 YAML 的 2 FPS、框阈值 0.5、热图阈值 0.1、
检测 batch 8、ROI batch 1 生成到 `datasets/generated/rm26_two_stage/recvlnoyv3arV6_test/`：

| 指标 | 数量 |
| --- | ---: |
| 采样帧 | 1320 |
| 有效候选帧 | 295 |
| 无检测帧 | 1024 |
| 有框但无有效点的帧 | 1 |
| 分界间隔排除帧 | 17 |
| train / val 图片 | 227 / 51 |
| pillar / exchange 目标 | 212 / 255 |
| 有效关键点 | 2297 |
| 总用时 | 45.26 秒 |

这组检测器权重不同于旧 HF 一阶段模型，样本数不要求与此前生成结果一致。
完整运行已通过 Ultralytics 加载；结果、预览和 GPU 审计保存在该输出目录。

- 全片均匀 20 帧冒烟：6 张有效图、13 个目标，4/2 划分，保存 20 张抽检预览。
- 190–200 秒带标注视频：250 帧，1920×1080、25 FPS、10 秒；100/26 张训练/验证图，235 个目标。
- GPU5/6 并行处理两个视频 190–210 秒：80 帧，26/7 张训练/验证图，49 个目标。
  输入为首个视频及同目录的 `recvlmvHBsvQ1S_红方工程第一视角.flv`，输出 `parallel_test/`。
- 对 198 秒画面的两个 marker ROI 和一个 station ROI，使用源项目的原始 visibility head
  与本地移植头进行对照：原图点、热图分数、在框分数差异均为 0。
  marker 单 ROI 与两 ROI 批处理最大坐标差为 0.00434 像素（FP32 内核数值差异）。
  证据在完整输出目录的 `native_parity.json`。
- 旧检测数据集 `rm26_exchange/recvlnoyv3arV6_typed_test` 和旧 pose 数据集
  `rm26_exchange_pose/recvlnoyv3arV6` 重新通过完整校验及 Ultralytics 加载。

核心测试覆盖任意类别映射、共享模型、ROI 批顺序、原图坐标、padding 与边界、分数超过 1、
在框筛选、缺失槽位、未映射类别、结构异常、JSON 往返、离线校验和导出不改变原记录。
所有核心函数有类型标注和注释；第三方无类型接口在适配模块内收窄。

开发检查：371 个测试通过（含现有浏览器测试），mypy 严格检查、Ruff 检查与格式检查通过。

```bash
uv run --locked mypy src/video_dataset_pipeline
uv run --locked pytest -q
uv run --locked ruff check src tests
uv run --locked ruff format --check src tests
```

模块分工：`inference/single_stage.py` 负责 YOLO detect/pose；`inference/two_stage.py`
负责按类别调度；`inference/litehrnet.py` 负责 MMPose 适配；`roi.py` 负责无模型依赖的几何与筛选。
`pipeline.py`、`parallel.py` 仅通过有类型的 `InferencePipeline` 接口调用选中的实现。
权重、视频、数据集和运行记录不提交 Git。

全国赛红蓝双方全部 259 个视频的生成结果和复现命令见
[全国赛两阶段运行记录](national-two-stage-20261007.md)。
