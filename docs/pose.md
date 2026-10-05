# YOLO pose 数据集生成

输入视频和 Ultralytics YOLO pose `.pt` 权重，输出标准 YOLO pose 伪标签。与检测模式共用解码、帧记录、时间划分、预览和视频保存流程；不包含模型训练、LiteHR 或其他两阶段姿态模型。

## 使用

从仓库根目录执行：

```bash
uv sync --locked
uv run --locked video-dataset download-model --task pose
uv run --locked video-dataset run --config configs/rm26_exchange_pose.yaml
```

下载仍使用固定 revision `afab41a8b61c399bf71480f0fd52352ed660b994`，官方站点失败后尝试镜像；可用 `--endpoint https://hf-mirror.com` 指定镜像。仅下载权重和 YAML，不生成清单、哈希或锁。已有文件直接复用。

示例配置使用以下文件：

```yaml
model: models/RM26_engineer_exchange_model/yolopose/v11.02/best.pt
model_yaml: models/RM26_engineer_exchange_model/yolopose/v11.02/openvino_best/metadata.yaml
output: datasets/generated/rm26_exchange_pose/recvlnoyv3arV6
confidence: 0.5
keypoint_confidence: 0.5
sample_fps: 2.0
device: 4
save_video: false
```

输入沿用原视频。启动时从权重读取任务类型，不需要另设 `task`；YAML 的 `names` 和 `kpt_shape` 必须与权重一致，显式提供的 `task` 也必须一致。`keypoint_confidence` 范围为 `[0,1]`，可用 `--keypoint-confidence` 覆盖。修改配置后用新的 `--output` 目录生成；不支持 resume。

首个模型实测为 YOLO11s-pose，类别 `0: pillar`、`1: exchange`，`kpt_shape: [12, 3]`。其 YAML 明确定义点顺序：

```text
0 TL          1 TR        2 BL         3 BR
4 ring        5 light_BR  6 light_TR   7 shell_R
8 shell_M     9 shell_L  10 light_TL  11 light_BL
```

`kpt_names` 可使用通用点名列表、点索引到名称的映射，或类别到点名列表的映射；保存到生成的数据 YAML 时统一为类别映射。未提供点名时，预览显示从 0 开始的点索引。只有显式提供合法 `flip_idx` 才导出翻转映射；必须是完整、翻转两次可还原的置换。当前模型未提供该映射，不猜测左右关系；Ultralytics 在缺少该映射时会禁用相关翻转增强。不推断骨架或按类别强行屏蔽某组点。

## 筛选与标签

模型返回的框和点均使用原图坐标；NMS 同时筛选二者，保持实例对应。先筛选框，再独立筛选每个点：

- 框继续遵守 `confidence`、`iou` 和坐标规则。
- 坐标合法且置信度达到 `keypoint_confidence` 的点，导出归一化 `(x, y, 2)`。
- 低置信度、非法分数、非有限坐标或越界点，导出 `(0, 0, 0)`，保留原始预测和原因。这里检查 Ultralytics 返回的原图坐标，不重新裁剪点到图像边缘。
- 模型输出 `[K,2]` 时没有单点分数，只检查坐标，元数据中分数为 `null`；不虚构置信度。标签仍统一输出三维点。
- `v=2` 表示本次接受的伪标签，不代表人工确认可见；不推断遮挡标记 `v=1`。
- 点数或维度不一致是模型接口错误，立即终止，不静默跳过。

每帧至少有一个合格框包含有效关键点才能入选。同帧其他合格框仍保留，即使其点全部为零，避免将这些框从标注中遗漏。只有框而没有有效点的帧记为 `no_keypoints`；没有合格框则为 `no_detection`。

标准标签为：

```text
class_id cx cy w h x1 y1 v1 ... xK yK vK
```

所有坐标归一化到 `[0,1]`，标签不保存连续置信度。`data.yaml` 使用 `kpt_shape: [K,3]`，12 点模型每行共 41 列。检测模式仍输出五列标签。

仅对入选帧按时间顺序进行数量 80/20 划分，再剔除分界两侧各 5 秒内的入选帧。`included_frames`、`no_detection_frames`、`no_keypoints_frames`、`split_gap_frames` 四项互斥，合计为候选帧数；pose 的 `confidence_qualified_frames` 计数同时通过框与关键点规则的帧。关键点统计仅覆盖最终 included 帧，包含按点索引的有效数量、总有效/缺失点数及缺失原因。

## 记录、预览与视频

检测模式继续使用 schema 4，省略 `keypoints`；旧记录无需迁移。pose 使用 schema 5，`run.json` 记录任务、模型的原始点维度和可选点名/翻转映射。`FrameRecord` 的检测对象保存不可变的 `RawKeypoint`/`Keypoint`，JSONL 保留点原始坐标、分数、归一化坐标、标记和筛选原因。

预览及视频绘制有效点和点名，缺失点不绘制。增加无有效点和部分缺失点的抽检样例。框和关键点都是模型伪标签，抽检材料用于检查模型输出，数据加载校验不能证明识别精度。

**`save_video: true` 时禁止抽帧**，忽略 `sample_fps`，逐帧推理并按源时间戳保存视频；与正数 `smoke_frames` 不能同时使用。短视频检查使用时间区间：

```bash
uv run --locked video-dataset run --config configs/rm26_exchange_pose.yaml \
  --save-video --start 200 --end 210 --split-gap-seconds 0 \
  --output datasets/generated/rm26_exchange_pose/recvlnoyv3arV6_video_test
```

## 验证结果：2026-10-05

GPU4 上完成以下检查，生成目录均位于 `datasets/generated/rm26_exchange_pose/`：

| 输出目录 | 处理帧 | 通过筛选 | 训练/验证图片 | 目标框 | 有效关键点 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `recvlnoyv3arV6_smoke` | 20 | 9 | 7 / 2 | 17 | 103 |
| `recvlnoyv3arV6` | 1320 | 464 | 361 / 83 | 767 | 4627 |
| `recvlnoyv3arV6_video_test` | 250 | 210 | 168 / 42 | 382 | 2328 |

全片顺序解码 16499 帧；853 帧采样无合格框、3 帧没有有效关键点、20 帧因分界间隔剔除，最终收录 444 张图片，生成 40 张预览。767 个框包含 354 个 pillar 和 413 个 exchange；缺失点 4577 个，均低于单点置信度阈值。点名顺序、原图叠加和部分缺失情况已抽检，未对模型真实精度作评测。

短视频实测 H.264、1920×1080、25 FPS、10 秒、250 帧，源帧号 5000–5249 连续，时间戳间隔 40 ms。三套数据都用 `YOLODataset(task="pose")` 加载并检查逐个标签的关键点坐标、标记和样本张量掩码。全片和视频测试具有 `COMPLETE.json`；20 帧抽检仅有 `SMOKE_COMPLETE.json`。

复现冒烟及独立校验：

```bash
uv run --locked video-dataset run --config configs/rm26_exchange_pose.yaml \
  --smoke-frames 20 --output datasets/generated/rm26_exchange_pose/recvlnoyv3arV6_smoke
uv run --locked video-dataset validate datasets/generated/rm26_exchange_pose/recvlnoyv3arV6
```

再次执行生成命令时需更换输出目录。启动、执行、退出 GPU 快照保存在各输出目录；推理进程已退出，暂存图片、下载临时文件、编码半成品和 Ultralytics 标签缓存均已清理。日志维持 10 MiB 轮转、保留 3 份。

核心包 102 项测试及严格 mypy、Ruff 检查通过，覆盖 NMS 框点对齐、二维/三维模型、阈值、元数据错误、序列化诊断、空关键点帧、同帧其他框保留、实际 pose 加载、视频逐帧处理及失败清理。旧 schema 4 数据集 `recvlnoyv3arV6_typed_test` 也通过现有校验器和 Ultralytics 加载，仍为 351/80 张图片、676 个框。

为避开并行开发的网页编辑功能尚未提交的改动，另将本次核心提交 `fbe5051` 与新增 pose 测试放入临时源码目录，复用 uv 锁定环境并通过 `PYTHONPATH` 隔离源码。全仓检查结果：**166 项测试通过**，核心 mypy、全仓 Ruff 检查和格式检查通过。测试恢复 Ultralytics 对 Pillow 的全局读取函数替换，避免污染独立网页模块的坏图测试；不新增 HEIF 依赖。临时源码目录已清理。
