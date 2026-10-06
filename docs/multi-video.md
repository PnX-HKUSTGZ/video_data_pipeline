# 多视频输入

一次运行使用同一个模型，将多个视频的伪标签合并到一个 YOLO 检测或 pose 数据集。模型只加载一次，在配置的设备上顺序处理视频；不并行占用额外 GPU。

## 配置和命令

`input` 兼容原来的单文件字符串，也支持目录、文件列表和目录列表，文件与目录可以混合。相对路径仍以执行命令的工作目录为基准。

```yaml
input:
  - datasets/videos/比赛A.flv
  - datasets/videos/北部赛区/红方
  - datasets/videos/南部赛区/红方
recursive: true
output: datasets/generated/rm26_exchange_pose/multi_video
model: models/RM26_engineer_exchange_model/yolopose/v11.02/best.pt
model_yaml: models/RM26_engineer_exchange_model/yolopose/v11.02/openvino_best/metadata.yaml
device: 4
sample_fps: 2.0
confidence: 0.5
keypoint_confidence: 0.5
train_fraction: 0.8
split_gap_seconds: 5.0
save_video: false
```

目录默认只扫描当前层；`recursive: true` 包含子目录。支持 `.mp4`、`.mkv`、`.avi`、`.mov`、`.flv`、`.webm`、`.m4v`、`.ts`、`.mts`、`.m2ts`，扩展名不区分大小写。显式指定的文件交给 ffprobe 检查，不受扩展名限制。

使用仓库中的两视频 pose 示例：

```bash
uv run --locked video-dataset run --config configs/rm26_exchange_multi.yaml \
  --output datasets/generated/rm26_exchange_pose/multi_video_new
```

CLI 重复指定 `--input`，**整体替换** YAML 输入列表，不与配置追加合并。`--recursive` / `--no-recursive` 覆盖递归设置：

```bash
uv run --locked video-dataset run --config configs/rm26_exchange_pose.yaml \
  --input datasets/videos/A.flv \
  --input datasets/videos/B.mp4 \
  --input datasets/videos/北部赛区 \
  --recursive --output datasets/generated/rm26_exchange_pose/mixed
```

示例目录需替换为实际路径。每次运行要求空输出目录，已有结果不会被覆盖。

## 输入展开和处理

- 按 `input` 顺序展开，目录内按路径排序，再按解析后的绝对路径去重。重复文件、文件与目录重叠、指向同一文件的符号链接只处理一次。
- 不计算内容哈希，因此不同路径下的内容副本仍视为不同视频。递归扫描不跟随目录符号链接；显式输入的目录符号链接先解析为实际路径。
- 排除本次输出目录及指向其中的文件。不存在的输入、无视频的目录、无法扫描的目录或无法探测的视频会报错，不静默跳过。
- 在 GPU 推理前固定清单并检查所有视频及处理区间。清单写入 `run.json.videos`，之后新增到输入目录的视频不会加入本次运行。
- `start/end` 分别作用于每段视频，以该视频起始时间为零点，范围为 `[start,end)`；实际结束时间不超过视频时长。空区间报错。
- `smoke_frames` 是每段视频的抽检数量。开启视频保存时禁止抽帧，忽略 `sample_fps`，也不能设置正数 `smoke_frames`。

逐帧流式解码，不保留整段视频图像。每个视频完成后导出并清理它的暂存图片，只保留最终图片、预览、帧记录及可选推理视频。`batch_size` 可控制单次推理帧数（默认 1），批次不跨视频；见 [批量推理说明](batch-inference.md)。

## 筛选与划分

每个视频先筛选有效帧，再按时间顺序对有效帧数量做 80/20 划分，最后排除分界两侧各 5 秒。pose 有效帧还必须至少包含一个有效关键点。不同视频互不影响分界，也不在两个视频之间计算时间间隔。

各视频的 train/val 合并到共享目录，间隔排除和整数取整后总体比例不保证恰好 80/20。只有一个有效帧的视频将它分到训练集；没有有效帧的视频只保留统计和预览。单个视频允许一个划分为空，最终合并的数据集必须有训练和验证样本，否则失败。这仍是同一视频内部的时间划分，并非按比赛完全隔离的评测集；标签来自模型预测。

## 产物与兼容性

新数据集使用 schema 6；旧 schema 4/5 数据集无需迁移，仍支持 `validate`。

```text
<output>/
  images/{train,val}/v0001_比赛A_f00000125.jpg
  labels/{train,val}/v0001_比赛A_f00000125.txt
  previews/v0001_比赛A_f00000125.jpg
  previews/index.html
  data.yaml
  config.yaml
  run.json
  inference.jsonl
  frames.jsonl
  report.json
  validation.json
  COMPLETE.json
```

`v0001` 等来源编号按本次清单顺序生成，相同清单对应相同编号；添加、删除或重排输入可能改变后续编号。编号不是跨运行的内容标识。`FrameRecord.source_id` 对应 `run.json.videos[].source_id`，帧唯一键是 `(source_id, frame_index)`。时间戳、尺寸、解码状态、处理区间、分界和推理视频路径按来源记录。

`report.json.videos` 保存逐视频统计；顶层保存整体统计。每段视频独立选择预览，预览索引显示来源编号。多视频整体的 `decode.last_timestamp` 为 `null`，末帧时间请查看各视频的 `decode`。

`save_video: true` 时，多视频分别保存为 `inference_videos/v0001_比赛A.mp4` 等，保留各自分辨率和时间间隔，不拼接；单视频继续使用根目录 `inference.mp4`。例如每段只处理 10 秒：

```bash
uv run --locked video-dataset run --config configs/rm26_exchange_multi.yaml \
  --save-video --start 200 --end 210 --split-gap-seconds 0 \
  --output datasets/generated/rm26_exchange_pose/multi_video_clip_new
```

运行失败时，`run.json`、`report.json` 和日志记录失败来源与原因，不写成功标记。已完成视频的导出产物保留；当前视频的暂存图片和编码半成品清理。不支持续跑，重新运行需指定新输出目录。日志仍为 10 MiB 轮转、保留 3 份；没有新增锁、哈希或 resume。

## 实测验收（2026-10-05，GPU4）

完整处理示例配置中的两段原始视频：`recvlnoyv3arV6_红方工程第一视角.flv` 与 `recvlbkCGRQpqN_红方工程第一视角.flv`。分别采样 1320 / 1361 帧，完整解码合计 33498 帧。所有运行使用 `device: 4`，每次只加载一次模型。

| 任务与输出目录 | 采样/处理帧 | 通过筛选 | 训练 / 验证图片 | 目标框 | 有效关键点 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 检测：`datasets/generated/rm26_exchange/multi_video_test` | 2681 | 599 | 461 / 102 | 879 | — |
| pose：`datasets/generated/rm26_exchange_pose/multi_video_test` | 2681 | 575 | 444 / 101 | 935 | 5511 |
| pose 保存视频：`datasets/generated/rm26_exchange_pose/multi_video_clip_test` | 500 | 212 | 169 / 43 | 384 | 2337 |

完整检测运行有 2082 帧无检测、36 帧因分界间隔排除；pose 有 2092 帧无检测、14 帧无有效关键点、30 帧因分界间隔排除。逐视频计数、划分和类别目标数之和均与整体报告一致。

保存视频测试对两个视频分别处理 `[200,210)`，间隔设为 0；实际使用“显式视频文件 + 两个临时目录（其中一个有子目录）”作为输入，并重复引用首个视频，确认递归和去重只生成两个来源。目录只包含符号链接，测试后已清理，未复制原视频。每个输出均为 1920×1080、25 FPS、10 秒、250 帧；源帧号 5000–5249 连续，视频时间戳每 40 ms 递增，没有抽帧。删除临时输入目录后独立 `validate` 仍通过。

三个数据集均通过图片标签配对、逐来源帧号/时间戳/划分、标签坐标/格式、文件集合及真实 Ultralytics 加载检查，具有 `COMPLETE.json` 和 `validation.json`。保存视频另通过逐帧解码、尺寸、时间戳和时长核对。

单视频回归：将两次完整多视频运行中的 `v0001` 与此前的独立数据集逐帧比较。检测 1320 条记录的预测、筛选状态、划分和 431 个标签文件完全一致，仍为 351/80 图片、676 个框；pose 同样 1320 条记录和 444 个标签一致，仍为 361/83 图片、767 个框、4627 个有效点。旧 schema 4 检测数据集及 schema 5 pose 数据集均再次通过真实加载校验。

全仓 **274 项测试通过**；核心包严格 mypy、全仓 Ruff 检查和格式检查通过。覆盖文件/目录混合、递归、中文路径、同名文件、去重、多个来源帧号重叠、模型只加载一次、独立划分、零检测来源、逐视频冒烟、不同帧率/尺寸、半途失败清理，以及旧格式兼容。测试有一条现有 Starlette/httpx 弃用提示。

完整进程表和设备映射保存在各输出目录的 `gpu_*.txt`，退出后的核对见 `gpu_exited.txt`。本次推理进程已退出，无残留 GPU 上下文；暂存图片、视频编码半成品、临时目录及 Ultralytics 标签缓存已清理。三套验证产物约占 889 MiB，作为本次验收结果保留。

复现检测运行与校验：

```bash
uv run --locked video-dataset run --config configs/rm26_exchange_multi.yaml \
  --model models/RM26_engineer_exchange_model/yolo/v10.12/best.pt \
  --model-yaml models/RM26_engineer_exchange_model/yolo/v10.12/openvino_best/metadata.yaml \
  --output datasets/generated/rm26_exchange/multi_video_new
uv run --locked video-dataset validate datasets/generated/rm26_exchange_pose/multi_video_test
```
