# 视频 → YOLO 检测 / 关键点数据集

使用可配置的预训练模型处理视频并生成 **伪标签**。支持 YOLO 检测和 YOLO pose 关键点格式；模型生成流程无需先转码视频，也不包含训练。

检测模式使用 `configs/rm26_exchange.yaml`；关键点模式使用 `configs/rm26_exchange_pose.yaml`，详见 [YOLO pose 使用与验证](docs/pose.md)。多个视频或目录可合并生成一个数据集，示例为 `configs/rm26_exchange_multi.yaml`，详见 [多视频输入与验证](docs/multi-video.md)。

两阶段 YOLO ROI + LiteHRNet 使用 `configs/rm26_two_stage.yaml`，与一阶段共用 CLI；模型和类别映射由 YAML 指定，详见 [两阶段使用与验证](docs/two-stage.md)。

## 环境

Linux，Python 3.12，系统需提供 C/C++ 编译器和 `ffprobe`；可选下载命令使用 `curl`。CUDA 推理使用支持 CUDA 12.8 的 NVIDIA 驱动。依赖全部由 uv 管理：

```bash
bash scripts/setup_env.sh  # 首次安装：构建匹配当前 Torch/NumPy 的原生依赖
uv sync --locked
uv run --locked video-dataset --help
```

PyTorch CUDA 包从官方 cu128 索引获取；首次安装约数 GB。网络较慢时使用 `UV_HTTP_TIMEOUT=180 UV_CONCURRENT_DOWNLOADS=4 uv sync --locked`，保留 uv 缓存以便重试。没有 GPU 时通过 `--device cpu` 运行。

模型文件中含训练时的 Albumentations 对象，因此锁文件显式包含该依赖。运行时禁止 Ultralytics 自动安装依赖，缓存放在本项目 `.cache/`。

## 下载固定模型

```bash
uv run --locked video-dataset download-model
# 直连不可用时自动尝试镜像；也可直接指定：
uv run --locked video-dataset download-model --endpoint https://hf-mirror.com
```

- 仓库：<https://huggingface.co/hkustenterprize/RM26_engineer_exchange_model>
- revision：`afab41a8b61c399bf71480f0fd52352ed660b994`
- 权重：`models/RM26_engineer_exchange_model/yolo/v10.12/best.pt`
- 类别来自权重和发布元数据，必须一致：`0: pillar`、`1: exchange`。
- 发布的纯检测模型实际为 **YOLO26s**；YOLO11 训练可以读取其生成的数据集。一阶段姿态模型使用独立的 pose 配置；LiteHRNet 使用两阶段配置。

这是可选的便捷下载命令，仅下载权重和类别 YAML；已有文件直接复用。下载不加锁，不生成额外的模型清单，也不进行哈希校验。下载失败会清理临时文件，下次从头下载。模型也可以自行准备，启动时只读取配置中的两个本地文件：

```yaml
model: models/RM26_engineer_exchange_model/yolo/v10.12/best.pt
model_yaml: models/RM26_engineer_exchange_model/yolo/v10.12/openvino_best/metadata.yaml
```

`model_yaml` 需要包含 `names` 类别名称，可使用列表或从 0 开始的连续整数映射。运行时确认其类别顺序与权重一致，防止生成错误类别标签；无需模型清单或特定目录结构。

## 生成

从项目根目录执行；配置中的相对路径以当前工作目录为基准：

```bash
# 默认视频、采样范围与 GPU 见 configs/rm26_exchange.yaml
uv run --locked video-dataset run --config configs/rm26_exchange.yaml

# 均匀抽取 20 帧做冒烟检查，输出到独立目录
uv run --locked video-dataset run --config configs/rm26_exchange.yaml \
  --smoke-frames 20 --output datasets/generated/rm26_exchange/smoke

# 参数覆盖示例：时间单位为距首个视频时间戳的秒数，结束时间不包含在内
uv run --locked video-dataset run --config configs/rm26_exchange.yaml \
  --sample-fps 5 --start 100 --end 400 \
  --output datasets/generated/rm26_exchange/custom

# 同时保存带检测框的视频，输出目录需为空
uv run --locked video-dataset run --config configs/rm26_exchange.yaml \
  --save-video --output datasets/generated/rm26_exchange/with_video

# 复核已生成的数据集
uv run --locked video-dataset validate datasets/generated/rm26_exchange/recvlnoyv3arV6
```

未启用视频保存时，默认 2 FPS 抽帧；置信度阈值 0.5，IoU 阈值 0.7，推理尺寸 640，FP32，保存原始分辨率 JPEG（质量 95）。采用 PTS 顺序抽帧，不通过帧号猜测时间；例如 25 FPS 视频中的 0.5 秒采样点会落到 0.52 秒帧。模型负责 letterbox 与原图坐标映射，额外进行类别内 NMS，使 YOLO26 end-to-end 检测头也遵守配置的 IoU 阈值。

`batch_size` 控制每次推理的帧数，默认 `1`；可在 YAML 设置 `batch_size: 8`，或通过 `--batch-size 8` 覆盖。每批来自同一个视频，末批不足时直接处理，按原帧顺序导出；开启视频保存时也不会抽帧。检测和 pose 均支持，详见 [批量推理与实测](docs/batch-inference.md)。

配置中用 `device: 4` 选择 `nvidia-smi` 显示的 GPU4，也可以写 `device: cpu`。命令行通过 `--device 4` 或 `--device cpu` 覆盖。默认配置使用 GPU4；视频在 CPU 解码，不创建显示窗口或 GPU 渲染上下文。

多个视频可通过 `--devices 0 2 3 7` 在指定物理 GPU 上并行，每卡一个进程，复用模型并动态领取完整视频。该参数覆盖 YAML 的单卡 `device`，与 CLI `--device` 互斥；设备列表和进程号保存在 `parallel.json`。使用前先检查这些卡是否空闲，程序不会自动占用其他卡。

```bash
uv run --locked video-dataset run --config configs/rm26_national_red_pose.yaml \
  --devices 0 2 3 7
```

每个进程独立写入该视频产物，主进程按固定来源顺序合并到同一个数据集，逐视频划分规则不变。任一子进程失败会停止本次运行并回收其他子进程，保留诊断与已导出文件，清理暂存文件，不写完成标记。运行日志按 10 MiB 轮转、保留 3 份，GPU 快照与逐视频诊断保存在输出的 `workers/` 下。相同 GPU 不得重复指定，不支持恢复。

全国赛红方全部 129 个视频的四卡生成结果与复现命令见 [运行验证记录](docs/national-red-20261007.md)。

`save_video` 默认是 `false`。设为 `true` 或添加 `--save-video` 后，**禁止抽帧，逐帧推理并保存** `[start, end)` 区间内的全部源视频帧，单视频在数据集根目录生成 `inference.mp4`，多视频分别保存到 `inference_videos/`。此时忽略 `sample_fps`；不允许同时设置 `smoke_frames > 0`，短片段测试请使用 `start/end`。`--no-save-video` 可覆盖配置将其关闭，恢复按配置抽帧。

视频使用 H.264/MP4、无音频，保留源帧时间戳和播放速度，最后一帧显示至处理区间结尾；25 FPS 源视频每秒处理并保存约 25 帧，变帧率视频也不会丢弃密集帧。视频包含无检测帧和随后被划分间隔排除的帧，叠加有效检测框、类别、置信度、源时间戳和推理阶段状态。数据集候选同样使用全部处理帧，再按置信度和时间划分规则筛选，因此启用后推理量及图片数量会增加。编码在 CPU 上流式进行，保持原图尺寸；奇数宽高仅在右侧或底部补一个黑像素以兼容播放器。关闭时不会创建编码器或视频文件。

短片段验证示例（输出目录需为空；短片段设间隔为 0，以免训练/验证帧全部被排除）：

```bash
uv run --locked video-dataset run --config configs/rm26_exchange.yaml \
  --save-video --start 100 --end 110 --split-gap-seconds 0 \
  --output datasets/generated/rm26_exchange/recvlnoyv3arV6_video_fullrate_test
```

2026-10-05 使用 GPU4 验证该片段：250 次逐帧推理，输出 1920×1080、25 FPS、10 秒的 250 帧视频，时间戳间隔 40 ms；数据集 10 张训练图、3 张验证图，通过 Ultralytics 加载及视频完整性检查。核心包 65 个测试及 mypy、Ruff 检查通过，覆盖逐帧模式、变帧率、时间区间、冒烟抽帧冲突和异常清理。

## 输出与标注规则

```text
<output>/
  images/{train,val}/*.jpg
  labels/{train,val}/*.txt
  data.yaml
  config.yaml              # 生效的绝对路径与参数
  run.json                 # videos 固定清单、逐视频解码/划分状态、模型信息
  inference.jsonl          # 全视频逐帧推理提交记录
  inference.mp4            # 可选：单视频逐帧带框视频
  inference_videos/*.mp4    # 可选：多视频逐个保存，不拼接
  frames.jsonl             # 筛选划分后的时间、框、置信度、文件路径
  report.json              # 数量、类别、置信度统计
  validation.json          # 完整性与真实 Ultralytics 数据加载结果
  previews/index.html      # 全部抽检预览
  previews/contact_sheet.jpg
  gpu_*.txt                # 物理 UUID、PID、显存与完整进程表
  COMPLETE.json            # 只有完整数据集校验通过后才存在
```

一阶段新生成数据使用 schema 6，两阶段使用 schema 7，图片与标签以“来源编号_视频文件名_f帧号”命名，例如 `v0001_recvlnoyv3arV6_红方工程第一视角_f00000000.jpg`。帧记录携带 `source_id`，帧号和时间戳保留源视频含义；旧 schema 4/5 数据集仍可校验。

检测模式每个标签严格为 `class_id x_center y_center width height`，坐标以原图宽高归一化到 `[0,1]`。置信度仅保存在元数据，不写入训练标签。默认不收录没有有效检测框的帧，这些帧可能存在漏检，不能直接视为负样本。

每个视频独立执行以下规则，再合并到同一数据集。检测模式先识别完整视频（默认 2 FPS 抽帧，保存视频时逐帧处理），仅选择至少有一个框达到置信度阈值的画面，然后按这些入选画面的时间顺序，以数量划分前 80% / 后 20%。切分位置取两组相邻画面的时间中点，再剔除分界前后各 5 秒内的入选帧。`train_fraction` 与 `split_gap_seconds` 均可配置；显式指定 `start/end` 时只处理该区间。报告中的 `included_frames`、`no_detection_frames`、`split_gap_frames` 互斥，其和等于候选帧数；`confidence_qualified_frames` 是分界间隔剔除前通过阈值的画面数。80/20 是间隔剔除前的入选图片比例；剔除后数量和各类别比例可能变化。片头片尾的误检仍可能通过阈值，需结合抽检预览判断。

预览覆盖 20 个时间桶、每个出现类别、低置信度和无检测样例。模型仍可能误检或漏检；本数据是伪标签，不是人工真值，验证集不能用于宣称模型真实精度。未自动剪裁画面中的赛事 UI 或小窗。

## 运行、校验与资源

- 推理阶段原子写入候选图片和预览后才提交 `inference.jsonl` 行，逐帧刷盘。
- 每次运行都从头执行。输出目录必须为空；已有数据用 `validate` 检查，重新生成时通过 `--output` 指定新目录。
- 源视频保持原样；中间仅暂存通过阈值的候选图像，导出通过硬链接复用图像字节。成功或失败退出时均清理暂存目录。
- 原始视频截断、解码失败、没有有效检测或任一划分为空，都会保留诊断并拒绝写 `COMPLETE.json`。
- `validate` 检查归一化框、配对、原始分辨率、时间顺序、逐帧记录数量、文件集合，并实际用 `YOLODataset` 加载两组数据和样本张量。只操作本输出目录，临时的 Ultralytics 标签缓存会清除。
- 启用 `save_video` 时，还会解码检查视频帧数、尺寸、时间戳和时长。视频编码结束后才发布最终 MP4；推理或编码中断时清理视频半成品。
- 冒烟结果仅生成 `SMOKE_COMPLETE.json`，不宣称得到完整训练数据集。
- `pipeline.log` 按 10 MiB 轮转并保留 3 个备份；运行结束清除本任务 `.tmp`，失败保留诊断记录。进程退出释放 GPU 上下文。
- Git 忽略输入视频、生成数据、模型、环境和缓存；现有视频下载任务不属于本 Pipeline，不会被修改。

## 通用 YOLO 数据集浏览器

`src/yolo_dataset_viewer` 支持浏览标准 YOLO 检测、分割、OBB、姿态数据集，并可差分编辑检测和姿态数据。使用方法见[模块 Readme](src/yolo_dataset_viewer/Readme.md)。

## 开发检查

```bash
uv run --locked mypy src/video_dataset_pipeline
uv run --locked pytest -q
uv run --locked ruff check src tests
uv run --locked ruff format --check src tests
```

测试覆盖时间戳采样、边界间隔、坐标裁剪/归一化、直接指定权重与 YAML、下载失败清理、中文视频路径、非空输出目录保护、失败状态、空数据集和不完整解码，以及记录序列化兼容性、错误记录行号、导出时保留原推理记录。

核心数据流使用 `records.py` 中的 `RawDetection`、`Detection`、`RejectedDetection` 和 `FrameRecord`。检测对象不可变，框为四元素元组；帧记录保存处理状态及产物路径。读取记录后通过 `record.timestamp`、`record.detections` 等属性访问，导出通过 `dataclasses.replace` 复制帧记录。图像仍是 uint8 BGR 数组，采样器仍返回 `(帧号, 时间戳, 图像)`。

`metadata.py` 使用 `TypedDict` 描述视频、划分、运行状态和报告，运行时仍是普通字典。`FrameRecord.to_dict/from_dict` 仅用于 JSON/JSONL 边界转换，检测模式保留 schema 4 的字段、坐标数组和可选字段省略规则；pose 使用 schema 5，新增强类型关键点字段，已有数据集无需迁移。

mypy 对整个核心包启用严格检查并禁止显式 `Any`，开发依赖由 uv 锁定。JSON/YAML 和第三方返回值在读取处进行类型收窄；局部类型忽略仅用于未提供类型标记的 TorchVision/MMPose 导入及自定义热图头的无类型基类。PyTorch/Ultralytics 继续在设备配置后延迟导入。

2026-10-05 类型重构回归：48 个测试及上述开发检查通过。使用当前配置、GPU4 对整段视频重新生成：

```bash
uv run --locked video-dataset run --config configs/rm26_exchange.yaml \
  --output datasets/generated/rm26_exchange/recvlnoyv3arV6_typed_test
```

结果为 1320 帧采样、450 帧通过阈值、19 帧因分界间隔排除，最终训练集 351 张、验证集 80 张，包含 676 个目标框和 36 张预览。与 `recvlnoyv3arV6_nohash_test` 相比，两份逐帧清单各 1320 条记录的全部字段一致，431 份标签内容逐字节一致；两套数据均通过 Ultralytics 加载。对比结果保存在新输出目录的 `comparison.json`。再次运行上述命令时需换用新的输出目录。
