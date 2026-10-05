# 视频 → YOLO 检测数据集

使用指定 RM26 工程机器人模型对视频抽帧并生成 **伪标签**。输出为 YOLO11/Ultralytics 通用检测格式；无需先转码视频，也不包含训练或人工审核界面。

## 环境

Linux，Python 3.12，系统需提供 `ffprobe`；可选下载命令使用 `curl`。CUDA 推理使用支持 CUDA 12.8 的 NVIDIA 驱动。依赖全部由 uv 管理：

```bash
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
- 发布的纯检测模型实际为 **YOLO26s**；YOLO11 训练可以读取其生成的数据集。姿态模型和 LiteHR 权重不参与本流程。

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

# 复核已生成的数据集
uv run --locked video-dataset validate datasets/generated/rm26_exchange/recvlnoyv3arV6
```

默认 2 FPS，置信度阈值 0.5，IoU 阈值 0.7，推理尺寸 640，FP32，保存原始分辨率 JPEG（质量 95）。采用 PTS 顺序抽帧，不通过帧号猜测时间；例如 25 FPS 视频中的 0.5 秒采样点会落到 0.52 秒帧。模型负责 letterbox 与原图坐标映射，额外进行类别内 NMS，使 YOLO26 end-to-end 检测头也遵守配置的 IoU 阈值。

配置中用 `device: 4` 选择 `nvidia-smi` 显示的 GPU4，也可以写 `device: cpu`。命令行通过 `--device 4` 或 `--device cpu` 覆盖。默认配置使用 GPU4；视频在 CPU 解码，不创建显示窗口或 GPU 渲染上下文。

## 输出与标注规则

```text
<output>/
  images/{train,val}/*.jpg
  labels/{train,val}/*.txt
  data.yaml
  config.yaml              # 生效的绝对路径与参数
  run.json                 # 视频信息、模型和 YAML 路径、类别、解码状态
  inference.jsonl          # 全视频逐帧推理提交记录
  frames.jsonl             # 筛选划分后的时间、框、置信度、文件哈希
  report.json              # 数量、类别、置信度统计
  validation.json          # 完整性与真实 Ultralytics 数据加载结果
  previews/index.html      # 全部抽检预览
  previews/contact_sheet.jpg
  gpu_*.txt                # 物理 UUID、PID、显存与完整进程表
  COMPLETE.json            # 只有完整数据集校验通过后才存在
```

每个标签严格为 `class_id x_center y_center width height`，坐标以原图宽高归一化到 `[0,1]`。置信度仅保存在元数据，不写入训练标签。默认不收录没有有效检测框的帧，这些帧可能存在漏检，不能直接视为负样本。

先识别完整视频（默认 2 FPS），仅选择至少有一个框达到置信度阈值的画面，然后按这些入选画面的时间顺序，以数量划分前 80% / 后 20%。切分位置取两组相邻画面的时间中点，再剔除分界前后各 5 秒内的入选帧。`train_fraction` 与 `split_gap_seconds` 均可配置；显式指定 `start/end` 时只处理该区间。报告中的 `included_frames`、`no_detection_frames`、`split_gap_frames` 互斥，其和等于候选帧数；`confidence_qualified_frames` 是分界间隔剔除前通过阈值的画面数。80/20 是间隔剔除前的入选图片比例；剔除后数量和各类别比例可能变化。片头片尾的误检仍可能通过阈值，需结合抽检预览判断。

预览覆盖 20 个时间桶、每个出现类别、低置信度和无检测样例。模型仍可能误检或漏检；本数据是伪标签，不是人工真值，验证集不能用于宣称模型真实精度。未自动剪裁画面中的赛事 UI 或小窗。

## 运行、校验与资源

- 推理阶段原子写入候选图片和预览后才提交 `inference.jsonl` 行，逐帧刷盘。
- 每次运行都从头执行。输出目录必须为空；已有数据用 `validate` 检查，重新生成时通过 `--output` 指定新目录。
- 源视频保持原样；中间仅暂存通过阈值的候选图像，导出通过硬链接复用图像字节。成功或失败退出时均清理暂存目录。
- 原始视频截断、解码失败、没有有效检测或任一划分为空，都会保留诊断并拒绝写 `COMPLETE.json`。
- `validate` 检查归一化框、配对、原始分辨率、时间顺序、逐帧记录完整性、跨划分重复图片，并实际用 `YOLODataset` 加载两组数据和样本张量。只操作本输出目录，临时的 Ultralytics 标签缓存会清除。
- 冒烟结果仅生成 `SMOKE_COMPLETE.json`，不宣称得到完整训练数据集。
- `pipeline.log` 按 10 MiB 轮转并保留 3 个备份；运行结束清除本任务 `.tmp`，失败保留诊断记录。进程退出释放 GPU 上下文。
- Git 忽略输入视频、生成数据、模型、环境和缓存；现有视频下载任务不属于本 Pipeline，不会被修改。

## 开发检查

```bash
uv run --locked pytest -q
uv run --locked ruff check src tests
uv run --locked ruff format --check src tests
```

测试覆盖时间戳采样、边界间隔、坐标裁剪/归一化、直接指定权重与 YAML、下载失败清理、中文视频路径、非空输出目录保护、失败状态、空数据集和不完整解码。
