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

# 同时保存带检测框的视频，输出目录需为空
uv run --locked video-dataset run --config configs/rm26_exchange.yaml \
  --save-video --output datasets/generated/rm26_exchange/with_video

# 复核已生成的数据集
uv run --locked video-dataset validate datasets/generated/rm26_exchange/recvlnoyv3arV6
```

默认 2 FPS，置信度阈值 0.5，IoU 阈值 0.7，推理尺寸 640，FP32，保存原始分辨率 JPEG（质量 95）。采用 PTS 顺序抽帧，不通过帧号猜测时间；例如 25 FPS 视频中的 0.5 秒采样点会落到 0.52 秒帧。模型负责 letterbox 与原图坐标映射，额外进行类别内 NMS，使 YOLO26 end-to-end 检测头也遵守配置的 IoU 阈值。

配置中用 `device: 4` 选择 `nvidia-smi` 显示的 GPU4，也可以写 `device: cpu`。命令行通过 `--device 4` 或 `--device cpu` 覆盖。默认配置使用 GPU4；视频在 CPU 解码，不创建显示窗口或 GPU 渲染上下文。

`save_video` 默认是 `false`。设为 `true` 或添加 `--save-video` 后，会在数据集根目录生成 `inference.mp4`；`--no-save-video` 可覆盖配置将其关闭。视频包含所有实际推理的采样帧，包括无检测帧和随后被划分间隔排除的帧，叠加有效检测框、类别、置信度、源时间戳和推理阶段状态。

视频使用 H.264/MP4、无音频，保留采样帧之间的原时间间隔，最后一帧显示至处理区间结尾。默认 2 FPS 推理时，画面也按约 2 FPS 更新；冒烟模式仅包含指定的少量采样帧。保存视频不会增加推理次数或改变数据集筛选规则。编码在 CPU 上流式进行，保持原图尺寸；奇数宽高仅在右侧或底部补一个黑像素以兼容播放器。关闭时不会创建编码器或视频文件。

## 输出与标注规则

```text
<output>/
  images/{train,val}/*.jpg
  labels/{train,val}/*.txt
  data.yaml
  config.yaml              # 生效的绝对路径与参数
  run.json                 # 视频信息、模型和 YAML 路径、类别、解码状态
  inference.jsonl          # 全视频逐帧推理提交记录
  inference.mp4            # 可选：save_video 开启时保存带框采样视频
  frames.jsonl             # 筛选划分后的时间、框、置信度、文件路径
  report.json              # 数量、类别、置信度统计
  validation.json          # 完整性与真实 Ultralytics 数据加载结果
  previews/index.html      # 全部抽检预览
  previews/contact_sheet.jpg
  gpu_*.txt                # 物理 UUID、PID、显存与完整进程表
  COMPLETE.json            # 只有完整数据集校验通过后才存在
```

图片与标签以“视频文件名_f帧号”命名，例如 `recvlnoyv3arV6_红方工程第一视角_f00000000.jpg`。

每个标签严格为 `class_id x_center y_center width height`，坐标以原图宽高归一化到 `[0,1]`。置信度仅保存在元数据，不写入训练标签。默认不收录没有有效检测框的帧，这些帧可能存在漏检，不能直接视为负样本。

先识别完整视频（默认 2 FPS），仅选择至少有一个框达到置信度阈值的画面，然后按这些入选画面的时间顺序，以数量划分前 80% / 后 20%。切分位置取两组相邻画面的时间中点，再剔除分界前后各 5 秒内的入选帧。`train_fraction` 与 `split_gap_seconds` 均可配置；显式指定 `start/end` 时只处理该区间。报告中的 `included_frames`、`no_detection_frames`、`split_gap_frames` 互斥，其和等于候选帧数；`confidence_qualified_frames` 是分界间隔剔除前通过阈值的画面数。80/20 是间隔剔除前的入选图片比例；剔除后数量和各类别比例可能变化。片头片尾的误检仍可能通过阈值，需结合抽检预览判断。

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

## 通用 YOLO 网页浏览器

独立的 `src/yolo_dataset_viewer` 模块可只读浏览任意来源的标准 YOLO **检测、分割、OBB、姿态**数据集。只读取图片、标签及可选的数据集 YAML，不需要本仓库的 `run.json`、`frames.jsonl`、模型或其他生成记录，也不会导入流水线、Torch 或 Ultralytics。

```bash
uv run --locked --extra viewer yolo-viewer --host 127.0.0.1 --port 8765
```

打开 <http://127.0.0.1:8765>，填写**服务所在机器**的数据集目录或 `.yaml` / `.yml` 文件路径，选择标注任务后打开。目录里有多个 YAML 时，网页会列出候选供选择。服务器上运行时，可以从自己的电脑建立 SSH 转发：

```bash
ssh -N -L 8765:127.0.0.1:8765 <user>@<server>
```

然后在电脑浏览器访问同一个本地地址。服务默认仅监听回环地址，面向单用户；没有账号或多用户会话管理。每次打开都会替换当前数据集，取消旧索引并释放旧缓存。

### 标准格式与路径

- YAML 支持 `path`、`train`、`val`、`test`（另接受 `valid`、`validation`、`minival`）、`names` 列表或连续整数映射、`nc`、姿态任务的 `kpt_shape`。划分来源支持目录、图片清单 TXT、多来源列表；无需同时具有训练集和验证集。
- 没有 YAML 时，识别 `images/train ↔ labels/train`、`train/images ↔ train/labels`、`images ↔ labels`、按划分排列的图片/标签同目录，以及无划分的同目录图片/标签。保持嵌套路径，按图片同名 `.txt` 配对；无类别名称时显示 `class_<id>`。
- YAML 相对 `path` 以 YAML 所在目录为基准；没有 `path` 则使用 YAML 所在目录。划分来源相对数据集根目录。网页「路径设置 → 根目录覆盖」可重新指定根目录；配置中的绝对来源仍原样使用。
- TXT 内绝对图片路径原样使用，`./`、`../` 路径相对 TXT 文件；其他相对路径分别检查 TXT 目录和数据集根目录，存在两个不同文件时提示歧义。全部路径规则独立于训练启动目录和 Ultralytics 全局设置。
- 读取本地已解压数据，不下载文件或执行 YAML 的 `download` 内容。常见 JPG/PNG/BMP/TIFF/WebP 等由 Pillow 解码；具体编解码器不支持的格式、损坏文件会在查看时提示，不中止其余样本的索引。

| 任务 | 每行标准标签 | 显示方式 |
| --- | --- | --- |
| 检测 | `class cx cy w h` | 矩形框与类别 |
| 分割 | `class x1 y1 … xn yn`，至少 3 个点 | 多边形轮廓与填充 |
| OBB | `class x1 y1 … x4 y4` | 四点旋转框 |
| 姿态 | `class cx cy w h` + N 组 `(x,y)` 或 `(x,y,v)` | 框、关键点与编号 |

所有坐标为相对图片宽高的归一化坐标。OBB 与四点分割的列数相同，必须选择正确任务，浏览器不猜测。姿态优先使用网页中填写的数量/维度，否则读取 `kpt_shape: [N, 2或3]`；缺少则提示补充。可见性 `0` 不显示，`1` 用空心点，`2` 用实心点；二维标签的负坐标按缺失关键点处理。关键点编号从 0 开始，不推测骨架连接。

### 浏览、统计与资源

- 按划分、类别、标签状态和文件名/相对路径筛选；每页 48 张缩略图。点击图片进入单图视图，可滚轮缩放、拖动、适应窗口，使用按钮或左右方向键翻页。
- 可以切换标注与文字，点击目标轮廓内或侧栏目标，查看原始标签行和归一化坐标。原图和缩略图都应用 EXIF 方向，标注以方向校正后的图片为坐标系。
- 类别筛选选择「包含该类别」的图片；统计和单图仍包含这些图片的全部有效目标。类别统计分别显示实例数与包含该类的图片数。跨划分重复图片按各划分条目计数，另显示唯一图片数；同一划分内的重复来源会去重。
- 空标签与缺失标签分别记录，均可浏览。非法类别、数值/坐标或列数错误定位至文件与行；同一文件的有效行继续渲染并计数。异常标注不会被修改、裁剪或写回。标准标签没有置信度，因此没有置信度统计。
- 后台索引仅扫描路径和标签，不解码图片、不保留全部目标几何；查看单图时才读取完整标注和解码图片。最多同时解码 2 张图片，缩略图内存缓存上限 32 MiB；图片接口仅接受当前索引登记的样本编号，不支持任意文件下载。
- 源数据变化后点击「重新扫描」。若标签自索引后改变，单图接口提示重新扫描，避免旧统计与新标注混用。一个服务仅保留一个活动数据集。
- 不创建磁盘缩略图或数据集标签缓存，不写入数据集。日志默认放在 `${XDG_CACHE_HOME:-~/.cache}/yolo-dataset-viewer/viewer.log`，5 MiB 轮转、保留 3 份；可通过 `--log-file` 修改。退出时取消索引任务并释放工作线程、内存索引和缩略图缓存。
- API 类型文档见服务的 `/docs`：打开数据集 `POST /api/datasets`；查询索引状态、分页样本、统计、单图详情和图片均使用返回的数据集 ID。

### 浏览器模块验证

```bash
uv run --locked --extra viewer pytest -q tests/test_viewer.py
uv run --locked --extra viewer mypy

# 真实 Chromium 交互测试；首次需要安装测试浏览器
uv run --locked --extra viewer --group browser playwright install chromium
CUDA_VISIBLE_DEVICES='' uv run --locked --extra viewer --group browser pytest -q
```

测试使用只含标准文件的外部风格数据，覆盖四种任务、目录和 TXT 解析、异常诊断、EXIF、只读性、索引取消及万级样本索引；浏览器测试覆盖渲染、缩放拖动、筛选分页和数据集切换。未安装 Playwright 或 Chromium 时浏览器测试跳过。测试截图保存在项目 `.cache/viewer-checks/`。

## 开发检查

```bash
uv run --locked mypy src/video_dataset_pipeline
uv run --locked pytest -q
uv run --locked ruff check src tests
uv run --locked ruff format --check src tests
```

测试覆盖时间戳采样、边界间隔、坐标裁剪/归一化、直接指定权重与 YAML、下载失败清理、中文视频路径、非空输出目录保护、失败状态、空数据集和不完整解码，以及记录序列化兼容性、错误记录行号、导出时保留原推理记录。

核心数据流使用 `records.py` 中的 `RawDetection`、`Detection`、`RejectedDetection` 和 `FrameRecord`。检测对象不可变，框为四元素元组；帧记录保存处理状态及产物路径。读取记录后通过 `record.timestamp`、`record.detections` 等属性访问，导出通过 `dataclasses.replace` 复制帧记录。图像仍是 uint8 BGR 数组，采样器仍返回 `(帧号, 时间戳, 图像)`。

`metadata.py` 使用 `TypedDict` 描述视频、划分、运行状态和报告，运行时仍是普通字典。`FrameRecord.to_dict/from_dict` 仅用于 JSON/JSONL 边界转换，保留 schema 4 的字段、坐标数组和可选字段省略规则，已有数据集无需迁移。

mypy 对整个核心包启用严格检查并禁止显式 `Any`，开发依赖由 uv 锁定。JSON/YAML 和第三方返回值在读取处进行类型收窄；唯一的局部导入忽略用于尚未提供 `py.typed` 标记的 `torchvision.ops`。PyTorch/Ultralytics 继续在设备配置后延迟导入。

2026-10-05 类型重构回归：48 个测试及上述开发检查通过。使用当前配置、GPU4 对整段视频重新生成：

```bash
uv run --locked video-dataset run --config configs/rm26_exchange.yaml \
  --output datasets/generated/rm26_exchange/recvlnoyv3arV6_typed_test
```

结果为 1320 帧采样、450 帧通过阈值、19 帧因分界间隔排除，最终训练集 351 张、验证集 80 张，包含 676 个目标框和 36 张预览。与 `recvlnoyv3arV6_nohash_test` 相比，两份逐帧清单各 1320 条记录的全部字段一致，431 份标签内容逐字节一致；两套数据均通过 Ultralytics 加载。对比结果保存在新输出目录的 `comparison.json`。再次运行上述命令时需换用新的输出目录。
