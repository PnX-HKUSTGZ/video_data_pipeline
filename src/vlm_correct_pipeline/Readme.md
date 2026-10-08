# YOLO 检测与 Pose 数据集的 VLM 自动标注与修正

通过 OpenAI Responses API，让视觉语言模型参考固定权威样例，在“观察 → 编辑检测框或关键点 → 查看渲染结果 → 再编辑”的交互循环中完成标注。程序维护标注状态并执行受限工具，模型使用完整图片的归一化坐标操作框与点，也可以明确将图片标记为无目标。

## 1. 当前状态与第一版范围

**第一版已实现**：提供 `validate`、`probe`、`run` 和 `audit-lighting` CLI 命令，支持逐框编辑、局部查看、显式负样本、光照剔除、检查点恢复和独立检测数据集导出。图片处理使用 CPU，不初始化训练模型或 GPU。

**Pose 扩展已实现**：新增 `dataset.input_task: pose`，联合复核框与关键点，支持逐实例批量点编辑、逐点确认状态、分工明确的框／关键点样例及独立 pose 导出。使用与实际验证情况见第 11 节。

| 内容 | 当前状态 |
| --- | --- |
| 本文的检测标注流水线设计与 CLI | 已实现 |
| 模块 `.env` 与 `.env.example` | 已存在，使用 `BASE_URL`、`API_KEY`；`.env` 已配置 Git 忽略 |
| 仓库已有 YOLO 读取能力 | 已通过适配层复用 |
| 工具执行、图片渲染、Responses 会话、恢复与导出 | 已实现，包含离线测试 |
| Responses 连通性、多图及多轮工具调用验证 | `gpt-6-astra` 合成场景和完整五组权威样例上下文的两轮预检已通过；每次有待处理图片时仍执行预检 |
| 真实场景标注质量评估 | national 的 50 张限定处理已结束，并发 2；导出 31 张并通过完整性校验，完成 Codex 视觉检查及 5 张针对性重检；仍缺独立人工真值 |

支持 YOLO 轴对齐检测框及 `[K,3]` pose，包括无标签图片的自动标注和已有标签的修正。`pose_boxes` 仍仅提取检测框，`pose` 才联合修正并导出关键点，源文件保持原样。分割、旋转框、二维无可见性字段的 pose、遮挡点推断、三维位姿解算、自动类别扩展、样例检索和人工编辑界面仍为未实现能力。

### 使用

在仓库根目录运行。依赖通过 `vlm` extra 提供，配置模板在本模块的 [config.example.yaml](config.example.yaml)。本地配置与 `.env` 均被忽略，不会提交凭据或个人路径。

```bash
cp -n src/vlm_correct_pipeline/config.example.yaml src/vlm_correct_pipeline/config.local.yaml
# 修改 config.local.yaml 的 dataset.source、output.directory，并确认标注规则。
uv run --locked --extra vlm vlm-correct validate --config src/vlm_correct_pipeline/config.local.yaml
uv run --locked --extra vlm vlm-correct probe --config src/vlm_correct_pipeline/config.local.yaml
uv run --locked --extra vlm vlm-correct run --config src/vlm_correct_pipeline/config.local.yaml
```

`validate` 解码并校验本地输入及全部样例，不访问 API；输出不合法图片的路径。`probe` 携带完整固定样例上下文，用少量合成图片检查多图与多轮能力；配置已提供类别表时不读取待标注数据。`run` 在存在待处理图片时自动执行同样的能力探测，再开始标注；重复运行同一配置即恢复运行。

退出码：`0` 为全部完成，`2` 为存在无效输入或未完成图片，`1` 为配置、服务或运行失败，`130` 为用户中断。进度记录在输出目录的 `logs/pipeline.log`，单图操作和最终结果分别见 `events/`、`samples/` 与 `report.json`。终端摘要只打印未处理数量 `not_processed_count`，完整未处理 ID 清单保留在报告的 `not_processed`，避免大批次中断时向终端输出数万个 ID。

## 2. 输入与权威样例

### 待处理数据

- 输入为本地 YOLO 数据集 YAML 或图片目录，配对标签使用标准的 `class_id cx cy w h` 文本格式。通过适配层复用现有数据集发现和标签读取能力，不导入训练模型。
- 类别表固定为从 0 开始的连续整数编号及名称。优先读取数据集 YAML；没有类别表的图片目录必须在任务配置中提供。两处同时提供时必须一致。
- 已有标签只是初始草稿，模型仍需检查所有目标。缺失标签与空标签分别记录，两者都不能直接认定为负样本。
- 图片无法解码、标签格式非法或类别不合法时，将该图片记为 `invalid_input` 并跳过。即使部分标签行有效，也不静默丢弃错误行后继续标注。
- 保留原有 split；无划分的输入归入 `train`。同一源图片被多个 split 引用时只进行一次模型标注，导出时保留各 split 的引用关系。
- `dataset.input_task` 默认是 `detect`；读取 pose 时必须显式指定 `pose_boxes`，校验原始完整关键点行后只提取检测框。输出 YAML 不含 `kpt_shape`，不修改或生成关键点。
- 可通过 `dataset.selection_file` 提供非空 JSON 图片路径数组，只处理这份固定清单；数组中的相对路径以源数据集根目录为基准，重复或未注册路径会报错。清单本身计入运行指纹。

### 固定权威样例

每个样例由原图、已确认正确的 YOLO 标签及可选说明组成。程序生成带框图和结构化标签，并将同一组样例提供给每张待处理图片的独立会话。样例与任务共用类别表，样例只读。

当前模板使用用户指定的五视角权威样例，完整副本位于 `datasets/vlm_correct_pipeline/references/yolo_five_views_20261007/`，来自 `/data/home/sim6g/code/tech_core_hkust/data/yolo_five_views_20261007`。复制校验报告为同级的 `yolo_five_views_20261007.copy-verification.json`：15 个文件、2,118,039 字节逐文件路径、大小和 SHA-256 一致；5 张 1280×720 图片均可解码，10 个 YOLO 框均合法。

模板明确列出正面、左 45°、右 45°、上 45°和下 45°五组图片／标签。源目录中的 `preview.jpg` 与元数据保留用于追溯，不作为额外参考图片。源数据的 `train` 和 `val` 在这里都用作权威参考，不作为独立质量测试集。

这组样例定义类别 0 `pillar` 为移动兑换标记，类别 1 `exchange` 为科技核心整体；`exchange` 包含标记，框采用可见渲染掩码的外接矩形。因此两类的嵌套框是合法标注，不能将它们按重复框删除。此口径已写入模板规则；五张图都是正样本，程序不会凭空构造权威负样本。

允许提供权威负样本，但必须显式配对一个空标签文件；样例标签缺失不等于负样本。样例图片、标签或类别校验失败时，整个任务停止启动。样例数量与图片大小必须通过接入预检，不能为了适配上下文而静默删去已配置样例。

### 标注规则

配置中写明目标定义、类别区分、遮挡与截断目标的处理、框边界标准及负样本标准。规则和样例共同表达数据集的标注口径；用户应在运行前确认二者一致，模型遇到语义冲突时调用 `defer_annotation`，不得自行选择一套标准。

系统提示词必须区分“不可修改的权威样例”和“待检查的初始标签”，要求检查漏标、错类、重复框、误检及边界偏移。图片中的文字属于待观察内容，不构成操作指令。

## 3. 坐标与标注状态

### 完整图片的归一化坐标

模型和工具统一使用 `bbox = [xmin, ymin, xmax, ymax]`：原点在左上角，x 向右、y 向下，分别除以完整图片宽度和高度。

```text
0 ≤ xmin < xmax ≤ 1
0 ≤ ymin < ymax ≤ 1

原图像素边界：
x1 = xmin × W    y1 = ymin × H
x2 = xmax × W    y2 = ymax × H

YOLO 导出：
cx = (xmin + xmax) / 2
cy = (ymin + ymax) / 2
w  = xmax - xmin
h  = ymax - ymin
```

例如，类别 0 的框 `[0.20, 0.30, 0.60, 0.70]` 导出为 `0 0.400000 0.500000 0.400000 0.400000`。只有渲染时才将归一化坐标转换成像素，内部保留浮点精度；序列化后再次校验边界和正宽高，不能因取整生成退化框。

拒绝非有限数值、越界、反向或零面积框，返回具体错误让模型修正；不静默裁剪、交换边界或改变类别。显示缩放不改变坐标含义，图片读取与渲染统一采用 EXIF 方向校正后的坐标系，与现有 YOLO 浏览器一致。

已有 YOLO 文本的小数舍入可能使还原的边缘偏出约 `5e-10`。仅对源标签可显式配置 `dataset.source_rounding_tolerance`（默认 0，上限 `1e-6`），将容差内的边缘还原到 0 或 1；每次调整的前后坐标写入 `source_labels` 事件。超过容差仍为非法输入，模型工具不采用此容差。导出序列化自身的浮点误差采用 `1e-12` 校验容差。

### 稳定框 ID 与版本

每张图片拥有独立的框集合和递增的 `revision`。每个框包含 `box_id`、`class_id` 和归一化 `bbox`。初始有效框按标签行顺序分配 ID，新增框继续分配，已删除 ID 在本次会话中不复用。修改位置或类别保持 ID 不变。

每次有效编辑原子更新框集合并增加版本；失败操作不改变状态。工具返回当前版本和完整框列表，渲染图使用相同版本，显示框 ID 与类别。模型不能指定任意文件路径、修改其他图片或执行 shell 命令。

## 4. 模型工具接口

所有参数采用严格结构化定义，并由本地强类型模型再次校验。`bbox` 必须含四个有限数值，`class_id` 必须属于固定类别表，`box_id` 必须存在于当前图片，`reason` 为非空简短说明。

| 工具 | 必需参数 | 行为与返回 |
| --- | --- | --- |
| `add_box` | `class_id, bbox` | 新增框，返回新 ID、完整框列表及最新整图渲染 |
| `update_box` | `box_id, class_id, bbox` | 提交该框的完整类别和坐标，返回更新后的列表及整图渲染 |
| `set_keypoints`（仅 pose） | `box_id, updates[{index,x,y,visibility}]` | 原子提交同一实例的一个或多个适用点，同时确认这些点；返回整图及目标局部对照 |
| `delete_box` | `box_id` | 删除指定框，返回最新列表及整图渲染；删除最后一个框不自动完成 |
| `inspect_region` | `bbox` | 返回局部原图、局部带框图和区域在完整图片中的范围，不修改标注 |
| `finish_annotation` | `reason` | 确认当前非空且合法的框集合已完成，终态为 `completed` |
| `mark_empty` | `reason` | 明确确认没有任务类别的目标，清空全部框并完成，终态为 `empty` |
| `defer_annotation` | `reason` | 无法可靠判断时结束，终态为 `deferred`，保留草稿供复核 |
| `discard_image` | `category, reason` | `category` 只能为 `poor_lighting`；因目标区域欠曝、眩光或过曝无法可靠标注而剔除整图，保留草稿和原因，不生成标签 |

工具失败返回可纠正的错误码、错误说明、当前版本和框列表，例如未知 ID 或越界坐标；不会产生成功图片或空标签。

`inspect_region` 的参数仍是完整图片的归一化坐标。返回内容应标注实际裁剪范围，包括像素取整后的边界及其对应的归一化范围。局部图仅辅助观察，**所有后续编辑仍以完整原图为坐标系**，不接受以裁剪图为基准的框坐标。

`finish_annotation` 不接受空框集合；模型必须使用 `mark_empty` 明确表达无目标。`mark_empty` 可以直接清空已有错误标签并结束，无需逐个删除。四个结束工具成功后立即关闭该图片的编辑会话。看不清目标不能据此确认无目标，背景较暗但目标清晰也不是剔除理由。

## 5. Responses 交互循环

每张图片建立独立会话，不继承上一张待处理图片的标签或推理上下文。

1. **初始化**：校验输入，加载初始框，注入系统提示词、规则、类别表和工具定义。
2. **提供样例**：按固定顺序提交每个权威样例的原图、带框图、正确标签及说明，明确图片与标签的配对关系。
3. **提供当前图片**：提交原图、当前带框图、方向校正后的宽高、版本及框列表。无框时明确说明初始标签状态，仍提供可检查的整图。
4. **请求模型**：通过 Responses API 发送上下文和工具定义，关闭并行工具调用。
5. **执行工具**：一次响应最多执行一个工具，先完整校验再修改状态。工具结果通过对应的 `call_id` 返回；编辑后提交最新框列表及带框图，裁剪查看后提交局部图与范围说明。图片使用 Responses 原生的多模态 `function_call_output` 内容，与该次工具结果一起返回。
6. **继续或结束**：模型查看新结果后继续操作，或显式调用结束工具。若达到调用上限、API 重试耗尽或发生中断，记录未完成原因。

原图始终保留在会话上下文中供对照，不能只让模型观察覆盖了目标细节的带框图。每次有效编辑都需要返回实际渲染结果，不能仅用文字宣称编辑成功。

普通文字回复不是完成信号。没有工具调用时提示模型使用约定工具继续，仍消耗调用预算；若服务端返回多个工具调用，则整组不执行，逐个返回协议错误并要求一次只调用一个工具。重复收到同一 `call_id` 时返回已保存结果，不重复执行编辑。

最后一次请求若只完成编辑，尚未经模型看到反馈并确认，就达到调用上限，则记为 `max_calls`；保留草稿但不导出为成功标签。

### 完整循环示例

假设图片尺寸为 1000 × 800，类别表为 `{0: "target"}`。用户的初始标签为：

```text
0 0.400000 0.500000 0.400000 0.400000
```

程序将它转换成 `box_0001 = [0.20, 0.30, 0.60, 0.70]`，版本为 0，并发送权威样例、待处理原图、当前带框图和框列表。之后的每一行代表一次模型响应和程序反馈：

| 轮次 | 模型调用 | 程序反馈 |
| --- | --- | --- |
| 1 | `inspect_region({"bbox": [0.10, 0.20, 0.70, 0.80]})` | 返回局部原图、带框图及实际裁剪范围；版本仍为 0 |
| 2 | `update_box({"box_id": "box_0001", "class_id": 0, "bbox": [0.18, 0.28, 0.62, 0.72]})` | 返回版本 1、完整框列表和更新后的整图渲染 |
| 3 | `add_box({"class_id": 0, "bbox": [0.75, 0.40, 0.90, 0.65]})` | 分配 `box_0002`，返回版本 2 和两框整图渲染 |
| 4 | `delete_box({"box_id": "box_0002"})` | 模型根据反馈判断第二处为误检并删除；返回版本 3 和单框整图渲染 |
| 5 | `finish_annotation({"reason": "已检查整图与局部，目标框已修正，未发现其他目标。"})` | 校验当前非空框集合，记录 `completed` 并写入导出暂存区 |

最终 YOLO 标签为：

```text
0 0.400000 0.500000 0.440000 0.440000
```

如果整张图片没有任务类别的目标，模型可直接调用 `mark_empty({"reason": "整图中没有符合类别定义的目标。"})`，终态为 `empty`，导出图片及零字节标签文件。如果无法区分类别，则调用 `defer_annotation`，该图片仅保留在报告和草稿中。两种情况不能互相替代。

## 6. API 与配置

使用 OpenAI Python SDK 的 Responses 接口。模块已有的环境变量格式为：

```dotenv
export BASE_URL="https://example.com"
export API_KEY="sk-xxxx"
```

这段只表示 `.env.example` 的占位格式，不是实际服务地址或可用密钥。使用支持 `export` 语法的 dotenv 解析器读取，不执行 `.env` 中的 shell 内容。显式进程环境变量优先，模块 `.env` 补齐缺失变量；通过配置中的变量名读取服务地址和密钥，分别传给 SDK 的 `base_url`、`api_key`。`BASE_URL` 按服务商给定的完整 API 基址使用，不自动猜测或追加 `/v1`。

模型名必须在任务配置中明确提供；模板已采用用户选定的 GPT-6-Astra 在当前服务中对应的模型标识 `gpt-6-astra`。任务配置、日志、指纹和报告均不得记录密钥，也不保存含认证头的原始请求。

观察图默认采用 JPEG 质量 95、4:4:4 色度采样，最长边上限为 2048；可设置 `image_format: png` 使用无损编码。`coordinate_grid: true` 在带框反馈图绘制原图归一化 x/y 网格，局部裁剪仍显示原图坐标值；无覆盖原图不加网格。national 配置启用此选项，并使用更清楚的框 ID 字体。编码和显示缩放仅影响发送给模型的观察图，源图片和导出副本保持原字节，坐标仍相对完整图片。图像编码设置属于运行指纹，恢复时不会静默更改。

以下是模块配置模板。所有相对路径以配置文件所在目录为基准；五视角参考已复制就位，`dataset.source` 仍需改为实际待标注数据路径。完整可复制文件见 [config.example.yaml](config.example.yaml)：

```yaml
api:
  env_file: .env
  base_url_env: BASE_URL
  api_key_env: API_KEY
  model: gpt-6-astra
  timeout_seconds: 120
  max_output_tokens: 8192
  stream: true
dataset:
  source: ../../datasets/to_annotate/data.yaml
  names:
    0: pillar
    1: exchange
  input_task: detect
  source_rounding_tolerance: 0
references:
  - image: ../../datasets/vlm_correct_pipeline/references/yolo_five_views_20261007/images/train/front.jpg
    label: ../../datasets/vlm_correct_pipeline/references/yolo_five_views_20261007/labels/train/front.txt
    description: 正面权威样例，pillar 是移动兑换标记，exchange 框包含标记和科技核心整体。
  - image: ../../datasets/vlm_correct_pipeline/references/yolo_five_views_20261007/images/train/left_45.jpg
    label: ../../datasets/vlm_correct_pipeline/references/yolo_five_views_20261007/labels/train/left_45.txt
    description: 左侧 45 度权威样例。
  - image: ../../datasets/vlm_correct_pipeline/references/yolo_five_views_20261007/images/train/right_45.jpg
    label: ../../datasets/vlm_correct_pipeline/references/yolo_five_views_20261007/labels/train/right_45.txt
    description: 右侧 45 度权威样例。
  - image: ../../datasets/vlm_correct_pipeline/references/yolo_five_views_20261007/images/train/up_45.jpg
    label: ../../datasets/vlm_correct_pipeline/references/yolo_five_views_20261007/labels/train/up_45.txt
    description: 上方 45 度权威样例。
  - image: ../../datasets/vlm_correct_pipeline/references/yolo_five_views_20261007/images/val/down_45.jpg
    label: ../../datasets/vlm_correct_pipeline/references/yolo_five_views_20261007/labels/val/down_45.txt
    description: 下方 45 度权威样例；源数据的 val 划分在本任务中同样作为参考。
annotation_rules: |
  类别 0 pillar 是移动兑换标记，类别 1 exchange 是科技核心整体。
  exchange 的框包含科技核心及其移动标记，允许与 pillar 框嵌套，不能将二者视为重复框删除。
  按可见目标轮廓取紧致的轴对齐框，遵循五视角权威样例的分割掩码边界标准。
  不根据背景或样例中的固定位置臆造目标；只有确认整图没有上述目标时才标记为空。
  无法可靠判断类别、目标存在性或遮挡后的可见边界时，提交待复核。
run:
  image_concurrency: 1
  max_model_calls: 30
  max_retries: 3
  max_consecutive_api_errors: 3
image_max_edge: 2048
image_format: jpeg
jpeg_quality: 95
coordinate_grid: false
output:
  directory: ../../outputs/vlm_correct/example_run
```

`max_model_calls` 是每张图片的实际 Responses 请求上限，包含网络重试。`max_retries` 表示同一逻辑请求在首次发送后最多额外重试 3 次，且不能突破单图总请求上限。超时、限流与可恢复服务端错误采用有界退避；认证失败或不支持的协议能力不重复重试，停止批次并记录原因。启动能力探测的请求单独统计，使用同一 `max_retries` 配置；两轮探测的总请求上限为 `2 × (max_retries + 1)`。

`api.stream: true` 使用 SDK 的 Responses 流式传输，收到完整响应后才执行工具，不执行半段参数。`false` 使用普通 JSON 响应。两种方式都保持相同的工具和上下文协议；流式传输不保证第三方网关一定支持多图多轮。`run.max_consecutive_api_errors` 默认 3，达到连续失败图片数后停止派发新图片；未开始的图片列入 `not_processed`，恢复时重新处理。

`api.timeout_seconds` 同时约束整次请求的总时长。内部使用异步 SDK 和可取消的超时上下文，每次请求独立关闭连接，外部仍是同步的图片工作线程；服务持续发送保活信息也不能无限延长请求。流中途断开按临时传输故障重试，不执行不完整工具参数。

### 会话与接入预检

- 客户端维护完整的必要多模态历史，保留正式输出中的工具调用及所选模型要求回传的推理项目，正确关联工具返回；不依赖服务端会话存储，不要求模型输出内部推理过程。
- 流式 SDK 的 `parsed_arguments`／`parsed` 属于本地解析结果，不能放入下一轮 API 输入；序列化时移除这些字段和未设置的空值，同时保留推理项目、工具调用 ID 和消息 `phase`。回归测试检查实际发送到第二轮请求的 JSON。
- 工具声明使用严格 schema，本地执行器再次校验。初始图片通过多模态消息提交；后续观察图与结构化状态共同放入 `function_call_output.output` 的文本／图片数组，明确关联工具调用和标注版本。
- 启动时携带完整固定样例上下文，用小型合成图片验证多图输入、严格工具参数、关闭并行调用及连续多轮行为：模型调用工具后收到匹配结果与新图，能继续下一轮并结束。这同时检查配置的整组样例能否被端点接收；该探测通过后才处理真实数据。
- 接口兼容性不能仅靠成功列出模型或完成一次文字回复判断；不满足必要能力时停止启动并给出具体失败项。

2026-10-07 已通过当前服务的 `gpt-6-astra` 和 OpenAI Python SDK 2.54.0 验证原生工具图像返回、多图计数、随机工具回执及下一轮新图观察。早期将反馈图片放在独立后续消息中的探测出现旧图计数，实际实现采用已通过探测的多模态工具返回方式；不把不同兼容服务视为具有相同能力。

官方文档页面在当前环境返回 HTTP 403，因此本次接口验证以已安装 SDK 类型、HTTP 协议测试和真实调用为依据。参考入口：[Responses API](https://platform.openai.com/docs/api-reference/responses)、[Function calling](https://platform.openai.com/docs/guides/function-calling)、[Images and vision](https://platform.openai.com/docs/guides/images-vision)。更换模型或端点后仍需通过启动预检。

## 7. 终态与独立数据集导出

| 单图状态 | 含义 | 是否进入成功数据集 |
| --- | --- | --- |
| `completed` | 模型明确确认当前非空框集合完成，本地校验通过 | 是，写入非空 YOLO 标签 |
| `empty` | 模型通过 `mark_empty` 明确确认无目标 | 是，保留图片并写入零字节 `.txt` |
| `discarded` | 光照不合格，保留 `poor_lighting` 类别及具体原因 | 否；已确认终态，恢复时跳过 |
| `deferred` | 模型无法可靠判断，附待复核原因 | 否 |
| `max_calls` | 请求预算耗尽，未得到明确完成结论 | 否 |
| `api_error` | 接口失败且不可恢复或重试耗尽 | 否 |
| `invalid_input` | 待处理图片或原始标签不合法 | 否 |
| `interrupted` | 任务中断，当前图片未完成 | 否 |

运行中另有 `pending` 和 `running` 状态，不属于终态。启动级别的配置、样例或能力校验失败记录在运行报告中，不伪造任何单图成功结果。

输出结构如下，`output.directory` 指向运行目录：

```text
example_run/
  dataset -> .exports/<snapshot_id>/
    data.yaml
    images/<split>/.../<image_name>
    labels/<split>/.../<image_stem>.txt
  run.json                     # 配置快照（不含密钥）、运行指纹与总体状态
  capability_probe.json        # 启动能力探测的请求、用量、工具数量与失败证据
  progress.json                # 当前阶段、已处理/剩余图片数和更新时间
  report.json                  # 状态计数、源文件到输出文件映射、异常及用量
  samples/<sample_id>.json      # 单图状态、当前/最终标签和简短原因
  events/<sample_id>.jsonl      # 工具操作、版本、标签变化及调用用量
  logs/pipeline.log
  .exports/<snapshot_id>/       # 完整数据集快照，含校验清单 .vlm-export.json
  .tmp/                        # 本次运行拥有的临时文件，结束时清理
```

- 只复制成功图片，保持原文件内容与元数据；导出标签和固定类别表，保留原 split。保留源图片树内的嵌套路径，移除原布局的 `images` 和重复 split 前缀，使用稳定样本 ID 消歧同名来源，映射写入报告。
- 输出必须独立于输入和权威样例目录，不能覆盖原图、原标签或写回权威样例。
- `data.yaml` 只引用已完成且文件写入完整的样本。输出校验通过后才发布数据集；没有成功图片时只保留报告，不发布可训练数据集。
- 文件写入和单图终态采用暂存加原子替换。整个数据集验证完成后，原子切换相对符号链接 `dataset`，使中断不会暴露一半旧文件、一半新文件；发布成功后清理旧快照。对外入口为 `dataset/data.yaml`，复制数据集到其他地方时需解引用 `dataset` 链接，或复制完整运行目录。
- 发布前核验图片与标签一一对应。磁盘写入失败必须记录为运行失败，不能把内存中的完成结论当作已成功导出。
- 报告按唯一源图片统计单图状态，按 split 单独统计导出条目数，明确负样本、排除图片及待复核原因。零字节标签只能来自 `empty`。

## 8. 恢复、资源与模块边界

### 持久化与恢复

逐图片记录源图片与标签、配置、类别规则、固定样例的指纹，以及模型名和 API 服务配置；密钥不进入指纹。恢复时核验运行身份和输出完整性，只跳过已完成或已确认剔除、且输入与配置均未变化的图片。`completed`、`empty`、`discarded` 都属于已处理终态，但只有前两种进入导出数据集。

同一输出目录使用进程锁，避免两个运行同时写入。已成功发布且校验未变的快照直接复用，不重复复制图片或调用 API；已完成单图但发布前中断时，可由检查点重新构建输出。用户手动修改已发布文件会阻止覆盖，需保留原运行并使用新的输出目录。

未完成图片保留原尝试的审计记录，新增尝试从原始图片和初始标签重新开始，不把半轮草稿自动视为正确结果。配置、样例或源内容改变时拒绝复用旧运行目录，使用新输出目录，避免不同任务结果混合。进程异常退出留下的 `running` 视为中断，不能发布。

### 资源管理

图片读取、裁剪和渲染使用 CPU；本模块不初始化 GPU、Torch 或 Ultralytics 推理。默认同一时间只处理一张图片；提高并发时，每张图片仍独立维护会话、状态和预算。

运行前按全部候选图片的复制体积估算输出空间，并为标签、审计和暂存文件预留空间，空间不足时停止。图片按需读取，不把整个数据集解码到内存；日志不保存图片 base64。诊断日志采用 5 MiB 轮转、保留 3 份；单图操作记录按样本分文件保存，受请求预算约束，作为可追溯结果保留。

正常结束或取消时释放图像对象、文件句柄和工作线程，清理本次运行拥有的临时裁剪与渲染文件；异常退出后的残留由恢复流程识别清理。保留必要的单图检查点与审计记录，清理范围不能扩展到源数据集或其他运行。

### 实现分层

| 子系统 | 职责 |
| --- | --- |
| 数据适配 | 通过独立适配层复用现有 YOLO 读取、路径与 split 发现能力；执行本模块输入校验 |
| 提示词与样例 | 读取规则和固定权威样例，构造有明确角色的多模态输入 |
| 工具与状态 | 强类型参数、框 ID、版本、原子编辑、结束条件和调用去重 |
| 图片渲染 | 方向校正后的原图、带框整图、局部裁剪及范围映射 |
| Responses 客户端 | 环境配置、能力探测、请求预算、重试和多轮上下文 |
| 运行与导出 | 样本调度、持久化、恢复、报告、独立 YOLO 数据集发布及资源清理 |

核心函数与类提供注释，核心接口保持强类型。模块配置放在本模块目录，使用方法仅记录在本 README。已接入仓库的 uv 构建、`vlm` 依赖组、`vlm-correct` 入口及 mypy 检查；发行包排除 `.env`。

## 9. 验证与验收清单

以下为实现的验收项目。离线测试位于 `tests/test_vlm_pipeline.py`，与现有 YOLO 读取／导出回归测试一同运行；真实场景质量评估仍需独立人工真值集。

```bash
uv run --locked --extra vlm --extra viewer pytest -q tests/test_vlm_pipeline.py tests/test_vlm_pose_*.py tests/test_viewer.py tests/test_viewer_exports.py
uv run --locked --extra vlm mypy
uv run --locked --extra vlm ruff check src/vlm_correct_pipeline tests/test_vlm_pipeline.py
```

| 验证类别 | 用例与验收条件 |
| --- | --- |
| 坐标与标签 | xyxy 与 YOLO 往返转换、四边边界、非法数值、退化框、类别与 ID 校验；序列化后仍合法 |
| 图像与局部查看 | 非正方形图片、缩放、边缘裁剪与 EXIF 方向；局部查看后编辑仍对应完整原图 |
| 工具闭环 | 新增、移框、改类、删除、裁剪；下一轮得到相同版本的框列表与真实渲染图；错误后状态不变且可纠正 |
| Responses 协议 | 多图输入、工具调用与返回配对、推理项目回传、一次一个工具、重复调用去重、多轮继续 |
| 结束状态 | 直接确认已有标签、从零标注、明确清空为负样本、无法判断、普通文本、调用超限与 API 失败；只有 `mark_empty` 可生成负样本 |
| 输入异常 | 缺失标签与空标签均需检查；非法待处理标签只跳过对应图片，非法权威样例阻止启动 |
| 恢复与落盘 | 编辑后中断、完成后中断、写盘失败、原子发布、输出缺失与指纹变化；未完成结果不能混入成功数据集 |
| 导出兼容性 | 原图和原标签未变；成功图片与标签一一对应，类别与 split 保持一致，同名图片不覆盖；现有 YOLO 浏览器可以重新打开结果 |
| 资源与保密 | 不初始化 GPU，日志轮转与临时文件清理有效；配置快照、错误日志、报告与 Git 变更不含密钥 |

真实效果使用一组**未放入权威样例集**的小型人工核验集评估，覆盖漏标、错类、框偏移、小目标、遮挡和负样本。比较修正前后的类别正确性、漏检、误检和 IoU，同时记录成功率、待复核率、调用量和耗时。新增标注任务没有初始结果时，直接与人工真值比较。

模型声明完成只代表交互流程结束，不等于标注正确。第一版不以模型自报置信度代替人工核验，也不预先承诺自动修正一定提升准确率；真实质量结果与协议、格式测试分别报告。

截至 2026-10-08，相关离线与 YOLO 浏览／导出回归测试共 139 项通过，仓库 mypy 检查通过。真实 `gpt-6-astra` 合成端到端测试完成了新增框、修正错误框和清空错误框三个样本，输出为 2 张正样本、1 张负样本；两个正样本与合成真值的 IoU 均为 1.0，导出结果可由现有浏览器读取。目标图片共调用 5 次，另有 2 次启动探测；恢复运行未增加模型调用。

本地合成验证记录为 `.cache/vlm-correct-checks/smoke/verification.json`，仅证明接口、坐标和流程在这些简单场景中成立，不代表真实 `pillar`／`exchange` 数据的标注准确率。发行包已检查排除 `.env`、本地配置及 `.checks` 测试文件。

## 10. national 数据集光照审计与试运行

源数据集为 `datasets/generated/rm26_two_stage/national_20261007_conf08_kpt05/data.yaml`，包含 **23,620 张图片、14,907,431,994 图片字节**，train 为 19,587 张，val 为 4,033 张。原标签是 `kpt_shape: [12, 3]` 的 pose 格式。本次配置按已确认要求显式使用 `pose_boxes`，只输出检测框，保留源 pose 数据。

先运行只读的 CPU 光照审计：

```bash
uv run --locked --extra vlm vlm-correct audit-lighting \
  --source datasets/generated/rm26_two_stage/national_20261007_conf08_kpt05/data.yaml \
  --output outputs/vlm_correct/national_20261007_lighting_audit \
  --workers 4
```

输出 `records.jsonl`、`summary.json` 和 `COMPLETE.json`；重复审计需使用新的空输出目录。亮度指标基于缩略图，分别统计整图及已有检测框区域。候选规则为：某个目标区域暗像素（亮度 ≤16）占比 ≥60% 且 p90 <65，或亮像素（亮度 ≥245）占比 ≥60% 且 p10 >180。没有原始框时才用整图指标。规则只用于筛选视觉检查候选，既不确认光照合格，也不自动删除或生成负样本。

本次已完整扫描全部图片：**89 张欠曝候选、0 张过曝候选、23,531 张未触发阈值**，没有记录到图片读取或标签错误。未触发阈值不代表标注正确或光照合格，尤其需继续检查小目标、局部眩光及原始漏标。

已准备 [config.national_pilot.yaml](config.national_pilot.yaml) 和固定清单 [national_pilot32.json](national_pilot32.json)：8 张欠曝候选、6 张目标偏暗、6 张亮部较强、6 张小目标、6 张普通场景，合计 train 22 张、val 10 张，尽量覆盖不同视频。选择依据与预览位于审计目录的 `pilot32-selection.json`、`pilot32-1.jpg`、`pilot32-2.jpg`；这不是人工真值集。

```bash
uv run --locked --extra vlm vlm-correct validate --config src/vlm_correct_pipeline/config.national_pilot.yaml
uv run --locked --extra vlm vlm-correct run --config src/vlm_correct_pipeline/config.national_pilot.yaml
```

试运行采用 `gpt-6-astra`、完整五视角参考、最长边 1280 的 JPEG 质量 85、流式 Responses、图片并发 4、单图最多 30 次调用。目标原图尺寸与字节不变，可通过裁剪查看更细的局部。光照判断基于目标是否可可靠标注；严重欠曝、过曝或眩光使用 `discard_image`，其他不确定情况提交待复核。源转播画面按主视角标注，画中画和 UI 不重复标注。

初次试运行停在 API 预检，非流式请求曾出现 HTTP 504、超时及非标准响应。2026-10-08 继续诊断发现客户端回传了流式 SDK 的本地解析字段，导致第二轮 HTTP 400。修正序列化后，完整五组样例的两轮预检通过。标注规则明确：框包围可见核心壳体及标记，排除外部支撑、机器人和围墙，正常遮挡按可见范围标注，不要求恢复像素级掩码。

最终试运行输出为 `outputs/vlm_correct/national_20261008_pilot32_grid/`，旧目录保留诊断记录。32 张全部处理完毕：14 张 `completed`、1 张 `empty`、12 张光照 `discarded`、5 张 `deferred`，无 API 失败或未派发图片；整体状态为 `partial`，因为待复核图片未完成标注。目标图片共 98 次请求、1,674,477 输入 token 和 5,867 输出 token，启动探测另计。该清单刻意覆盖困难样本，其剔除率不能外推到全量数据。

试运行导出 15 张检测图片（train 12、val 3）。`verification.json` 记录原图片和 pose 标签哈希未变、权威样例哈希未变、输出哈希、五列检测标签、显式负样本、split、排除待复核和剔除图片、浏览器读取及临时文件清理检查。修正前后视觉对照保留在 `review/`；这些检查证明流程与导出完整性，不构成独立人工真值评估。

结果与终态以 `report.json`、`capability_probe.json` 和 `samples/` 为准；运行中报告可能仍是上一次尝试的终态，应结合 `run.json`、`progress.json` 与实际进程判断。网络异常和未完成草稿不会发布为训练数据。

### 已完成的限定任务：50 张，并发 2

按用户最新要求，当前只处理 **50 张输入图片，并发为 2**，不继续派发全量数据。配置为 [config.national_50_c2.yaml](config.national_50_c2.yaml)，固定清单为 [national_50.json](national_50.json)：包含原有 32 张困难样本，另从其他视频中分散选取 18 张，共覆盖 50 个视频，train 37 张、val 13 张。这是定性检查样本，不是独立人工真值集或无偏随机样本。

本次使用 `gpt-6-astra`、五组固定权威样例和 2 路图片并发，输出目录为 `outputs/vlm_correct/national_20261008_50_c2/`。**50 张均已处理至明确终态，最终导出 31 张**，入口为 [dataset/data.yaml](../../outputs/vlm_correct/national_20261008_50_c2/dataset/data.yaml)。检测数据含 train 23 张、val 8 张，共 57 个框（`pillar` 27 个、`exchange` 30 个），不含关键点。

| 终态 | 图片数 | 导出行为 |
| --- | ---: | --- |
| `completed` | 30 | 导出图片及非空检测标签 |
| `empty` | 1 | 导出图片及空标签；由模型明确确认无目标 |
| `discarded` | 13 | 目标区域光照不足以可靠标注，排除整图 |
| `deferred` | 6 | 目标与机械结构或 UI 遮挡后的边界无法可靠确定，排除整图，保留草稿与原因 |

`progress.json` 为 `finished`，剩余图片和待重检数量均为 0；`report.json` 的 `not_processed` 为空，没有 API 失败、调用超限或仍在运行的图片。运行总状态仍为 `partial`，表示 6 张待复核图片没有完成标注，不表示后台仍在处理。所有 19 张排除图片及逐图原因见 [exclusions.json](../../outputs/vlm_correct/national_20261008_50_c2/exclusions.json)，完整结果见 [report.json](../../outputs/vlm_correct/national_20261008_50_c2/report.json)。原 pose 数据保留。

目标图片各次尝试和网络重试累计 **253 次请求**，服务端已返回的用量合计 3,526,881 输入 token、20,190 输出 token；能力探测另计，失败请求可能没有返回 token 用量。未额外处理清单外图片。

首轮结果经 Codex 视觉检查后，将 5 张可能有误的图片附带具体观察反馈重新交给 Responses 模型核验；所有最终标签仍由模型通过既定工具提交，没有直接手写修正坐标。重检纠正了一张误判为空的图片，恢复漏掉的标记，调整核心框的上边界，并将一张边界不明的图片转为 `deferred`。最终五张原图／带框局部对照和检查记录位于 [review/final-visual-review.json](../../outputs/vlm_correct/national_20261008_50_c2/review/final-visual-review.json)。`review/positive-*.jpg` 是首轮修正前的对照图，最终结果以当前标签和 `review/final-*.jpg` 为准。

这是本批数据的一次针对性质量检查，**尚未实现通用自动质量门禁**。视觉检查由 Codex 完成，不是独立人工真值评估；这批样本包含刻意选择的困难图，不能据此推算全量剔除率，也未测得类别准确率、漏检率或 IoU 的改善。

主运行请求超时为 120 秒；针对性重检遇到网络超时后，仅最后一次重检将请求时限改为 300 秒，图片并发始终为 2。重检复用了同一端点与配置已通过的多轮能力探测。反馈、运行脚本副本及 SHA-256、重检前快照、探测来源和超时覆盖值均记入 `review/targeted-recheck-102334/`，并由 `run.json`／`report.json` 的 `quality_review` 引用；早期失败尝试继续保留用于追溯。

[full-verification.json](../../outputs/vlm_correct/national_20261008_50_c2/full-verification.json) 已在最终重检发布后通过：核对全部 50 张源输入指纹和 split、权威样例、终态工具与事件、检查点、请求用量、31 组导出文件的内容与哈希、独立图片副本、5 列检测标签、显式负样本、排除清单，以及现有 YOLO 浏览器加载结果。临时目录为空；校验脚本副本保存在 `review/final-verifier.py`。另有全量 23,620 张原图和 pose 标签的源完整性复核，以及全部 15 个权威样例文件的原目录／副本哈希复核，记录分别为 `review/full-source-integrity.json`、`review/reference-reverification.json`。

执行本轮使用以下命令。当前任务已结束，查看结果直接打开导出 YAML；再次运行会重新尝试 `deferred` 图片并更新结果，不是只读查询：

```bash
uv run --locked --extra vlm vlm-correct run --config src/vlm_correct_pipeline/config.national_50_c2.yaml
```

此前四路的 50 张尝试 `outputs/vlm_correct/national_20261008_50/` 在多轮预检第二轮连续遇到 HTTP 504，未开始目标图片处理，也未发布数据集。该目录及原配置保留用于诊断；本轮有效结果以 `national_20261008_50_c2/` 为准。

### 已停止的全量尝试

全量配置 [config.national_full.yaml](config.national_full.yaml) 和 [config.national_full_c16.yaml](config.national_full_c16.yaml) 保留供以后使用，当前均未运行。四路尝试实测吞吐约 1.3 张／分钟，已正常中断；十六路尝试出现持续 HTTP 504、连接及总时长超时，由连续失败保护自动停止，未发布数据集。记录分别保留在 `outputs/vlm_correct/national_20261008_full/` 与 `outputs/vlm_correct/national_20261008_full_c16/`，不能作为完整优化结果。

配置发生变化时使用新的输出目录，不跨配置拼接检查点。完整光照审计中的 89 张候选不等于最终剔除名单，最终以视觉检查的 `discarded` 结果为准。

2026-10-08 定位校准：一次试标把核心上方的弯曲安装结构纳入框中。单独增加网格没有纠正该错误；明确“盾形壳体不包括上方管线和回环支架”后，同一真实图片的上边界从 0.157 修正为 0.228。受控回归记录位于 `.cache/vlm-correct-checks/grid_case_housing/output/quality_check.json`，仅验证这一已观察错误的修正，不替代独立人工质量评估。

## 11. Pose 联合标注

### 输入、点序与可见性

`dataset.input_task: pose` 必须同时提供 `pose` 配置：`kpt_shape: [K,3]`、`kpt_names`、`class_keypoint_indices`、逐点物理定义 `point_definitions`，以及可选的 `flip_idx`、`skeleton`。点数、点序及翻转／连线元数据与源 YAML 冲突时停止启动。无标签图片目录可以由配置补充类别及点数。

national 配置采用固定的 12 槽：

| 类别 | 适用索引（从 0 开始） | 点名 |
| --- | --- | --- |
| `pillar` | 0–4 | TL、TR、BL、BR、ring |
| `exchange` | 5–11 | light_BR、light_TR、shell_R、shell_M、shell_L、light_TL、light_BL |

点名属于物体自身，不能随画面旋转按屏幕左右重新编号。每个实例始终保留全部 12 槽，不适用槽位严格为 `0 0 0`；源标签违反此规则则该图片为 `invalid_input`。

所有点坐标均相对完整原图归一化。最终标签只包含可可靠定位的可见点 `v=2`，以及不可见、遮挡、出画或位置不可靠的 `0 0 0`。源 `v=1` 可以作为草稿读取，但模型必须明确改为 `v=0/2`；适用槽位的源 `v=0` 若带旧坐标，规范为零并记录 `source_labels` 事件。非适用槽位不采用该规范化例外。ring 是离开四角平面的独立几何参考点，不是四角中心；无法直接观察时置零，不能进行几何推算。

### 固定样例的作用范围

参考项支持 `scope: annotation | boxes | keypoints`，默认 `annotation`。pose 运行中的 `annotation` 同时约束框与点，`boxes` 读取检测标签，仅约束框，`keypoints` 读取 pose 标签，仅约束点。

本批使用十组参考：原五视角检测样例继续定义 `pillar` 面板框和 `exchange` 核心框；另五张来自 HKUST `experiment_v2/exchange_pose` 的样例，仅定义关键点。pose 样例副本位于 `datasets/vlm_correct_pipeline/references/exchange_pose_visible5/`，文件为 `val_frame_003488`、`002498`、`000395`、`000479`、`000607`。原标签和适配标签分开保存，55 个可见点坐标原样保留，仅五个 ring 遮挡点规范为零，来源、哈希与转换记录见 `manifest.json`、`copy-verification.json`。

新的 pose 样例源框包含伸出的装配柱，与本任务的面板框定义不同。因此 `keypoints` 参考的结构化内容、图像覆盖和裁剪逻辑均不使用其源框；模型只收到原图、点身份和编号点图。生成器的几何无遮挡状态不保证肉眼可定位，样例准备阶段还需视觉检查。真实场景仍按可见性规则重新判断。

### 点编辑与完成条件

每个适用点保存独立的 `reviewed` 状态。已有 `v=2` 和 `v=0` 均为待复核草稿，新建实例的适用点也全部待标注；不适用槽位直接确认缺失。模型使用 `set_keypoints` 提交全部适用点，可以分多次提交，也可以一次提交一个实例的全部点：

```json
{
  "box_id": "box_0001",
  "updates": [
    {"index": 0, "x": 0.42, "y": 0.36, "visibility": 2},
    {"index": 1, "x": 0.49, "y": 0.37, "visibility": 2},
    {"index": 2, "x": 0.41, "y": 0.46, "visibility": 2},
    {"index": 3, "x": 0.48, "y": 0.47, "visibility": 2},
    {"index": 4, "x": 0, "y": 0, "visibility": 0}
  ]
}
```

这是参数示例，不是任何真实图片的真值。重复索引、非适用索引、越界／非有限坐标、`v=1` 或非零坐标的 `v=0` 会拒绝整次修改，状态与版本保持不变。

同类改框不平移或缩放关键点；改类别后重置全部点，必须复核新类别适用槽位。删除实例连同其点一起删除。可见点在框外时返回警告并要求模型复查，不自动吸附或扩大框。

每次有效框／点编辑后反馈最新整图，及编辑实例周围的原图／带框点局部对照。查看区域按框与可见点的并集外扩 10%，局部图仍显示完整原图坐标。关键点连线仅连接当前实例中可见的端点，原图始终保留供对照。

`finish_annotation` 要求所有适用点均已明确复核，每个实例至少有一个可靠可见点。部分点不可定位可以置零后保留其余点；任一真实目标没有可定位点则整图待复核，光照导致的问题按原规则剔除。不能删除真实目标或标为空来规避缺点检查。普通文字及调用超限均不算完成。

### 运行、导出与验证

```bash
uv run --locked --extra vlm vlm-correct validate --config src/vlm_correct_pipeline/config.national_pose_50_c2.yaml
uv run --locked --extra vlm vlm-correct run --config src/vlm_correct_pipeline/config.national_pose_smoke5.yaml
uv run --locked --extra vlm vlm-correct run --config src/vlm_correct_pipeline/config.national_pose_50_c2.yaml
```

主配置仍读取原始 national pose 数据及 `national_50.json`，包含 train 37 张、val 13 张。冒烟清单选自同一 50 张，包含此前的 1 张负样本和 4 张正样本；此前检测结果不作为本轮真值。两轮均采用 `gpt-6-astra`、并发 2、每张每次尝试最多 30 次请求和最多 3 次临时故障重试，输出分别为 `outputs/vlm_correct/national_pose_smoke5_c2/` 与 `outputs/vlm_correct/national_pose_50_c2/`。

真实 pose 请求携带十组参考及关键点状态，初始 120 秒时限连续触发超时；该尝试已正常中断，诊断保留于 `national_pose_smoke5_c2_timeout120/`，没有发布数据集。300 秒、JPEG95 的尝试也遇到超时和 HTTP 502，原始事件与检查点保留于 `national_pose_smoke5_c2_timeout300_jpeg95/`，未确认草稿没有导出。

当前两份 pose 配置的请求总时限为 300 秒，传输 JPEG 质量为 90，其他调用预算与并发不变。`reference_max_edge: 768` 仅缩放参考图的传输副本；`image_max_edge: 1280` 继续用于目标整图与局部反馈。参考源文件、标签和完整原图归一化坐标均不变。该字段默认空，沿用 `image_max_edge` 并从运行身份中剔除，因此不改变旧运行指纹；显式设置则纳入指纹。

传输诊断中，同一端点的最小工具请求耗时约 3 秒；20 张 1280 像素参考图请求已完整上传，却在 300 秒内未收到 SDK 响应事件。相同十组参考缩至 640／768 像素后均通过两轮能力检查，分别约 73／126 秒。原 1280 像素 JPEG90 尝试保留于 `national_pose_smoke5_c2_reference1280_jpeg90/`。较早 JPEG95 请求首次超过约 16 MiB 后出现快速 502，但未证实具体服务限制；缩放减轻请求负载，不构成精度改善证据。

流式客户端收到 `response.completed` 后使用 SDK 已解析的完整响应并关闭连接，不再额外等待 EOF；没有完成事件时仍执行总时限，半截工具参数不会执行。超时诊断仅记录已观察 SDK 事件数量、首事件耗时及白名单内的末事件类型，不记录请求内容。事件数为零不能区分无响应与被 SDK 忽略的 SSE 心跳；该兼容性修复是否解释真实服务超时，需要实际请求证据。

仅 `completed`、`empty` 导出到新的数据集，保留 split；非空标签每行 `5+3K` 列（本批 41 列），YAML 写入逐类 `kpt_names`、`kpt_shape`、已配置的翻转与显示连线。源文件及上一轮检测导出不变。

pose 使用运行格式版本 2，点定义、作用范围和适配参考的哈希都属于运行身份。旧检测配置继续使用原格式版本 1 的身份序列化，不因新增默认字段失去恢复能力。已完成 pose 结果可直接恢复，未完成图片重新从源标签开始；输入、点语义或配置改变则需新输出目录。

当前离线集成与既有浏览器回归共 197 项通过，包含真实旧版检测检查点的零 API 恢复、pose 完成门槛、原子批量编辑、裁剪反馈、输入冲突及浏览器读取，以及流完成后持续心跳、异常终态和独立参考图尺寸；仓库 46 个源文件的 mypy 及相关 Ruff 检查通过。

真实 Responses 合成验证已完成：一张四点图从偏移草稿修正至四个已知圆心，误差均为 0 像素；一张白图明确清空错误实例并确认为负样本。目标图共 3 次请求，另有启动预检，记录在 `.cache/vlm-correct-checks/pose-synthetic/verification.json`。这只证明该简单场景的工具与坐标闭环；尚无独立人工点真值，不能宣称准确率或定位精度改善。

### 固定 50 张 pose 运行结果

2026-10-08 完成模型处理，2026-10-09 完成导出完整性核验。先行 5 张冒烟运行得到 1 张正样本、1 张负样本、2 张光照剔除和 1 张待复核，目标请求共 13 次；全部检查通过，记录位于 `outputs/vlm_correct/national_pose_smoke5_c2/full-verification.json`。主运行重新检查原固定清单的全部 50 张，没有复用冒烟或旧检测标签作为真值。

| 最终状态 | 图片数 | 是否导出 |
|---|---:|---|
| `completed` | 13 | 是 |
| `empty` | 1 | 是，标签为空 |
| `discarded`（光照） | 12 | 否 |
| `deferred` | 24 | 否，需人工复核 |
| API 失败／调用超限／未处理 | 0 | 否 |

输出数据集为 [national_pose_50_c2/dataset/data.yaml](../../outputs/vlm_correct/national_pose_50_c2/dataset/data.yaml)，共 **14 张**，train 12、val 2。`progress.json` 为 `finished`、剩余 0，运行状态 `partial` 表示有待复核图片，并非后台仍在处理。原始 pose 数据和上一轮检测输出均保留。

成功的 13 张正样本包含 24 个实例、113 个可见适用点和 33 个置零适用点；不适用槽位全部为零，输出每行 41 列。模型通过 `add_box` 补充过两个源标签遗漏实例，最终保留点均经过显式 `set_keypoints` 确认。主运行目标请求 147 次，记录输入 token 3,802,460、输出 token 11,428；启动预检另计 2 次。期间 4 次 HTTP 502 和 2 次超时在重试预算内恢复，无失败终态；失败请求未返回的用量不能由客户端确定。

[完整性核验](../../outputs/vlm_correct/national_pose_50_c2/full-verification.json) 核对全部 50 张源图片及标签、split、配置身份、20 个参考文件哈希、终态工具与事件、全部请求用量、检查点、14 组导出文件的哈希和独立副本、pose 元数据、浏览器读取及临时文件清理。源输入还与上一轮固定 50 张基线逐项一致。逐图终态见 [report.json](../../outputs/vlm_correct/national_pose_50_c2/report.json)，36 张排除图片及原因见 [exclusions.json](../../outputs/vlm_correct/national_pose_50_c2/exclusions.json)。验证脚本副本及原图／框点对照保存在 `review/`。

全部 50 张及其目标局部已完成视觉查看，共 130 张原图／标注对照，记录与最终检查点和图片哈希绑定，见 [visual-review.json](../../outputs/vlm_correct/national_pose_50_c2/review/visual-review.json)。未发现证据明确、必须重跑的错误；`a6894207d5bf` 在倾斜视角下的壳体顶点身份仍建议人工抽检，逐图意见保留在三个 `visual-part*.json` 中。

本轮待复核比例较高，主要涉及遮挡、UI 重叠及目标边界或点身份不可靠；允许部分可见点并不代表可以忽略无法确定的实例边界。视觉检查属于 Codex 定性复核，不能代替独立人工真值，也不能据此宣称定位精度提高。
