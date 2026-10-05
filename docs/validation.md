# 首次实测验收（2026-10-05，Asia/Shanghai）

最终采用：全视频按 2 FPS 推理 → 保留至少一个框置信度 ≥ 0.5 的画面 → 按入选画面时间顺序做数量 80/20 划分 → 剔除边界两侧各 5 秒。

## 固定输入与运行

- 输入：`datasets/rm26_fps_videos/videos/北部赛区/红方/recvlnoyv3arV6_红方工程第一视角.flv`
- 分辨率 1920×1080，25 FPS，有效时间轴 660 秒，解码 16,499 帧，末帧时间 659.96 秒。
- 模型 revision：`afab41a8b61c399bf71480f0fd52352ed660b994`
- 本次仅使用物理 GPU4：`GPU-8ba6e5a3-9634-eee2-9681-f22387d0e62f`，GPU0 当时被其他任务占用。
- 推理 PID 816674 退出后，其 GPU 上下文已释放；完整进程表见输出目录 `gpu_*.txt`。
- run ID：`6a3da3eecbc9495ff59242b38ea5b64aaeacbc665c506a06b5ae7913a8e8a7e3`

## 数据集结果

| 项目 | 数量 |
| --- | ---: |
| 候选帧 | 1320 |
| 通过置信度筛选 | 450 |
| 无有效检测 | 870 |
| 分界间隔剔除 | 19 |
| 最终图片 | 431 |
| 训练图片 | 351 |
| 验证图片 | 80 |
| pillar 框 | 282 |
| exchange 框 | 394 |
| 总框数 | 676 |
| 抽检预览 | 36 |

划分前名义数量为 360/90，边界时间 374.26 秒；剔除间隔后为 351/80。训练集包含 pillar 267、exchange 317 个框；验证集包含 pillar 15、exchange 77 个框。全片推理及导出约 34.74 秒，不包括后续完整性校验和最初依赖下载。

标签为模型伪标签。抽查确认片头/片尾仍有 5 张画面通过 0.5 阈值，主要是宣传画面上的误检；按约定的全片阈值规则保留，未偷偷裁剪或删除。可通过预览审核，或在新输出目录提高 `--confidence` 后重新生成。

## 已通过的检查

- 30 项自动测试：抽帧、坐标、类别、阈值、先筛选后划分、本地权重/YAML 配置、GPU 数字编号、非空输出保护、下载失败清理和失败状态等。
- Ruff 检查及格式检查。
- 最新流程的 20 帧真实冒烟检查：8 张通过阈值，6 张训练、2 张验证；两类均出现。
- 全视频正常解码；图片与标签配对、分辨率、五列格式、归一化边界、时间顺序及文件集合检查。
- Ultralytics `YOLODataset` 实际加载训练集 351 图片/584 框、验证集 80 图片/92 框，并读取样本张量。
- 已生成的数据集通过独立 `validate` 命令校验；每次新运行要求空输出目录。
- 暂存目录和 `.tmp` 已清理；`COMPLETE.json`、`validation.json` 均存在。

## 复现

```bash
uv sync --locked
uv run --locked video-dataset download-model
uv run --locked video-dataset run --config configs/rm26_exchange.yaml --output datasets/generated/rm26_exchange/recvlnoyv3arV6_new
uv run --locked video-dataset validate datasets/generated/rm26_exchange/recvlnoyv3arV6
```

生成到新位置时通过 `--output` 指定空目录；采样 FPS、阈值、设备、划分比例均可覆盖。后续默认以 `configs/rm26_exchange.yaml` 为准。
