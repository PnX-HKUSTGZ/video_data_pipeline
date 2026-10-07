#!/usr/bin/env bash
# 在本项目 Python 3.12 环境构建与锁定 Torch/NumPy 匹配的原生扩展。
set -euo pipefail
cd "$(dirname "$0")/.."
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"
export MAX_JOBS="${MAX_JOBS:-4}"
export NO_ALBUMENTATIONS_UPDATE=1
# 先装好非隔离构建使用的 Torch、NumPy 与构建工具，额外参数原样传给 uv。
uv sync --locked --no-install-package mmcv --no-install-package xtcocotools \
  --no-install-package chumpy "$@"
# LiteHRNet 不使用 MMCV CUDA 算子；CPU 扩展满足 MMPose 的导入要求。
CUDA_VISIBLE_DEVICES="" FORCE_CUDA=0 MMCV_WITH_OPS=1 uv sync --locked "$@"
