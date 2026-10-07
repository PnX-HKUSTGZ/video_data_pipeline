from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Literal, TypedDict

# 保留原有 Python 导入位置；运行调度通过 inference.build_pipeline 选择实现。
from .inference.single_stage import SingleStagePipeline as Detector
from .model_metadata import load_model_names

__all__ = ["Detector", "download_model", "load_model_names"]

REPO_ID = "hkustenterprize/RM26_engineer_exchange_model"
REVISION = "afab41a8b61c399bf71480f0fd52352ed660b994"
WEIGHTS = "yolo/v10.12/best.pt"
METADATA = "yolo/v10.12/openvino_best/metadata.yaml"
POSE_WEIGHTS = "yolopose/v11.02/best.pt"
POSE_METADATA = "yolopose/v11.02/openvino_best/metadata.yaml"
LOG = logging.getLogger(__name__)


class ModelPaths(TypedDict):
    model: str
    model_yaml: str


def download_model(
    destination: str | Path,
    revision: str = REVISION,
    endpoint: str | None = None,
    task: Literal["detect", "pose"] = "detect",
) -> ModelPaths:
    """下载指定 revision 的权重和类别 YAML，返回两个本地文件的绝对路径。

    已有文件直接复用；未指定 endpoint 时依次尝试官方站点与镜像。
    下载通过临时文件完成，失败时清理临时文件并报错。
    """
    if task not in ("detect", "pose"):
        raise ValueError("Download task must be detect or pose")
    weights, metadata = (POSE_WEIGHTS, POSE_METADATA) if task == "pose" else (WEIGHTS, METADATA)
    destination = Path(destination).resolve()
    endpoints = [endpoint] if endpoint else ["https://huggingface.co", "https://hf-mirror.com"]
    for filename in (weights, metadata):
        local = destination / filename
        if local.is_file():
            continue
        local.parent.mkdir(parents=True, exist_ok=True)
        errors = []
        for host in endpoints:
            with tempfile.NamedTemporaryFile(
                dir=local.parent, suffix=".tmp", delete=False
            ) as stream:
                temporary = Path(stream.name)
            try:
                LOG.info("Downloading %s via %s", filename, host)
                subprocess.run(
                    [
                        "curl",
                        "--fail",
                        "--location",
                        "--silent",
                        "--show-error",
                        "--connect-timeout",
                        "10",
                        "--max-time",
                        "900",
                        "--retry",
                        "2",
                        "--retry-all-errors",
                        "--output",
                        str(temporary),
                        f"{host}/{REPO_ID}/resolve/{revision}/{filename}?download=true",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                if not temporary.stat().st_size:
                    raise ValueError(f"Downloaded file is empty: {filename}")
                temporary.replace(local)
                break
            except (subprocess.CalledProcessError, OSError, ValueError) as error:
                errors.append(f"{host}: {error}")
                LOG.warning("Download failed via %s: %s", host, error)
            finally:
                temporary.unlink(missing_ok=True)
        else:
            raise RuntimeError("Model download failed: " + "; ".join(errors))
    return {"model": str(destination / weights), "model_yaml": str(destination / metadata)}
