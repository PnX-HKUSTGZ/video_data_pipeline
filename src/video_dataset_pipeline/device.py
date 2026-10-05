import os
import subprocess
from pathlib import Path
from typing import Literal

from .util import atomic_bytes


def parse_device(value: object) -> int | Literal["cpu"]:
    """接受非负 GPU 编号、其数字字符串或 'cpu'，拒绝布尔值等其他输入。"""
    if value == "cpu":
        return "cpu"
    if isinstance(value, str) and value.isascii() and value.isdecimal():
        value = int(value)
    if type(value) is not int or value < 0:
        raise ValueError("device must be a nonnegative GPU index or 'cpu'")
    return value


def gpu_snapshot(output: str | Path, phase: str) -> None:
    """保存完整 GPU 状态，并检查本进程是否出现在未选中的物理 GPU 上。

    CPU 模式跳过检查；同一 phase 的快照写入同名文件并覆盖旧内容。
    快照包含其他进程以供核对，但只对本进程的设备位置作判断。
    """
    output = Path(output)
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        return
    text = subprocess.check_output(["nvidia-smi"], text=True)
    mapping = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,utilization.gpu", "--format=csv"],
        text=True,
    )
    processes = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv"],
        text=True,
    )
    atomic_bytes(output / f"gpu_{phase}.txt", (mapping + "\n" + processes + "\n" + text).encode())
    selected = os.environ["CUDA_VISIBLE_DEVICES"]
    for line in processes.splitlines()[1:]:
        fields = [part.strip() for part in line.split(",")]
        if fields[0] == str(os.getpid()) and fields[1] != selected:
            raise RuntimeError(f"Inference PID is on an unexpected GPU: {line}")


def configure_device(device: int | str) -> Literal["cpu", "0"]:
    """在导入 PyTorch/Ultralytics 之前设置设备可见性和运行环境。

    输入编号对应 nvidia-smi 的物理 GPU，内部按 UUID 隔离后返回框架内编号 '0'；
    CPU 模式隐藏所有 GPU 并返回 'cpu'。设备不存在或空闲显存不足 4 GiB 时报错。
    """
    # This module deliberately does not import torch or ultralytics.
    device = parse_device(device)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(".cache/ultralytics").resolve()))
    Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    if device != "cpu":
        rows = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid,memory.free", "--format=csv,noheader,nounits"],
            text=True,
        ).splitlines()
        matches = [row.split(",") for row in rows if row.split(",")[0].strip() == str(device)]
        if len(matches) != 1:
            raise ValueError(f"GPU index not available: {device}")
        if int(matches[0][2]) < 4096:
            raise RuntimeError(f"Selected GPU has less than 4 GiB free: {device}")
        # Resolve the nvidia-smi index internally so CUDA enumeration cannot select another card.
        os.environ["CUDA_VISIBLE_DEVICES"] = matches[0][1].strip()
    return "cpu" if device == "cpu" else "0"
