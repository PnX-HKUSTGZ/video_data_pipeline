import os
import subprocess
from pathlib import Path

from .util import atomic_bytes


def gpu_snapshot(output, phase):
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


def configure_device(device):
    # This module deliberately does not import torch or ultralytics.
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "" if device == "cpu" else device
    os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(".cache/ultralytics").resolve()))
    Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    if device != "cpu":
        rows = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid,memory.free", "--format=csv,noheader,nounits"],
            text=True,
        ).splitlines()
        matches = [row for row in rows if row.split(",")[0].strip() == device]
        if len(matches) != 1:
            raise ValueError(f"GPU UUID not available: {device}")
        if int(matches[0].split(",")[1]) < 4096:
            raise RuntimeError(f"Selected GPU has less than 4 GiB free: {device}")
    return "cpu" if device == "cpu" else "0"
