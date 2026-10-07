"""轻量模型元数据读取；不初始化推理框架。"""

from pathlib import Path
from typing import cast

import yaml


def load_model_names(path: str | Path) -> dict[int, str]:
    """读取 YAML 的 names 列表或映射，要求类别编号从 0 连续排列且非空。"""
    metadata: object = yaml.safe_load(Path(path).read_text())
    names: object = metadata.get("names") if isinstance(metadata, dict) else None
    if isinstance(names, list):
        names = dict(enumerate(names))
    if not isinstance(names, dict) or not names:
        raise ValueError("Model YAML must contain a nonempty 'names' mapping or list")
    result: dict[int, str] = {}
    for key, value in cast(dict[object, object], names).items():
        if not isinstance(key, (str, int)):
            raise TypeError("Model YAML class IDs must be integers")
        result[int(key)] = str(value)
    if sorted(result) != list(range(len(result))):
        raise ValueError("Model YAML class IDs must be contiguous and start at zero")
    return result
