import json
import os
from pathlib import Path
from typing import cast


def atomic_bytes(path: str | Path, data: bytes) -> None:
    """在目标目录写临时文件并刷盘，再原子替换目标；结束时清理临时文件。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path: str | Path, value: object) -> None:
    """将可 JSON 序列化的值原子写入文件，保留中文并添加末尾换行。"""
    atomic_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode())


def read_json(path: str | Path) -> object:
    """读取 JSON 原始值；具体结构由调用处检查，避免动态类型进入核心流程。"""
    return json.loads(Path(path).read_text())


def as_mapping(value: object) -> dict[str, object]:
    """确认输入是字符串键字典并收窄类型，字典中的值仍需逐项解析。"""
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise TypeError("Expected an object with string keys")
    return cast(dict[str, object], value)


def as_list(value: object) -> list[object]:
    """确认输入是列表并收窄类型，不隐式转换字符串或其他可迭代对象。"""
    if not isinstance(value, list):
        raise TypeError("Expected an array")
    return cast(list[object], value)


def as_int(value: object) -> int:
    """读取整数，特别排除 Python 中属于 int 子类的 bool。"""
    if type(value) is not int:
        raise TypeError("Expected an integer")
    return value


def as_float(value: object) -> float:
    """接受整数或浮点数并转为 float；排除 bool，保留非有限值供后续业务校验。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("Expected a number")
    return float(value)


def as_str(value: object) -> str:
    """要求输入本身就是字符串，不隐式转换其他类型，避免掩盖元数据错误。"""
    if not isinstance(value, str):
        raise TypeError("Expected a string")
    return value
