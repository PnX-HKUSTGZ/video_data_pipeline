import json
import os
from pathlib import Path
from typing import cast


def atomic_bytes(path: str | Path, data: bytes) -> None:
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
    atomic_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode())


def read_json(path: str | Path) -> object:
    return json.loads(Path(path).read_text())


def as_mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise TypeError("Expected an object with string keys")
    return cast(dict[str, object], value)


def as_list(value: object) -> list[object]:
    if not isinstance(value, list):
        raise TypeError("Expected an array")
    return cast(list[object], value)


def as_int(value: object) -> int:
    if type(value) is not int:
        raise TypeError("Expected an integer")
    return value


def as_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("Expected a number")
    return float(value)


def as_str(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("Expected a string")
    return value
