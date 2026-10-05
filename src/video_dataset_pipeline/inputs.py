"""展开文件和目录输入，固定本次运行的视频顺序。"""

from dataclasses import dataclass
from pathlib import Path

VIDEO_EXTENSIONS = frozenset(
    {".mp4", ".mkv", ".avi", ".mov", ".flv", ".webm", ".m4v", ".ts", ".mts", ".m2ts"}
)


@dataclass(frozen=True)
class SourceVideo:
    """单个源视频；编号只在本次固定输入清单内唯一，不表示内容哈希。"""

    source_id: str
    path: Path


def discover_videos(
    inputs: str | list[str], output: str | Path, recursive: bool = False
) -> list[SourceVideo]:
    """按输入顺序展开目录，目录内按路径排序，并按解析后的绝对路径去重。

    目录只收录已知视频扩展名；显式文件交给后续 ffprobe 检查。递归扫描不跟随
    目录符号链接。输出目录及其内容始终排除，空输入或无视频的目录会明确报错。
    """
    entries = [inputs] if isinstance(inputs, str) else inputs
    if not entries:
        raise ValueError("input must contain at least one file or directory")
    root = Path(output).resolve()
    paths: dict[Path, None] = {}
    for entry in entries:
        path = Path(entry).resolve()
        if path.is_relative_to(root):
            raise ValueError(f"Input is inside output directory: {path}")
        if path.is_file():
            candidates = [path]
        elif path.is_dir():
            candidates = []
            # 直接裁剪输出子树，避免在已有大型产物目录中遍历。
            if recursive:
                for directory, directories, filenames in path.walk(on_error=_raise_scan_error):
                    directories[:] = sorted(
                        name
                        for name in directories
                        if not (directory / name).resolve().is_relative_to(root)
                    )
                    candidates.extend(directory / name for name in filenames)
            else:
                candidates = list(path.iterdir())
            candidates = sorted(
                p
                for p in candidates
                if p.suffix.lower() in VIDEO_EXTENSIONS
                and not p.resolve().is_relative_to(root)
                and p.is_file()
            )
            if not candidates:
                raise ValueError(f"No videos found in directory: {path} (recursive={recursive})")
        else:
            raise ValueError(f"Input file or directory missing: {path}")
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved.is_relative_to(root):
                continue
            paths.setdefault(resolved, None)
    if not paths:
        raise ValueError("No input videos remain after discovery")
    return [SourceVideo(f"v{index:04d}", path) for index, path in enumerate(paths, 1)]


def _raise_scan_error(error: OSError) -> None:
    """无法读取目录时终止发现，避免将不完整的视频清单作为完整输入。"""
    raise error
