"""Command line entry point for the CPU-only local viewer."""

import argparse
import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="只读浏览标准 YOLO 数据集（检测/分割/OBB/姿态）")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    cache = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
    parser.add_argument(
        "--log-file", type=Path, default=cache / "yolo-dataset-viewer" / "viewer.log"
    )
    arguments = parser.parse_args()
    if not 1 <= arguments.port <= 65535:
        parser.error("port 必须在 1 到 65535 之间")
    try:
        import uvicorn

        from .app import create_app
    except ImportError:
        parser.error("缺少网页依赖，请运行 uv sync --extra viewer")
    arguments.log_file.parent.mkdir(parents=True, exist_ok=True)
    rotating = RotatingFileHandler(
        arguments.log_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), rotating],
        force=True,
    )
    try:
        uvicorn.run(
            create_app(),
            host=arguments.host,
            port=arguments.port,
            log_config=None,
            access_log=False,
        )
    finally:
        logging.shutdown()


if __name__ == "__main__":
    main()
