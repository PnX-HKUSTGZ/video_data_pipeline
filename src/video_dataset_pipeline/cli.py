import argparse
import json
import logging
from pathlib import Path
from typing import Literal, cast

from .config import load_config
from .device import parse_device
from .model import REVISION, download_model
from .util import as_str


def main() -> None:
    """解析命令行，执行模型下载、数据生成或校验，并将结果输出为 JSON。"""
    parser = argparse.ArgumentParser(
        description="Generate YOLO detection or pose pseudo-labels from video"
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    download = subcommands.add_parser(
        "download-model", help="Download a pinned RM26 detection or pose model"
    )
    download.add_argument("--output", default="models/RM26_engineer_exchange_model")
    download.add_argument("--revision", default=REVISION)
    download.add_argument("--endpoint", help="Optional Hugging Face API endpoint")
    download.add_argument("--task", choices=("detect", "pose"), default="detect")
    run = subcommands.add_parser("run", help="Sample, label, export and validate")
    run.add_argument("--config", required=True)
    for field in ("input", "output", "model", "model-yaml"):
        run.add_argument(f"--{field}")
    run.add_argument("--device", type=parse_device, help="GPU index (e.g. 4) or cpu")
    run.add_argument(
        "--save-video",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Infer and save every source frame to inference.mp4; disables sampling",
    )
    for field in (
        "sample-fps",
        "start",
        "end",
        "confidence",
        "keypoint-confidence",
        "iou",
        "train-fraction",
        "split-gap-seconds",
    ):
        run.add_argument(f"--{field}", type=float)
    for field in ("imgsz", "jpeg-quality", "smoke-frames"):
        run.add_argument(f"--{field}", type=int)
    validate = subcommands.add_parser(
        "validate", help="Check labels, artifacts and Ultralytics loading"
    )
    validate.add_argument("dataset", type=Path)
    arguments: dict[str, object] = vars(parser.parse_args())
    command = arguments.pop("command")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result: object
    try:
        if command == "download-model":
            result = download_model(
                as_str(arguments["output"]),
                as_str(arguments["revision"]),
                as_str(arguments["endpoint"]) if arguments["endpoint"] is not None else None,
                task=cast(Literal["detect", "pose"], arguments["task"]),
            )
        elif command == "run":
            from .pipeline import run_pipeline

            config_path = as_str(arguments.pop("config"))
            result = run_pipeline(load_config(config_path, arguments))
        else:
            from .device import configure_device

            configure_device("cpu")
            from .dataset import validate_dataset

            result = validate_dataset(str(arguments["dataset"]))
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except (ValueError, TypeError, RuntimeError, OSError) as error:
        logging.getLogger(__name__).error("%s", error)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
