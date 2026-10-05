import argparse
import json
import logging
from pathlib import Path

from .config import load_config
from .model import REVISION, download_model


def main():
    parser = argparse.ArgumentParser(description="Generate YOLO detection pseudo-labels from video")
    subcommands = parser.add_subparsers(dest="command", required=True)
    download = subcommands.add_parser(
        "download-model", help="Download the pinned RM26 detection model"
    )
    download.add_argument("--output", default="models/RM26_engineer_exchange_model")
    download.add_argument("--revision", default=REVISION)
    download.add_argument("--endpoint", help="Optional Hugging Face API endpoint")
    run = subcommands.add_parser("run", help="Sample, label, export and validate")
    run.add_argument("--config", required=True)
    for field in ("input", "output", "model", "model-yaml", "device"):
        run.add_argument(f"--{field}")
    for field in (
        "sample-fps",
        "start",
        "end",
        "confidence",
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
    arguments = vars(parser.parse_args())
    command = arguments.pop("command")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        if command == "download-model":
            result = download_model(
                arguments["output"], arguments["revision"], arguments["endpoint"]
            )
        elif command == "run":
            from .pipeline import run_pipeline

            config_path = arguments.pop("config")
            result = run_pipeline(load_config(config_path, arguments))
        else:
            from .device import configure_device

            configure_device("cpu")
            from .dataset import validate_dataset

            result = validate_dataset(arguments["dataset"])
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except (ValueError, RuntimeError, OSError) as error:
        logging.getLogger(__name__).error("%s", error)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
