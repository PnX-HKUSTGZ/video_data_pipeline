"""CLI entrypoints for offline validation, capability probes, and resumable annotation."""

import argparse
import json
import signal
from pathlib import Path
from threading import Event
from types import FrameType

from pydantic import ValidationError

from .client import APIProblem, ResponsesBackend
from .config import credentials, load_config, local_path
from .data import discover, load_boxes
from .engine import CallLimit, RunCancelled, probe
from .pipeline import guard_output, run
from .prompts import references
from .quality import audit_lighting
from .render import load_image


def main() -> int:
    """Keep credentials out of diagnostics and close the HTTP client on every exit path."""
    parser = argparse.ArgumentParser(description="Responses-based YOLO annotation and correction")
    parser.add_argument("command", choices=("validate", "probe", "run", "audit-lighting"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    stop = Event()

    def interrupt(signum: int, frame: FrameType | None) -> None:
        stop.set()

    old_int = signal.signal(signal.SIGINT, interrupt)
    old_term = signal.signal(signal.SIGTERM, interrupt)
    backend: ResponsesBackend | None = None
    try:
        if args.command == "audit-lighting":
            if not args.source or not args.output or args.workers < 1:
                parser.error("audit-lighting requires --source, --output and positive --workers")
            print(
                json.dumps(
                    audit_lighting(args.source, args.output, workers=args.workers, cancel=stop)
                )
            )
            return 0
        if not args.config:
            parser.error("--config is required for validate, probe and run")
        path = args.config.resolve()
        config = load_config(path)
        base = path.parent
        if args.command == "validate":
            data = discover(config, base)
            guard_output(local_path(config.output.directory, base), data, config, base)
            _, hashes = references(config, base, data.names)
            invalid: list[str] = []
            for sample in data.samples:
                try:
                    load_boxes(
                        sample.label,
                        data.names,
                        task=data.task,
                        kpt_shape=data.kpt_shape,
                        rounding_tolerance=config.dataset.source_rounding_tolerance,
                    )
                    with load_image(sample.image):
                        pass
                except (OSError, ValueError):
                    invalid.append(str(sample.image))
            print(
                json.dumps(
                    {
                        "status": "invalid_inputs" if invalid else "valid",
                        "images": len(data.samples),
                        "classes": data.names,
                        "reference_files": len(hashes),
                        "invalid_images": invalid,
                    },
                    ensure_ascii=False,
                )
            )
            return 2 if invalid else 0
        auth = credentials(config.api, base)
        backend = ResponsesBackend(config.api, auth)
        if args.command == "probe":
            names = config.dataset.names
            if names is None:
                names = discover(config, base).names
            context, _ = references(config, base, names)
            result = probe(
                backend, cancel=stop, reference_context=context, max_retries=config.run.max_retries
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        report = run(config, base, backend, endpoint_identity=auth.base_url, cancel=stop)
        not_processed = report.get("not_processed", [])
        print(
            json.dumps(
                {
                    **{
                        key: value
                        for key, value in report.items()
                        if key not in {"mapping", "samples", "capability_probe", "not_processed"}
                    },
                    "not_processed_count": len(not_processed)
                    if isinstance(not_processed, list)
                    else 0,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return {"completed": 0, "partial": 2, "failed": 1, "interrupted": 130}[
            str(report["status"])
        ]
    except ValidationError as error:
        print(
            "Configuration validation failed: "
            + "; ".join(str(item["msg"]) for item in error.errors(include_input=False))
        )
        return 1
    except (RunCancelled, KeyboardInterrupt):
        print("Run interrupted")
        return 130
    except (APIProblem, CallLimit, OSError, ValueError, TypeError) as error:
        print(f"{type(error).__name__}: {error}")
        return 1
    finally:
        if backend:
            backend.close()
        signal.signal(signal.SIGINT, old_int)
        signal.signal(signal.SIGTERM, old_term)


if __name__ == "__main__":
    raise SystemExit(main())
