import fcntl
import hashlib
import logging
import re
import subprocess
import time
from pathlib import Path

import yaml

from .util import read_json, sha256, write_json

REPO_ID = "hkustenterprize/RM26_engineer_exchange_model"
REVISION = "afab41a8b61c399bf71480f0fd52352ed660b994"
WEIGHTS = "yolo/v10.12/best.pt"
METADATA = "yolo/v10.12/openvino_best/metadata.yaml"
FILES = [WEIGHTS, METADATA, "README.md"]
LOG = logging.getLogger(__name__)


def _curl(url, path, *, resume=False, timeout=60):
    """Bound each transfer; retries re-read the current offset (curl --retry does not)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "curl",
        "--fail",
        "--location",
        "--silent",
        "--show-error",
        "--connect-timeout",
        "10",
        "--max-time",
        str(timeout),
        "--speed-limit",
        "1024",
        "--speed-time",
        "45",
    ]
    if resume:
        command += ["--continue-at", "-"]
    command += ["--output", str(path), url]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode:
        raise RuntimeError(f"curl exit {completed.returncode}: {completed.stderr.strip()}")


def download_model(destination, revision=REVISION, endpoint=None):
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Pin --revision to a full 40-character Hugging Face commit SHA")
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / ".download.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another model downloader owns this directory") from None
        return _download_locked(destination, revision, endpoint)


def _download_locked(destination, revision, endpoint):
    manifest_path = destination / "model_manifest.json"
    if manifest_path.exists():
        existing = verify_model(destination)
        if existing["revision"] == revision:
            return existing
        raise ValueError(
            "Model directory already contains a different revision; choose another directory"
        )
    endpoints = [endpoint] if endpoint else ["https://huggingface.co", "https://hf-mirror.com"]
    errors = []
    cache = destination / ".cache" / "direct"
    cache.mkdir(parents=True, exist_ok=True)
    for host in endpoints:
        for attempt in range(4):
            try:
                metadata_path = cache / "hub.json"
                info = read_json(metadata_path) if metadata_path.exists() else {}
                if info.get("sha") != revision or not all(
                    "size" in f for f in info.get("siblings", [])
                ):
                    temporary = cache / "hub.json.tmp"
                    try:
                        _curl(
                            f"{host}/api/models/{REPO_ID}/revision/{revision}?blobs=true", temporary
                        )
                        info = read_json(temporary)
                        if info.get("sha") != revision or info.get("id") != REPO_ID:
                            raise ValueError(
                                "Hub metadata does not match the requested model revision"
                            )
                        temporary.replace(metadata_path)
                    finally:
                        temporary.unlink(missing_ok=True)
                if info.get("id") != REPO_ID or not info.get("siblings"):
                    raise ValueError(
                        "Cached Hub metadata does not identify the expected repository"
                    )
                remote = {f["rfilename"]: f for f in info["siblings"]}
                records = []
                for filename in FILES:
                    item = remote[filename]
                    local = destination / filename
                    partial = cache / (Path(filename).name + ".part")
                    expected_size = item["size"]
                    if not local.exists():
                        LOG.info(
                            "Downloading %s (%s bytes), resume offset %s",
                            filename,
                            expected_size,
                            partial.stat().st_size if partial.exists() else 0,
                        )
                        if not partial.exists() or partial.stat().st_size < expected_size:
                            _curl(
                                f"{host}/{REPO_ID}/resolve/{revision}/{filename}?download=true",
                                partial,
                                resume=True,
                                timeout=900,
                            )
                        if partial.stat().st_size != expected_size:
                            raise ValueError(f"Downloaded byte count mismatch: {filename}")
                        local.parent.mkdir(parents=True, exist_ok=True)
                        partial.replace(local)
                    digest = sha256(local)
                    lfs_hash = item.get("lfs", {}).get("sha256")
                    if local.stat().st_size != expected_size or (lfs_hash and digest != lfs_hash):
                        raise ValueError(
                            f"Downloaded file fails Hub size/SHA-256 verification: {filename}"
                        )
                    if not lfs_hash:
                        content = local.read_bytes()
                        blob = hashlib.sha1(
                            b"blob " + str(len(content)).encode() + b"\0" + content
                        ).hexdigest()
                        if blob != item["blobId"]:
                            raise ValueError(
                                f"Downloaded file fails Git blob verification: {filename}"
                            )
                    records.append(
                        {
                            "path": filename,
                            "size": expected_size,
                            "sha256": digest,
                            "remote_lfs_sha256": lfs_hash,
                        }
                    )
                manifest = {
                    "repo_id": REPO_ID,
                    "revision": revision,
                    "endpoint": host,
                    "weights": WEIGHTS,
                    "metadata": METADATA,
                    "files": records,
                    "transport": "curl-resumable-https",
                }
                write_json(manifest_path, manifest)
                return manifest
            except (OSError, ValueError, RuntimeError) as error:
                errors.append(f"{host}: {type(error).__name__}: {error}")
                LOG.warning("Model transfer attempt %s via %s failed: %s", attempt + 1, host, error)
                if host == "https://huggingface.co" and endpoint is None:
                    break
                time.sleep(1 + attempt)
    raise RuntimeError("Model transfer failed; partial downloads retained.\n" + "\n".join(errors))


def verify_model(directory):
    directory = Path(directory)
    manifest = read_json(directory / "model_manifest.json")
    if manifest["repo_id"] != REPO_ID or manifest["weights"] != WEIGHTS:
        raise ValueError("This adapter requires the published RM26 v10.12 detection checkpoint")
    for item in manifest["files"]:
        path = directory / item["path"]
        if (
            not path.is_file()
            or path.stat().st_size != item["size"]
            or sha256(path) != item["sha256"]
        ):
            raise ValueError(f"Model artifact integrity check failed: {path}")
    if set(FILES) != {item["path"] for item in manifest["files"]}:
        raise ValueError("Incomplete model manifest")
    return manifest


class Detector:
    def __init__(self, directory, config, device):
        import torch
        from ultralytics import YOLO

        torch.set_num_threads(4)
        self.config, self.device = config, device
        self.manifest = verify_model(directory)
        self.model = YOLO(str(Path(directory) / self.manifest["weights"]))
        if self.model.task != "detect":
            raise ValueError(f"Expected detection model, got {self.model.task}")
        metadata = yaml.safe_load((Path(directory) / self.manifest["metadata"]).read_text())
        self.names = {int(k): str(v) for k, v in self.model.names.items()}
        published = {int(k): str(v) for k, v in metadata["names"].items()}
        if self.names != published or sorted(self.names) != list(range(len(self.names))):
            raise ValueError("Checkpoint and published metadata disagree about class ordering")
        if config.imgsz != 640:
            LOG.info("Using configured image size %s (published model size: 640)", config.imgsz)

    def predict(self, image):
        import torch
        from torchvision.ops import batched_nms

        result = self.model.predict(
            image,
            imgsz=self.config.imgsz,
            conf=self.config.confidence,
            iou=self.config.iou,
            device=self.device,
            verbose=False,
            save=False,
            quantize=32,
            max_det=300,
        )[0]
        if result.orig_shape != image.shape[:2]:
            raise ValueError("Detector did not return coordinates in the source image space")
        boxes = result.boxes.data.detach().cpu()
        if boxes.numel():
            # YOLO26 end-to-end heads do not apply the configurable NMS threshold.
            # Explicit class-aware NMS gives this pipeline's iou option consistent semantics.
            keep = batched_nms(
                boxes[:, :4], boxes[:, 4], boxes[:, 5].to(torch.int64), self.config.iou
            )
            boxes = boxes[keep]
        return [
            {
                "class_id": int(row[5]),
                "confidence": float(row[4]),
                "xyxy": [float(v) for v in row[:4]],
            }
            for row in boxes.tolist()
        ]
