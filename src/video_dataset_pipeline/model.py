import logging
import subprocess
import tempfile
from pathlib import Path

import yaml

REPO_ID = "hkustenterprize/RM26_engineer_exchange_model"
REVISION = "afab41a8b61c399bf71480f0fd52352ed660b994"
WEIGHTS = "yolo/v10.12/best.pt"
METADATA = "yolo/v10.12/openvino_best/metadata.yaml"
LOG = logging.getLogger(__name__)


def download_model(destination, revision=REVISION, endpoint=None):
    """Optional download helper. Inference only needs the two configured local files."""
    destination = Path(destination).resolve()
    endpoints = [endpoint] if endpoint else ["https://huggingface.co", "https://hf-mirror.com"]
    for filename in (WEIGHTS, METADATA):
        local = destination / filename
        if local.is_file():
            continue
        local.parent.mkdir(parents=True, exist_ok=True)
        errors = []
        for host in endpoints:
            with tempfile.NamedTemporaryFile(
                dir=local.parent, suffix=".tmp", delete=False
            ) as stream:
                temporary = Path(stream.name)
            try:
                LOG.info("Downloading %s via %s", filename, host)
                subprocess.run(
                    [
                        "curl",
                        "--fail",
                        "--location",
                        "--silent",
                        "--show-error",
                        "--connect-timeout",
                        "10",
                        "--max-time",
                        "900",
                        "--retry",
                        "2",
                        "--retry-all-errors",
                        "--output",
                        str(temporary),
                        f"{host}/{REPO_ID}/resolve/{revision}/{filename}?download=true",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                if not temporary.stat().st_size:
                    raise ValueError(f"Downloaded file is empty: {filename}")
                temporary.replace(local)
                break
            except (subprocess.CalledProcessError, OSError, ValueError) as error:
                errors.append(f"{host}: {error}")
                LOG.warning("Download failed via %s: %s", host, error)
            finally:
                temporary.unlink(missing_ok=True)
        else:
            raise RuntimeError("Model download failed: " + "; ".join(errors))
    return {"model": str(destination / WEIGHTS), "model_yaml": str(destination / METADATA)}


def load_model_names(path):
    metadata = yaml.safe_load(Path(path).read_text())
    names = metadata.get("names") if isinstance(metadata, dict) else None
    if isinstance(names, list):
        names = dict(enumerate(names))
    if not isinstance(names, dict) or not names:
        raise ValueError("Model YAML must contain a nonempty 'names' mapping or list")
    names = {int(k): str(v) for k, v in names.items()}
    if sorted(names) != list(range(len(names))):
        raise ValueError("Model YAML class IDs must be contiguous and start at zero")
    return names


class Detector:
    def __init__(self, config, device):
        import torch
        from ultralytics import YOLO

        torch.set_num_threads(4)
        self.config, self.device = config, device
        self.model = YOLO(config.model)
        if self.model.task != "detect":
            raise ValueError(f"Expected detection model, got {self.model.task}")
        self.names = {int(k): str(v) for k, v in self.model.names.items()}
        published = load_model_names(config.model_yaml)
        if self.names != published or sorted(self.names) != list(range(len(self.names))):
            raise ValueError("Checkpoint and configured YAML disagree about class ordering")
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
