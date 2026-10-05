import html
import json
import math
import os
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml

from .detection import label_text, split_selected_frames
from .util import atomic_bytes, read_json, sha256, write_json


def read_records(path):
    path = Path(path)
    if not path.exists():
        return []
    data = path.read_bytes()
    records = []
    lines = data.splitlines(keepends=True)
    for i, line in enumerate(lines):
        try:
            record = json.loads(line)
            if not line.endswith(b"\n"):
                raise ValueError("Uncommitted trailing record")
        except (ValueError, UnicodeDecodeError):
            raise ValueError(f"Invalid frame record {i + 1}") from None
        records.append(record)
    ids = [r["frame_index"] for r in records]
    if ids != sorted(set(ids)):
        raise ValueError("Manifest frame IDs must be unique and ordered")
    return records


def append_record(stream, record):
    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def save_jpeg(path, image, quality):
    success, data = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not success:
        raise ValueError(f"JPEG encoding failed: {path}")
    atomic_bytes(path, data.tobytes())


def export_records(root, inference, train_fraction, gap):
    assignments, split_info = split_selected_frames(inference, train_fraction, gap)
    records = []
    for source in inference:
        record = {k: v for k, v in source.items() if not k.startswith("candidate_image")}
        split = assignments.get(source["frame_index"])
        record["split"] = split
        record["status"] = (
            "no_detection"
            if not record["detections"]
            else "split_gap"
            if split is None
            else "included"
        )
        if record["status"] == "included":
            name = Path(source["candidate_image"]).stem
            image_path = Path("images") / split / f"{name}.jpg"
            label_path = Path("labels") / split / f"{name}.txt"
            # Both directories share a filesystem, so image bytes need not be duplicated.
            os.link(root / source["candidate_image"], root / image_path)
            atomic_bytes(root / label_path, label_text(record["detections"]).encode())
            record.update(
                image=str(image_path),
                label=str(label_path),
                image_sha256=source["candidate_image_sha256"],
                label_sha256=sha256(root / label_path),
            )
        records.append(record)
    atomic_bytes(
        root / "frames.jsonl",
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records).encode(),
    )
    return records, split_info


def check_record_files(root, record):
    for key in ("image", "label", "preview", "candidate_image"):
        if key in record:
            path = root / record[key]
            if not path.is_file() or sha256(path) != record[f"{key}_sha256"]:
                raise ValueError(f"Missing or modified committed artifact: {path}")


class Previews:
    def __init__(self, root, names, start, end, smoke=False):
        self.root, self.names, self.start, self.end = root, names, start, end
        self.smoke = smoke
        self.counts = Counter()

    def save(self, name, image, record):
        reasons = []
        bucket = min(19, int(20 * (record["timestamp"] - self.start) / (self.end - self.start)))
        if self.counts[f"time_{bucket}"] == 0:
            reasons.append(f"time_{bucket}")
        if self.smoke:
            reasons.append("smoke")
        detections = record["detections"]
        for class_id in sorted({d["class_id"] for d in detections}):
            if self.counts[f"class_{class_id}"] < 2:
                reasons.append(f"class_{class_id}")
        if (
            detections
            and min(d["confidence"] for d in detections) < 0.6
            and self.counts["low_confidence"] < 10
        ):
            reasons.append("low_confidence")
        if not detections and self.counts["no_detection"] < 5:
            reasons.append("no_detection")
        if not reasons:
            return {}
        canvas = image.copy()
        for d in detections:
            x1, y1, x2, y2 = map(round, d["xyxy"])
            color = (40, 220, 40) if d["class_id"] % 2 == 0 else (0, 170, 255)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                canvas,
                f"{self.names[d['class_id']]} {d['confidence']:.3f}",
                (x1, max(22, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )
        cv2.putText(
            canvas,
            f"PSEUDO LABEL | t={record['timestamp']:.3f}s | {record['status']}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )
        path = Path("previews") / f"{name}.jpg"
        save_jpeg(self.root / path, canvas, 90)
        self.counts.update(reasons)
        return {
            "preview": str(path),
            "preview_sha256": sha256(self.root / path),
            "preview_reasons": reasons,
        }


def summarize(records, names):
    counts = Counter(r["status"] for r in records)
    splits = Counter(r["split"] for r in records if r["status"] == "included")
    classes = Counter(
        d["class_id"] for r in records if r["status"] == "included" for d in r["detections"]
    )
    scores = [
        d["confidence"] for r in records if r["status"] == "included" for d in r["detections"]
    ]
    return {
        "candidate_frames": len(records),
        "confidence_qualified_frames": sum(bool(r["detections"]) for r in records),
        "included_frames": counts["included"],
        "no_detection_frames": counts["no_detection"],
        "split_gap_frames": counts["split_gap"],
        "no_detection_including_gap_frames": sum(not r["detections"] for r in records),
        "splits": {s: splits[s] for s in ("train", "val")},
        "class_instances": {names[k]: classes[k] for k in names},
        "class_instances_by_split": {
            split: {
                names[k]: sum(
                    d["class_id"] == k
                    for r in records
                    if r["status"] == "included" and r["split"] == split
                    for d in r["detections"]
                )
                for k in names
            }
            for split in ("train", "val")
        },
        "preview_frames": sum("preview" in r for r in records),
        "confidence": (
            {
                "minimum": min(scores),
                "median": float(np.median(scores)),
                "maximum": max(scores),
                "below_0_6": sum(s < 0.6 for s in scores),
            }
            if scores
            else None
        ),
    }


def validate_dataset(root, ultralytics_check=True):
    root = Path(root).resolve()
    run = read_json(root / "run.json")
    config = run["config"]
    if run.get("inference_sha256") != sha256(root / "inference.jsonl"):
        raise ValueError("Inference manifest hash mismatch")
    records = read_records(root / "frames.jsonl")
    if run.get("candidate_frames") != len(records) or run.get("manifest_sha256") != sha256(
        root / "frames.jsonl"
    ):
        raise ValueError("Manifest count or hash differs from completed decode record")
    names = {int(k): v for k, v in run["names"].items()}
    data = yaml.safe_load((root / "data.yaml").read_text())
    if data["names"] != names or data["train"] != "images/train" or data["val"] != "images/val":
        raise ValueError("data.yaml disagrees with run metadata")
    if Path(data["path"]).resolve() != root:
        raise ValueError("data.yaml path does not point to this dataset")
    if not run.get("decode", {}).get("decode_complete"):
        raise ValueError("Video decode did not complete")
    expected_files = set()
    seen_content = {"train": set(), "val": set()}
    previous = -1
    assignments, split_info = split_selected_frames(
        records, config["train_fraction"], config["split_gap_seconds"]
    )
    if run.get("split") != split_info:
        raise ValueError("Split metadata does not match confidence-qualified frame selection")
    for record in records:
        check_record_files(root, record)
        if record["timestamp"] <= previous:
            raise ValueError("Manifest timestamps must be increasing")
        previous = record["timestamp"]
        expected_split = assignments.get(record["frame_index"])
        expected_status = (
            "no_detection"
            if not record["detections"]
            else "split_gap"
            if expected_split is None
            else "included"
        )
        if record["split"] != expected_split or record["status"] != expected_status:
            raise ValueError("Manifest split or inclusion rule mismatch")
        if record["status"] != "included":
            if "image" in record or "label" in record:
                raise ValueError("Excluded frame has training artifacts")
            continue
        expected_files.update((record["image"], record["label"]))
        image = cv2.imdecode(np.fromfile(root / record["image"], dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None or list(image.shape[:2]) != [record["height"], record["width"]]:
            raise ValueError(f"Image decode/dimension failure: {record['image']}")
        if Path(record["image"]).parts[1] != record["split"]:
            raise ValueError("Image stored in wrong split")
        if Path(record["image"]).stem != Path(record["label"]).stem:
            raise ValueError("Image/label basename mismatch")
        label = (root / record["label"]).read_text()
        if label != label_text(record["detections"]):
            raise ValueError("Label does not match manifest")
        lines = label.splitlines()
        if not lines:
            raise ValueError("Positive-only dataset contains an empty label")
        for line in lines:
            fields = line.split()
            if len(fields) != 5 or not fields[0].isdigit() or int(fields[0]) not in names:
                raise ValueError(f"Invalid YOLO class/columns: {line}")
            x, y, w, h = map(float, fields[1:])
            if not all(math.isfinite(v) and 0 <= v <= 1 for v in (x, y, w, h)) or min(w, h) <= 0:
                raise ValueError(f"Invalid normalized box: {line}")
            if min(x - w / 2, y - h / 2) < -1e-8 or max(x + w / 2, y + h / 2) > 1 + 1e-8:
                raise ValueError("YOLO box lies outside image")
        seen_content[record["split"]].add(record["image_sha256"])
    actual_files = {
        str(p.relative_to(root))
        for folder in ("images", "labels")
        for p in (root / folder).rglob("*")
        if p.is_file()
    }
    if actual_files != expected_files:
        raise ValueError(f"Dataset artifact set mismatch: {actual_files ^ expected_files}")
    overlap = seen_content["train"] & seen_content["val"]
    if overlap:
        raise ValueError(f"Identical image content crosses train/val: {len(overlap)} hashes")
    summary = summarize(records, names)
    if min(summary["splits"].values()) == 0:
        raise ValueError(f"Dataset requires positive samples in both splits: {summary['splits']}")
    loader_counts = {}
    if ultralytics_check:
        from ultralytics.data.dataset import YOLODataset

        # Use the actual training dataset class without the unrelated online font download.
        checked = {**data, "nc": len(names)}
        for split in ("train", "val"):
            try:
                dataset = YOLODataset(
                    img_path=str(root / "images" / split),
                    data=checked,
                    task="detect",
                    imgsz=config["imgsz"],
                    augment=False,
                    cache=False,
                    batch_size=1,
                    prefix=f"{split}: ",
                )
                if len(dataset) != summary["splits"][split]:
                    raise ValueError("Ultralytics dropped dataset images")
                instance_count = sum(len(label["cls"]) for label in dataset.labels)
                if instance_count != sum(summary["class_instances_by_split"][split].values()):
                    raise ValueError("Ultralytics dropped or changed labels")
                item = dataset[0]
                if item["img"].numel() == 0:
                    raise ValueError("Ultralytics returned an empty image tensor")
                loader_counts[split] = {"images": len(dataset), "instances": instance_count}
            finally:
                (root / "labels" / f"{split}.cache").unlink(missing_ok=True)
    result = {"valid": True, **summary, "ultralytics_loader": loader_counts}
    write_json(root / "validation.json", result)
    return result


def write_inspection_report(root, records, report):
    preview_records = [record for record in records if "preview" in record]
    cards, tiles = [], []
    for record in preview_records:
        filename = html.escape(Path(record["preview"]).name)
        caption = f"{record['timestamp']:.2f}s | {record['status']} | {', '.join(record['preview_reasons'])}"
        cards.append(
            f'<figure><a href="{filename}"><img loading="lazy" src="{filename}"></a>'
            f"<figcaption>{html.escape(caption)}</figcaption></figure>"
        )
    indices = np.linspace(0, len(preview_records) - 1, min(24, len(preview_records)), dtype=int)
    for index in indices:
        record = preview_records[index]
        image = cv2.imdecode(
            np.fromfile(root / record["preview"], dtype=np.uint8), cv2.IMREAD_COLOR
        )
        tile = np.zeros((210, 320, 3), dtype=np.uint8)
        tile[:180] = cv2.resize(image, (320, 180))
        cv2.putText(
            tile,
            f"{record['timestamp']:.2f}s | {record['status']}",
            (5, 201),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
        )
        tiles.append(tile)
    if tiles:
        sheet = np.zeros((210 * math.ceil(len(tiles) / 4), 1280, 3), dtype=np.uint8)
        for i, tile in enumerate(tiles):
            sheet[(i // 4) * 210 : (i // 4 + 1) * 210, (i % 4) * 320 : (i % 4 + 1) * 320] = tile
        save_jpeg(root / "previews" / "contact_sheet.jpg", sheet, 90)
    document = (
        '<!doctype html><meta charset="utf-8"><title>RM26 pseudo-label inspection</title>'
        "<style>body{font-family:sans-serif;margin:24px;background:#eee}"
        "main{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:16px}"
        "figure{margin:0;background:white;padding:12px}img{width:100%}pre{white-space:pre-wrap}</style>"
        "<h1>自动标注抽检 · Pseudo labels</h1><p>标签来自模型预测，未经人工确认；不能作为真实精度评测依据。</p>"
        "<pre>" + html.escape(json.dumps(report, ensure_ascii=False, indent=2)) + "</pre>"
        "<main>" + "".join(cards) + "</main>"
    )
    atomic_bytes(root / "previews" / "index.html", document.encode())
