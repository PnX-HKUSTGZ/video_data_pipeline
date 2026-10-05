import json
import math
from copy import deepcopy

import numpy as np
import pytest

from video_dataset_pipeline.dataset import append_record, export_records, read_records, save_jpeg
from video_dataset_pipeline.records import FrameRecord


@pytest.fixture
def empty_record():
    # Existing schema-4 JSON shape, including a null split and omitted artifact fields.
    return {
        "frame_index": 0,
        "timestamp": 0.0,
        "width": 100,
        "height": 80,
        "split": None,
        "status": "no_detection",
        "detections": [],
        "rejected_detections": [],
    }


@pytest.fixture
def candidate_record(empty_record):
    return {
        **empty_record,
        "status": "candidate",
        "detections": [
            {
                "class_id": 1,
                "confidence": 0.9,
                "xyxy": [0.0, 20.0, 100.0, 60.0],
                "source_xyxy": [-10.0, 20.0, 110.0, 60.0],
                "xywhn": [0.5, 0.5, 1.0, 0.5],
            }
        ],
        "candidate_image": ".staging/中文帧.jpg",
        "preview": "previews/中文帧.jpg",
        "preview_reasons": ["time_0", "class_1"],
    }


@pytest.mark.parametrize("stage", ["empty", "candidate", "included", "rejected"])
def test_schema4_jsonl_round_trip(tmp_path, empty_record, candidate_record, stage):
    payload = candidate_record if stage in {"candidate", "included"} else empty_record
    if stage == "included":
        del payload["candidate_image"]
        payload.update(
            split="train",
            status="included",
            image="images/train/中文帧.jpg",
            label="labels/train/中文帧.txt",
        )
    if stage == "rejected":
        payload["rejected_detections"] = [
            {
                "class_id": 0,
                "confidence": 0.3,
                "xyxy": [1.0, 1.0, 20.0, 20.0],
                "reason": "below_confidence",
            }
        ]
    record = FrameRecord.from_dict(payload)
    path = tmp_path / "frames.jsonl"
    with path.open("w") as stream:
        append_record(stream, record)
    assert json.loads(path.read_text()) == payload
    assert read_records(path) == [record]
    if record.detections:
        assert isinstance(record.detections[0].xyxy, tuple)
    assert ("candidate_image" in record.to_dict()) == (stage == "candidate")


def test_rejected_nonfinite_prediction_remains_readable(empty_record):
    empty_record["rejected_detections"] = [
        {
            "class_id": 0,
            "confidence": float("nan"),
            "xyxy": [1.0, 1.0, 20.0, 20.0],
            "reason": "nonfinite_coordinates_or_confidence",
        }
    ]
    record = FrameRecord.from_dict(json.loads(json.dumps(empty_record)))
    assert math.isnan(record.rejected_detections[0].confidence)
    assert record.rejected_detections[0].reason == "nonfinite_coordinates_or_confidence"


@pytest.mark.parametrize(
    "field,value",
    [
        ("frame_index", True),
        ("timestamp", "1.5"),
        ("width", None),
        ("split", "test"),
        ("status", "finished"),
        ("detections", {}),
        ("preview", 42),
        ("preview_reasons", [1]),
    ],
)
def test_invalid_record_types_report_line_number(tmp_path, empty_record, field, value):
    invalid = {**empty_record, "frame_index": 1, field: value}
    path = tmp_path / "frames.jsonl"
    content = json.dumps(empty_record) + "\n" + json.dumps(invalid) + "\n"
    path.write_text(content)
    with pytest.raises(ValueError, match="Invalid frame record 2"):
        read_records(path)
    assert path.read_text() == content


@pytest.mark.parametrize("mutation", ["missing_field", "short_box", "bad_coordinate", "bad_reason"])
def test_invalid_nested_records_report_line_number(
    tmp_path, empty_record, candidate_record, mutation
):
    if mutation == "missing_field":
        del candidate_record["detections"][0]["source_xyxy"]
    elif mutation == "short_box":
        candidate_record["detections"][0]["xyxy"] = [1, 2, 3]
    elif mutation == "bad_coordinate":
        candidate_record["detections"][0]["xyxy"] = [1, "2", 3, 4]
    else:
        candidate_record["rejected_detections"] = [
            {
                "class_id": 0,
                "confidence": 0.3,
                "xyxy": [1, 2, 3, 4],
                "reason": "unknown",
            }
        ]
    path = tmp_path / "frames.jsonl"
    path.write_text(json.dumps(empty_record) + "\n" + json.dumps(candidate_record) + "\n")
    with pytest.raises(ValueError, match="Invalid frame record 2"):
        read_records(path)


def test_export_copies_records_and_keeps_inference_state(tmp_path, candidate_record, empty_record):
    inference = []
    for i in range(4):
        record = FrameRecord.from_dict(candidate_record)
        record.frame_index = i
        record.timestamp = float(i)
        record.candidate_image = f".staging/中文帧{i}.jpg"
        save_jpeg(tmp_path / record.candidate_image, np.zeros((80, 100, 3), dtype=np.uint8), 95)
        inference.append(record)
    inference.append(FrameRecord.from_dict({**empty_record, "frame_index": 4, "timestamp": 4.0}))
    before = deepcopy(inference)
    for kind in ("images", "labels"):
        for split in ("train", "val"):
            (tmp_path / kind / split).mkdir(parents=True)
    exported, info = export_records(tmp_path, inference, 0.5, 0.6)
    assert info["boundary_timestamp"] == 1.5
    assert [r.status for r in exported] == [
        "included",
        "split_gap",
        "split_gap",
        "included",
        "no_detection",
    ]
    assert inference == before
    assert all(source is not final for source, final in zip(inference, exported, strict=True))
    assert all(r.candidate_image is None for r in exported)
    assert all("candidate_image" not in r.to_dict() for r in exported)
    assert read_records(tmp_path / "frames.jsonl") == exported
    for final in (exported[0], exported[3]):
        assert (tmp_path / final.label).read_text() == (
            "1 0.500000000 0.500000000 1.000000000 0.500000000\n"
        )
    assert len(list((tmp_path / "images").rglob("*.jpg"))) == 2
