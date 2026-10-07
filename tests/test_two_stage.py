"""两阶段的配置、实例对应、原图坐标与持久化契约。"""

import json
from dataclasses import replace
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
import yaml

from video_dataset_pipeline import pipeline
from video_dataset_pipeline.config import (
    Config,
    KeypointModelConfig,
    load_config,
    parse_keypoint_models,
)
from video_dataset_pipeline.dataset import read_records, validate_dataset
from video_dataset_pipeline.detection import filter_detections
from video_dataset_pipeline.inference import build_pipeline, two_stage
from video_dataset_pipeline.records import FrameRecord, NativeKeypoint, RawDetection, ROIResult
from video_dataset_pipeline.roi import expand_roi, filter_roi_keypoints, map_keypoints


def head(**kwargs):
    return KeypointModelConfig("model.py", "weights.pth", (1, 3), **kwargs)


def native(box=(10.0, 20.0, 50.0, 60.0), index=0):
    return ROIResult(
        index,
        box,
        (
            NativeKeypoint("a", (30.0, 40.0), (30.0 - box[0], 40.0 - box[1]), 1.2, 0.9),
            NativeKeypoint("b", (40.0, 50.0), (40.0 - box[0], 50.0 - box[1]), 0.2, 0.05),
        ),
    )


def test_heatmap_scores_slots_and_in_frame_gate():
    roi = native()
    points = map_keypoints(roi, head(), 4)
    filtered = filter_roi_keypoints(points, roi, head(), 100, 100)
    assert [p.visibility for p in filtered] == [0, 2, 0, 2]
    assert filtered[1].source.confidence == 1.2
    assert filtered[0].reason == "not_applicable"
    assert filtered[1].xyn == (0.3, 0.4)
    gated = filter_roi_keypoints(points, roi, head(in_frame_confidence=0.5), 100, 100)
    assert gated[3].reason == "below_in_frame_confidence"
    assert gated[0].visibility == 0
    with pytest.raises(ValueError, match="mapping disagree"):
        filter_roi_keypoints(points, replace(roi, xyxy=(0, 0, 50, 60)), head(), 100, 100)


@pytest.mark.parametrize(
    "xy,score,reason",
    [
        ((-1.0, 30.0), 2.0, "outside_image"),
        ((30.0, 40.0), -0.1, "below_keypoint_confidence"),
        ((30.0, 40.0), float("nan"), "invalid_confidence"),
        ((float("nan"), 40.0), 0.9, "nonfinite_coordinates"),
    ],
)
def test_invalid_native_points(xy, score, reason):
    p = NativeKeypoint("a", xy, (xy[0] - 10, xy[1] - 20), score, 0.9)
    roi = replace(native(), keypoints=(p, native().keypoints[1]))
    filtered = filter_roi_keypoints(map_keypoints(roi, head(), 4), roi, head(), 100, 100)
    assert filtered[1].visibility == 0
    assert filtered[1].reason == reason


def test_unmapped_class_and_record_roundtrip():
    roi = native()
    raw = [
        RawDetection(0, 0.9, (10, 20, 50, 60), map_keypoints(roi, head(), 4), roi),
        RawDetection(2, 0.9, (10, 20, 50, 60)),
    ]
    accepted, rejected = filter_detections(
        raw, 100, 100, 3, 0.5, (4, 3), keypoint_models={0: head()}
    )
    assert rejected[0].reason == "unmapped_class"
    record = FrameRecord(0, 0.0, 100, 100, None, "candidate", accepted, rejected)
    assert FrameRecord.from_dict(json.loads(json.dumps(record.to_dict()))) == record
    assert accepted[0].source_xyxy == (10, 20, 50, 60)
    assert expand_roi((1, 2, 90, 80), 100, 100, 0.5) == (0, 0, 100, 100)


@pytest.mark.parametrize(
    "changes",
    [
        {"output_indices": [1, 1]},
        {"output_indices": [-1]},
        {"output_indices": [True]},
        {"batch_size": 0},
        {"confidence": float("inf")},
        {"in_frame_confidence": 1.1},
        {"crop_margin": -1},
        {"oops": 0},
    ],
)
def test_invalid_head_config(changes):
    value = {"config": "a.py", "checkpoint": "a.pth", "output_indices": [0], **changes}
    with pytest.raises((ValueError, TypeError)):
        parse_keypoint_models({0: value})


def test_load_two_stage_config_paths_and_roundtrip(tmp_path):
    for filename in ("weights.pt", "names.yaml", "head.py", "head.pth"):
        (tmp_path / filename).touch()
    config = Config(
        "input.mp4",
        str(tmp_path / "out"),
        str(tmp_path / "weights.pt"),
        str(tmp_path / "names.yaml"),
        pipeline="two_stage",
        output_keypoints=("a", "b"),
        keypoint_models={
            0: KeypointModelConfig(str(tmp_path / "head.py"), str(tmp_path / "head.pth"), (0, 1))
        },
    )
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config.to_dict()))
    actual = load_config(path, {"keypoint_confidence": 1.2})
    assert actual.keypoint_models[0].confidence == 1.2
    assert actual.output_keypoints == ("a", "b")
    assert (
        parse_keypoint_models(json.loads(json.dumps(config.to_dict()))["keypoint_models"])
        == config.keypoint_models
    )
    with pytest.raises(ValueError, match="Missing keypoint model"):
        parse_keypoint_models({0: {"config": "foo"}})


def fake_models(monkeypatch):
    class Detector:
        def __init__(self, *args):
            self.names = {0: "c0", 1: "c1", 2: "c2", 3: "unmapped"}
            self.task = "detect"

        def predict_batch(self, images):
            return [
                [RawDetection(i, 0.9, (10, 20, 50, 60)) for i in (2, 3, 0, 1, 0)] for _ in images
            ]

    class Head:
        loads: ClassVar[list] = []
        batches: ClassVar[list] = []

        def __init__(self, config, device):
            self.names = ("a", "b")
            Head.loads.append(config)

        def predict(self, image, boxes):
            Head.batches.append(len(boxes))
            return [native(box).keypoints for box in boxes]

    monkeypatch.setattr(two_stage, "SingleStagePipeline", Detector)
    monkeypatch.setattr(two_stage, "LiteHRNet", Head)
    return Head


def test_arbitrary_classes_shared_model_roi_batch_and_order(monkeypatch):
    Head = fake_models(monkeypatch)
    config = Config(
        "in",
        "out",
        "m",
        "yaml",
        pipeline="two_stage",
        output_keypoints=("unused", "a", "unused2", "b"),
        keypoint_models={i: head(batch_size=2, crop_margin=0.1) for i in (0, 1, 2)},
    )
    engine = build_pipeline(config, "cpu")
    result = engine.predict_batch([np.zeros((100, 100, 3), dtype=np.uint8)])[0]
    assert len(Head.loads) == 1
    assert Head.batches == [2, 1, 1]
    assert [d.class_id for d in result] == [2, 3, 0, 1, 0]
    assert [d.roi.detection_index if d.roi else None for d in result] == [0, None, 2, 3, 4]
    assert result[0].roi.xyxy == (6.0, 16.0, 54.0, 64.0)
    assert result[0].keypoints[1].xy == (30, 40)
    assert result[0].xyxy == (10, 20, 50, 60)


def test_two_stage_full_export_and_offline_validation(synthetic_run, monkeypatch):
    config, _ = synthetic_run
    fake_models(monkeypatch)
    path = Path(config.model_yaml)
    path.write_text(yaml.safe_dump({"names": {0: "c0", 1: "c1", 2: "c2", 3: "unmapped"}}))
    monkeypatch.setattr(pipeline, "build_pipeline", build_pipeline)
    config = replace(
        config,
        pipeline="two_stage",
        output_keypoints=("unused", "a", "unused2", "b"),
        keypoint_models={i: head() for i in (0, 1, 2)},
    )
    pipeline.run_pipeline(config)
    root = Path(config.output)
    assert json.loads((root / "run.json").read_text())["schema_version"] == 7
    inferred = read_records(root / "inference.jsonl")
    exported = read_records(root / "frames.jsonl")
    assert all(r.status == "candidate" and r.image is None for r in inferred)
    assert any(r.status == "included" for r in exported)
    # 原模型不再存在时仍能独立校验数值与文件。
    assert validate_dataset(root, False)["valid"]
    lines = (root / "frames.jsonl").read_text().splitlines()
    broken = json.loads(lines[0])
    broken["detections"][0]["roi"]["xyxy"][0] += 1
    lines[0] = json.dumps(broken)
    (root / "frames.jsonl").write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="ROI geometry"):
        validate_dataset(root, False)


def test_shape_and_model_failures_stop_run(synthetic_run, monkeypatch):
    config, _ = synthetic_run
    fake_models(monkeypatch)
    config = replace(
        config,
        pipeline="two_stage",
        output_keypoints=("unused", "a", "unused2", "b"),
        keypoint_models={0: head()},
    )
    engine = build_pipeline(config, "cpu")
    monkeypatch.setattr(engine.heads[0], "predict", lambda image, boxes: [])
    with pytest.raises(ValueError, match="ROI batch"):
        engine.predict_batch([np.zeros((100, 100, 3), dtype=np.uint8)])
    with pytest.raises(ValueError, match="names/order"):
        build_pipeline(replace(config, output_keypoints=("unused", "b", "unused2", "a")), "cpu")
    with pytest.raises(ValueError, match="absent from detector"):
        build_pipeline(replace(config, keypoint_models={4: head()}), "cpu")


def test_roi_record_type_error_has_line_number(tmp_path):
    roi = native()
    raw = RawDetection(0, 0.9, roi.xyxy, map_keypoints(roi, head(), 4), roi)
    accepted, rejected = filter_detections(
        [raw], 100, 100, 1, 0.5, (4, 3), keypoint_models={0: head()}
    )
    record = FrameRecord(0, 0, 100, 100, None, "candidate", accepted, rejected)
    data = record.to_dict()
    data["detections"][0]["roi"]["detection_index"] = "oops"
    path = tmp_path / "frames.jsonl"
    path.write_text(json.dumps(data) + "\n")
    with pytest.raises(ValueError, match=r"frame record 1"):
        read_records(path)
