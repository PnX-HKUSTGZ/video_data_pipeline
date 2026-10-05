from dataclasses import replace

import av
import numpy as np
import pytest
import yaml

from video_dataset_pipeline import pipeline
from video_dataset_pipeline.config import Config
from video_dataset_pipeline.dataset import read_records, validate_dataset
from video_dataset_pipeline.detection import filter_detections, label_text, split_selected_frames
from video_dataset_pipeline.util import read_json
from video_dataset_pipeline.video import TimestampSampler


def test_timestamp_sampling_handles_irregular_frames_without_duplicates():
    sampler = TimestampSampler(2, 0, 2)
    times = [0, 0.04, 0.52, 0.56, 1.04, 1.08, 1.8, 1.9, 2]
    assert [t for t in times if sampler.accepts(t)] == [0, 0.52, 1.04, 1.8]


def test_smoke_samples_cover_interval():
    sampler = TimestampSampler(2, 1, 11, 5)
    assert [t for t in range(12) if sampler.accepts(t)] == [1, 3, 5, 7, 9]


@pytest.mark.parametrize(
    ("timestamp", "expected"), [(74, "train"), (75, None), (84, None), (85, "val")]
)
def test_time_split_embargo(timestamp, expected):
    records = [{"frame_index": i, "timestamp": i, "detections": [{}]} for i in range(100)]
    assignments, info = split_selected_frames(records, 0.8, 5)
    assert assignments[timestamp] == expected
    assert info["nominal_train_frames"] == 80


def test_split_counts_only_confidence_qualified_frames():
    records = [
        {"frame_index": i, "timestamp": i, "detections": [{}] if 10 <= i < 20 else []}
        for i in range(1000)
    ]
    assignments, info = split_selected_frames(records, 0.8, 0)
    assert info["qualified_frames"] == 10
    assert info["boundary_timestamp"] == 17.5
    assert sum(s == "train" for s in assignments.values()) == 8
    assert sum(s == "val" for s in assignments.values()) == 2
    assert 0 not in assignments


def test_boxes_are_clipped_normalized_and_invalid_predictions_rejected():
    raw = [
        {"class_id": 0, "confidence": 0.8, "xyxy": [-20, 20, 220, 80]},
        {"class_id": 2, "confidence": 0.8, "xyxy": [1, 1, 10, 10]},
        {"class_id": 0, "confidence": 0.4, "xyxy": [1, 1, 10, 10]},
        {"class_id": 0, "confidence": 0.8, "xyxy": [20, 20, 10, 10]},
        {"class_id": 0, "confidence": float("nan"), "xyxy": [1, 1, 10, 10]},
    ]
    accepted, rejected = filter_detections(raw, 200, 100, 2, 0.5)
    assert len(accepted) == 1
    assert accepted[0]["xywhn"] == pytest.approx([0.5, 0.5, 1, 0.6])
    assert accepted[0]["source_xyxy"] == [-20, 20, 220, 80]
    assert len(label_text(accepted).split()) == 5
    assert len(rejected) == 4


@pytest.mark.parametrize(
    "changes",
    [
        {"sample_fps": 0},
        {"confidence": 1.1},
        {"start": 5, "end": 4},
        {"imgsz": 641},
        {"device": "0"},
        {"sample_fps": float("nan")},
    ],
)
def test_invalid_config(changes):
    with pytest.raises(ValueError):
        Config(input="a", output="b", **changes)


def test_manifest_only_repairs_uncommitted_tail(tmp_path):
    path = tmp_path / "frames.jsonl"
    path.write_bytes(b'{"frame_index": 0}\n{"frame_index":')
    assert read_records(path, repair_tail=True) == [{"frame_index": 0}]
    assert path.read_bytes() == b'{"frame_index": 0}\n'
    path.write_bytes(b'broken\n{"frame_index": 1}\n')
    with pytest.raises(ValueError):
        read_records(path, repair_tail=True)


@pytest.fixture
def synthetic_run(tmp_path, monkeypatch):
    video = tmp_path / "中文视频.mp4"
    with av.open(str(video), "w") as container:
        stream = container.add_stream("mpeg4", rate=10)
        stream.width, stream.height, stream.pix_fmt = 160, 96, "yuv420p"
        for i in range(120):
            image = np.full((96, 160, 3), i * 2, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(image, format="bgr24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    model = tmp_path / "model"
    model.mkdir()
    (model / "metadata.yaml").write_text(yaml.safe_dump({"names": {0: "pillar", 1: "exchange"}}))
    manifest = {"revision": "synthetic", "files": [], "metadata": "metadata.yaml"}
    monkeypatch.setattr(pipeline, "verify_model", lambda _: manifest)
    monkeypatch.setattr(pipeline, "validate_dataset", lambda root: validate_dataset(root, False))

    class FakeDetector:
        fail_at = None
        calls = 0

        def __init__(self, *args):
            self.names = {0: "pillar", 1: "exchange"}

        def predict(self, image):
            FakeDetector.calls += 1
            if FakeDetector.calls == FakeDetector.fail_at:
                raise RuntimeError("simulated interruption")
            if image.mean() < 20:
                return []
            return [{"class_id": 0, "confidence": 0.9, "xyxy": [20, 10, 80, 70]}]

    monkeypatch.setattr(pipeline, "Detector", FakeDetector)
    config = Config(
        input=str(video),
        output=str(tmp_path / "out"),
        model=str(model),
        device="cpu",
        split_gap_seconds=0.5,
    )
    return config, FakeDetector


def test_pipeline_recovers_interrupted_run_without_relabeling_committed_frames(synthetic_run):
    config, detector = synthetic_run
    detector.fail_at = 7
    with pytest.raises(RuntimeError, match="simulated interruption"):
        pipeline.run_pipeline(config)
    root = __import__("pathlib").Path(config.output)
    assert not (root / "COMPLETE.json").exists()
    assert len(read_records(root / "inference.jsonl")) == 6
    detector.fail_at = None
    detector.calls = 0
    report = pipeline.run_pipeline(config, resume=True)
    assert detector.calls == 18
    assert report["candidate_frames"] == 24
    assert report["no_detection_frames"] > 0
    assert report["split_gap_frames"] == 2
    assert report["splits"]["val"] > 0
    assert (root / "COMPLETE.json").exists()
    assert read_json(root / "run.json")["decode"]["decoded_frames"] == 120
    with pytest.raises(ValueError, match="configuration changed"):
        pipeline.run_pipeline(replace(config, confidence=0.7), resume=True)
    positive = next(r for r in read_records(root / "frames.jsonl") if r["status"] == "included")
    (root / positive["label"]).write_text("0 0.5 0.5 1 1 0.9\n")
    with pytest.raises(ValueError, match="modified"):
        validate_dataset(root, False)


def test_empty_dataset_is_not_marked_complete(synthetic_run):
    config, detector = synthetic_run
    detector.predict = lambda self, image: []
    with pytest.raises(ValueError, match="positive samples"):
        pipeline.run_pipeline(config)
    from pathlib import Path

    root = Path(config.output)
    assert not (root / "COMPLETE.json").exists()
    assert read_json(root / "report.json")["included_frames"] == 0


def test_export_interruption_resumes_without_inference(synthetic_run, monkeypatch):
    config, detector = synthetic_run
    original = pipeline.export_records

    def crash_after_export(*args):
        original(*args)
        raise RuntimeError("export interrupted")

    monkeypatch.setattr(pipeline, "export_records", crash_after_export)
    with pytest.raises(RuntimeError, match="export interrupted"):
        pipeline.run_pipeline(config)
    calls = detector.calls
    monkeypatch.setattr(pipeline, "export_records", original)
    report = pipeline.run_pipeline(config, resume=True)
    assert detector.calls == calls
    assert report["candidate_frames"] == 24
    from pathlib import Path

    assert not (Path(config.output) / ".staging").exists()
    assert (Path(config.output) / "COMPLETE.json").exists()


def test_incomplete_decode_is_not_marked_complete(synthetic_run, monkeypatch):
    config, _ = synthetic_run
    original = pipeline.VideoFrames.__iter__

    def broken(self):
        for i, frame in enumerate(original(self)):
            if i == 4:
                raise ValueError("Decode ended early")
            yield frame

    monkeypatch.setattr(pipeline.VideoFrames, "__iter__", broken)
    with pytest.raises(ValueError, match="Decode ended early"):
        pipeline.run_pipeline(config)
    from pathlib import Path

    root = Path(config.output)
    assert not (root / "COMPLETE.json").exists()
    assert read_json(root / "run.json")["decode"]["decode_complete"] is False
