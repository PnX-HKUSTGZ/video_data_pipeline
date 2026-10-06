import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import av
import numpy as np
import pytest
import yaml

from video_dataset_pipeline import model, pipeline
from video_dataset_pipeline.config import Config
from video_dataset_pipeline.dataset import annotate_frame, read_records, validate_dataset
from video_dataset_pipeline.detection import candidate_status, filter_detections, label_text
from video_dataset_pipeline.pose import check_keypoint_shape, filter_keypoints, parse_pose_metadata
from video_dataset_pipeline.records import FrameRecord, RawDetection, RawKeypoint


@pytest.fixture(autouse=True)
def preserve_pillow_reader(monkeypatch):
    # Ultralytics patches Image.open on import; do not leak its optional HEIF dependency
    # into unrelated viewer tests sharing this Python process.
    from PIL import Image

    monkeypatch.setattr(Image, "open", Image.open)


def test_keypoint_filtering_and_label_visibility():
    raw = (
        RawKeypoint((50, 25), 0.5),
        RawKeypoint((100, 50), 0.49),
        RawKeypoint((-1, 10), 0.9),
        RawKeypoint((float("nan"), 10), 0.9),
        RawKeypoint((10, 10), float("nan")),
        RawKeypoint((200, 100), 1.0),
    )
    accepted, _ = filter_detections(
        [RawDetection(0, 0.9, (0, 0, 200, 100), raw)], 200, 100, 1, 0.5, (6, 3), 0.5
    )
    points = accepted[0].keypoints
    assert [p.visibility for p in points] == [2, 0, 0, 0, 0, 2]
    assert [p.reason for p in points] == [
        None,
        "below_keypoint_confidence",
        "outside_image",
        "nonfinite_coordinates",
        "invalid_confidence",
        None,
    ]
    assert points[0].xyn == (0.25, 0.25)
    assert points[-1].xyn == (1.0, 1.0)
    assert all(p.xyn == (0, 0) for p in points[1:-1])
    fields = label_text(accepted).split()
    assert len(fields) == 23
    assert fields[5:8] == ["0.250000000", "0.250000000", "2"]
    assert fields[8:11] == ["0.000000000", "0.000000000", "0"]
    assert raw[1].confidence == 0.49


def test_two_dimensional_model_does_not_invent_confidence():
    raw = [RawKeypoint((0, 0), None), RawKeypoint((20, 10), None)]
    check_keypoint_shape(raw, (2, 2))
    points = filter_keypoints(raw, 200, 100, 1.0)
    assert [p.visibility for p in points] == [2, 2]
    assert [p.source.confidence for p in points] == [None, None]
    assert points[1].xyn == (0.1, 0.1)


@pytest.mark.parametrize(
    "points,shape",
    [
        (None, (1, 3)),
        ((), (1, 3)),
        ((RawKeypoint((0, 0), None),), (1, 3)),
        ((RawKeypoint((0, 0), 0.8),), (1, 2)),
        ((RawKeypoint((0, 0), 0.8),), None),
    ],
)
def test_keypoint_structure_errors_are_not_silently_dropped(points, shape):
    with pytest.raises(ValueError):
        filter_detections([RawDetection(0, 0.1, (0, 0, 10, 10), points)], 20, 20, 1, 0.5, shape)


@pytest.mark.parametrize(
    "names",
    [["left", "right"], {0: "left", 1: "right"}, {0: ["left", "right"], 1: ["left", "right"]}],
)
def test_pose_metadata_preserves_names_and_explicit_flip(names):
    pose = parse_pose_metadata({"kpt_shape": [2, 3], "kpt_names": names, "flip_idx": [1, 0]}, 2)
    assert pose == {
        "source_kpt_shape": (2, 3),
        "kpt_names": {0: ["left", "right"], 1: ["left", "right"]},
        "flip_idx": [1, 0],
    }
    assert parse_pose_metadata(json.loads(json.dumps(pose)), 2) == pose
    assert parse_pose_metadata({"kpt_shape": [2, 2]}, 1) == {"source_kpt_shape": (2, 2)}


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {"kpt_shape": [0, 3]},
        {"kpt_shape": [2, 4]},
        {"kpt_shape": [True, 3]},
        {"kpt_shape": [2, 3], "kpt_names": ["one"]},
        {"kpt_shape": [2, 3], "kpt_names": {1: "one", 2: "two"}},
        {"kpt_shape": [2, 3], "flip_idx": [0, 0]},
        {"kpt_shape": [3, 3], "flip_idx": [1, 2, 0]},
    ],
)
def test_invalid_pose_metadata(metadata):
    with pytest.raises((ValueError, TypeError)):
        parse_pose_metadata(metadata, 1)


def test_pose_json_roundtrip_and_line_diagnostics(tmp_path):
    raw = [
        RawDetection(0, score, (0, 0, 100, 80), (RawKeypoint((10, 20), 0.4),))
        for score in (0.9, 0.2)
    ]
    detections, rejected = filter_detections(raw, 100, 80, 1, 0.5, (1, 3))
    record = FrameRecord(
        0,
        0.0,
        100,
        80,
        None,
        candidate_status(detections),
        detections,
        rejected,
        preview="previews/frame.jpg",
        preview_reasons=["no_keypoints"],
    )
    assert record.status == "no_keypoints"
    payload = record.to_dict()
    assert FrameRecord.from_dict(json.loads(json.dumps(payload))) == record
    payload["detections"][0]["keypoints"][0]["visibility"] = True
    path = tmp_path / "frames.jsonl"
    path.write_text(json.dumps(record.to_dict()) + "\n" + json.dumps(payload) + "\n")
    with pytest.raises(ValueError, match="Invalid frame record 2"):
        read_records(path)


@pytest.fixture
def pose_run(synthetic_run):
    config, detector = synthetic_run
    original_init = detector.__init__

    def initialize(self, *args):
        original_init(self, *args)
        self.task = "pose"
        self.pose = {"source_kpt_shape": (2, 3)}

    def predict(self, image):
        detector.calls += 1
        if image.mean() < 20:
            return []
        score = 0.2 if image.mean() < 40 else 0.8
        return [
            RawDetection(
                0, 0.9, (20, 10, 80, 70), (RawKeypoint((30, 20), score), RawKeypoint((40, 40), 0.1))
            ),
            RawDetection(
                1, 0.9, (85, 10, 145, 70), (RawKeypoint((90, 20), 0.1), RawKeypoint((100, 40), 0.1))
            ),
        ]

    detector.__init__, detector.predict = initialize, predict
    return config, detector


@pytest.mark.parametrize("save_video", [False, True])
def test_pose_pipeline_retains_boxes_and_excludes_frames_without_points(pose_run, save_video):
    config, detector = pose_run
    report = pipeline.run_pipeline(replace(config, save_video=save_video))
    root = Path(config.output)
    records = read_records(root / "frames.jsonl")
    assert detector.calls == report["candidate_frames"] == (120 if save_video else 24)
    assert report["no_keypoints_frames"] > 0
    assert report["no_detection_frames"] > 0
    assert (
        sum(
            report[k]
            for k in (
                "no_detection_frames",
                "no_keypoints_frames",
                "split_gap_frames",
                "included_frames",
            )
        )
        == report["candidate_frames"]
    )
    assert len([r for r in records if r.status == "no_keypoints"]) == report["no_keypoints_frames"]
    included = [r for r in records if r.status == "included"]
    assert all(len(r.detections) == 2 for r in included)
    assert all(
        len(line.split()) == 11
        for r in included
        for line in (root / r.label).read_text().splitlines()
    )
    assert report["keypoints"]["valid_points"] == len(included)
    assert report["keypoints"]["missing_points"] == 3 * len(included)
    assert yaml.safe_load((root / "data.yaml").read_text())["kpt_shape"] == [2, 3]
    assert json.loads((root / "run.json").read_text())["schema_version"] == 6
    assert (root / "COMPLETE.json").exists()
    assert not (root / ".staging").exists()
    result = validate_dataset(root)  # Real Ultralytics pose loader, including masks.
    assert sum(c["valid_keypoints"] for c in result["ultralytics_loader"].values()) == len(included)
    if save_video:
        with av.open(str(root / "inference.mp4")) as container:
            assert sum(1 for _ in container.decode(video=0)) == 120


@pytest.mark.parametrize("smoke_frames", [0, 6])
def test_pose_empty_keypoints_cannot_mark_success(pose_run, smoke_frames):
    config, _ = pose_run
    with pytest.raises(ValueError, match="positive samples"):
        pipeline.run_pipeline(replace(config, keypoint_confidence=1.0, smoke_frames=smoke_frames))
    assert not (Path(config.output) / "COMPLETE.json").exists()
    assert not (Path(config.output) / "SMOKE_COMPLETE.json").exists()


def test_pose_validator_rejects_manifest_and_yaml_tampering(pose_run):
    config, _ = pose_run
    pipeline.run_pipeline(config)
    root = Path(config.output)
    path = root / "data.yaml"
    original = path.read_text()
    data = yaml.safe_load(original)
    data["kpt_shape"] = [2, 2]
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="kpt_shape"):
        validate_dataset(root, False)
    path.write_text(original)
    path = root / "frames.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    candidate = next(r for r in records if r["status"] == "included")
    candidate["detections"][0]["keypoints"][0]["xyn"] = [0.5, 0.5]
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    with pytest.raises(ValueError, match="Keypoint filtering mismatch"):
        validate_dataset(root, False)


def test_pose_preview_draws_only_valid_points():
    raw = [
        RawDetection(
            0, 0.9, (10, 40, 150, 90), (RawKeypoint((50, 60), 0.8), RawKeypoint((100, 60), 0.1))
        )
    ]
    detections, _ = filter_detections(raw, 160, 96, 1, 0.5, (2, 3))
    record = FrameRecord(0, 0.0, 160, 96, None, "candidate", detections)
    image = np.zeros((96, 160, 3), dtype=np.uint8)
    canvas = annotate_frame(image, record, {0: "object"})
    assert not image.any()
    assert tuple(canvas[60, 50]) == (255, 0, 255)
    assert not canvas[60, 100].any()


@pytest.fixture
def fake_yolo(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("YOLO_OFFLINE", "true")
    import torch
    import ultralytics.models.yolo.model as yolo_module
    from ultralytics.engine.results import Results

    image = np.zeros((100, 200, 3), dtype=np.uint8)
    boxes = torch.tensor(
        [[0, 0, 100, 80, 0.8, 0], [1, 1, 101, 81, 0.9, 0], [0, 0, 100, 80, 0.7, 1]]
    )
    points = torch.tensor(
        [
            [[10, 10, 0.6], [20, 20, 0.2]],
            [[30, 30, 0.9], [40, 40, 0.1]],
            [[50, 50, 0.8], [60, 60, 0.3]],
        ]
    )
    result = Results(image, "unused", {0: "pillar", 1: "exchange"}, boxes=boxes, keypoints=points)
    yolo = SimpleNamespace(
        task="pose",
        names={0: "pillar", 1: "exchange"},
        model=SimpleNamespace(kpt_shape=[2, 3]),
        predict=lambda *a, **kw: [result],
    )
    monkeypatch.setattr(yolo_module, "YOLO", lambda path: yolo)
    path = tmp_path / "metadata.yaml"
    path.write_text(yaml.safe_dump({"names": yolo.names, "task": "pose", "kpt_shape": [2, 3]}))
    config = Config(
        input="unused", output="unused", model="unused.pt", model_yaml=str(path), device="cpu"
    )
    return config, yolo, result, image


def test_detector_nms_keeps_matching_keypoints_and_classes(fake_yolo):
    config, _, _, image = fake_yolo
    predictions = model.Detector(config, "cpu").predict(image)
    assert [d.class_id for d in predictions] == [0, 1]
    assert [d.keypoints[0].xy for d in predictions] == [(30, 30), (50, 50)]
    assert predictions[0].keypoints[1].confidence == pytest.approx(0.1)


@pytest.mark.parametrize(
    "field,value",
    [("task", "detect"), ("kpt_shape", [3, 3]), ("names", {0: "wrong", 1: "exchange"})],
)
def test_detector_rejects_conflicting_yaml(fake_yolo, field, value):
    config, _, _, _ = fake_yolo
    path = Path(config.model_yaml)
    metadata = yaml.safe_load(path.read_text())
    metadata[field] = value
    path.write_text(yaml.safe_dump(metadata))
    with pytest.raises(ValueError, match="disagree"):
        model.Detector(config, "cpu")


def test_pose_download_selects_correct_files(tmp_path, monkeypatch):
    def download(command, **kwargs):
        Path(command[command.index("--output") + 1]).write_bytes(b"data")

    monkeypatch.setattr(model.subprocess, "run", download)
    paths = model.download_model(tmp_path, task="pose")
    assert Path(paths["model"]).relative_to(tmp_path).as_posix() == model.POSE_WEIGHTS
    assert Path(paths["model_yaml"]).relative_to(tmp_path).as_posix() == model.POSE_METADATA


@pytest.mark.parametrize("confidence", [-0.1, 1.1, float("nan")])
def test_invalid_keypoint_threshold(confidence):
    with pytest.raises(ValueError):
        Config(
            input="a", output="b", model="c.pt", model_yaml="d.yaml", keypoint_confidence=confidence
        )


@pytest.mark.parametrize("mode", ["empty", "two_dimensions", "mismatch", "missing"])
def test_pose_adapter_result_shapes(fake_yolo, mode):
    config, yolo, result, image = fake_yolo
    if mode == "empty":
        yolo.predict = lambda *a, **kw: [result[:0]]
        assert model.Detector(config, "cpu").predict(image) == []
    elif mode == "two_dimensions":
        yolo.model.kpt_shape = [2, 2]
        path = Path(config.model_yaml)
        metadata = yaml.safe_load(path.read_text())
        metadata["kpt_shape"] = [2, 2]
        path.write_text(yaml.safe_dump(metadata))
        result.keypoints.data = result.keypoints.data[..., :2]
        predictions = model.Detector(config, "cpu").predict(image)
        assert all(p.confidence is None for d in predictions for p in d.keypoints)
    else:
        if mode == "missing":
            result.keypoints = None
        else:
            result.keypoints.data = result.keypoints.data[:1]
        with pytest.raises(ValueError, match="keypoints|shape"):
            model.Detector(config, "cpu").predict(image)


def test_batch_adapter_preserves_image_order_and_independent_nms(fake_yolo):
    config, yolo, result, image = fake_yolo
    from ultralytics.engine.results import Results

    second = Results(
        image,
        "second",
        result.names,
        boxes=result.boxes.data[[0]].clone(),
        keypoints=result.keypoints.data[[0]].clone(),
    )
    second.boxes.data[0, :4] += 5
    second.keypoints.data[..., :2] += 5
    calls = []

    def predict(images, **kwargs):
        calls.append((len(images), kwargs))
        assert images[0] is image
        return [result, result[:0], second]

    yolo.predict = predict
    predictions = model.Detector(config, "cpu").predict_batch([image, image.copy(), image.copy()])
    assert len(calls) == 1
    assert calls[0][0] == calls[0][1]["batch"] == 3
    assert calls[0][1]["rect"] is True
    assert [len(p) for p in predictions] == [2, 0, 1]
    assert [d.class_id for d in predictions[0]] == [0, 1]
    assert predictions[0][0].keypoints[0].xy == (30, 30)
    assert predictions[2][0].xyxy == (5, 5, 105, 85)
    assert predictions[2][0].keypoints[0].xy == (15, 15)


def test_batch_adapter_checks_result_count_empty_input_and_dimensions(fake_yolo):
    config, _, _, image = fake_yolo
    detector = model.Detector(config, "cpu")
    assert detector.predict_batch([]) == []
    with pytest.raises(ValueError, match="result count"):
        detector.predict_batch([image, image.copy()])
    with pytest.raises(ValueError, match="same dimensions"):
        detector.predict_batch([image, image[:50]])


def test_pose_pipeline_supports_partial_batches(pose_run):
    config, detector = pose_run
    config = replace(config, batch_size=7)
    report = pipeline.run_pipeline(config)
    records = read_records(Path(config.output) / "frames.jsonl")
    assert len(records) == detector.calls == 24
    assert report["no_keypoints_frames"] > 0
    assert report["keypoints"]["valid_points"] > 0
    assert validate_dataset(config.output)["valid"]
