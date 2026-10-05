import json
import shutil
from dataclasses import replace
from pathlib import Path

import av
import numpy as np
import pytest
import yaml

from video_dataset_pipeline import cli, pipeline
from video_dataset_pipeline.config import load_config
from video_dataset_pipeline.dataset import read_records, validate_dataset
from video_dataset_pipeline.inference_video import InferenceVideoWriter
from video_dataset_pipeline.inputs import discover_videos
from video_dataset_pipeline.records import FrameRecord, RawKeypoint


@pytest.fixture(autouse=True)
def preserve_pillow_reader(monkeypatch):
    from PIL import Image

    monkeypatch.setattr(Image, "open", Image.open)


def test_discovery_mixed_inputs_order_recursion_and_resolved_path_dedup(tmp_path):
    for relative in ("one/B.MP4", "one/A.flv", "one/sub/C.mov", "two/D.mkv", "one/readme.txt"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    (tmp_path / "alias.mp4").symlink_to(tmp_path / "one/A.flv")
    (tmp_path / "one/loop").symlink_to(tmp_path / "one", target_is_directory=True)
    inputs = [str(tmp_path / p) for p in ("one/B.MP4", "one", "alias.mp4", "two")]
    direct = discover_videos(inputs, tmp_path / "out")
    assert [v.path.name for v in direct] == ["B.MP4", "A.flv", "D.mkv"]
    recursive = discover_videos(inputs, tmp_path / "out", True)
    assert [v.path.name for v in recursive] == ["B.MP4", "A.flv", "C.mov", "D.mkv"]
    assert [v.source_id for v in recursive] == ["v0001", "v0002", "v0003", "v0004"]


def test_discovery_excludes_output_and_links_to_output(tmp_path):
    source = tmp_path / "input.mp4"
    source.touch()
    output = tmp_path / "generated"
    output.mkdir()
    (output / "inference.mp4").touch()
    (tmp_path / "alias.mp4").symlink_to(output / "inference.mp4")
    assert [v.path for v in discover_videos(str(tmp_path), output, True)] == [source]
    with pytest.raises(ValueError, match="inside output"):
        discover_videos(str(output / "inference.mp4"), output)


@pytest.mark.parametrize("kind", ["missing", "empty", "no_video", "empty_list"])
def test_discovery_rejects_invalid_entries(tmp_path, kind):
    path = tmp_path / "input"
    if kind in {"empty", "no_video"}:
        path.mkdir()
    if kind == "no_video":
        (path / "text.txt").touch()
    with pytest.raises(ValueError):
        discover_videos([] if kind == "empty_list" else str(path), tmp_path / "out")


@pytest.mark.parametrize("value", [None, [], [""], "", ["a", " "], [42], 42])
def test_config_rejects_invalid_input_types(synthetic_run, tmp_path, value):
    config, _ = synthetic_run
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({**config.to_dict(), "input": value}))
    with pytest.raises((ValueError, TypeError)):
        load_config(path)


def test_yaml_and_repeated_cli_inputs_replace_list(synthetic_run, tmp_path, monkeypatch, capsys):
    config, _ = synthetic_run
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump({**config.to_dict(), "input": [config.input], "recursive": True})
    )
    loaded = load_config(path)
    assert loaded.input == [config.input]
    assert loaded.recursive
    monkeypatch.setattr(pipeline, "run_pipeline", lambda config: config.to_dict())
    monkeypatch.setattr(
        "sys.argv",
        [
            "video-dataset",
            "run",
            "--config",
            str(path),
            "--input",
            "a.mp4",
            "--input",
            "中文目录",
            "--no-recursive",
        ],
    )
    cli.main()
    result = json.loads(capsys.readouterr().out)
    assert result["input"] == [str(Path("a.mp4").resolve()), str(Path("中文目录").resolve())]
    assert result["recursive"] is False


@pytest.fixture
def multi_run(synthetic_run, tmp_path):
    config, detector = synthetic_run
    second = tmp_path / "other" / Path(config.input).name
    second.parent.mkdir()
    shutil.copyfile(config.input, second)
    config = replace(config, input=[config.input, str(second.parent)], recursive=True)
    return config, detector


@pytest.mark.parametrize("task", ["detect", "pose"])
def test_multi_video_independent_splits_unique_names_and_one_model(multi_run, task):
    config, detector = multi_run
    detector.initializations = 0
    original_init = detector.__init__
    original_predict = detector.predict

    def initialize(self, *args):
        detector.initializations += 1
        original_init(self, *args)
        self.task = task
        self.pose = {"source_kpt_shape": (1, 3)} if task == "pose" else None

    def predict(self, image):
        raw = original_predict(self, image)
        return (
            [replace(d, keypoints=(RawKeypoint((30, 20), 0.9),)) for d in raw]
            if task == "pose"
            else raw
        )

    detector.__init__, detector.predict = initialize, predict
    report = pipeline.run_pipeline(config)
    root = Path(config.output)
    records = read_records(root / "frames.jsonl")
    assert detector.initializations == 1
    assert detector.calls == report["candidate_frames"] == 48
    groups = [[r for r in records if r.source_id == sid] for sid in ("v0001", "v0002")]
    assert [r.frame_index for r in groups[0]] == [r.frame_index for r in groups[1]]
    assert [r.split for r in groups[0]] == [r.split for r in groups[1]]
    assert [r.status for r in groups[0]] == [r.status for r in groups[1]]
    assert report["split"]["v0001"] == report["split"]["v0002"]
    assert report["split_gap_frames"] == 4
    assert len(report["videos"]) == 2
    assert all(v["candidate_frames"] == 24 for v in report["videos"])
    images = [r.image for r in records if r.image]
    assert len(images) == len(set(images)) == report["included_frames"]
    assert all(Path(r.image).name.startswith(r.source_id) for r in records if r.image)
    assert all(v["preview_frames"] > 0 for v in report["videos"])
    result = validate_dataset(root)  # Real detection and pose Ultralytics loaders.
    assert result["valid"]
    assert sum(v["instances"] for v in result["ultralytics_loader"].values()) == sum(
        report["class_instances"].values()
    )
    assert (root / "COMPLETE.json").is_file()
    assert not (root / ".staging").exists()


def test_video_without_detections_keeps_statistics(multi_run):
    config, detector = multi_run
    predict = detector.predict

    def only_first(self, image):
        predictions = predict(self, image)
        return predictions if detector.calls <= 24 else []

    detector.predict = only_first
    report = pipeline.run_pipeline(config)
    assert report["candidate_frames"] == 48
    assert report["videos"][1]["included_frames"] == 0
    assert report["videos"][1]["no_detection_frames"] == 24
    assert report["split"]["v0002"]["boundary_timestamp"] is None
    assert (Path(config.output) / "COMPLETE.json").exists()


def test_smoke_count_is_per_video(multi_run):
    config, detector = multi_run
    report = pipeline.run_pipeline(replace(config, smoke_frames=6, split_gap_seconds=0))
    assert detector.calls == report["candidate_frames"] == 12
    assert all(v["candidate_frames"] == 6 for v in report["videos"])
    assert (Path(config.output) / "SMOKE_COMPLETE.json").is_file()
    assert not (Path(config.output) / "COMPLETE.json").exists()


def test_failure_in_later_video_preserves_first_and_cleans_temporary_files(multi_run):
    config, detector = multi_run
    detector.fail_at = 124
    with pytest.raises(RuntimeError, match="simulated interruption"):
        pipeline.run_pipeline(replace(config, save_video=True))
    root = Path(config.output)
    run = json.loads((root / "run.json").read_text())
    report = json.loads((root / "report.json").read_text())
    assert run["status"] == "failed"
    assert [v["status"] for v in run["videos"]] == ["complete", "failed"]
    assert "v0002" in run["error"]
    assert run["videos"][1]["input"] in run["error"]
    assert report["candidate_frames"] == 123
    assert (root / run["videos"][0]["inference_video"]).is_file()
    assert not (root / run["videos"][1]["inference_video"]).exists()
    assert not (root / "COMPLETE.json").exists()
    assert not (root / ".staging").exists()
    assert not list(root.rglob("*.tmp"))
    assert all(r.source_id == "v0001" for r in read_records(root / "frames.jsonl"))


def test_multi_video_saving_keeps_individual_rates_dimensions_and_full_frames(
    synthetic_run, tmp_path
):
    config, detector = synthetic_run
    source_dir = tmp_path / "second"
    source_dir.mkdir()
    with InferenceVideoWriter(source_dir, 128, 80, 5, 6) as writer:
        for i in range(30):
            writer.write(np.full((80, 128, 3), 100, dtype=np.uint8), i / 5)
    config = replace(
        config, input=[config.input, str(source_dir)], save_video=True, sample_fps=100, end=10
    )
    report = pipeline.run_pipeline(config)
    root = Path(config.output)
    run = json.loads((root / "run.json").read_text())
    assert [v["end"] for v in run["videos"]] == [10, 6]
    assert detector.calls == report["candidate_frames"] == 130
    for source, count, size, fps in zip(
        run["videos"], (100, 30), ((160, 96), (128, 80)), (10, 5), strict=True
    ):
        with av.open(str(root / source["inference_video"])) as container:
            stream = container.streams.video[0]
            assert (stream.width, stream.height) == size
            assert float(stream.average_rate) == fps
            assert sum(1 for _ in container.decode(stream)) == count
    assert not (root / "inference.mp4").exists()


@pytest.mark.parametrize("kind", ["corrupt", "empty_interval"])
def test_bad_source_is_rejected_before_model_loading(synthetic_run, tmp_path, kind):
    config, detector = synthetic_run
    if kind == "corrupt":
        source = tmp_path / "broken.mp4"
        source.write_bytes(b"not a video")
        config = replace(config, input=[config.input, str(source)])
    else:
        config = replace(config, start=13)
    with pytest.raises(ValueError, match="Cannot process v000"):
        pipeline.run_pipeline(config)
    assert detector.calls == 0
    assert not Path(config.output).exists()


@pytest.mark.parametrize(
    "mutation",
    [
        "unknown_source",
        "duplicate_id",
        "wrong_split",
        "incomplete_decode",
        "wrong_count",
        "timestamp",
        "source_order",
    ],
)
def test_validator_checks_each_video(multi_run, mutation):
    config, _ = multi_run
    pipeline.run_pipeline(config)
    root = Path(config.output)
    run_path = root / "run.json"
    frame_path = root / "frames.jsonl"
    run = json.loads(run_path.read_text())
    records = [json.loads(line) for line in frame_path.read_text().splitlines()]
    first = next(i for i, r in enumerate(records) if r["source_id"] == "v0002")
    if mutation == "unknown_source":
        records[first]["source_id"] = "v9999"
    elif mutation == "duplicate_id":
        records[first + 1]["frame_index"] = records[first]["frame_index"]
    elif mutation == "wrong_split":
        next(r for r in records[first:] if r["split"] == "val")["split"] = "train"
    elif mutation == "incomplete_decode":
        run["videos"][1]["decode"]["decode_complete"] = False
    elif mutation == "wrong_count":
        run["videos"][1]["candidate_frames"] += 1
    elif mutation == "timestamp":
        records[first + 1]["timestamp"] = records[first]["timestamp"]
    else:
        records = records[first:] + records[:first]
    run_path.write_text(json.dumps(run))
    frame_path.write_text("".join(json.dumps(r) + "\n" for r in records))
    with pytest.raises(ValueError):
        validate_dataset(root, False)


def test_source_id_round_trip_and_invalid_type(tmp_path):
    record = FrameRecord(0, 0, 160, 96, None, "no_detection", source_id="v0002")
    assert FrameRecord.from_dict(record.to_dict()) == record
    path = tmp_path / "frames.jsonl"
    data = record.to_dict()
    data["source_id"] = 42
    path.write_text(json.dumps(data) + "\n")
    with pytest.raises(TypeError):
        FrameRecord.from_dict(data)
    with pytest.raises(ValueError, match="Invalid frame record 1"):
        read_records(path)


def test_old_schema4_dataset_still_validates(synthetic_run):
    config, _ = synthetic_run
    pipeline.run_pipeline(config)
    root = Path(config.output)
    path = root / "run.json"
    run = json.loads(path.read_text())
    video = run.pop("videos")[0]
    run.update(schema_version=4, video=video["video"], end=video["end"], split=video["split"])
    path.write_text(json.dumps(run))
    path = root / "frames.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    for record in records:
        del record["source_id"]
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    assert validate_dataset(root, False)["valid"]
