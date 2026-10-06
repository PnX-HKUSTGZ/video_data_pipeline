import json
import shutil
from dataclasses import replace
from pathlib import Path

import av
import pytest
import yaml

from video_dataset_pipeline import cli, pipeline
from video_dataset_pipeline.config import load_config
from video_dataset_pipeline.dataset import read_records


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "8", None])
def test_batch_size_requires_positive_integer(synthetic_run, value):
    config, _ = synthetic_run
    with pytest.raises((ValueError, TypeError), match="integer"):
        replace(config, batch_size=value)


def test_batch_size_yaml_and_cli_override(synthetic_run, tmp_path, monkeypatch, capsys):
    config, _ = synthetic_run
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({**config.to_dict(), "batch_size": 8}))
    assert load_config(path).batch_size == 8
    monkeypatch.setattr(pipeline, "run_pipeline", lambda config: {"batch_size": config.batch_size})
    monkeypatch.setattr(
        "sys.argv", ["video-dataset", "run", "--config", str(path), "--batch-size", "4"]
    )
    cli.main()
    assert json.loads(capsys.readouterr().out) == {"batch_size": 4}
    path.write_text(yaml.safe_dump({**config.to_dict(), "batch_size": "8"}))
    with pytest.raises(TypeError, match="integer"):
        load_config(path)


@pytest.mark.parametrize("batch_size", [1, 7, 24, 32])
def test_batch_predictions_keep_order_tail_and_dataset(synthetic_run, tmp_path, batch_size):
    config, detector = synthetic_run
    baseline = replace(config, output=str(tmp_path / "baseline"))
    expected_report = pipeline.run_pipeline(baseline)
    expected_records = read_records(Path(baseline.output) / "frames.jsonl")
    predict_batch = detector.predict_batch
    calls = []

    def capture(self, images):
        calls.append(len(images))
        assert len(images) <= batch_size
        return predict_batch(self, images)

    detector.predict_batch = capture
    report = pipeline.run_pipeline(replace(config, batch_size=batch_size))
    actual_records = read_records(Path(config.output) / "frames.jsonl")
    assert actual_records == expected_records
    assert calls == [min(batch_size, 24 - i) for i in range(0, 24, batch_size)]
    assert sum(calls) == report["candidate_frames"] == 24
    assert report["splits"] == expected_report["splits"]
    for record in actual_records:
        if record.label:
            assert (Path(config.output) / record.label).read_bytes() == (
                Path(baseline.output) / record.label
            ).read_bytes()


def test_batch_is_flushed_at_each_video_boundary(synthetic_run, tmp_path):
    config, detector = synthetic_run
    second = tmp_path / "other.mp4"
    shutil.copyfile(config.input, second)
    calls = []
    predict_batch = detector.predict_batch

    def capture(self, images):
        calls.append(len(images))
        return predict_batch(self, images)

    detector.predict_batch = capture
    report = pipeline.run_pipeline(replace(config, input=[config.input, str(second)], batch_size=7))
    assert calls == [7, 7, 7, 3] * 2
    assert report["candidate_frames"] == 48
    records = read_records(Path(config.output) / "frames.jsonl")
    assert [r.source_id for r in records] == ["v0001"] * 24 + ["v0002"] * 24


def test_batch_video_keeps_every_frame_and_timestamp(synthetic_run):
    config, detector = synthetic_run
    calls = []
    predict_batch = detector.predict_batch

    def capture(self, images):
        calls.append(len(images))
        return predict_batch(self, images)

    detector.predict_batch = capture
    report = pipeline.run_pipeline(replace(config, save_video=True, batch_size=16))
    assert calls == [16] * 7 + [8]
    assert report["candidate_frames"] == detector.calls == 120
    with av.open(str(Path(config.output) / "inference.mp4")) as container:
        frames = list(container.decode(video=0))
        assert len(frames) == 120
        assert [float(f.pts * f.time_base) for f in frames] == pytest.approx(
            [i / 10 for i in range(120)]
        )
    assert (Path(config.output) / "COMPLETE.json").exists()


def test_batch_smoke_keeps_requested_sample_count(synthetic_run):
    config, detector = synthetic_run
    report = pipeline.run_pipeline(
        replace(config, batch_size=8, smoke_frames=6, split_gap_seconds=0)
    )
    assert report["candidate_frames"] == detector.calls == 6
    assert (Path(config.output) / "SMOKE_COMPLETE.json").exists()


@pytest.mark.parametrize("failure", ["model", "result_count"])
def test_failed_batch_does_not_commit_partial_predictions(synthetic_run, failure):
    config, detector = synthetic_run
    predict_batch = detector.predict_batch
    batches = 0

    def fail(self, images):
        nonlocal batches
        batches += 1
        if batches == 2:
            if failure == "model":
                raise RuntimeError("simulated CUDA out of memory")
            return [[]]
        return predict_batch(self, images)

    detector.predict_batch = fail
    with pytest.raises((ValueError, RuntimeError), match="batch size|out of memory"):
        pipeline.run_pipeline(replace(config, batch_size=4, save_video=True))
    root = Path(config.output)
    assert len(read_records(root / "inference.jsonl")) == 4
    assert json.loads((root / "run.json").read_text())["status"] == "failed"
    assert not (root / "COMPLETE.json").exists()
    assert not (root / "inference.mp4").exists()
    assert not (root / ".staging").exists()
    assert not list(root.rglob("*.tmp"))
