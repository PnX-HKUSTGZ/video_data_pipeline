import json
import multiprocessing as mp
import shutil
import time
from dataclasses import replace
from pathlib import Path

import pytest

from video_dataset_pipeline import parallel, pipeline
from video_dataset_pipeline.dataset import read_records, validate_dataset


@pytest.fixture
def parallel_run(synthetic_run, tmp_path, monkeypatch):
    config, detector = synthetic_run
    other = tmp_path / "other" / Path(config.input).name
    other.parent.mkdir()
    shutil.copyfile(config.input, other)
    config = replace(config, input=[config.input, str(other)], batch_size=8)
    # 测试进程只继承 CPU 假模型；生产使用 spawn，不继承 CUDA 上下文。
    context = mp.get_context("fork")
    monkeypatch.setattr(parallel.mp, "get_context", lambda _: context)
    monkeypatch.setattr(parallel, "build_pipeline", detector)
    monkeypatch.setattr(parallel, "configure_device", lambda _: "cpu")
    monkeypatch.setattr(parallel, "gpu_snapshot", lambda *_: None)
    return config, detector


@pytest.mark.parametrize("save_video", [False, True])
def test_parallel_merges_in_input_order_and_matches_serial(parallel_run, monkeypatch, save_video):
    config, _ = parallel_run
    config = replace(config, save_video=save_video)
    reference = replace(config, output=config.output + "_serial")
    expected = pipeline.run_pipeline(reference)
    original = parallel._process_source

    def delayed(config, root, run, source, detector):
        if source["source_id"] == "v0001":
            time.sleep(0.5)
        return original(config, root, run, source, detector)

    monkeypatch.setattr(parallel, "_process_source", delayed)
    result = pipeline.run_pipeline(config, [0, 2])
    root = Path(config.output)
    records = read_records(root / "frames.jsonl")
    assert records == read_records(Path(reference.output) / "frames.jsonl")
    assert result["class_instances"] == expected["class_instances"]
    assert result["splits"] == expected["splits"]
    assert validate_dataset(root, False)["valid"]
    for record in records:
        if record.label:
            assert (root / record.label).read_bytes() == (
                Path(reference.output) / record.label
            ).read_bytes()
    assert (root / "COMPLETE.json").exists()
    assert not list(root.rglob(".staging"))
    assert not list((root / "workers").glob("v*/images"))
    assert not mp.active_children()


def test_parallel_zero_detection_source_is_allowed(parallel_run, monkeypatch):
    config, _ = parallel_run
    original = parallel._process_source

    def empty_second(config, root, run, source, detector):
        if source["source_id"] == "v0002":
            detector.predict_batch = lambda images: [[] for _ in images]
        return original(config, root, run, source, detector)

    monkeypatch.setattr(parallel, "_process_source", empty_second)
    report = pipeline.run_pipeline(config, [0])
    assert report["videos"][0]["included_frames"] > 0
    assert report["videos"][1]["no_detection_frames"] == 24
    assert report["videos"][1]["included_frames"] == 0


@pytest.mark.parametrize("initialization", [False, True])
def test_parallel_failure_reaps_workers_and_removes_staging(
    parallel_run, monkeypatch, initialization
):
    config, detector = parallel_run
    if initialization:

        def fail(*_):
            raise RuntimeError("initialization failed")

        monkeypatch.setattr(parallel, "build_pipeline", fail)
    else:
        detector.fail_at = 12
    with pytest.raises(RuntimeError, match="GPU worker failed"):
        pipeline.run_pipeline(config, [0, 2])
    root = Path(config.output)
    assert json.loads((root / "run.json").read_text())["status"] == "failed"
    assert not (root / "COMPLETE.json").exists()
    assert not list(root.rglob(".staging"))
    assert not list(root.rglob("*.tmp"))
    assert not mp.active_children()


@pytest.mark.parametrize("devices", [[], [0, 0], [-1], [True]])
def test_invalid_parallel_devices_rejected_before_creating_output(synthetic_run, devices):
    config, _ = synthetic_run
    with pytest.raises(ValueError, match="distinct nonnegative"):
        pipeline.run_pipeline(config, devices)
    assert not Path(config.output).exists()
