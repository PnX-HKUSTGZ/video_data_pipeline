import json
import os

import pytest
import yaml

from video_dataset_pipeline import cli, device, pipeline
from video_dataset_pipeline.config import load_config


@pytest.mark.parametrize(("value", "expected"), [(0, 0), (4, 4), ("4", 4), ("cpu", "cpu")])
def test_device_config_and_cli_use_gpu_indices(tmp_path, monkeypatch, capsys, value, expected):
    for name in ("video.mp4", "model.pt", "model.yaml"):
        (tmp_path / name).touch()
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "input": str(tmp_path / "video.mp4"),
                "output": str(tmp_path / "out"),
                "model": str(tmp_path / "model.pt"),
                "model_yaml": str(tmp_path / "model.yaml"),
                "device": value,
            }
        )
    )
    assert load_config(path).device == expected
    monkeypatch.setattr(pipeline, "run_pipeline", lambda config: {"device": config.device})
    monkeypatch.setattr(
        "sys.argv", ["video-dataset", "run", "--config", str(path), "--device", str(value)]
    )
    cli.main()
    assert json.loads(capsys.readouterr().out)["device"] == expected


def test_gpu_index_selects_requested_physical_card(tmp_path, monkeypatch):
    monkeypatch.setenv("YOLO_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "old-selection")
    monkeypatch.setattr(
        device.subprocess, "check_output", lambda *a, **k: "0, GPU-zero, 24000\n4, GPU-four, 8192\n"
    )
    assert device.configure_device(4) == "0"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-four"
    with pytest.raises(ValueError, match="GPU index not available: 9"):
        device.configure_device(9)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
    assert device.configure_device("cpu") == "cpu"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
