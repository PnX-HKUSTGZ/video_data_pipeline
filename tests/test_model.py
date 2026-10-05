import subprocess
from pathlib import Path

import pytest
import yaml

from video_dataset_pipeline import cli, model
from video_dataset_pipeline.config import load_config


def test_download_only_produces_model_and_yaml(tmp_path, monkeypatch):
    def fake_download(command, **kwargs):
        path = Path(command[command.index("--output") + 1])
        path.write_bytes(b"downloaded file")

    monkeypatch.setattr(model.subprocess, "run", fake_download)
    result = model.download_model(tmp_path, endpoint="https://example.invalid")
    assert set(result) == {"model", "model_yaml"}
    assert Path(result["model"]).is_file()
    assert Path(result["model_yaml"]).is_file()
    assert {p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file()} == {
        model.WEIGHTS,
        model.METADATA,
    }


def test_failed_download_cleans_temporary_file(tmp_path, monkeypatch):
    def fail(command, **kwargs):
        Path(command[command.index("--output") + 1]).write_bytes(b"partial")
        raise subprocess.CalledProcessError(28, command)

    monkeypatch.setattr(model.subprocess, "run", fail)
    with pytest.raises(RuntimeError, match="download failed"):
        model.download_model(tmp_path, endpoint="https://example.invalid")
    assert not any(p.is_file() for p in tmp_path.rglob("*"))


@pytest.mark.parametrize("names", [{0: "pillar", 1: "exchange"}, ["pillar", "exchange"]])
def test_class_names_from_configured_yaml(tmp_path, names):
    path = tmp_path / "arbitrary.yaml"
    path.write_text(yaml.safe_dump({"names": names}))
    assert model.load_model_names(path) == {0: "pillar", 1: "exchange"}


def test_config_accepts_independent_model_paths(tmp_path):
    for filename in ("video.mp4", "custom.pt", "custom.yaml"):
        (tmp_path / filename).touch()
    config_path = tmp_path / "run.yaml"
    values = {
        "input": str(tmp_path / "video.mp4"),
        "output": str(tmp_path / "out"),
        "model": str(tmp_path / "custom.pt"),
        "model_yaml": str(tmp_path / "custom.yaml"),
    }
    config_path.write_text(yaml.safe_dump(values))
    config = load_config(config_path)
    assert config.model == values["model"]
    assert config.model_yaml == values["model_yaml"]
    del values["model_yaml"]
    config_path.write_text(yaml.safe_dump(values))
    with pytest.raises(ValueError, match="model_yaml"):
        load_config(config_path)


def test_cli_rejects_removed_option(monkeypatch):
    monkeypatch.setattr("sys.argv", ["video-dataset", "run", "--config", "unused.yaml", "--resume"])
    with pytest.raises(SystemExit) as result:
        cli.main()
    assert result.value.code == 2
