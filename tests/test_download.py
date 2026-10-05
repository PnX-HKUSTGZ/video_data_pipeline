import hashlib

import pytest

from video_dataset_pipeline import model
from video_dataset_pipeline.util import sha256, write_json


def test_download_resumes_and_checks_remote_hashes(tmp_path, monkeypatch):
    payloads = {
        model.WEIGHTS: b"published model weights",
        model.METADATA: b"names: {0: pillar}\n",
        "README.md": b"model card\n",
    }
    siblings = []
    for filename, data in payloads.items():
        item = {
            "rfilename": filename,
            "size": len(data),
            "blobId": hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest(),
        }
        if filename == model.WEIGHTS:
            item["lfs"] = {"sha256": hashlib.sha256(data).hexdigest()}
        siblings.append(item)
    cache = tmp_path / ".cache/direct"
    cache.mkdir(parents=True)
    write_json(
        cache / "hub.json", {"id": model.REPO_ID, "sha": model.REVISION, "siblings": siblings}
    )
    (cache / "best.pt.part").write_bytes(payloads[model.WEIGHTS][:4])
    calls = []

    def fake_curl(url, path, resume=False, timeout=60):
        calls.append((url, resume, path.read_bytes() if path.exists() else b""))
        filename = next(name for name in payloads if name in url)
        path.write_bytes(payloads[filename])

    monkeypatch.setattr(model, "_curl", fake_curl)
    manifest = model.download_model(tmp_path, endpoint="https://example.invalid")
    assert calls[0][1:] == (True, b"publ")
    assert manifest["files"][0]["sha256"] == sha256(tmp_path / model.WEIGHTS)
    assert model.verify_model(tmp_path) == manifest
    assert model.download_model(tmp_path) == manifest
    assert len(calls) == 3
    (tmp_path / model.WEIGHTS).write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="integrity"):
        model.verify_model(tmp_path)


def test_transfer_recomputes_resume_offset_on_each_invocation(monkeypatch, tmp_path):
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        return type("Completed", (), {"returncode": 28, "stderr": "timeout"})()

    monkeypatch.setattr(model.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="timeout"):
        model._curl("https://example.invalid/a", tmp_path / "a.part", resume=True)
    assert "--continue-at" in commands[0]
    assert "--retry" not in commands[0]


def test_download_rejects_mutable_revision(tmp_path):
    with pytest.raises(ValueError, match="commit SHA"):
        model.download_model(tmp_path, revision="main")
