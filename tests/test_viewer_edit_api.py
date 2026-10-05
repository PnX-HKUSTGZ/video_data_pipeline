"""Exercise durable edits and exports through the actual HTTP/service boundaries."""

import hashlib
import time
from pathlib import Path
from threading import Event

import pytest
import yaml
from PIL import Image

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from yolo_dataset_viewer.app import create_app


def dataset(root):
    (root / "images/train").mkdir(parents=True)
    (root / "labels/train").mkdir(parents=True)
    for name in ("a", "b"):
        Image.new("RGB", (80, 60), "green").save(root / f"images/train/{name}.png")
        (root / f"labels/train/{name}.txt").write_text("0 .4 .6 .3 .4\n")
    (root / "data.yaml").write_text(yaml.safe_dump({"train": "images/train", "names": ["old"]}))
    return root


def fingerprint(root):
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file() and not path.name.startswith(".yolo-viewer-draft")
    }


def opened(client, root, task="detect"):
    response = client.post("/api/datasets", json={"path": str(root), "task": task})
    assert response.status_code == 202, response.text
    base = "/api/datasets/" + response.json()["id"]
    for _ in range(300):
        status = client.get(base).json()
        if status["state"] != "indexing":
            assert status["state"] == "ready", status
            return base
        time.sleep(0.01)
    pytest.fail("index timed out")


def edit(client, base, operation):
    revision = client.get(base + "/draft").json()["revision"]
    response = client.post(
        base + "/draft/operations",
        json={
            "expected_revision": revision,
            "operation": operation,
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def exported(client, base, mode, destination=None):
    request = {"expected_revision": client.get(base + "/draft").json()["revision"], "mode": mode}
    if destination is not None:
        request["destination"] = str(destination)
    preview = client.post(base + "/exports/preview", json=request)
    assert preview.status_code == 200, preview.text
    assert not preview.json()["errors"], preview.json()
    start = client.post(base + "/exports", json=request)
    assert start.status_code == 202, start.text
    for _ in range(500):
        result = client.get(base + "/exports/" + start.json()["id"]).json()
        if result["state"] != "running":
            assert result["state"] == "complete", result
            return result
        time.sleep(0.01)
    pytest.fail("export timed out")


def test_api_durable_edit_snapshot_and_new_export(tmp_path):
    root = dataset(tmp_path / "source")
    before = fingerprint(root)
    with TestClient(create_app()) as client:
        base = opened(client, root)
        assert not (root / ".yolo-viewer-draft.sqlite3").exists()
        edit(client, base, {"kind": "class_rename", "class_id": 0, "name": "物体"})
        state = edit(
            client,
            base,
            {
                "kind": "keypoints_set",
                "class_id": 0,
                "keypoints": [{"name": "tip"}, {"name": "base"}],
            },
        )
        detail = client.get(base + "/samples/0").json()
        assert detail["annotations"][0]["kind"] == "pose"
        annotation = detail["annotations"][0]
        edit(
            client,
            base,
            {
                "kind": "annotation_upsert",
                "sample_id": 0,
                "annotation_id": annotation["uid"],
                "class_id": 0,
                "box": annotation["box"],
                "keypoints": [
                    {"x": 0.1, "y": 0.2, "visibility": 2},
                    {"x": 0.8, "y": 0.9, "visibility": 1},
                ],
            },
        )
        saved = edit(client, base, {"kind": "sample_delete", "sample_id": 1})
        assert saved["deleted_images"] == 1
        page = client.get(base + "/samples").json()
        stats = client.get(base + "/stats").json()
        assert page["revision"] == stats["revision"] == saved["revision"]
        assert page["total"] == stats["images"] == 1
        assert client.get(base + "/samples?edit_state=deleted").json()["total"] == 1
        assert fingerprint(root) == before
        rejected = client.post(base + "/draft/undo", json={"expected_revision": state["revision"]})
        assert rejected.status_code == 409
        rejected = client.post(
            base + "/draft/undo",
            headers={"Origin": "https://evil.test"},
            json={"expected_revision": saved["revision"]},
        )
        assert rejected.status_code == 403
    with TestClient(create_app()) as client:
        base = opened(client, root)
        draft = client.get(base + "/draft").json()
        assert draft["revision"] == saved["revision"]
        assert draft["task"] == "pose"
        result = exported(client, base, "new", tmp_path / "exported")
        assert result["result"]["task"] == "pose"
        assert fingerprint(root) == before
        assert client.get(base + "/draft").json()["dirty"]
        base = opened(client, result["result"]["config_path"], "pose")
        detail = client.get(base + "/samples/0").json()
        assert detail["annotations"][0]["keypoints"][0]["x"] == 0.1
        assert client.get(base + "/draft").json()["classes"][0]["keypoints"][1]["name"] == "base"
        assert not client.get(base + "/draft").json()["dirty"]


def test_inplace_rebases_session_and_keeps_backup(tmp_path):
    root = dataset(tmp_path / "source")
    before = fingerprint(root)
    with TestClient(create_app()) as client:
        base = opened(client, root)
        edit(client, base, {"kind": "keypoints_set", "class_id": 0, "keypoints": [{"name": "tip"}]})
        edit(client, base, {"kind": "sample_delete", "sample_id": 1})
        result = exported(client, base, "inplace")
        assert result["reopen"]["task"] == "pose"
        assert client.get(base + "/samples").status_code == 409
        base = opened(client, result["reopen"]["path"], result["reopen"]["task"])
        assert result["result"]["backup_path"]
        backup = Path(result["result"]["backup_path"])
        assert backup.is_dir() and backup.parent == root.parent
        assert (backup / "manifest.json").is_file()
        assert not (root / "images/train/b.png").exists()
        assert not (root / "labels/train/b.txt").exists()
        assert not client.get(base + "/draft").json()["dirty"]
        assert client.get(base + "/stats").json()["images"] == 1
        assert client.get(base + "/samples/0").json()["annotations"][0]["kind"] == "pose"
        assert client.get(base + "/exports/recoveries").json() == []
        assert fingerprint(root) != before


def test_export_progress_does_not_lock_polling_and_rejects_edits(tmp_path, monkeypatch):
    from yolo_dataset_viewer import service

    root = dataset(tmp_path / "source")
    entered, proceed = Event(), Event()
    original = service.Exporter.run

    def held(self, *args):
        entered.set()
        assert proceed.wait(5)
        return original(self, *args)

    monkeypatch.setattr(service.Exporter, "run", held)
    with TestClient(create_app()) as client:
        base = opened(client, root)
        response = client.post(
            base + "/exports",
            json={"expected_revision": 0, "mode": "new", "destination": str(tmp_path / "out")},
        )
        assert response.status_code == 202
        assert entered.wait(2)
        try:
            assert (
                client.get(base + "/exports/" + response.json()["id"]).json()["state"] == "running"
            )
            assert (
                client.post(base + "/draft/undo", json={"expected_revision": 0}).status_code == 409
            )
            assert client.post("/api/datasets", json={"path": str(root)}).status_code == 409
        finally:
            proceed.set()


def test_discard_reindexes_external_changes_without_reusing_revision(tmp_path):
    root = dataset(tmp_path / "source")
    with TestClient(create_app()) as client:
        base = opened(client, root)
        saved = edit(client, base, {"kind": "class_rename", "class_id": 0, "name": "draft"})
        (root / "labels/train/a.txt").write_text("0 .5 .5 .2 .3\n0 .2 .2 .1 .1\n")
        assert client.get(base + "/draft").json()["conflicts"]
        response = client.post(
            base + "/draft/discard", json={"expected_revision": saved["revision"]}
        )
        assert response.status_code == 200, response.text
        status = response.json()
        assert client.get(base + "/samples").status_code == 409
        base = "/api/datasets/" + status["session_id"]
        assert not status["conflicts"] and not status["dirty"]
        assert status["revision"] > saved["revision"]
        assert status["classes"][0]["name"] == "old"
        assert client.get(base + "/stats").json()["instances"] == 3
        assert len(client.get(base + "/samples/0").json()["annotations"]) == 2


def test_recovery_available_even_if_task_schema_was_changed_before_crash(tmp_path):
    from yolo_dataset_viewer.dataset import build_index
    from yolo_dataset_viewer.edit_models import EditRequest
    from yolo_dataset_viewer.editing import DraftEditor
    from yolo_dataset_viewer.exports import Exporter, ExportRequest
    from yolo_dataset_viewer.models import OpenRequest

    root = dataset(tmp_path / "source")
    (root / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "train": "images/train",
                "names": ["old"],
                "kpt_shape": [1, 3],
            }
        )
    )
    for label in (root / "labels/train").glob("*.txt"):
        label.write_text("0 .4 .6 .3 .4 .2 .3 2\n")
    original = fingerprint(root)
    request = OpenRequest(path=str(root), task="pose")
    data = build_index(request, Event(), lambda *_: None)
    editor = DraftEditor(data, request)
    editor.apply(
        EditRequest.model_validate(
            {
                "expected_revision": 0,
                "operation": {
                    "kind": "keypoints_set",
                    "class_id": 0,
                    "keypoints": [],
                },
            }
        )
    )
    result = Exporter(data, editor, request).run(
        ExportRequest(expected_revision=1, mode="inplace"),
        lambda *_: None,
        Event(),
    )
    # Simulate process loss after files were applied, before reindex and finalization.
    with TestClient(create_app()) as client:
        response = client.post("/api/datasets", json=request.model_dump())
        base = "/api/datasets/" + response.json()["id"]
        for _ in range(300):
            status = client.get(base).json()
            if status["state"] != "indexing":
                break
            time.sleep(0.01)
        assert status["state"] == "error" and status["recovery_available"]
        pending = client.get(base + "/exports/recoveries").json()
        assert pending[0]["path"] == result.backup_path
        restored = client.post(
            base + "/exports/recover",
            json={
                "expected_revision": 0,
                "backup_path": pending[0]["path"],
            },
        )
        assert restored.status_code == 202, restored.text
        assert restored.json()["request"]["task"] == "pose"
        base = "/api/datasets/" + restored.json()["id"]
        for _ in range(300):
            status = client.get(base).json()
            if status["state"] != "indexing":
                break
            time.sleep(0.01)
        assert status["state"] == "ready", status
        assert fingerprint(root) == original
        assert client.get(base + "/draft").json()["dirty"]
        assert client.get(base + "/exports/recoveries").json() == []
