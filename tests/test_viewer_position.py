"""Filtered inspector navigation must follow the same order and revision as the gallery."""

import time
from threading import Event

import pytest
import yaml
from fastapi.testclient import TestClient
from PIL import Image

from yolo_dataset_viewer.app import create_app
from yolo_dataset_viewer.dataset import build_index
from yolo_dataset_viewer.edit_models import EditRequest
from yolo_dataset_viewer.editing import DraftEditor
from yolo_dataset_viewer.models import OpenRequest
from yolo_dataset_viewer.service import ViewerService


def source_dataset(root, count=6, *, shared_split=False):
    """Create only portable YOLO files, without any pipeline metadata."""
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    for number in range(count):
        Image.new("RGB", (24, 16), (number, 70, 90)).save(root / f"images/样本_{number:03}.png")
        (root / f"labels/样本_{number:03}.txt").write_text("0 .5 .5 .4 .4\n")
    config = {"train": "images", "names": ["original", "other"]}
    if shared_split:
        config["val"] = "images"
    (root / "data.yaml").write_text(yaml.safe_dump(config))
    return root


def open_service(service, root):
    session = service.open(OpenRequest(path=str(root))).id
    for _ in range(500):
        status = service.status(session)
        if status.state != "indexing":
            assert status.state == "ready", status.error
            return session
        time.sleep(0.005)
    pytest.fail("Dataset index did not finish")


@pytest.fixture
def service():
    instance = ViewerService()
    try:
        yield instance
    finally:
        instance.close()


def apply(service, session, **operation):
    revision = service.draft(session).revision
    return service.edit(
        session,
        EditRequest.model_validate(
            {
                "expected_revision": revision,
                "operation": operation,
            }
        ),
    )


def position(service, session, sample_id, **filters):
    return service.position(
        session,
        sample_id,
        filters.get("split"),
        filters.get("class_id"),
        filters.get("state"),
        filters.get("search", ""),
        filters.get("edit_state"),
    )


def page(service, session, offset=0, limit=200, **filters):
    return service.page(
        session,
        filters.get("split"),
        filters.get("class_id"),
        filters.get("state"),
        filters.get("search", ""),
        offset,
        limit,
        filters.get("edit_state"),
    )


@pytest.mark.parametrize("removed", [0, 2, 5])
def test_deleted_sample_insertion_points_preserve_both_neighbors(tmp_path, service, removed):
    session = open_service(service, source_dataset(tmp_path / "data"))
    assert position(service, session, removed).position == removed
    saved = apply(service, session, kind="sample_delete", sample_id=removed)
    result = position(service, session, removed)
    assert result.model_dump() == {
        "position": None,
        "insertion": removed,
        "total": 5,
        "revision": saved.revision,
    }
    remaining = [sample.id for sample in page(service, session).items]
    assert remaining == [sample_id for sample_id in range(6) if sample_id != removed]
    if removed < 5:
        assert page(service, session, offset=result.insertion, limit=1).items[0].id == removed + 1
    if removed > 0:
        assert (
            page(service, session, offset=result.insertion - 1, limit=1).items[0].id == removed - 1
        )
    saved = apply(service, session, kind="sample_delete", sample_id=removed, deleted=False)
    restored = position(service, session, removed)
    assert restored.position == restored.insertion == removed
    assert restored.total == 6 and restored.revision == saved.revision


def test_class_filter_removal_does_not_skip_next_remaining_sample_across_pages(tmp_path, service):
    session = open_service(service, source_dataset(tmp_path / "data", count=53))
    # The inspector is at the last gallery card, with another gallery page after it.
    current = service.detail(session, 47).annotations[0]
    saved = apply(
        service,
        session,
        kind="annotation_upsert",
        sample_id=47,
        annotation_id=current.uid,
        class_id=1,
        box=current.box.model_dump(),
    )
    result = position(service, session, 47, class_id=0)
    assert result.position is None and result.insertion == 47 and result.total == 52
    assert result.revision == saved.revision
    assert page(service, session, offset=result.insertion, limit=1, class_id=0).items[0].id == 48
    assert (
        page(service, session, offset=result.insertion - 1, limit=1, class_id=0).items[0].id == 46
    )
    other = position(service, session, 47, class_id=1)
    assert other.position == other.insertion == 0 and other.total == 1


def test_state_filter_and_empty_result_after_deleting_last_object(tmp_path, service):
    session = open_service(service, source_dataset(tmp_path / "data", count=1))
    current = service.detail(session, 0).annotations[0]
    saved = apply(
        service, session, kind="annotation_delete", sample_id=0, annotation_id=current.uid
    )
    result = position(service, session, 0, state="valid")
    assert result.model_dump() == {
        "position": None,
        "insertion": 0,
        "total": 0,
        "revision": saved.revision,
    }
    assert page(service, session, state="valid").items == []
    empty = position(service, session, 0, state="empty")
    assert empty.position == 0 and empty.total == 1
    service.history(session, "undo", saved.revision)
    assert position(service, session, 0, state="valid").position == 0
    assert position(service, session, 0, state="empty").position is None


def test_modified_and_deleted_filters_update_after_undo_and_restore(tmp_path, service):
    session = open_service(service, source_dataset(tmp_path / "data"))
    first = apply(service, session, kind="sample_delete", sample_id=1)
    second = apply(service, session, kind="sample_delete", sample_id=4)
    assert second.revision > first.revision
    result = position(service, session, 4, edit_state="deleted")
    assert result.position == 1 and result.total == 2
    saved = apply(service, session, kind="sample_delete", sample_id=1, deleted=False)
    removed_from_deleted = position(service, session, 1, edit_state="deleted")
    assert removed_from_deleted.position is None and removed_from_deleted.insertion == 0
    assert removed_from_deleted.total == 1
    assert removed_from_deleted.revision == saved.revision
    assert position(service, session, 1, edit_state="modified").position == 0
    service.history(session, "undo", saved.revision)
    assert position(service, session, 1, edit_state="modified").total == 0
    assert position(service, session, 1, edit_state="deleted").position == 0


def test_shared_split_memberships_keep_distinct_positions_and_shared_deletion(tmp_path, service):
    session = open_service(service, source_dataset(tmp_path / "data", count=3, shared_split=True))
    # Sample IDs are global discovery order, but positions are local to the chosen split.
    assert position(service, session, 4, split="val").position == 1
    assert position(service, session, 1, split="val").position is None
    assert position(service, session, 1, split="val").insertion == 0
    saved = apply(service, session, kind="sample_delete", sample_id=1)
    for sample_id, split in ((1, "train"), (4, "val")):
        result = position(service, session, sample_id, split=split)
        assert result.position is None and result.insertion == 1 and result.total == 2
        assert result.revision == saved.revision
    assert position(service, session, 4, split="val", edit_state="deleted").position == 0


def test_search_insertion_and_external_writer_revision_match_gallery(tmp_path, service):
    root = source_dataset(tmp_path / "data")
    session = open_service(service, root)
    assert position(service, session, 2, search="样本_004").insertion == 0
    assert position(service, session, 4, search="样本_004").position == 0
    assert position(service, session, 5, search="样本_004").insertion == 1
    request = OpenRequest(path=str(root))
    external = DraftEditor(build_index(request, Event(), lambda *_: None), request)
    saved = external.apply(
        EditRequest.model_validate(
            {
                "expected_revision": 0,
                "operation": {"kind": "sample_delete", "sample_id": 4},
            }
        )
    )
    result = position(service, session, 4, search="样本_004")
    gallery = page(service, session, search="样本_004")
    assert result.position is None and result.total == gallery.total == 0
    assert result.revision == gallery.revision == saved.revision
    external.close()


def test_position_http_errors_filters_and_stale_session(tmp_path):
    root = source_dataset(tmp_path / "data", count=3)
    with TestClient(create_app()) as client:
        opened = client.post("/api/datasets", json={"path": str(root)}).json()
        base = "/api/datasets/" + opened["id"]
        for _ in range(500):
            status = client.get(base).json()
            if status["state"] != "indexing":
                break
            time.sleep(0.005)
        assert status["state"] == "ready"
        assert client.get(base + "/samples/1/position?class_id=0").json() == {
            "position": 1,
            "insertion": 1,
            "total": 3,
            "revision": 0,
        }
        assert client.get(base + "/samples/1/position?search=absent").json() == {
            "position": None,
            "insertion": 0,
            "total": 0,
            "revision": 0,
        }
        assert client.get(base + "/samples/-1/position").status_code == 404
        assert client.get(base + "/samples/3/position").status_code == 404
        assert client.get(base + "/samples/noninteger/position").status_code == 422
        assert client.get(base + "/samples/1/position?state=wrong").status_code == 422
        assert client.get(base + "/samples/1/position?edit_state=wrong").status_code == 422
        changed = client.post(
            base + "/draft/operations",
            json={
                "expected_revision": 0,
                "operation": {"kind": "sample_delete", "sample_id": 1},
            },
        )
        assert changed.status_code == 200
        assert client.get(base + "/samples/1/position").json() == {
            "position": None,
            "insertion": 1,
            "total": 2,
            "revision": 1,
        }
        replacement = client.post("/api/datasets", json={"path": str(root)}).json()
        assert replacement["id"] != opened["id"]
        assert client.get(base + "/samples/1/position").status_code == 409
