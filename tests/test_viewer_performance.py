"""Loading optimizations preserve exact draft snapshots and invalidate changed images."""

import time
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from yolo_dataset_viewer.app import create_app
from yolo_dataset_viewer.dataset import build_index
from yolo_dataset_viewer.edit_models import EditRequest
from yolo_dataset_viewer.editing import DraftConflict, DraftEditor
from yolo_dataset_viewer.models import OpenRequest
from yolo_dataset_viewer.service import SessionError, ViewerService


def create_dataset(root, count=3):
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    for number in range(count):
        Image.new("RGB", (240, 120), (number, 100, 120)).save(root / f"images/{number:04d}.jpg")
        (root / f"labels/{number:04d}.txt").write_text("0 .5 .5 .4 .4\n")
    return root


def opened(service, root):
    status = service.open(OpenRequest(path=str(root)))
    for _ in range(500):
        result = service.status(status.id)
        if result.state != "indexing":
            assert result.state == "ready", result
            return result.id
        time.sleep(0.005)
    pytest.fail("index timeout")


def operation(revision, **kwargs):
    return EditRequest.model_validate({"expected_revision": revision, "operation": kwargs})


@pytest.fixture
def service():
    instance = ViewerService()
    try:
        yield instance
    finally:
        instance.close()


def test_view_is_one_selection_and_coherent_after_external_writer(tmp_path, service, monkeypatch):
    root = create_dataset(tmp_path / "data")
    session = opened(service, root)
    editor = service.editor(session)
    original = editor.selected
    calls = []

    def counted(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(editor, "selected", counted)
    first = service.view(session, None, None, None, "", 0, 2)
    next_page = service.view(session, None, None, None, "", 2, 2)
    assert first.page.total == 3 and len(next_page.page.items) == 1
    assert len(calls) == 1
    request = OpenRequest(path=str(root))
    other = DraftEditor(build_index(request, Event(), lambda *_: None), request)
    other.apply(operation(0, kind="sample_delete", sample_id=0, deleted=True))
    changed = service.view(session, None, None, None, "", 0, 2)
    assert changed.page.revision == changed.stats.revision == changed.draft.revision == 1
    assert changed.page.total == changed.stats.images == 2
    assert changed.draft.deleted_images == 1
    assert len(calls) == 2


def test_repeated_reads_skip_sqlite_payload_reparse_and_detail_global_scan(
    tmp_path, service, monkeypatch
):
    from yolo_dataset_viewer import editing

    root = create_dataset(tmp_path / "data")
    session = opened(service, root)
    service.edit(session, operation(0, kind="class_rename", class_id=0, name="renamed"))
    editor = service.editor(session)
    service.view(session, None, None, None, "", 0, 2)

    def forbidden(*args, **kwargs):
        pytest.fail("reparsed whole draft or scanned all sources while opening one image")

    monkeypatch.setattr(editing.Baseline, "model_validate_json", forbidden)
    monkeypatch.setattr(editor, "check_sources", forbidden)
    assert service.detail(session, 0).revision == 1
    assert service.detail(session, 1).revision == 1
    assert service.view(session, None, None, None, "", 0, 2).draft.revision == 1


def test_detail_then_original_share_one_decode_and_stamp_invalidates(
    tmp_path, service, monkeypatch
):
    root = create_dataset(tmp_path / "data")
    session = opened(service, root)
    original = Image.open
    calls = []

    def counted(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(Image, "open", counted)
    detail = service.detail(session, 0)
    assert detail.width == 240 and detail.height == 120
    payload = service.image(session, 0, False, detail.image_version)
    assert payload and len(calls) == 1
    assert service.detail(session, 0).image_version == detail.image_version
    assert len(calls) == 1
    Image.new("RGB", (400, 200), "blue").save(root / "images/0000.jpg")
    with pytest.raises(SessionError, match="版本"):
        service.image(session, 0, False, detail.image_version)
    updated = service.detail(session, 0)
    assert updated.width == 400 and updated.image_version != detail.image_version
    assert len(calls) == 2


def test_parallel_prefetch_and_detail_coalesce_decode(tmp_path, service, monkeypatch):
    root = create_dataset(tmp_path / "data")
    session = opened(service, root)
    original = Image.open
    started, release = Event(), Event()
    calls = []

    def slow_image(*args, **kwargs):
        calls.append(True)
        started.set()
        assert release.wait(timeout=5)
        return original(*args, **kwargs)

    monkeypatch.setattr(Image, "open", slow_image)
    with ThreadPoolExecutor() as pool:
        image = pool.submit(service.image, session, 0, False)
        assert started.wait(timeout=5)
        detail = pool.submit(service.detail, session, 0)
        release.set()
        assert image.result(timeout=5)
        assert detail.result(timeout=5).width == 240
    assert len(calls) == 1


def test_combined_image_cache_is_byte_bounded_and_switch_clears_metadata(tmp_path):
    service = ViewerService(cache_bytes=4000)
    try:
        session = opened(service, create_dataset(tmp_path / "data", 8))
        for sample_id in range(8):
            service.image(session, sample_id, False)
            service.image(session, sample_id, True)
        assert service._cache_size == sum(map(len, service._cache.values())) <= 4000
        assert service._image_meta
        replacement = opened(service, create_dataset(tmp_path / "other", 1))
        assert replacement != session
        assert not service._cache and not service._image_meta and service._cache_size == 0
    finally:
        service.close()


def test_view_conflict_ttl_never_weakens_write_source_check(tmp_path, service):
    root = create_dataset(tmp_path / "data")
    session = opened(service, root)
    service.view(session, None, None, None, "", 0, 2)
    (root / "labels/0000.txt").write_text("0 .5 .5 .2 .3\n")
    with pytest.raises(DraftConflict, match="源文件"):
        service.edit(session, operation(0, kind="class_rename", class_id=0, name="blocked"))
    assert service.draft(session).conflicts
    assert service.editor(session).revision == 0


def test_http_view_and_versioned_image_contract(tmp_path):
    root = create_dataset(tmp_path / "data")
    with TestClient(create_app()) as client:
        result = client.post("/api/datasets", json={"path": str(root)}).json()
        base = "/api/datasets/" + result["id"]
        for _ in range(500):
            status = client.get(base).json()
            if status["state"] != "indexing":
                break
            time.sleep(0.005)
        response = client.get(base + "/view?limit=2&offset=1")
        assert response.status_code == 200
        view = response.json()
        assert view["page"]["revision"] == view["stats"]["revision"] == view["draft"]["revision"]
        assert len(view["page"]["items"]) == 2
        version = view["page"]["items"][0]["image_version"]
        assert version
        assert client.get(base + f"/samples/1/image?version={version}").status_code == 200
        assert client.get(base + "/samples/1/image?version=wrong").status_code == 409


def test_cached_view_refreshes_only_visible_image_versions(tmp_path, service, monkeypatch):
    from yolo_dataset_viewer import service as module

    root = create_dataset(tmp_path / "data", 8)
    session = opened(service, root)
    first = service.view(session, None, None, None, "", 0, 2)
    Image.new("RGB", (300, 140), "orange").save(root / "images/0000.jpg")
    original = module.file_stamp
    checked = []

    def count_stamps(path):
        checked.append(path)
        return original(path)

    monkeypatch.setattr(module, "file_stamp", count_stamps)
    changed = service.view(session, None, None, None, "", 0, 2)
    assert len(checked) == 2
    assert first.page.items[0].image_version != changed.page.items[0].image_version
    assert first.page.items[1].image_version == changed.page.items[1].image_version
    assert changed.page.revision == changed.stats.revision == 0


@pytest.mark.parametrize("existing", [False, True])
def test_failed_commit_does_not_leave_phantom_cached_edits(
    tmp_path, service, monkeypatch, existing
):
    import sqlite3

    from yolo_dataset_viewer import editing

    root = create_dataset(tmp_path / "data")
    session = opened(service, root)
    if existing:
        service.edit(session, operation(0, kind="class_rename", class_id=0, name="committed"))
    before = service.draft(session)
    original = sqlite3.connect
    rejected = False

    class FailedCommit(sqlite3.Connection):
        def execute(self, sql, parameters=()):
            nonlocal rejected
            if sql == "COMMIT" and not rejected:
                rejected = True
                raise sqlite3.OperationalError("injected first commit failure")
            return super().execute(sql, parameters)

    def connect(*args, **kwargs):
        return original(*args, **kwargs, factory=FailedCommit)

    monkeypatch.setattr(editing.sqlite3, "connect", connect)
    with pytest.raises(editing.DraftError, match="commit failure"):
        service.edit(
            session, operation(before.revision, kind="class_rename", class_id=0, name="phantom")
        )
    snapshot = service.view(session, None, None, None, "", 0, 2)
    assert snapshot.draft.revision == before.revision
    assert snapshot.draft.classes == before.classes
    assert snapshot.draft.dirty == before.dirty
    assert snapshot.stats.classes[0].name == before.classes[0].name
    saved = service.edit(
        session, operation(before.revision, kind="class_rename", class_id=0, name="retry")
    )
    assert saved.revision == before.revision + 1 and saved.classes[0].name == "retry"
