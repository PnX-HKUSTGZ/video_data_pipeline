"""Durable overlays are tested using standalone YOLO inputs and no pipeline metadata."""

import hashlib
import os
import sqlite3
from pathlib import Path
from threading import Event

import pytest
import yaml
from PIL import Image

from yolo_dataset_viewer.dataset import build_index
from yolo_dataset_viewer.edit_models import EditRequest
from yolo_dataset_viewer.editing import DraftConflict, DraftEditor, DraftError
from yolo_dataset_viewer.models import OpenRequest, Pose


def make_editor(root, rows=None, shape=None, names=None, extra=None, task=None):
    root.mkdir(exist_ok=True)
    for name, row in (rows or {"a": "0 .5 .5 .4 .4\n1 .3 .3 .2 .2"}).items():
        image = root / "images/train" / f"{name}.png"
        image.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (80, 60), (60, 140, 200)).save(image)
        label = root / "labels/train" / f"{name}.txt"
        label.parent.mkdir(parents=True, exist_ok=True)
        if row is not None:
            label.write_text(row)
    data = {"train": "images/train", "names": names or ["first", "second"]}
    if shape:
        data["kpt_shape"] = shape
    data.update(extra or {})
    config = root / "data.yaml"
    config.write_text(yaml.safe_dump(data))
    request = OpenRequest(path=str(config), task=task or ("pose" if shape else "detect"))
    dataset = build_index(request, Event(), lambda *args: None)
    return DraftEditor(dataset, request)


def reopen(editor):
    data = build_index(editor.request, Event(), lambda *args: None)
    return DraftEditor(data, editor.request)


def apply(editor, kind, **kwargs):
    return editor.apply(
        EditRequest.model_validate(
            {"expected_revision": editor.status().revision, "operation": {"kind": kind, **kwargs}}
        )
    )


def objects(editor, index=0):
    return editor.labels(editor.dataset.samples[index]).annotations


def box(cx=0.5, cy=0.5, width=0.3, height=0.3):
    return {"cx": cx, "cy": cy, "width": width, "height": height}


def source_hashes(root):
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file() and not path.name.startswith(".yolo-viewer")
    }


def test_lazy_durable_history_and_source_unchanged(tmp_path):
    editor = make_editor(tmp_path)
    before = source_hashes(tmp_path)
    assert not editor.status().dirty
    assert not editor.path.exists()
    apply(editor, "class_rename", class_id=0, name="renamed")
    assert editor.path.exists()
    assert not Path(str(editor.path) + "-journal").exists()
    target = objects(editor)[0]
    apply(editor, "annotation_upsert", sample_id=0, annotation_id=target.uid, class_id=0, box=box())
    editor = reopen(editor)
    assert objects(editor)[0].box.width == 0.3
    assert editor.status().classes[0].name == "renamed"
    revision = editor.status().revision
    editor.undo(revision)
    assert objects(editor)[0].box.width == 0.4
    editor.undo(editor.status().revision)
    assert not editor.status().dirty
    assert editor.status().can_redo
    editor = reopen(editor)
    editor.redo(editor.status().revision)
    assert editor.status().classes[0].name == "renamed"
    assert source_hashes(tmp_path) == before


def test_stable_ids_and_new_class_never_reuses_deleted_id(tmp_path):
    editor = make_editor(tmp_path)
    initial = objects(editor)[0].uid
    apply(editor, "class_add", name="new")
    old_id = editor.status().classes[-1].id
    apply(editor, "class_delete", class_id=old_id)
    apply(editor, "class_add", name="other")
    assert editor.status().classes[-1].id > old_id
    assert objects(reopen(editor))[0].uid == initial
    with pytest.raises(DraftError, match="重复"):
        apply(editor, "class_add", name="first")


def test_class_delete_filters_targets_and_keeps_images(tmp_path):
    editor = make_editor(tmp_path, {"a": "0 .5 .5 .2 .2", "b": "1 .5 .5 .2 .2"})
    apply(editor, "class_delete", class_id=0)
    selected = editor.selected(None, None, None, "")
    assert len(selected) == 2
    assert selected[0].state == "empty"
    stats = editor.stats(selected)
    assert stats.instances == 1
    assert stats.classes[0].id == 1
    editor.undo(editor.status().revision)
    assert len(objects(editor)) == 1


def test_keypoints_migration_preserves_identity_and_box_does_not_move_points(tmp_path):
    editor = make_editor(tmp_path, {"a": "0 .5 .5 .4 .4 .2 .3 1 .6 .7 2"}, [2, 3])
    status = editor.status()
    left, right = status.classes[0].keypoints
    apply(
        editor,
        "keypoints_set",
        class_id=0,
        keypoints=[
            {"id": right.id, "name": "renamed"},
            {"id": left.id, "name": "left"},
            {"name": "new"},
        ],
    )
    item = objects(editor)[0]
    assert isinstance(item, Pose)
    assert [(point.x, point.y, point.visibility) for point in item.keypoints] == [
        (0.6, 0.7, 2),
        (0.2, 0.3, 1),
        (0, 0, 0),
    ]
    apply(
        editor,
        "annotation_upsert",
        sample_id=0,
        annotation_id=item.uid,
        class_id=0,
        box=box(0.4, 0.4),
    )
    assert objects(editor)[0].keypoints == item.keypoints
    points = editor.status().classes[0].keypoints
    apply(editor, "keypoints_set", class_id=0, keypoints=[{"id": points[1].id, "name": "left"}])
    assert len(objects(editor)[0].keypoints) == 1
    assert objects(editor)[0].keypoints[0].x == 0.2
    editor.undo(editor.status().revision)
    assert len(objects(editor)[0].keypoints) == 3


def test_detection_to_pose_and_target_class_name_mapping(tmp_path):
    editor = make_editor(tmp_path)
    apply(
        editor, "keypoints_set", class_id=0, keypoints=[{"name": "shared"}, {"name": "first-only"}]
    )
    apply(editor, "keypoints_set", class_id=1, keypoints=[{"name": "other"}, {"name": "shared"}])
    item = objects(editor)[0]
    assert all(point.visibility == 0 for point in item.keypoints)
    apply(
        editor,
        "annotation_upsert",
        sample_id=0,
        annotation_id=item.uid,
        class_id=0,
        box=box(),
        keypoints=[{"x": 0.2, "y": 0.3, "visibility": 1}, {"x": 0.7, "y": 0.6, "visibility": 2}],
    )
    apply(editor, "annotation_upsert", sample_id=0, annotation_id=item.uid, class_id=1, box=box())
    updated = objects(editor)[0]
    assert [(point.x, point.visibility) for point in updated.keypoints] == [(0, 0), (0.2, 1)]
    assert editor.status().task == "pose"


def test_import_variable_keypoint_names_and_2d_missing_sentinel(tmp_path):
    editor = make_editor(
        tmp_path,
        {"a": "0 .5 .5 .4 .4 -1 -1 .6 .7\n1 .5 .5 .4 .4 .2 .3 .4 .5"},
        [2, 2],
        extra={"kpt_names": {0: ["tip", "end"], 1: []}},
    )
    assert [point.name for point in editor.status().classes[0].keypoints] == ["tip", "end"]
    assert objects(editor)[0].keypoints[0].visibility == 0
    assert objects(editor)[0].keypoints[1].visibility == 2
    assert objects(editor)[1].keypoints == []


def test_new_targets_issue_correction_and_invalid_rows_never_disappear(tmp_path):
    editor = make_editor(tmp_path, {"a": "0 .5 .5 .4 .4\nbad row\n0 nan .2 .3 .4"})
    assert editor.labels(editor.dataset.samples[0]).issues[0].raw == "bad row"
    apply(editor, "annotation_upsert", sample_id=0, class_id=1, box=box(), issue_line=2)
    labels = editor.labels(editor.dataset.samples[0])
    assert len(labels.annotations) == 2
    assert [issue.line for issue in labels.issues] == [3]
    apply(editor, "issue_delete", sample_id=0, line=3)
    assert editor.info(editor.dataset.samples[0]).state == "valid"
    apply(editor, "annotation_delete", sample_id=0, annotation_id=objects(editor)[0].uid)
    assert len(objects(editor)) == 1
    editor.undo(editor.status().revision)
    assert len(objects(editor)) == 2


def test_sample_delete_duplicate_split_membership_and_statistics(tmp_path):
    editor = make_editor(
        tmp_path, {"a": "0 .5 .5 .4 .4\n0 .3 .3 .2 .2"}, extra={"val": "images/train"}
    )
    selected = editor.selected(None, None, None, "")
    assert len(selected) == 2
    stats = editor.stats(selected)
    assert stats.unique_images == 1 and stats.images == 2 and stats.instances == 4
    assert stats.classes[0].images == 2
    apply(editor, "sample_delete", sample_id=0)
    assert not editor.selected(None, None, None, "")
    deleted = editor.selected(None, None, None, "", "deleted")
    assert len(deleted) == 2
    assert editor.status().deleted_images == 1
    apply(editor, "sample_delete", sample_id=1, deleted=False)
    assert len(editor.selected(None, 0, "valid", "a")) == 2


def test_revision_cas_and_source_conflicts_keep_draft(tmp_path):
    editor = make_editor(tmp_path)
    other = reopen(editor)
    apply(editor, "class_rename", class_id=0, name="new")
    with pytest.raises(DraftConflict, match="版本"):
        other.apply(
            EditRequest.model_validate(
                {"expected_revision": 0, "operation": {"kind": "class_add", "name": "third"}}
            )
        )
    label = editor.dataset.samples[0].label
    original_stat = label.stat()
    label.write_text(label.read_text().replace(".5", ".4", 1))
    os.utime(label, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    assert editor.check_sources()
    assert reopen(editor).status().conflicts
    with pytest.raises(DraftConflict, match="源文件"):
        apply(editor, "class_add", name="third")
    assert editor.status().classes[0].name == "new"


def test_selected_configuration_has_independent_draft(tmp_path):
    editor = make_editor(tmp_path)
    second = tmp_path / "other.yaml"
    second.write_text((tmp_path / "data.yaml").read_text())
    request = OpenRequest(path=str(second))
    other = DraftEditor(build_index(request, Event(), lambda *args: None), request)
    apply(editor, "class_rename", class_id=0, name="edited")
    assert other.status().classes[0].name == "first"
    apply(other, "class_rename", class_id=0, name="separate")
    editor.clear_after_export()
    assert other.status().classes[0].name == "separate"


@pytest.mark.parametrize(
    "invalid_box", [box(width=0), box(cx=0.99), box(width=float("nan")), box(height=-1)]
)
def test_invalid_geometry_is_not_saved(tmp_path, invalid_box):
    editor = make_editor(tmp_path)
    with pytest.raises((DraftError, ValueError)):
        apply(editor, "annotation_upsert", sample_id=0, class_id=0, box=invalid_box)
    assert editor.status().revision == 0
    assert not editor.status().dirty


def test_sqlite_failure_rolls_back_and_retry_succeeds(tmp_path):
    editor = make_editor(tmp_path)
    apply(editor, "class_rename", class_id=0, name="new")
    with sqlite3.connect(editor.path) as connection:
        connection.execute(
            "CREATE TRIGGER reject_history BEFORE INSERT ON history BEGIN SELECT RAISE(ABORT, 'test disk failure'); END"
        )
    with pytest.raises(DraftError, match="test disk failure"):
        apply(editor, "class_add", name="third")
    assert editor.status().revision == 1
    assert len(editor.status().classes) == 2
    with sqlite3.connect(editor.path) as connection:
        connection.execute("DROP TRIGGER reject_history")
    apply(editor, "class_add", name="third")
    assert editor.status().revision == 2


def test_schema_changes_do_not_read_all_label_geometry(tmp_path, monkeypatch):
    editor = make_editor(tmp_path, {str(i): "0 .5 .5 .4 .4" for i in range(100)})
    import yolo_dataset_viewer.editing as module

    monkeypatch.setattr(module, "read_labels", lambda *args: pytest.fail("eager geometry read"))
    apply(editor, "class_rename", class_id=0, name="renamed")
    apply(editor, "keypoints_set", class_id=0, keypoints=[{"name": "tip"}])
    selected = editor.selected(None, 0, None, "", "modified")
    assert len(selected) == 100
    assert editor.stats(selected).instances == 100


def test_class_id_is_not_reused_after_undo_new_branch(tmp_path):
    editor = make_editor(tmp_path)
    added = apply(editor, "class_add", name="third").classes[-1].id
    editor.undo(editor.status().revision)
    replacement = apply(editor, "class_add", name="another").classes[-1].id
    assert replacement > added
    assert not editor.status().can_redo


@pytest.mark.parametrize(
    "point",
    [
        {"x": float("nan"), "y": 0, "visibility": 0},
        {"x": -1, "y": -1, "visibility": 0},
        {"x": 1.1, "y": 0.5, "visibility": 2},
    ],
)
def test_invalid_keypoints_are_not_saved_even_when_missing(tmp_path, point):
    editor = make_editor(tmp_path)
    apply(editor, "keypoints_set", class_id=0, keypoints=[{"name": "tip"}])
    with pytest.raises((DraftError, ValueError)):
        apply(editor, "annotation_upsert", sample_id=0, class_id=0, box=box(), keypoints=[point])
    assert editor.status().revision == 1


def test_discard_fresh_index_rebases_external_changes_and_keeps_revision(tmp_path):
    editor = make_editor(tmp_path)
    apply(editor, "class_rename", class_id=0, name="draft-name")
    apply(editor, "sample_delete", sample_id=0)
    label = editor.dataset.samples[0].label
    label.write_text("0 .4 .4 .2 .2")
    new_image = tmp_path / "images/train/new.png"
    Image.new("RGB", (80, 60)).save(new_image)
    (tmp_path / "labels/train/new.txt").write_text("0 .5 .5 .4 .4")
    config = yaml.safe_load((tmp_path / "data.yaml").read_text())
    config["names"] = ["source-name", "second"]
    (tmp_path / "data.yaml").write_text(yaml.safe_dump(config))
    assert editor.status().conflicts
    fresh = reopen(editor)
    status = fresh.discard(2)
    assert status.revision == 3
    assert status.classes[0].name == "source-name"
    assert not status.dirty and not status.conflicts
    assert not status.can_redo and not status.can_undo
    assert len(fresh.selected(None, None, None, "")) == 2
    assert objects(fresh)[0].box.cx == 0.4
    assert not reopen(fresh).status().conflicts


def test_recovery_restores_only_selected_workspace_and_rejects_new_edits(tmp_path):
    from yolo_dataset_viewer.editing import restore_workspace_from_backup

    editor = make_editor(tmp_path)
    other_config = tmp_path / "other.yaml"
    other_config.write_text((tmp_path / "data.yaml").read_text())
    request = OpenRequest(path=str(other_config))
    other = DraftEditor(build_index(request, Event(), lambda *args: None), request)
    apply(editor, "class_rename", class_id=0, name="archived")
    backup = tmp_path / "backup.sqlite3"
    with sqlite3.connect(editor.path) as connection, sqlite3.connect(backup) as archived:
        connection.backup(archived)
    editor.clear_after_export()
    apply(other, "class_rename", class_id=0, name="independent")
    restore_workspace_from_backup(editor.path, backup, editor.workspace, check_only=True)
    assert editor.status().classes[0].name == "first"
    restore_workspace_from_backup(editor.path, backup, editor.workspace)
    assert editor.status().classes[0].name == "archived"
    assert other.status().classes[0].name == "independent"
    apply(editor, "class_rename", class_id=0, name="new edits")
    with pytest.raises(DraftConflict, match="导出后已修改"):
        restore_workspace_from_backup(editor.path, backup, editor.workspace, check_only=True)
    with pytest.raises(DraftConflict, match="导出后已修改"):
        restore_workspace_from_backup(editor.path, backup, editor.workspace)
    assert editor.status().classes[0].name == "new edits"


def test_ten_thousand_edits_use_lazy_geometry_and_small_sample_overrides(tmp_path, monkeypatch):
    from yolo_dataset_viewer import editing as module

    images, labels = tmp_path / "images", tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    for number in range(10000):
        (images / f"{number:05d}.jpg").touch()
        (labels / f"{number:05d}.txt").write_text("0 .5 .5 .2 .2")
    monkeypatch.setattr(Image, "open", lambda *args, **kwargs: pytest.fail("eager image decode"))
    request = OpenRequest(path=str(tmp_path))
    dataset = build_index(request, Event(), lambda *args: None)
    editor = DraftEditor(dataset, request)
    assert not editor.path.exists()
    original = module.read_labels
    reads = []

    def counted(*args, **kwargs):
        reads.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "read_labels", counted)
    apply(editor, "keypoints_set", class_id=0, keypoints=[{"name": "tip"}])
    assert editor.stats(editor.selected(None, None, None, "")).instances == 10000
    assert reads == []
    apply(editor, "annotation_upsert", sample_id=42, class_id=0, box=box())
    assert len(reads) == 1
    assert editor.stats(editor.selected(None, None, None, "")).instances == 10001
    with sqlite3.connect(editor.path) as connection:
        count, size = connection.execute(
            "SELECT COUNT(*),SUM(length(data)) FROM samples"
        ).fetchone()
    assert count == 1 and size < 10000


def test_dataset_lock_is_reentrant_nonblocking_and_released_on_error(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    from yolo_dataset_viewer.editing import dataset_lock

    editor = make_editor(tmp_path)

    def competing_write():
        with pytest.raises(DraftConflict, match="正在导出"):
            apply(editor, "class_add", name="blocked")

    with pytest.raises(RuntimeError), dataset_lock(tmp_path, exclusive=True):
        with dataset_lock(tmp_path, exclusive=True), dataset_lock(tmp_path):
            apply(editor, "class_rename", class_id=0, name="allowed")
        with ThreadPoolExecutor() as pool:
            pool.submit(competing_write).result(timeout=5)
        raise RuntimeError("export failed")
    apply(editor, "class_add", name="after export")
    assert len(editor.status().classes) == 3
    with (
        dataset_lock(tmp_path),
        pytest.raises(DraftConflict, match="事务内"),
        dataset_lock(tmp_path, exclusive=True),
    ):
        pytest.fail("lock upgraded")


def test_dataset_export_lock_blocks_an_independent_process(tmp_path):
    import subprocess
    import sys

    from yolo_dataset_viewer.editing import dataset_lock

    script = "from pathlib import Path; from yolo_dataset_viewer.editing import dataset_lock; import sys\nwith dataset_lock(Path(sys.argv[1])): pass"
    with dataset_lock(tmp_path, exclusive=True):
        blocked = subprocess.run(
            [sys.executable, "-c", script, str(tmp_path)],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        assert blocked.returncode != 0 and "正在导出" in blocked.stderr
    success = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert success.returncode == 0
    assert list(tmp_path.iterdir()) == []


def test_write_acknowledges_its_exact_revision_when_another_editor_commits(tmp_path, monkeypatch):
    editor = make_editor(tmp_path)
    other = reopen(editor)
    original_status = editor.status
    raced = False

    def interleaved_status(*, refresh=True):
        nonlocal raced
        if not raced and editor.revision == 1:
            raced = True
            apply(other, "class_rename", class_id=1, name="concurrent")
        return original_status(refresh=refresh)

    monkeypatch.setattr(editor, "status", interleaved_status)
    acknowledged = apply(editor, "class_rename", class_id=0, name="first-write")
    assert raced
    assert acknowledged.revision == 1
    assert acknowledged.classes[1].name == "second"
    assert editor.status().revision == 2
    assert editor.status().classes[1].name == "concurrent"
    with pytest.raises(DraftConflict, match="版本"):
        editor.apply(
            EditRequest.model_validate(
                {
                    "expected_revision": acknowledged.revision,
                    "operation": {"kind": "class_add", "name": "stale followup"},
                }
            )
        )


def test_metadata_read_is_one_sqlite_snapshot_during_concurrent_write(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    editor = make_editor(tmp_path)
    apply(editor, "class_rename", class_id=0, name="first-write")
    other = reopen(editor)
    writing = Event()
    original_sqlite = other._sqlite_connection
    from contextlib import contextmanager

    @contextmanager
    def traced_connection(*, write=False):
        with original_sqlite(write=write) as connection:
            if write:
                connection.set_trace_callback(
                    lambda sql: writing.set() if sql.startswith("COMMIT") else None
                )
            yield connection

    other._sqlite_connection = traced_connection
    with ThreadPoolExecutor() as pool:
        with editor._connection() as connection:
            revision = connection.execute(
                "SELECT revision FROM workspaces WHERE id=?", (editor.workspace,)
            ).fetchone()[0]
            future = pool.submit(
                apply, other, "annotation_upsert", sample_id=0, class_id=0, box=box()
            )
            assert writing.wait(timeout=5)
            # A pending concurrent COMMIT cannot introduce newer overrides into this snapshot.
            overrides = connection.execute(
                "SELECT COUNT(*) FROM samples WHERE workspace=?", (editor.workspace,)
            ).fetchone()[0]
            assert revision == 1 and overrides == 0
        assert future.result(timeout=5).revision == 2
    assert editor.status().revision == 2
    assert editor.stats(editor.selected(None, None, None, "")).instances == 3
