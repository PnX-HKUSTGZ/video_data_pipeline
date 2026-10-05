"""Exports are portable snapshots; source updates are backed up and recoverable."""

import hashlib
from pathlib import Path
from threading import Event

import pytest
import yaml
from PIL import Image

from yolo_dataset_viewer import exports
from yolo_dataset_viewer.dataset import Cancelled, DatasetError, build_index, label_path
from yolo_dataset_viewer.edit_models import EditRequest
from yolo_dataset_viewer.editing import DraftEditor
from yolo_dataset_viewer.exports import (
    Exporter,
    ExportRequest,
    discover_recoveries,
    finalize_backup,
    recover_backup,
    recovery_request,
)
from yolo_dataset_viewer.models import OpenRequest


def dataset(root, rows=None, task="detect", shape=None, **metadata):
    root.mkdir(parents=True)
    for name, row in (rows or {"a": "0 .3 .4 .2 .3\n"}).items():
        path = root / "images" / "train" / f"{name}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (80, 50), (20, 70, 120)).save(path)
        if row is not None:
            label = label_path(path)
            label.parent.mkdir(parents=True, exist_ok=True)
            label.write_text(row)
    config = {"path": ".", "train": "images/train", "names": ["first", "second"]}
    if shape:
        config["kpt_shape"] = shape
    config.update(metadata)
    entry = root / "dataset.yaml"
    entry.write_text(yaml.safe_dump(config))
    request = OpenRequest(path=str(entry), task=task)
    index = build_index(request, Event(), lambda *_: None)
    return index, DraftEditor(index, request), request


def operation(editor, **data):
    return editor.apply(
        EditRequest.model_validate(
            {
                "expected_revision": editor.status().revision,
                "operation": data,
            }
        )
    )


def run(index, editor, opened, destination=None, mode="new", progress=lambda *_: None):
    request = ExportRequest(
        expected_revision=editor.status().revision,
        mode=mode,
        destination=str(destination) if destination else None,
    )
    return Exporter(index, editor, opened).run(request, progress, Event())


def hashes(root):
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file() and ".yolo-viewer" not in str(path)
    }


def reopen(result):
    request = OpenRequest(path=result.config_path, task=result.task)
    index = build_index(request, Event(), lambda *_: None)
    return index, DraftEditor(index, request)


def test_new_snapshot_pose_padding_remap_and_portability(tmp_path):
    source, editor, opened = dataset(
        tmp_path / "源数据", {"中文/同名": "0 .3 .4 .2 .3\n", "other/同名": "1 .6 .5 .2 .2\n"}
    )
    before = hashes(source.spec.root)
    operation(
        editor, kind="keypoints_set", class_id=0, keypoints=[{"name": "中心"}, {"name": "tip"}]
    )
    operation(editor, kind="class_rename", class_id=1, name="renamed")
    target = next(sample for sample in source.samples if 0 in sample.counts)
    annotation = editor.labels(target).annotations[0]
    operation(
        editor,
        kind="annotation_upsert",
        sample_id=target.id,
        annotation_id=annotation.uid,
        class_id=annotation.class_id,
        box=annotation.box.model_dump(),
        keypoints=[{"x": 0.2, "y": 0.7, "visibility": 1}, {"x": 0.8, "y": 0.3, "visibility": 2}],
    )
    result = run(source, editor, opened, tmp_path / "output")
    assert hashes(source.spec.root) == before
    assert editor.status().dirty
    exported, restored = reopen(result)
    assert exported.spec.kpt_shape == (2, 3)
    assert exported.names == {0: "first", 1: "renamed"}
    assert [[point.name for point in cls.keypoints] for cls in restored.status().classes] == [
        ["中心", "tip"],
        [],
    ]
    assert not restored.path.exists()
    assert not exported.spec.config_data.get("download")
    for sample in exported.samples:
        objects = restored.labels(sample).annotations
        if objects[0].class_id == 0:
            assert [(p.x, p.y, p.visibility) for p in objects[0].keypoints] == [
                (0.2, 0.7, 1),
                (0.8, 0.3, 2),
            ]
        else:
            assert len(objects[0].keypoints) == 0
            assert sample.label.read_text().split()[-6:] == ["0"] * 6
    moved = tmp_path / "移走后"
    Path(result.output_path).rename(moved)
    source.spec.config_path.unlink()
    assert (
        len(
            build_index(OpenRequest(path=str(moved), task="pose"), Event(), lambda *_: None).samples
        )
        == 2
    )


def test_class_deletion_remaps_ids_and_keeps_empty_image(tmp_path):
    source, editor, opened = dataset(
        tmp_path / "data", {"a": "0 .3 .4 .2 .3", "b": "1 .6 .5 .2 .2"}
    )
    operation(editor, kind="class_delete", class_id=0)
    result = run(source, editor, opened, tmp_path / "export")
    exported, _ = reopen(result)
    assert exported.names == {0: "second"}
    assert len(exported.samples) == 2
    assert {sample.state for sample in exported.samples} == {"empty", "valid"}
    assert exported.stats(exported.samples).instances == 1


def test_two_dimensional_points_and_schema_migration(tmp_path):
    source, editor, opened = dataset(
        tmp_path / "pose",
        {"a": "0 .3 .4 .2 .3 .25 .75 -1 -1"},
        task="pose",
        shape=[2, 2],
        flip_idx=[1, 0],
    )
    first, second = editor.status().classes[0].keypoints
    operation(
        editor,
        kind="keypoints_set",
        class_id=0,
        keypoints=[
            {"id": second.id, "name": "missing"},
            {"id": first.id, "name": "kept"},
            {"name": "added"},
        ],
    )
    request = ExportRequest(
        expected_revision=editor.status().revision, mode="new", destination=str(tmp_path / "out")
    )
    exporter = Exporter(source, editor, opened)
    assert any("flip_idx" in warning for warning in exporter.preview(request).warnings)
    exported, restored = reopen(exporter.run(request, lambda *_: None, Event()))
    assert "flip_idx" not in exported.spec.config_data
    points = restored.labels(exported.samples[0]).annotations[0].keypoints
    assert [(point.x, point.y, point.visibility) for point in points] == [
        (0, 0, 0),
        (0.25, 0.75, 2),
        (0, 0, 0),
    ]


@pytest.mark.parametrize("orientation", range(1, 9))
def test_exif_all_orientations_keep_image_bytes_and_edited_coordinates(tmp_path, orientation):
    source, editor, opened = dataset(
        tmp_path / "data", task="pose", shape=[1, 3], rows={"a": "0 .3 .4 .2 .3 .2 .7 1"}
    )
    image = source.samples[0].image
    exif = Image.Exif()
    exif[274] = orientation
    Image.new("RGB", (80, 50), (100, 20, 5)).save(image, exif=exif)
    # Reindex after fixture setup, before any user edit.
    source = build_index(opened, Event(), lambda *_: None)
    editor = DraftEditor(source, opened)
    obj = editor.labels(source.samples[0]).annotations[0]
    operation(
        editor,
        kind="annotation_upsert",
        sample_id=0,
        annotation_id=obj.uid,
        class_id=0,
        box={"cx": 0.7, "cy": 0.6, "width": 0.4, "height": 0.2},
        keypoints=[{"x": 0.65, "y": 0.53, "visibility": 2}],
    )
    exported, restored = reopen(run(source, editor, opened, tmp_path / "out"))
    output = exported.samples[0]
    assert output.image.read_bytes() == image.read_bytes()
    actual = restored.labels(output).annotations[0]
    assert actual.box.cx == 0.7 and actual.box.height == 0.2
    assert actual.keypoints[0].x == 0.65 and actual.keypoints[0].y == 0.53


def test_shared_split_references_export_one_file_and_keep_memberships(tmp_path):
    source, editor, opened = dataset(tmp_path / "data", val="images/train")
    result = run(source, editor, opened, tmp_path / "out")
    exported, _ = reopen(result)
    assert result.images == 1
    assert len(exported.samples) == 2
    assert len({sample.image for sample in exported.samples}) == 1
    assert exported.stats(exported.samples).splits == {"train": 1, "val": 1}


def test_inplace_delete_backup_and_recover(tmp_path):
    source, editor, opened = dataset(
        tmp_path / "data", {"a": "0 .3 .4 .2 .3", "b": "1 .6 .5 .2 .2"}
    )
    before = hashes(source.spec.root)
    operation(editor, kind="sample_delete", sample_id=0)
    operation(editor, kind="class_rename", class_id=1, name="after")
    result = run(source, editor, opened, mode="inplace")
    assert not source.samples[0].image.exists()
    assert Path(result.backup_path, "draft.sqlite3").is_file()
    exported, _ = reopen(result)
    assert len(exported.samples) == 1
    assert exported.names[1] == "after"
    pending = discover_recoveries(source.spec.root)
    assert len(pending) == 1 and pending[0].state == "applied"
    assert pending[0].files >= 4
    assert recovery_request(source.spec.root, pending[0].path) == opened
    recover_backup(source.spec.root, pending[0].path)
    assert hashes(source.spec.root) == before
    assert not discover_recoveries(source.spec.root)
    restored = build_index(opened, Event(), lambda *_: None)
    assert DraftEditor(restored, opened).status().dirty


def test_successful_inplace_finalization_keeps_backup_and_rebases(tmp_path):
    source, editor, opened = dataset(tmp_path / "data")
    operation(editor, kind="class_rename", class_id=0, name="changed")
    result = run(source, editor, opened, mode="inplace")
    editor.clear_after_export()
    exported, restored = reopen(result)
    assert not restored.status().dirty
    assert exported.names[0] == "changed"
    finalize_backup(result.backup_path)
    assert not discover_recoveries(source.spec.root)
    assert Path(result.backup_path, "manifest.json").exists()


def test_inplace_failure_restores_every_original(tmp_path, monkeypatch):
    source, editor, opened = dataset(
        tmp_path / "data", {"a": "0 .3 .4 .2 .3", "b": "1 .6 .5 .2 .2"}
    )
    before = hashes(source.spec.root)
    operation(editor, kind="sample_delete", sample_id=0)
    operation(editor, kind="class_rename", class_id=1, name="changed")
    original = exports._atomic_write
    failed = False

    def fail_once(path, content):
        nonlocal failed
        if path == source.samples[1].label and not failed:
            failed = True
            raise OSError("simulated disk failure")
        original(path, content)

    monkeypatch.setattr(exports, "_atomic_write", fail_once)
    with pytest.raises(OSError, match="simulated"):
        run(source, editor, opened, mode="inplace")
    assert failed
    assert hashes(source.spec.root) == before
    assert editor.status().dirty
    assert not discover_recoveries(source.spec.root)


def test_cancelled_export_removes_staging_directory(tmp_path):
    source, editor, opened = dataset(tmp_path / "data")
    cancel = Event()
    request = ExportRequest(expected_revision=0, mode="new", destination=str(tmp_path / "out"))

    def cancel_after_copy(phase, *_):
        if phase == "复制图片":
            cancel.set()

    with pytest.raises(Cancelled):
        Exporter(source, editor, opened).run(request, cancel_after_copy, cancel)
    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.glob(".out.viewer-export-*"))


@pytest.mark.parametrize(
    "problem",
    [
        "invalid",
        "broken_image",
        "no_classes",
        "source_changed",
        "occupied",
        "inside",
        "stale_revision",
    ],
)
def test_preflight_blocks_unsafe_export_without_writes(tmp_path, problem):
    source, editor, opened = dataset(
        tmp_path / "data", {"a": "0 .3 .4 .2 .3\nbad"} if problem == "invalid" else None
    )
    output = tmp_path / "out"
    revision = 0
    if problem == "broken_image":
        source.samples[0].image.write_bytes(b"not an image")
    elif problem == "no_classes":
        operation(editor, kind="class_delete", class_id=0)
        operation(editor, kind="class_delete", class_id=1)
        revision = editor.status().revision
    elif problem == "source_changed":
        source.samples[0].label.write_text("0 .2 .3 .1 .1")
    elif problem == "occupied":
        output.mkdir()
        (output / "keep").write_text("untouched")
    elif problem == "inside":
        output = source.spec.root / "out"
    elif problem == "stale_revision":
        operation(editor, kind="class_rename", class_id=0, name="changed")
    before = hashes(source.spec.root)
    exporter = Exporter(source, editor, opened)
    request = ExportRequest(expected_revision=revision, mode="new", destination=str(output))
    assert exporter.preview(request).errors
    with pytest.raises(DatasetError):
        exporter.run(request, lambda *_: None, Event())
    assert hashes(source.spec.root) == before
    assert not list(tmp_path.glob("*.viewer-export-*"))


@pytest.mark.parametrize("external", ["image", "config", "manifest"])
def test_external_references_allow_new_export_only(tmp_path, external):
    source, editor, opened = dataset(tmp_path / "data")
    config = source.spec.config_path
    raw = yaml.safe_load(config.read_text())
    if external == "image":
        outside = tmp_path / "elsewhere.jpg"
        source.samples[0].image.rename(outside)
        source.samples[0].image.symlink_to(outside)
    elif external == "config":
        config.rename(tmp_path / "outside.yaml")
        config = tmp_path / "outside.yaml"
        raw["path"] = str(source.spec.root)
        config.write_text(yaml.safe_dump(raw))
    else:
        manifest = tmp_path / "images.txt"
        manifest.write_text(str(source.samples[0].image) + "\n")
        raw["train"] = str(manifest)
        config.write_text(yaml.safe_dump(raw))
    opened = OpenRequest(path=str(config))
    source = build_index(opened, Event(), lambda *_: None)
    editor = DraftEditor(source, opened)
    exporter = Exporter(source, editor, opened)
    assert exporter.preview(ExportRequest(expected_revision=0, mode="inplace")).errors
    result = run(source, editor, opened, tmp_path / "out")
    assert len(reopen(result)[0].samples) == 1


def test_inplace_rejects_manifest_directory_symlink_outside_root(tmp_path):
    source, editor, opened = dataset(tmp_path / "data")
    outside = tmp_path / "outside"
    outside.mkdir()
    (source.spec.root / ".yolo-viewer-splits").symlink_to(outside, target_is_directory=True)
    preview = Exporter(source, editor, opened).preview(
        ExportRequest(expected_revision=0, mode="inplace")
    )
    assert any("根目录外" in error for error in preview.errors)
    assert not list(outside.iterdir())


def test_deleted_invalid_sample_does_not_block_export(tmp_path):
    source, editor, opened = dataset(tmp_path / "data", {"a": "broken", "b": "1 .6 .5 .2 .2"})
    operation(editor, kind="sample_delete", sample_id=0)
    result = run(source, editor, opened, tmp_path / "out")
    assert len(reopen(result)[0].samples) == 1


def test_missing_labels_and_empty_dataset_export(tmp_path):
    source, editor, opened = dataset(tmp_path / "data", {"a": None})
    result = run(source, editor, opened, tmp_path / "missing")
    assert reopen(result)[0].samples[0].state == "empty"
    operation(editor, kind="sample_delete", sample_id=0)
    result = run(source, editor, opened, tmp_path / "empty")
    assert reopen(result)[0].samples == []


def test_recovery_refuses_external_changes_and_corrupted_backups(tmp_path):
    source, editor, opened = dataset(tmp_path / "data")
    operation(editor, kind="class_rename", class_id=0, name="changed")
    result = run(source, editor, opened, mode="inplace")
    config = source.spec.config_path
    generated = config.read_bytes()
    config.write_text("external edit")
    unchanged_label = source.samples[0].label.read_bytes()
    with pytest.raises(DatasetError, match="外部修改"):
        recover_backup(source.spec.root, result.backup_path)
    assert source.samples[0].label.read_bytes() == unchanged_label
    assert config.read_text() == "external edit"
    config.write_bytes(generated)
    backup = Path(result.backup_path)
    saved = next((backup / "files").iterdir())
    saved.write_bytes(b"corrupt")
    with pytest.raises(DatasetError, match="备份文件内容变化"):
        recover_backup(source.spec.root, backup)
    assert config.read_bytes() == generated


def test_recovery_after_rebase_restores_only_selected_draft_workspace(tmp_path):
    source, editor, opened = dataset(tmp_path / "data")
    alternate = source.spec.root / "other.yaml"
    other_images = source.spec.root / "other_images"
    other_images.mkdir()
    Image.new("RGB", (20, 20)).save(other_images / "a.png")
    (other_images / "a.txt").write_text("0 .5 .5 .2 .2")
    other_config = yaml.safe_load(source.spec.config_path.read_text())
    other_config["train"] = "other_images"
    alternate.write_text(yaml.safe_dump(other_config))
    other_request = OpenRequest(path=str(alternate))
    other_source = build_index(other_request, Event(), lambda *_: None)
    other = DraftEditor(other_source, other_request)
    operation(editor, kind="class_rename", class_id=0, name="selected")
    operation(other, kind="class_rename", class_id=1, name="other-before")
    result = run(source, editor, opened, mode="inplace")
    editor.clear_after_export()
    # Another configuration's independent source files and draft were not exported.
    operation(other, kind="class_rename", class_id=1, name="other-after")
    recover_backup(source.spec.root, result.backup_path)
    restored = DraftEditor(build_index(opened, Event(), lambda *_: None), opened)
    other_restored = DraftEditor(
        build_index(other_request, Event(), lambda *_: None), other_request
    )
    assert restored.status().classes[0].name == "selected"
    assert other_restored.status().classes[1].name == "other-after"


def test_unchanged_keypoint_layout_preserves_training_and_descriptive_metadata(tmp_path):
    source, editor, opened = dataset(
        tmp_path / "data",
        task="pose",
        shape=[2, 3],
        rows={"a": "0 .3 .4 .2 .3 .2 .7 1 .5 .6 2"},
        flip_idx=[1, 0],
        skeleton=[[0, 1]],
        channels=1,
        description="example",
        download="do not execute",
    )
    operation(editor, kind="class_rename", class_id=0, name="renamed")
    exported, _ = reopen(run(source, editor, opened, tmp_path / "out"))
    assert exported.spec.config_data["flip_idx"] == [1, 0]
    assert exported.spec.config_data["skeleton"] == [[0, 1]]
    assert exported.spec.config_data["channels"] == 1
    assert exported.spec.config_data["description"] == "example"
    assert "download" not in exported.spec.config_data


def test_export_blocks_insufficient_space_and_cleans_after_source_changes(tmp_path, monkeypatch):
    source, editor, opened = dataset(tmp_path / "data")
    request = ExportRequest(expected_revision=0, mode="new", destination=str(tmp_path / "out"))
    exporter = Exporter(source, editor, opened)
    with monkeypatch.context() as patch:
        patch.setattr(exports, "_free_space", lambda _: 0)
        assert any("磁盘空间" in error for error in exporter.preview(request).errors)

    def edit_source(phase, *_):
        if phase == "复制图片":
            source.samples[0].label.write_text("0 .2 .3 .2 .2")

    with pytest.raises(DatasetError, match="变化"):
        exporter.run(request, edit_source, Event())
    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.glob(".out.viewer-export-*"))


def test_cancelled_inplace_application_rolls_back_deleted_files(tmp_path):
    source, editor, opened = dataset(tmp_path / "data")
    before = hashes(source.spec.root)
    operation(editor, kind="sample_delete", sample_id=0)
    cancel = Event()

    def stop_after_mutation(phase, *_):
        if phase == "应用修改":
            cancel.set()

    with pytest.raises(Cancelled):
        Exporter(source, editor, opened).run(
            ExportRequest(expected_revision=editor.status().revision, mode="inplace"),
            stop_after_mutation,
            cancel,
        )
    assert hashes(source.spec.root) == before
    assert not discover_recoveries(source.spec.root)


def test_same_physical_image_different_labels_can_export_separate_samples(tmp_path):
    source, _, _ = dataset(tmp_path / "data")
    alias = source.spec.root / "images" / "alias" / "a.jpg"
    alias.parent.mkdir()
    alias.symlink_to(source.samples[0].image)
    alias_label = label_path(alias)
    alias_label.parent.mkdir(parents=True)
    alias_label.write_text("1 .6 .5 .2 .2")
    config = source.spec.config_path
    raw = yaml.safe_load(config.read_text())
    raw["val"] = "images/alias"
    config.write_text(yaml.safe_dump(raw))
    opened = OpenRequest(path=str(config))
    source = build_index(opened, Event(), lambda *_: None)
    editor = DraftEditor(source, opened)
    result = run(source, editor, opened, tmp_path / "out")
    exported, _ = reopen(result)
    assert result.images == 2
    assert {tuple(sample.counts) for sample in exported.samples} == {(0,), (1,)}


def test_inplace_preflight_rejects_parent_file_collision(tmp_path):
    source, editor, opened = dataset(tmp_path / "data")
    (source.spec.root / ".yolo-viewer-splits").write_text("unrelated existing file")
    request = ExportRequest(expected_revision=0, mode="inplace")
    exporter = Exporter(source, editor, opened)
    assert any("父目录" in message for message in exporter.preview(request).errors)
    before = hashes(source.spec.root)
    with pytest.raises(DatasetError):
        exporter.run(request, lambda *_: None, Event())
    assert hashes(source.spec.root) == before
    assert not list(tmp_path.glob(".data.viewer-backup-*"))


def test_recovery_discovery_handles_dataset_names_with_glob_characters(tmp_path):
    source, editor, opened = dataset(tmp_path / "data [v1]")
    operation(editor, kind="class_rename", class_id=0, name="changed")
    result = run(source, editor, opened, mode="inplace")
    pending = discover_recoveries(source.spec.root)
    assert len(pending) == 1 and pending[0].path == result.backup_path
    recover_backup(source.spec.root, result.backup_path)
