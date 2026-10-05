"""Exercise real, visible editor gestures and exports against a live CPU-only server."""

import hashlib
from pathlib import Path

import pytest
import yaml

pytest.importorskip("fastapi")
pytest.importorskip("playwright")
from playwright.sync_api import expect
from test_viewer_browser import (  # noqa: F401
    browser,
    browser_dataset,
    open_from_page,
    web_server,
)
from test_viewer_browser import (
    page as viewer_page,
)

page = viewer_page


def revision(page):
    return page.evaluate("editor.draft.revision")


def saved(page, previous):
    page.wait_for_function(
        "previous => editor.draft.revision > previous && !editor.saving", arg=previous
    )
    assert page.evaluate("editor.failure") is None
    expect(page.locator("#save-status")).to_contain_text("已保存")


def source_hashes(root):
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file() and not path.name.startswith(".yolo-viewer-draft.sqlite3")
    }


def canvas_point(page, x, y):
    """Translate normalized image coordinates without changing the application state."""
    return page.evaluate(
        "([x,y]) => {const r=canvas.getBoundingClientRect(); return {"
        "x:r.x+view.x+x*view.image.width*view.scale,"
        "y:r.y+view.y+y*view.image.height*view.scale};}",
        [x, y],
    )


def drag(page, start, end):
    a, b = canvas_point(page, *start), canvas_point(page, *end)
    page.mouse.move(a["x"], a["y"])
    page.mouse.down()
    page.mouse.move(b["x"], b["y"], steps=6)
    page.mouse.up()


def select_first(page):
    page.locator("#gallery .sample-card").first.click()
    expect(page.locator("#image-message")).to_have_text("")
    page.locator(".annotation-row").first.click()


def test_full_edit_restore_and_both_exports(page, tmp_path):
    root = browser_dataset(tmp_path / "editable", count=3)
    before = source_hashes(root)
    page.on("dialog", lambda dialog: dialog.accept())
    open_from_page(page, root)
    assert not (root / ".yolo-viewer-draft.sqlite3").exists()

    page.locator("#manage-classes").click()
    page.locator("#class-add-name").fill("临时类别")
    old = revision(page)
    page.locator("#class-add-form button").click()
    saved(page, old)
    temporary = page.locator('.class-editor[data-class-id="1"]')
    temporary.locator('input[aria-label="类别 1 名称"]').fill("新类别")
    old = revision(page)
    temporary.get_by_role("button", name="重命名", exact=True).click()
    saved(page, old)
    first = page.locator('.class-editor[data-class-id="0"]')
    first.get_by_role("button", name="增加关键点", exact=True).click()
    first.get_by_role("button", name="增加关键点", exact=True).click()
    first.locator(".keypoint-definition-row input").nth(0).fill("left")
    first.locator(".keypoint-definition-row input").nth(1).fill("right")
    old = revision(page)
    first.get_by_role("button", name="保存关键点列表", exact=True).click()
    saved(page, old)
    expect(page.locator("#task-badge")).to_have_text("Pose")
    page.locator("#close-classes").click()

    select_first(page)
    page.locator('button[data-tool="points"]').click()
    p = canvas_point(page, 0.35, 0.4)
    old = revision(page)
    page.mouse.click(p["x"], p["y"])
    saved(page, old)
    row = page.locator(".point-edit-row").nth(1)
    row.get_by_role("button", name="1 · right", exact=True).click()
    p = canvas_point(page, 0.62, 0.65)
    old = revision(page)
    page.mouse.click(p["x"], p["y"])
    saved(page, old)
    row = page.locator(".point-edit-row").nth(1)
    row.locator("select").select_option("1")
    old = revision(page)
    row.get_by_role("button", name="保存点", exact=True).click()
    saved(page, old)
    points_before = page.evaluate("state.detail.annotations[0].keypoints")
    page.locator('button[data-tool="edit"]').click()
    old = revision(page)
    drag(page, (0.5, 0.5), (0.55, 0.52))
    saved(page, old)
    assert page.evaluate("state.detail.annotations[0].keypoints") == points_before
    page.locator("#box-cx").fill("0.5")
    old = revision(page)
    page.locator("#object-save").click()
    saved(page, old)
    assert page.evaluate("state.detail.annotations[0].box.cx") == 0.5
    page.locator('button[data-tool="draw"]').click()
    page.locator("#new-object-class").select_option("1")
    old = revision(page)
    drag(page, (0.05, 0.08), (0.2, 0.3))
    saved(page, old)
    expect(page.locator("#annotation-count")).to_have_text("2")
    page.locator("#object-delete").click()
    page.wait_for_function("state.detail.annotations.length === 1 && !editor.saving")
    output = Path(".cache/viewer-checks")
    output.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(output / "editor-image.png"))
    page.locator("#close-detail").click()

    page.locator("#manage-classes").click()
    first = page.locator('.class-editor[data-class-id="0"]')
    first.get_by_role("button", name="上移关键点 1", exact=True).click()
    first.locator(".keypoint-definition-row input").nth(0).fill("renamed_right")
    old = revision(page)
    first.get_by_role("button", name="保存关键点列表", exact=True).click()
    saved(page, old)
    temporary = page.locator('.class-editor[data-class-id="1"]')
    old = revision(page)
    temporary.get_by_role("button", name="删除类别", exact=True).click()
    saved(page, old)
    page.screenshot(path=str(output / "editor-classes.png"))
    page.locator("#close-classes").click()
    select_first(page)
    points_after = page.evaluate("state.detail.annotations[0].keypoints")
    assert points_after == list(reversed(points_before))
    old = revision(page)
    page.locator("#image-delete").click()
    saved(page, old)
    expect(page.locator("#detail-meta")).to_contain_text("已在草稿中删除")
    page.locator("#close-detail").click()
    expect(page.locator("#metric-images")).to_have_text("2")
    page.locator("#edit-filter").select_option("deleted")
    expect(page.locator(".sample-card")).to_have_count(1)
    old = revision(page)
    page.locator("#draft-undo").click()
    saved(page, old)
    expect(page.locator(".sample-card")).to_have_count(0)
    old = revision(page)
    page.locator("#draft-redo").click()
    saved(page, old)
    page.locator("#edit-filter").select_option("")
    assert source_hashes(root) == before

    current_revision = revision(page)
    page.reload()
    expect(page.locator("#status-text")).to_contain_text("索引完成", timeout=10000)
    expect(page.locator("#metric-images")).to_have_text("2")
    assert revision(page) == current_revision
    assert source_hashes(root) == before
    exported = tmp_path / "exported"
    page.locator("#export-open").click()
    page.locator("#export-destination").fill(str(exported))
    page.locator("#export-preview-button").click()
    expect(page.locator("#export-start")).to_be_visible()
    expect(page.locator("#export-preview")).to_contain_text("pose")
    page.screenshot(path=str(output / "editor-export.png"))
    page.locator("#export-start").click()
    expect(page.locator("#export-status")).to_contain_text("导出完成", timeout=20000)
    assert source_hashes(root) == before
    assert not (exported / ".yolo-viewer-draft.sqlite3").exists()
    configuration = yaml.safe_load((exported / "data.yaml").read_text())
    assert configuration["kpt_shape"] == [2, 3]
    page.locator("#export-view-result").click()
    expect(page.locator("#metric-images")).to_have_text("2")
    expect(page.locator("#task-badge")).to_have_text("Pose")
    page.locator("#manage-classes").click()
    expect(page.locator(".keypoint-definition-row input").first).to_have_value("renamed_right")
    page.locator("#close-classes").click()

    # Apply back to an independent copy, then inspect the reindexed baseline and backup.
    select_first(page)
    old = revision(page)
    page.locator("#object-delete").click()
    saved(page, old)
    page.locator("#close-detail").click()
    page.locator("#export-open").click()
    page.locator("#export-mode").select_option("inplace")
    page.locator("#export-preview-button").click()
    expect(page.locator("#export-start")).to_be_visible()
    page.locator("#export-start").click()
    expect(page.locator("#notice")).to_contain_text("原文件备份", timeout=20000)
    expect(page.locator("#metric-instances")).to_have_text("1")
    expect(page.locator("#save-status")).to_have_text("无待应用改动")
    assert list(tmp_path.glob(".exported.viewer-backup-*"))
    assert source_hashes(root) == before
    page.screenshot(path=str(output / "editor-applied.png"))


def test_save_failure_retains_operation_and_blocks_navigation(page, tmp_path):
    root = browser_dataset(tmp_path / "failure", count=2)
    open_from_page(page, root)
    select_first(page)
    page.route(
        "**/draft/operations",
        lambda route: route.fulfill(status=503, json={"detail": "disk temporarily unavailable"}),
    )
    page.locator("#box-cx").fill("0.4")
    page.locator("#object-save").click()
    expect(page.locator("#editor-error")).to_contain_text("disk temporarily unavailable")
    page.locator("#image-next").click()
    expect(page.locator("#detail-title")).to_contain_text("sample_000")
    assert page.evaluate("editor.queue.length") == 1
    page.unroute("**/draft/operations")
    old = revision(page)
    page.locator("#image-dialog .retry-pending").click()
    saved(page, old)
    assert page.evaluate("state.detail.annotations[0].box.cx") == 0.4
    page.locator("#image-next").click()
    expect(page.locator("#detail-title")).to_contain_text("sample_001")


def test_read_only_tasks_have_no_edit_controls(page, tmp_path):
    root = browser_dataset(tmp_path / "segment", "segment")
    open_from_page(page, root, "segment")
    expect(page.locator("#draft-bar")).not_to_be_visible()
    select_first(page)
    expect(page.locator("#edit-toolbar")).not_to_be_visible()
    assert not (root / ".yolo-viewer-draft.sqlite3").exists()


def test_multitab_revision_conflict_can_reload_without_losing_saved_edits(page, tmp_path):
    root = browser_dataset(tmp_path / "tabs", count=2)
    open_from_page(page, root)
    first_session = page.evaluate("state.session")
    other = page.context.new_page()
    other.goto(page.url)
    open_from_page(other, root)
    assert other.evaluate("state.session") == first_session
    select_first(page)
    select_first(other)
    page.locator("#box-cx").fill("0.4")
    old = revision(page)
    page.locator("#object-save").click()
    saved(page, old)
    other.locator("#box-cx").fill("0.6")
    other.locator("#object-save").click()
    expect(other.locator("#editor-error")).to_be_visible()
    expect(other.locator("#detail-save-status")).to_contain_text("保存失败")
    other.on("dialog", lambda dialog: dialog.accept())
    other.locator("#image-dialog .reload-pending").click()
    expect(other.locator("#box-cx")).to_have_value("0.4")
    assert other.evaluate("editor.queue.length") == 0
    assert (root / "labels/train/sample_000.txt").read_text() == "0 .5 .5 .4 .5\n"
    other.close()


def test_resize_keypoint_drag_class_migration_and_atomic_issue_repair(page, tmp_path):
    root = browser_dataset(tmp_path / "pose-editor", "pose", count=2)
    path = root / "labels/train/sample_000.txt"
    path.write_text(path.read_text() + "0 invalid row\n")
    page.on("dialog", lambda dialog: dialog.accept())
    open_from_page(page, root, "pose")
    select_first(page)
    expect(page.locator("#detail-issues")).to_contain_text("0 invalid row")
    points_before = page.evaluate("state.detail.annotations[0].keypoints")
    page.locator('button[data-tool="edit"]').click()
    old = revision(page)
    drag(page, (0.1, 0.1), (0.18, 0.2))
    saved(page, old)
    box = page.evaluate("state.detail.annotations[0].box")
    assert box["width"] == pytest.approx(0.72, abs=1e-6)
    assert box["height"] == pytest.approx(0.7, abs=1e-6)
    assert page.evaluate("state.detail.annotations[0].keypoints") == points_before
    page.locator('button[data-tool="points"]').click()
    old = revision(page)
    drag(page, (0.4, 0.4), (0.32, 0.35))
    saved(page, old)
    point = page.evaluate("state.detail.annotations[0].keypoints[0]")
    assert point["x"] == pytest.approx(0.32, abs=1e-6)
    assert point["visibility"] == 1
    old = revision(page)
    page.get_by_role("button", name="替换异常行 2", exact=True).click()
    drag(page, (0.02, 0.04), (0.12, 0.14))
    saved(page, old)
    expect(page.locator("#annotation-count")).to_have_text("2")
    expect(page.locator("#detail-issues")).to_have_text("")
    assert "0 invalid row" in path.read_text()
    page.locator("#close-detail").click()
    page.locator("#manage-classes").click()
    page.locator("#class-add-name").fill("另一种目标")
    old = revision(page)
    page.locator("#class-add-form button").click()
    saved(page, old)
    second = page.locator(".class-editor").last
    new_class_id = second.get_attribute("data-class-id")
    second.get_by_role("button", name="增加关键点", exact=True).click()
    second.locator(".keypoint-definition-row input").fill("keypoint_0")
    old = revision(page)
    second.get_by_role("button", name="保存关键点列表", exact=True).click()
    saved(page, old)
    page.locator("#close-classes").click()
    select_first(page)
    old = revision(page)
    page.locator("#object-class").select_option(new_class_id)
    saved(page, old)
    assert page.evaluate("state.detail.annotations[0].keypoints") == [point]
    page.locator("#close-detail").click()
    page.locator("#manage-classes").click()
    first = page.locator('.class-editor[data-class-id="0"]')
    first.locator(".keypoint-definition-row").first.get_by_role("button", name="移除").click()
    old = revision(page)
    first.get_by_role("button", name="保存关键点列表", exact=True).click()
    saved(page, old)
    expect(first.locator(".keypoint-definition-row")).to_have_count(2)
    page.locator("#close-classes").click()
    page.locator("#edit-filter").select_option("modified")
    expect(page.locator(".sample-card")).to_have_count(2)


def test_recover_interrupted_application_before_indexing_from_webpage(page, tmp_path):
    from threading import Event

    from yolo_dataset_viewer.dataset import build_index
    from yolo_dataset_viewer.edit_models import EditRequest
    from yolo_dataset_viewer.editing import DraftEditor
    from yolo_dataset_viewer.exports import Exporter, ExportRequest
    from yolo_dataset_viewer.models import OpenRequest

    root = browser_dataset(tmp_path / "interrupted", "pose", count=2)
    original = source_hashes(root)
    request = OpenRequest(path=str(root), task="pose")
    data = build_index(request, Event(), lambda *_: None)
    draft = DraftEditor(data, request)
    draft.apply(
        EditRequest.model_validate(
            {
                "expected_revision": 0,
                "operation": {"kind": "keypoints_set", "class_id": 0, "keypoints": []},
            }
        )
    )
    # Leave an applied manifest as if the process died before its final reindex.
    result = Exporter(data, draft, request).run(
        ExportRequest(expected_revision=1, mode="inplace"), lambda *_: None, Event()
    )
    draft.close()
    assert result.backup_path
    page.on("dialog", lambda dialog: dialog.accept())
    page.locator("#dataset-path").fill(str(root))
    page.locator("#task").select_option("pose")
    page.locator("#open-button").click()
    expect(page.locator("#export-dialog")).to_be_visible()
    expect(page.locator("#export-status")).to_contain_text("未完成")
    expect(page.locator("#export-recoveries")).to_contain_text(result.backup_path)
    page.get_by_role("button", name="恢复备份", exact=True).click()
    expect(page.locator("#status-text")).to_contain_text("索引完成", timeout=10000)
    expect(page.locator("#gallery .sample-card")).to_have_count(2)
    expect(page.locator("#save-status")).to_have_text("已保存到草稿")
    assert source_hashes(root) == original
    assert page.evaluate("state.request.task") == "pose"


def test_pending_point_save_locks_mutations_until_refreshed(page, tmp_path):
    root = browser_dataset(tmp_path / "rapid-points", "pose", count=1)
    open_from_page(page, root, "pose")
    select_first(page)
    point_a = page.locator(".point-edit-row").nth(0)
    point_b = page.locator(".point-edit-row").nth(1)
    point_a.locator("input").first.fill("0.25")
    pending = []
    page.route("**/draft/operations", lambda route: pending.append(route))
    old = revision(page)
    point_a.get_by_role("button", name="保存点", exact=True).click()
    expect(point_b.get_by_role("button", name="保存点", exact=True)).to_be_disabled()
    expect(page.locator("#object-save")).to_be_disabled()
    expect(page.locator('[data-tool="draw"]')).to_be_disabled()
    # Even an extra event queued by an input device cannot submit a stale full object.
    point_b.get_by_role("button", name="保存点", exact=True).dispatch_event("click")
    assert page.evaluate("editor.queue.length") == 1
    assert len(pending) == 1
    pending[0].continue_()
    page.unroute("**/draft/operations")
    saved(page, old)
    expect(point_b.get_by_role("button", name="保存点", exact=True)).to_be_enabled()
    point_b.locator("input").first.fill("0.75")
    old = revision(page)
    point_b.get_by_role("button", name="保存点", exact=True).click()
    saved(page, old)
    assert page.evaluate("state.detail.annotations[0].keypoints[0].x") == 0.25
    assert page.evaluate("state.detail.annotations[0].keypoints[1].x") == 0.75


def test_discard_reindexes_added_files_with_new_session_and_closed_inspector(page, tmp_path):
    import shutil

    root = browser_dataset(tmp_path / "discard", count=2)
    page.on("dialog", lambda dialog: dialog.accept())
    open_from_page(page, root)
    select_first(page)
    page.locator("#box-cx").fill("0.4")
    old = revision(page)
    page.locator("#object-save").click()
    saved(page, old)
    old_session = page.evaluate("state.session")
    shutil.copy2(root / "images/train/sample_000.png", root / "images/train/aaa_new.png")
    (root / "labels/train/aaa_new.txt").write_text("0 .3 .3 .2 .2\n")
    # Dispatch the global action with the inspector open to check cleanup of its old sample ID.
    page.locator("#draft-discard").dispatch_event("click")
    expect(page.locator("#save-status")).to_have_text("无待应用改动")
    expect(page.locator("#image-dialog")).not_to_be_visible()
    expect(page.locator("#metric-images")).to_have_text("3")
    assert page.evaluate("state.session") != old_session
    select_first(page)
    expect(page.locator("#detail-title")).to_contain_text("aaa_new.png")
    expect(page.locator("#box-cx")).to_have_value("0.3")
