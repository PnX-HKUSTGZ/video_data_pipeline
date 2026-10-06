"""Real browser regressions for inspector identity and independent class form edits."""

import os

import pytest
from PIL import Image

pytest.importorskip("fastapi")
pytest.importorskip("playwright")
from playwright.sync_api import expect
from test_viewer_browser import (  # noqa: F401
    browser,
    browser_dataset,
    open_from_page,
    web_server,
)
from test_viewer_browser import page as viewer_page
from test_viewer_editor_browser import revision, saved, select_first

page = viewer_page


def test_corrupt_image_can_be_deleted_and_undone_without_loading_pixels(page, tmp_path):
    root = browser_dataset(tmp_path / "broken", count=2)
    (root / "images/train/sample_000.png").write_bytes(b"corrupted image")
    page.on("dialog", lambda dialog: dialog.accept())
    open_from_page(page, root)
    page.locator(".sample-card").first.click()
    expect(page.locator("#image-message")).to_contain_text("图片无法读取")
    previous = revision(page)
    page.locator("#image-delete").click()
    saved(page, previous)
    assert page.evaluate("state.detail.deleted")
    assert not page.evaluate("!!view.image")
    expect(page.locator("#metric-images")).to_have_text("1")
    previous = revision(page)
    page.locator("#image-delete").click()
    saved(page, previous)
    assert not page.evaluate("state.detail.deleted")
    expect(page.locator("#metric-images")).to_have_text("2")


def test_corrupt_image_allows_invalid_row_removal_but_blocks_geometry_save(page, tmp_path):
    root = browser_dataset(tmp_path / "broken-label", count=1)
    (root / "images/train/sample_000.png").write_bytes(b"corrupted image")
    (root / "labels/train/sample_000.txt").write_text("0 .5 .5 .4 .5\nbad row\n")
    page.on("dialog", lambda dialog: dialog.accept())
    open_from_page(page, root)
    page.locator(".sample-card").first.click()
    expect(page.locator("#image-message")).to_contain_text("图片无法读取")
    previous = revision(page)
    page.get_by_role("button", name="删除异常行 2", exact=True).click()
    saved(page, previous)
    assert page.evaluate("state.detail.issues.every(issue => issue.line !== 2)")
    page.locator(".annotation-row").first.click()
    previous = revision(page)
    page.locator("#box-cx").fill("0.4")
    page.locator("#object-save").click()
    expect(page.locator("#editor-error")).to_contain_text("暂不能修改坐标")
    assert revision(page) == previous
    assert page.evaluate("state.detail.annotations[0].box.cx") == 0.5


def test_class_rename_and_keypoint_save_preserve_each_others_pending_form(page, tmp_path):
    root = browser_dataset(tmp_path / "class-form", "pose", count=1)
    open_from_page(page, root, "pose")
    page.locator("#manage-classes").click()
    first = page.locator('.class-editor[data-class-id="0"]')
    first.locator(".keypoint-definition-row input").first.fill("pending-point")
    first.locator('input[aria-label="类别 0 名称"]').fill("renamed-class")
    previous = revision(page)
    first.get_by_role("button", name="重命名", exact=True).click()
    saved(page, previous)
    expect(first.locator(".keypoint-definition-row input").first).to_have_value("pending-point")
    expect(first.locator(".class-form-pending")).to_be_visible()
    first.locator('input[aria-label="类别 0 名称"]').fill("next-class-name")
    previous = revision(page)
    first.get_by_role("button", name="保存关键点列表", exact=True).click()
    saved(page, previous)
    expect(first.locator('input[aria-label="类别 0 名称"]')).to_have_value("next-class-name")
    assert page.evaluate("editor.draft.classes[0].name") == "renamed-class"
    assert page.evaluate("editor.draft.classes[0].keypoints[0].name") == "pending-point"
    previous = revision(page)
    page.locator("#class-add-name").fill("another")
    page.locator("#class-add-form button").click()
    saved(page, previous)
    expect(first.locator('input[aria-label="类别 0 名称"]')).to_have_value("next-class-name")
    page.locator("#close-classes").click()
    page.locator("#manage-classes").click()
    expect(first.locator('input[aria-label="类别 0 名称"]')).to_have_value("next-class-name")


def test_new_keypoint_name_is_unique_after_middle_removal(page, tmp_path):
    root = browser_dataset(tmp_path / "point-names", "pose", count=1)
    page.on("dialog", lambda dialog: dialog.accept())
    open_from_page(page, root, "pose")
    page.locator("#manage-classes").click()
    first = page.locator(".class-editor").first
    first.locator(".keypoint-definition-row").nth(1).get_by_role(
        "button", name="移除", exact=True
    ).click()
    first.get_by_role("button", name="增加关键点", exact=True).click()
    names = first.locator(".keypoint-definition-row input").evaluate_all(
        "inputs => inputs.map(i => i.value)"
    )
    assert names == ["keypoint_0", "keypoint_2", "keypoint_1"]
    previous = revision(page)
    first.get_by_role("button", name="保存关键点列表", exact=True).click()
    saved(page, previous)
    assert page.evaluate("editor.draft.classes[0].keypoints.map(point => point.name)") == names


def test_latest_draft_after_other_tab_rescan_resets_old_sample_pixels(page, tmp_path):
    root = browser_dataset(tmp_path / "session", count=2)
    stamp = 1700000000000000000
    for png in (root / "images/train").glob("*.png"):
        with Image.open(png) as source:
            source.save(png.with_suffix(".bmp"))
        png.unlink()
        os.utime(png.with_suffix(".bmp"), ns=(stamp, stamp))
    open_from_page(page, root)
    select_first(page)
    page.evaluate("window.beforeImage = view.image; window.beforeKey = state.detail.key")
    original_session = page.evaluate("state.session")
    second = page.context.new_page()
    try:
        second.goto(page.url)
        open_from_page(second, root)
        Image.new("RGB", (600, 400), "red").save(root / "images/train/aaa.bmp")
        os.utime(root / "images/train/aaa.bmp", ns=(stamp, stamp))
        (root / "labels/train/aaa.txt").write_text("0 .25 .25 .2 .2\n")
        second.locator("#rescan").click()
        second.wait_for_function(
            "previous => state.session !== previous && editor.draft !== null", arg=original_session
        )
        expect(second.locator("#metric-images")).to_have_text("3")
        page.locator("#box-cx").fill("0.4")
        page.locator("#object-save").click()
        expect(page.locator("#editor-error")).to_contain_text("数据集已切换")
        page.on("dialog", lambda dialog: dialog.accept())
        page.locator("#image-dialog .reload-pending").click()
        expect(page.locator("#image-dialog")).not_to_be_visible()
        expect(page.locator("#metric-images")).to_have_text("3")
        assert page.evaluate("state.detail === null && view.image === null")
        select_first(page)
        expect(page.locator("#detail-title")).to_contain_text("aaa.bmp")
        assert page.evaluate("beforeImage !== view.image && beforeKey !== state.detail.key")
    finally:
        second.close()


def test_detail_reload_rejects_changed_sample_identity_even_with_same_image_stamp(page, tmp_path):
    root = browser_dataset(tmp_path / "identity", count=2)
    open_from_page(page, root)
    select_first(page)

    def swapped_detail(route):
        response = route.fetch()
        detail = response.json()
        detail["key"] = "another-sample-with-same-image-stamp"
        route.fulfill(response=response, json=detail)

    page.route("**/samples/0", swapped_detail)
    previous = revision(page)
    page.locator("#box-cx").fill("0.4")
    page.locator("#object-save").click()
    saved(page, previous)
    expect(page.locator("#image-dialog")).not_to_be_visible()
    expect(page.locator("#notice")).to_contain_text("图片索引已变化")
    assert page.evaluate("state.detail === null && view.image === null")
