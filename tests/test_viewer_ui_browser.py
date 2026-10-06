"""Real browser layout checks for the dataset workspace and compact image editor."""

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
from test_viewer_editor_browser import revision, saved

page = viewer_page


def assert_no_horizontal_overflow(page, selector="html"):
    assert page.locator(selector).evaluate("el => el.scrollWidth <= el.clientWidth + 1")


@pytest.mark.parametrize(
    "width,height",
    [(1366, 768), (1024, 768), (800, 768), (721, 768), (720, 768), (390, 844), (360, 740)],
)
def test_workspace_editor_and_settings_fit_small_screens(page, tmp_path, width, height):
    page.set_viewport_size({"width": width, "height": height})
    root = browser_dataset(tmp_path / "标准姿态数据集", "pose", count=12)
    config = yaml.safe_load((root / "data.yaml").read_text())
    config["names"] = ["目标 object · a deliberately long class name", "另一类别"]
    (root / "data.yaml").write_text(yaml.safe_dump(config, allow_unicode=True))
    open_from_page(page, root, "pose")
    assert_no_horizontal_overflow(page)
    if width >= 1024:
        assert page.locator(".sample-card").first.bounding_box()["y"] < 580
    output = Path(".cache/viewer-checks")
    output.mkdir(parents=True, exist_ok=True)
    page.locator(".sample-card").last.scroll_into_view_if_needed()
    page.locator(".sample-card").first.scroll_into_view_if_needed()
    page.wait_for_function(
        "[...document.querySelectorAll('.thumb-wrap img')].slice(0,2).every(i => i.complete && i.naturalWidth > 0)"
    )
    page.evaluate("window.scrollTo(0,0)")
    page.wait_for_function(
        "[...document.querySelectorAll('.thumb-wrap img[src]')].every(i => i.complete)"
    )
    page.screenshot(path=str(output / f"ui-{width}-workspace.png"), full_page=True)
    page.locator(".sample-card").first.click()
    expect(page.locator("#image-message")).to_have_text("")
    page.locator(".annotation-row").first.click()
    expect(page.locator("#image-dialog")).to_have_accessible_name("images/train/sample_000.png")
    assert_no_horizontal_overflow(page, "#image-dialog")
    before = page.locator("#canvas-stage").bounding_box()
    assert before["height"] >= 180
    page.locator("#box-cx").scroll_into_view_if_needed()
    if width < 721:
        after = page.locator("#canvas-stage").bounding_box()
        assert after["y"] == pytest.approx(before["y"], abs=1)
        assert after["height"] == pytest.approx(before["height"], abs=1)
        assert after["y"] + after["height"] <= height
    page.locator("#box-cx").fill("0.5")
    old = revision(page)
    page.locator("#object-save").click()
    saved(page, old)
    expect(page.locator("#close-detail")).to_be_in_viewport()
    page.screenshot(path=str(output / f"ui-{width}-editor.png"))
    page.locator("#close-detail").click()
    page.locator("#manage-classes").click()
    expect(page.locator("#classes-dialog")).to_have_accessible_name("类别与关键点")
    assert_no_horizontal_overflow(page, "#classes-dialog")
    page.locator(".class-editor").last.scroll_into_view_if_needed()
    expect(page.locator("#close-classes")).to_be_in_viewport()
    page.screenshot(path=str(output / f"ui-{width}-classes.png"))
    page.locator("#close-classes").click()
    page.locator("#export-open").click()
    expect(page.locator("#export-dialog")).to_have_accessible_name("导出与应用草稿")
    assert_no_horizontal_overflow(page, "#export-dialog")
    expect(page.locator("#export-preview-button")).to_be_in_viewport()
    page.locator("#close-export").click()
