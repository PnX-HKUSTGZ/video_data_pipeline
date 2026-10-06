"""Keep inspector navigation anchored to its sample after filter membership changes."""

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


@pytest.mark.parametrize("control", ["button", "keyboard"])
@pytest.mark.parametrize("change", ["class", "state", "deleted"])
def test_current_sample_leaving_filter_does_not_skip_next(page, tmp_path, control, change):
    root = browser_dataset(tmp_path / "filter-navigation", count=4)
    config = yaml.safe_load((root / "data.yaml").read_text())
    config["names"].append("其他类别")
    (root / "data.yaml").write_text(yaml.safe_dump(config, allow_unicode=True))
    page.on("dialog", lambda dialog: dialog.accept())
    open_from_page(page, root)
    if change == "class":
        page.locator("#class-filter").select_option("0")
    elif change == "state":
        page.locator("#state-filter").select_option("valid")
    page.locator(".sample-card").first.click()
    expect(page.locator("#image-message")).to_have_text("")
    page.locator(".annotation-row").click()
    old = revision(page)
    if change == "class":
        page.locator("#object-class").select_option("1")
    elif change == "state":
        page.locator("#object-delete").click()
    else:
        page.locator("#image-delete").click()
    saved(page, old)
    expect(page.locator("#detail-position")).to_contain_text("剩余 3 张")
    expect(page.locator("#image-prev")).to_be_disabled()
    expect(page.locator("#image-next")).to_be_enabled()
    if control == "button":
        page.locator("#image-next").click()
    else:
        page.locator("#image-next").focus()
        page.keyboard.press("ArrowRight")
    expect(page.locator("#detail-title")).to_contain_text("sample_001")
    expect(page.locator("#image-message")).to_have_text("")
    expect(page.locator("#detail-position")).to_have_text("1 / 3")


def test_filter_insertion_point_previous_and_across_gallery_pages(page, tmp_path):
    root = browser_dataset(tmp_path / "boundary-navigation", count=53)
    open_from_page(page, root)
    page.locator("#state-filter").select_option("valid")
    page.locator(".sample-card").nth(47).click()
    expect(page.locator("#image-message")).to_have_text("")
    page.locator(".annotation-row").click()
    old = revision(page)
    page.locator("#object-delete").click()
    saved(page, old)
    expect(page.locator("#detail-position")).to_contain_text("剩余 52 张")
    page.locator("#image-prev").click()
    expect(page.locator("#detail-title")).to_contain_text("sample_046")
    expect(page.locator("#image-message")).to_have_text("")
    page.locator("#image-next").click()
    expect(page.locator("#detail-title")).to_contain_text("sample_048")
    expect(page.locator("#image-message")).to_have_text("")
    expect(page.locator("#detail-position")).to_have_text("48 / 52")
    page.locator("#image-next").click()
    expect(page.locator("#detail-title")).to_contain_text("sample_049")


def test_removing_only_filter_match_disables_both_directions(page, tmp_path):
    root = browser_dataset(tmp_path / "empty-filter", count=1)
    open_from_page(page, root)
    page.locator("#state-filter").select_option("valid")
    page.locator(".sample-card").click()
    expect(page.locator("#image-message")).to_have_text("")
    page.locator(".annotation-row").click()
    old = revision(page)
    page.locator("#object-delete").click()
    saved(page, old)
    expect(page.locator("#detail-position")).to_contain_text("剩余 0 张")
    expect(page.locator("#image-prev")).to_be_disabled()
    expect(page.locator("#image-next")).to_be_disabled()
    page.locator("#close-detail").click()
    expect(page.locator("#empty-state")).to_be_visible()


def test_neighbor_prefetch_uses_insertion_point_after_current_leaves_filter(page, tmp_path):
    root = browser_dataset(tmp_path / "filtered-prefetch", count=5)
    open_from_page(page, root)
    page.locator("#state-filter").select_option("valid")
    page.locator(".sample-card").nth(1).click()
    expect(page.locator("#image-message")).to_have_text("")
    page.locator(".annotation-row").click()
    page.evaluate("loading.stopNeighbors(); loading.images.clear()")
    requests = []
    page.on(
        "request",
        lambda request: (
            requests.append(request.url)
            if "/image" in request.url and "thumbnail=true" not in request.url
            else None
        ),
    )
    old = revision(page)
    page.locator("#object-delete").click()
    saved(page, old)
    page.wait_for_function(
        "loading.images.stats().cached === 2 && loading.images.stats().active === 0"
    )
    assert any("/samples/2/image" in url for url in requests)
    assert any("/samples/0/image" in url for url in requests)
    assert not any("/samples/3/image" in url for url in requests)


def test_unloaded_mobile_thumbnails_do_not_show_broken_alt_placeholders(page, tmp_path):
    page.set_viewport_size({"width": 360, "height": 740})
    root = browser_dataset(tmp_path / "mobile-thumbnails", count=12)
    open_from_page(page, root)
    unloaded = page.locator(".thumb-wrap img:not([src])")
    assert unloaded.count() > 0
    assert unloaded.evaluate_all(
        "nodes => nodes.every(node => getComputedStyle(node).opacity === '0')"
    )
    page.locator(".sample-card").first.scroll_into_view_if_needed()
    page.wait_for_function(
        "[...document.querySelectorAll('.thumb-wrap img')].slice(0,2).every(i => i.complete && i.naturalWidth > 0 && getComputedStyle(i).opacity === '1')"
    )
    assert page.locator(".thumb-wrap .issue").count() == 0
