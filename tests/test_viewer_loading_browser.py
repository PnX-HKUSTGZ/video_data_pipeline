"""Browser regressions for bounded preloading and responsive editor refreshes."""

import pytest
from PIL import Image

pytest.importorskip("fastapi")
pytest.importorskip("playwright")
from playwright.sync_api import Error as PlaywrightError
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
from test_viewer_editor_browser import revision, saved, select_first

page = viewer_page


def test_editor_save_preserves_gallery_images_and_class_options(page, tmp_path):
    root = browser_dataset(tmp_path / "stable-dom", "pose", count=4)
    open_from_page(page, root, "pose")
    page.wait_for_function(
        "[...document.querySelectorAll('.thumb-wrap img')].every(i => i.complete)"
    )
    select_first(page)
    requests = []
    page.on("request", lambda request: requests.append(request.url))
    page.evaluate(
        """() => {
          window.savedGalleryNodes = [...document.querySelectorAll('.thumb-wrap img')];
          window.savedClassOptions = [...document.querySelectorAll('#class-filter option')];
          window.savedPointRows = [...document.querySelectorAll('.point-edit-row')];
        }"""
    )
    page.locator('[data-tool="points"]').click()
    page.locator('[data-tool="edit"]').click()
    assert page.evaluate(
        "savedPointRows.every((node,i) => node === document.querySelectorAll('.point-edit-row')[i])"
    )
    old = revision(page)
    page.locator("#box-cx").fill("0.4")
    page.locator("#object-save").click()
    saved(page, old)
    assert page.evaluate(
        "savedClassOptions.every((node,i) => node === document.querySelectorAll('#class-filter option')[i])"
    )
    assert page.evaluate(
        "savedGalleryNodes.every((node,i) => node === document.querySelectorAll('.thumb-wrap img')[i])"
    )
    assert not any("thumbnail=true" in url for url in requests)
    assert page.evaluate("state.detail.annotations[0].box.cx") == 0.4


def test_navigation_waits_for_committed_edit_detail_refresh(page, tmp_path):
    root = browser_dataset(tmp_path / "save-navigation", count=2)
    open_from_page(page, root)
    select_first(page)
    pending = []
    page.route("**/samples/0", lambda route: pending.append(route))
    page.locator("#box-cx").fill("0.4")
    page.locator("#object-save").click()
    page.wait_for_function("editor.queue.length === 0 && !!editor.saving")
    page.locator("#image-next").click()
    expect(page.locator("#detail-title")).to_contain_text("sample_000")
    assert len(pending) == 1
    pending[0].continue_()
    page.unroute("**/samples/0")
    expect(page.locator("#detail-title")).to_contain_text("sample_001")
    expect(page.locator("#image-message")).to_have_text("")
    page.locator("#image-prev").click()
    expect(page.locator("#detail-title")).to_contain_text("sample_000")
    expect(page.locator("#image-message")).to_have_text("")
    page.locator(".annotation-row").click()
    expect(page.locator("#box-cx")).to_have_value("0.4")


def test_latest_navigation_cancels_pending_detail_request(page, tmp_path):
    root = browser_dataset(tmp_path / "cancel-detail", count=5)
    open_from_page(page, root)
    pending = []
    failed = []
    page.on("requestfailed", lambda request: failed.append(request.url))
    page.route("**/samples/0", lambda route: pending.append(route))
    page.locator("#gallery .sample-card").first.click()
    page.wait_for_function("state.position === 0")
    page.locator("#image-next").click()
    expect(page.locator("#detail-title")).to_contain_text("sample_001")
    expect(page.locator("#image-message")).to_have_text("")
    page.wait_for_function("state.detail?.id === 1 && !!view.image")
    assert len(pending) == 1
    assert any(url.endswith("/samples/0") for url in failed)
    # A late response from the abandoned request must not change the current inspector.
    try:
        pending[0].continue_()
    except PlaywrightError as error:
        assert "handled" in str(error) or "closed" in str(error)
    page.unroute("**/samples/0")
    expect(page.locator("#detail-title")).to_contain_text("sample_001")
    assert page.evaluate("state.detail.id") == 1


def test_image_cache_revalidates_changed_source_on_reopen(page, tmp_path):
    root = browser_dataset(tmp_path / "changed-image", count=3)
    open_from_page(page, root)
    select_first(page)
    assert page.evaluate("view.image.width") == 600
    page.locator("#image-next").click()
    expect(page.locator("#detail-title")).to_contain_text("sample_001")
    expect(page.locator("#image-message")).to_have_text("")
    Image.new("RGB", (320, 240), "purple").save(root / "images/train/sample_000.png")
    page.locator("#image-prev").click()
    expect(page.locator("#detail-title")).to_contain_text("sample_000")
    expect(page.locator("#image-message")).to_have_text("")
    page.wait_for_function("view.image?.width === 320 && view.image?.height === 240")
    expect(page.locator("#detail-meta")).to_contain_text("320 × 240")


def test_neighbor_prefetch_is_bounded_and_back_navigation_reuses_pixels(page, tmp_path):
    root = browser_dataset(tmp_path / "neighbors", count=16)
    open_from_page(page, root)
    full_requests = []
    detail_requests = []

    def record(request):
        if "/image" in request.url and "thumbnail=true" not in request.url:
            full_requests.append(request.url)
        elif request.url.rsplit("/", 1)[-1].isdigit():
            detail_requests.append(request.url)

    page.on("request", record)
    select_first(page)
    page.wait_for_function(
        "loading.images.stats().cached >= 2 && loading.images.stats().active === 0"
    )
    assert len(full_requests) == 2  # Current image plus its immediate next neighbor.
    assert len(detail_requests) == 1  # Annotations are fresh reads, never speculative copies.
    page.locator("#image-next").click()
    expect(page.locator("#detail-title")).to_contain_text("sample_001")
    expect(page.locator("#image-message")).to_have_text("")
    page.wait_for_function(
        "loading.images.stats().cached >= 3 && loading.images.stats().active === 0"
    )
    image_count, detail_count = len(full_requests), len(detail_requests)
    page.locator("#image-prev").click()
    expect(page.locator("#detail-title")).to_contain_text("sample_000")
    expect(page.locator("#image-message")).to_have_text("")
    assert len(full_requests) == image_count
    assert len(detail_requests) == detail_count + 1
    for index in range(1, 11):
        page.locator("#image-next").click()
        expect(page.locator("#detail-title")).to_contain_text(f"sample_{index:03d}")
        expect(page.locator("#image-message")).to_have_text("")
        stats = page.evaluate("loading.images.stats()")
        assert stats["cached"] <= 4
        assert stats["bytes"] <= 96 * 1024 * 1024
        assert stats["active"] <= 2
    page.locator("#close-detail").click()
    page.wait_for_function(
        "loading.images.stats().active === 0 && loading.images.stats().queued === 0"
    )
    page.locator("#rescan").click()
    expect(page.locator("#status-text")).to_contain_text("索引完成")
    assert page.evaluate("loading.images.stats().cached") == 0


def test_pointer_bursts_coalesce_to_one_canvas_render_per_frame(page, tmp_path):
    root = browser_dataset(tmp_path / "canvas", count=1)
    open_from_page(page, root)
    select_first(page)
    counts = page.evaluate(
        """async () => {
          await new Promise(requestAnimationFrame);
          const original = ctx.drawImage.bind(ctx);
          let renders = 0;
          ctx.drawImage = (...args) => { renders++; return original(...args); };
          for (let i=0;i<80;i++) draw();
          const synchronous = renders;
          await new Promise(requestAnimationFrame);
          ctx.drawImage = original;
          return {synchronous, renders};
        }"""
    )
    assert counts == {"synchronous": 0, "renders": 1}


def test_thumbnail_requests_follow_viewport_and_scroll(page, tmp_path):
    root = browser_dataset(tmp_path / "viewport-thumbnails", count=53)
    thumbnails = []
    page.on(
        "request",
        lambda request: thumbnails.append(request.url) if "thumbnail=true" in request.url else None,
    )
    open_from_page(page, root)
    page.wait_for_function(
        "[...document.querySelectorAll('.thumb-wrap img[src]')].every(i => i.complete)"
    )
    initial = len(thumbnails)
    assert 0 < initial < 48
    assert page.locator(".thumb-wrap img:not([src])").count() > 0
    page.locator(".sample-card").nth(47).scroll_into_view_if_needed()
    page.wait_for_function(
        "document.querySelectorAll('.thumb-wrap img')[47].hasAttribute('src') && "
        "document.querySelectorAll('.thumb-wrap img')[47].complete"
    )
    assert len(thumbnails) > initial
    assert len(thumbnails) == len(set(thumbnails))


def test_explicit_latest_draft_reload_resets_unsaved_form_fields(page, tmp_path):
    root = browser_dataset(tmp_path / "reload-form", count=1)
    open_from_page(page, root)
    select_first(page)
    page.route(
        "**/draft/operations",
        lambda route: route.fulfill(status=503, json={"detail": "storage unavailable"}),
    )
    page.locator("#box-cx").fill("0.4")
    page.locator("#object-save").click()
    expect(page.locator("#editor-error")).to_contain_text("storage unavailable")
    page.on("dialog", lambda dialog: dialog.accept())
    page.locator("#image-dialog .reload-pending").click()
    expect(page.locator("#box-cx")).to_have_value("0.5")
    assert page.evaluate("editor.queue.length") == 0
