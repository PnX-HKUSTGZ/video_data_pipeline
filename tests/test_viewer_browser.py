"""Real browser acceptance tests: uv run --extra viewer --group browser pytest -q."""

import socket
import threading
import time
from pathlib import Path

import pytest
import yaml
from PIL import Image, ImageDraw

pytest.importorskip("fastapi")
pytest.importorskip("playwright")
import uvicorn
from playwright.sync_api import expect, sync_playwright

from yolo_dataset_viewer.app import create_app


@pytest.fixture(scope="module")
def web_server():
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(create_app(), log_level="error", access_log=False))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    try:
        for _ in range(300):
            if server.started:
                break
            time.sleep(0.01)
        assert server.started
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(5)
        sock.close()
        assert not thread.is_alive()


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as playwright:
        if not Path(playwright.chromium.executable_path).exists():
            pytest.skip("Install browser with: uv run --group browser playwright install chromium")
        browser = playwright.chromium.launch(
            headless=True, args=["--disable-gpu", "--disable-dev-shm-usage"]
        )
        yield browser
        browser.close()


@pytest.fixture
def page(browser, web_server):
    context = browser.new_context(viewport={"width": 1440, "height": 1000}, device_scale_factor=1)
    page = context.new_page()
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto(web_server)
    yield page
    context.close()
    assert not errors


def browser_dataset(root, task="detect", count=4):
    images, labels = root / "images/train", root / "labels/train"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    rows = {
        "detect": "0 .5 .5 .4 .5",
        "segment": "0 .3 .25 .7 .25 .7 .75 .3 .75",
        "obb": "0 .4 .2 .8 .4 .6 .8 .2 .6",
        "pose": "0 .5 .5 .8 .8 .4 .4 1 .6 .6 2 .0 .0 0",
    }
    for i in range(count):
        image = Image.new("RGB", (600, 400), "#e3e6dc")
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 250, 600, 400), fill="#c5cbb9")
        draw.rectangle((180, 100, 420, 300), fill=["#739d8a", "#caac77", "#9aaec0"][i % 3])
        draw.rectangle((205, 120, 394, 160), fill="#ffffff")
        draw.text((22, 25), f"External dataset / sample {i + 1:03}", fill="#3c5047")
        image.save(images / f"sample_{i:03d}.png")
        (labels / f"sample_{i:03d}.txt").write_text(rows[task] + "\n")
    config = {"train": "images/train", "names": ["目标 object"]}
    if task == "pose":
        config["kpt_shape"] = [3, 3]
    (root / "data.yaml").write_text(yaml.safe_dump(config, allow_unicode=True))
    return root


def open_from_page(page, root, task="detect"):
    page.locator("#dataset-path").fill(str(root))
    page.locator("#task").select_option(task)
    page.locator("#open-button").click()
    expect(page.locator("#status-text")).to_contain_text("索引完成", timeout=10000)
    expect(page.locator("#gallery .sample-card").first).to_be_visible()


@pytest.mark.parametrize("task", ["detect", "segment", "obb", "pose"])
def test_render_inspect_zoom_pan_and_annotations(page, tmp_path, task):
    root = browser_dataset(tmp_path / task, task)
    open_from_page(page, root, task)
    expect(page.locator("#metric-images")).to_have_text("4")
    page.locator(".sample-card").first.click()
    expect(page.locator("#image-dialog")).to_be_visible()
    expect(page.locator("#image-message")).to_have_text("")
    expect(page.locator("#annotation-count")).to_have_text("1")
    canvas = page.locator("#image-canvas")
    canvas.click()
    expect(page.locator("#selected-annotation")).to_be_visible()
    expect(page.locator("#raw-label")).to_contain_text("0 ")
    before = canvas.evaluate(
        "async el => { await new Promise(requestAnimationFrame); return el.toDataURL(); }"
    )
    page.locator("#show-annotations").uncheck()
    after = canvas.evaluate(
        "async el => { await new Promise(requestAnimationFrame); return el.toDataURL(); }"
    )
    assert before != after
    page.locator("#show-annotations").check()
    zoom = page.locator("#zoom-level").inner_text()
    page.locator("#zoom-in").click()
    expect(page.locator("#zoom-level")).not_to_have_text(zoom)
    box = canvas.bounding_box()
    before = canvas.evaluate(
        "async el => { await new Promise(requestAnimationFrame); return el.toDataURL(); }"
    )
    page.mouse.move(box["x"] + 100, box["y"] + 100)
    page.mouse.down()
    page.mouse.move(box["x"] + 160, box["y"] + 140, steps=4)
    page.mouse.up()
    assert before != canvas.evaluate(
        "async el => { await new Promise(requestAnimationFrame); return el.toDataURL(); }"
    )
    page.locator("#fit-image").click()
    page.locator("#image-next").click()
    expect(page.locator("#detail-title")).to_contain_text("sample_001")
    expect(page.locator("#image-message")).to_have_text("")
    if task == "pose":
        page.locator(".annotation-row").click()
        expect(page.locator("#annotation-coordinates")).to_contain_text('"visibility": 0')
    if task == "detect":
        output = Path(".cache/viewer-checks")
        output.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(output / "image-inspector.png"))
    page.locator("#close-detail").click()
    expect(page.locator("#image-dialog")).not_to_be_visible()


def test_filters_pagination_rescan_and_dataset_switch(page, tmp_path):
    root = browser_dataset(tmp_path / "many", count=53)
    (root / "labels/train/sample_001.txt").write_text("")
    (root / "labels/train/sample_002.txt").unlink()
    (root / "labels/train/sample_003.txt").write_text("0 .5 .5 .4 .5\n0 nan .5 .2 .2")
    open_from_page(page, root)
    expect(page.locator(".sample-card")).to_have_count(48)
    page.locator("#page-next").click()
    expect(page.locator(".sample-card")).to_have_count(5)
    page.locator(".sample-card").last.click()
    expect(page.locator("#detail-title")).to_contain_text("sample_052")
    expect(page.locator("#image-next")).to_be_disabled()
    page.locator("#close-detail").click()
    page.locator("#state-filter").select_option("invalid")
    expect(page.locator(".sample-card")).to_have_count(1)
    expect(page.locator("#metric-instances")).to_have_text("1")
    page.locator(".sample-card").click()
    expect(page.locator("#detail-issues")).to_contain_text(".txt:2")
    page.locator("#close-detail").click()
    page.locator("#state-filter").select_option("missing")
    expect(page.locator(".sample-card")).to_have_count(1)
    expect(page.locator(".sample-card")).to_contain_text("sample_002")
    page.locator("#state-filter").select_option("")
    page.locator("#class-filter").select_option("0")
    expect(page.locator("#metric-images")).to_have_text("51")
    page.locator("#search").fill("sample_010")
    expect(page.locator(".sample-card")).to_have_count(1)
    page.locator("#search").fill("does-not-exist")
    expect(page.locator("#empty-state")).to_be_visible()
    (root / "labels/train/sample_001.txt").write_text("0 .5 .5 .4 .5")
    page.locator("#rescan").click()
    expect(page.locator("#status-text")).to_contain_text("索引完成")
    expect(page.locator("#metric-instances")).to_have_text("52")
    other = browser_dataset(tmp_path / "other", count=2)
    open_from_page(page, other)
    expect(page.locator("#metric-images")).to_have_text("2")


def test_yaml_choices_errors_and_safe_text(page, tmp_path):
    root = browser_dataset(tmp_path / "choices")
    (root / "extra.yaml").write_text("train: images/train\nnames: ['<img src=x onerror=alert(1)>']")
    page.locator("#dataset-path").fill(str(root))
    page.locator("#open-button").click()
    expect(page.locator("#candidates")).to_be_visible()
    page.locator("#yaml-choice").select_option(str(root / "extra.yaml"))
    page.locator("#choose-yaml").click()
    expect(page.locator("#status-text")).to_contain_text("索引完成")
    expect(page.locator("#class-stats")).to_contain_text("<img src=x onerror=alert(1)>")
    assert page.locator("#class-stats img").count() == 0
    page.locator("#dataset-path").fill(str(tmp_path / "does-not-exist"))
    page.locator("#open-button").click()
    expect(page.locator("#notice")).to_contain_text("路径不存在")
    expect(page.locator("#split-filter")).to_be_disabled()


def test_gallery_layout_and_screenshot(page, tmp_path):
    root = browser_dataset(tmp_path / "standard-demo", count=12)
    open_from_page(page, root)
    page.wait_for_function(
        "[...document.querySelectorAll('.thumb-wrap img')].every(i => i.complete)"
    )
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    assert page.locator(".sample-card").first.bounding_box()["width"] > 130
    output = Path(".cache/viewer-checks")
    output.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(output / "dataset-browser.png"), full_page=True)
    page.set_viewport_size({"width": 700, "height": 1000})
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
