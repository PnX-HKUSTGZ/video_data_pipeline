"""External-style fixtures contain only standard images, labels and optional YAML."""

import os
import subprocess
import sys
import time
from io import BytesIO
from threading import Event

import pytest
import yaml
from PIL import Image

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from yolo_dataset_viewer.app import create_app
from yolo_dataset_viewer.dataset import DatasetError, build_index, label_path, resolve_spec
from yolo_dataset_viewer.labels import read_labels
from yolo_dataset_viewer.models import OpenRequest
from yolo_dataset_viewer.service import ViewerService


def picture(path, size=(120, 80), color=(40, 130, 90)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)
    return path


def standard_dataset(root, rows=None, shape=None):
    for name, content in (rows or {"a": "0 0.5 0.5 0.4 0.6\n"}).items():
        image = picture(root / "images" / "train" / f"{name}.png")
        label = label_path(image)
        label.parent.mkdir(parents=True, exist_ok=True)
        if content is not None:
            label.write_text(content)
    (root / "images" / "val").mkdir(parents=True, exist_ok=True)
    config = {
        "path": ".",
        "train": "images/train",
        "val": "images/val",
        "names": ["物体", "second"],
    }
    if shape:
        config["kpt_shape"] = shape
    (root / "dataset.yml").write_text(yaml.safe_dump(config, allow_unicode=True))
    return root


def index(entry, task="detect", **kwargs):
    return build_index(
        OpenRequest(path=str(entry), task=task, **kwargs), Event(), lambda *args: None
    )


def wait_ready(client, request):
    opened = client.post("/api/datasets", json=request)
    assert opened.status_code == 202
    session = opened.json()["id"]
    for _ in range(300):
        result = client.get(f"/api/datasets/{session}").json()
        if result["state"] != "indexing":
            return session, result
        time.sleep(0.01)
    pytest.fail("Index worker did not finish")


@pytest.mark.parametrize(
    ("task", "row", "shape", "kind"),
    [
        ("detect", "0 0.5 0.5 1 1", None, "detect"),
        ("segment", "1 0 0 1 0 .5 1", None, "segment"),
        ("obb", "0 .1 .2 .8 .1 .9 .8 .2 .9", None, "obb"),
        ("segment", "0 .1 .2 .8 .1 .9 .8 .2 .9", None, "segment"),
        ("pose", "0 .5 .5 1 1 .2 .3 .8 .7", [2, 2], "pose"),
        ("pose", "0 .5 .5 1 1 .2 .3 1 .8 .7 2 0 0 0", [3, 3], "pose"),
        ("pose", "0 .5 .5 1 1 -1 -1 .8 .7", [2, 2], "pose"),
    ],
)
def test_standard_tasks(tmp_path, task, row, shape, kind):
    root = standard_dataset(tmp_path / "外部数据集", {"nested/同名": row}, shape)
    data = index(root, task)
    assert len(data.samples) == 1
    result = read_labels(data.samples[0].label, task, data.spec.names, data.spec.kpt_shape)
    assert result.state == "valid"
    assert result.annotations[0].kind == kind
    assert result.annotations[0].raw == row
    assert data.stats(data.samples).instances == 1


@pytest.mark.parametrize(
    "row",
    [
        "-1 .5 .5 .2 .2",
        "1.0 .5 .5 .2 .2",
        "2 .5 .5 .2 .2",
        "0 nan .5 .2 .2",
        "0 inf .5 .2 .2",
        "0 .5 .5 0 .2",
        "0 .9 .5 .5 .2",
        "0 .5 .5 .2",
        "0 .5 .5 .2 .2 .8",
        "0 nope .5 .2 .2",
        "0 .5 .5 -1 .2",
    ],
)
def test_bad_rows_keep_other_valid_objects(tmp_path, row):
    path = tmp_path / "label.txt"
    path.write_text(f"\n0 .5 .5 .2 .2\n{row}\n", encoding="utf-8-sig")
    result = read_labels(path, "detect", {0: "ok"}, None)
    assert result.state == "invalid"
    assert len(result.annotations) == 1
    assert result.annotations[0].line == 2
    assert result.issues[0].line == 3
    assert result.issues[0].file == str(path)


@pytest.mark.parametrize(
    ("task", "row", "shape"),
    [
        ("segment", "0 .1 .2 .3 .4", None),
        ("segment", "0 .1 .2 .3 .4 .5 .6 .7", None),
        ("segment", "0 .1 .2 .3 .4 .5 1.1", None),
        ("obb", "0 .1 .2 .3 .4 .5 .6", None),
        ("pose", "0 .5 .5 1 1 .2 .3 3", (1, 3)),
        ("pose", "0 .5 .5 1 1 .2 .3", (1, 3)),
    ],
)
def test_malformed_geometry(tmp_path, task, row, shape):
    path = tmp_path / "a.txt"
    path.write_text(row)
    result = read_labels(path, task, {}, shape)
    assert result.state == "invalid"
    assert not result.annotations


@pytest.mark.parametrize("layout", ["images/train", "train/images", "train", "images", ""])
def test_layouts_without_yaml_and_nested_duplicate_names(tmp_path, layout):
    for folder, number in (("first", 0), ("second", 7)):
        image = picture(tmp_path / layout / folder / "同名.JPG")
        label = label_path(image)
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text(f"{number} .5 .5 .2 .2")
    data = index(tmp_path)
    assert len(data.samples) == 2
    assert data.names == {0: "class_0", 7: "class_7"}
    assert {tuple(s.counts) for s in data.samples} == {(0,), (7,)}
    assert {s.split for s in data.samples} == ({"train"} if "train" in layout else {"all"})


def test_yaml_sources_lists_absolute_paths_and_duplicates(tmp_path):
    root = standard_dataset(tmp_path / "data", {"a": "0 .5 .5 1 1", "b": "1 .5 .5 1 1"})
    lists = root / "lists"
    lists.mkdir()
    (lists / "train.txt").write_text(f"../images/train/a.png\n{root}/images/train/b.png\n")
    config = {
        "path": "data",
        "train": ["lists/train.txt", "images/train"],
        "test": "images/val",
        "names": {0: "物体", 1: "second"},
    }
    yaml_path = tmp_path / "custom.yaml"
    yaml_path.write_text(yaml.safe_dump(config))
    data = index(yaml_path)
    assert len(data.samples) == 2
    assert data.spec.root == root
    assert data.stats(data.samples).splits == {"train": 2, "test": 0}
    assert data.warnings


def test_manifest_plain_relative_path_ambiguity(tmp_path):
    picture(tmp_path / "a.png")
    picture(tmp_path / "lists" / "a.png")
    (tmp_path / "lists" / "images.txt").write_text("a.png\n")
    yaml_path = tmp_path / "set.yaml"
    yaml_path.write_text("train: lists/images.txt\nnames: [a]\n")
    with pytest.raises(DatasetError, match="多解"):
        index(yaml_path)
    (tmp_path / "lists" / "images.txt").write_text("./a.png\n")
    assert len(index(yaml_path).samples) == 1


def test_root_override_and_multiple_yaml_choices(tmp_path):
    root = standard_dataset(tmp_path / "data")
    config = yaml.safe_load((root / "dataset.yml").read_text())
    config["path"] = "/not/on/this/machine"
    (root / "dataset.yml").write_text(yaml.safe_dump(config))
    with pytest.raises(DatasetError, match="根目录"):
        index(root)
    assert len(index(root, root=str(root)).samples) == 1
    (root / "second.yaml").write_text("train: images/train")
    with pytest.raises(DatasetError) as exc:
        index(root)
    assert len(exc.value.candidates) == 2
    assert len(index(root / "dataset.yml", root=str(root)).samples) == 1


def test_pose_requires_shape_or_explicit_override(tmp_path):
    root = standard_dataset(tmp_path, {"a": "0 .5 .5 1 1 .2 .3"})
    with pytest.raises(DatasetError, match="kpt_shape"):
        index(root, "pose")
    assert index(root, "pose", kpt_shape=(1, 2)).samples[0].state == "valid"


def test_linked_image_directory_keeps_parallel_labels(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()
    picture(tmp_path / "external-pictures" / "a.png")
    (root / "images").symlink_to(tmp_path / "external-pictures", target_is_directory=True)
    (root / "labels").mkdir()
    (root / "labels/a.txt").write_text("0 .5 .5 .2 .2")
    (root / "data.yaml").write_text("train: images\nnames: [object]")
    data = index(root)
    assert data.samples[0].state == "valid"
    assert data.samples[0].label == root / "labels/a.txt"


def test_label_changes_during_index_are_reported(tmp_path, monkeypatch):
    from yolo_dataset_viewer import dataset as module

    root = standard_dataset(tmp_path)
    original = module.read_labels

    def modify_after_read(path, *args):
        result = original(path, *args)
        path.write_text("0 .5 .5 .1 .1\n")
        return result

    monkeypatch.setattr(module, "read_labels", modify_after_read)
    with pytest.raises(DatasetError, match="索引期间发生变化"):
        index(root)


@pytest.mark.parametrize(
    "config",
    [
        "train: images\nnames: {1: a}",
        "train: images\nnames: [a]\nnc: 2",
        "train: images\nnc: 0",
        "train: images\nnames: [1]",
        "[train, val]",
        "train: https://example.com/data",
        "train: images\nnc: true",
    ],
)
def test_invalid_yaml_configs(tmp_path, config):
    path = tmp_path / "data.yaml"
    path.write_text(config)
    with pytest.raises(DatasetError):
        resolve_spec(OpenRequest(path=str(path)))


def test_yaml_is_data_only(tmp_path):
    root = standard_dataset(tmp_path)
    path = root / "dataset.yml"
    with path.open("a") as stream:
        stream.write("download: \"__import__('pathlib').Path('SHOULD_NOT_EXIST').touch()\"\n")
    assert len(index(root).samples) == 1
    path.write_text("!!python/object/apply:os.system ['touch SHOULD_NOT_EXIST']")
    with pytest.raises(DatasetError):
        index(root)


def test_stats_filtering_missing_empty_invalid_and_shared_splits(tmp_path):
    root = standard_dataset(
        tmp_path,
        {
            "valid": "0 .5 .5 1 1\n0 .5 .5 .2 .2\n1 .5 .5 .2 .2",
            "empty": "\n",
            "missing": None,
            "bad": "0 .5 .5 .2 .2\n5 broken",
        },
    )
    data = index(root)
    stats = data.stats(data.samples)
    assert stats.images == 4 and stats.instances == 4
    assert stats.states == {"valid": 1, "empty": 1, "missing": 1, "invalid": 1}
    assert [(c.instances, c.images) for c in stats.classes] == [(3, 2), (1, 1)]
    assert len(data.select("train", 1, "valid", "VALID")) == 1
    assert not data.select("val", None, None, "")
    config = yaml.safe_load((root / "dataset.yml").read_text())
    config["test"] = config["train"]
    (root / "dataset.yml").write_text(yaml.safe_dump(config))
    data = index(root)
    assert data.stats(data.samples).images == 8
    assert data.stats(data.samples).unique_images == 4


def test_api_readonly_pagination_detail_and_file_change(tmp_path):
    root = standard_dataset(tmp_path, {"a": "0 .5 .5 1 1", "b": "", "c": None})
    before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    with TestClient(create_app()) as client:
        assert client.get("/").status_code == 200
        assert client.get("/app.js").status_code == 200
        session, ready = wait_ready(client, {"path": str(root)})
        assert ready["state"] == "ready"
        base = f"/api/datasets/{session}"
        page = client.get(f"{base}/samples?offset=1&limit=1").json()
        assert page["total"] == 3 and len(page["items"]) == 1
        assert client.get(f"{base}/samples?limit=0").status_code == 422
        assert client.get(f"{base}/samples?state=missing").json()["total"] == 1
        assert client.get(f"{base}/stats?state=empty").json()["images"] == 1
        detail = client.get(f"{base}/samples/0").json()
        assert (detail["width"], detail["height"]) == (120, 80)
        assert detail["annotations"][0]["box"]["width"] == 1
        assert (
            client.get(f"{base}/samples/0/image?thumbnail=true").headers["content-type"]
            == "image/jpeg"
        )
        assert client.get(f"{base}/samples/0/image").status_code == 200
        assert client.get(f"{base}/samples/999/image").status_code == 404
        assert client.get(f"{base}/samples/-1").status_code == 404
        after = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
        assert before == after
        label_path(root / "images/train/a.png").write_text("0 .5 .5 .2 .2")
        assert client.get(f"{base}/samples/0").status_code == 409
        newer, ready = wait_ready(client, {"path": str(root)})
        assert ready["state"] == "ready" and newer != session
        assert client.get(f"{base}/samples").status_code == 409
        assert (
            client.post(
                "/api/datasets",
                json={"path": str(root)},
                headers={"Origin": "https://other-site.example"},
            ).status_code
            == 403
        )


@pytest.mark.parametrize("orientation", range(1, 9))
def test_exif_dimensions_match_both_image_resolutions(tmp_path, orientation):
    path = tmp_path / "photo.jpg"
    exif = Image.Exif()
    exif[274] = orientation
    Image.new("RGB", (120, 80), "green").save(path, exif=exif)
    path.with_suffix(".txt").write_text("0 .5 .5 .4 .4")
    with TestClient(create_app()) as client:
        session, _ = wait_ready(client, {"path": str(tmp_path)})
        base = f"/api/datasets/{session}/samples/0"
        detail = client.get(base).json()
        expected = (80, 120) if orientation in (5, 6, 7, 8) else (120, 80)
        assert (detail["width"], detail["height"]) == expected
        for query in ("", "?thumbnail=true"):
            response = client.get(f"{base}/image{query}")
            assert response.status_code == 200
            with Image.open(BytesIO(response.content)) as rendered:
                assert rendered.size == expected
                assert rendered.getexif().get(274, 1) == 1


def test_corrupt_images_are_lazy_errors_not_index_failures(tmp_path):
    (tmp_path / "broken.jpg").write_bytes(b"not an image")
    with TestClient(create_app()) as client:
        session, status = wait_ready(client, {"path": str(tmp_path)})
        assert status["state"] == "ready"
        base = f"/api/datasets/{session}/samples/0"
        assert client.get(base).json()["image_error"]
        assert client.get(f"{base}/image").status_code == 422


def test_ten_thousand_sample_index_never_decodes_images(tmp_path, monkeypatch):
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    for i in range(10000):
        (images / f"{i:05d}.jpg").touch()
        (labels / f"{i:05d}.txt").write_text("0 .5 .5 .2 .2\n")

    def forbidden(*args, **kwargs):
        pytest.fail("Indexing decoded an image")

    monkeypatch.setattr(Image, "open", forbidden)
    data = index(tmp_path)
    assert len(data.samples) == 10000
    assert data.stats(data.samples).instances == 10000
    assert not hasattr(data.samples[0], "annotations")


def test_switch_cancels_old_index_and_cache_is_bounded(tmp_path, monkeypatch):
    from yolo_dataset_viewer import service as module

    root = standard_dataset(tmp_path)
    original = module.build_index
    started = Event()
    cancelled = Event()
    calls = 0

    def blocked(request, cancel, progress):
        nonlocal calls
        calls += 1
        if calls == 1:
            started.set()
            assert cancel.wait(3)
            cancelled.set()
        return original(request, cancel, progress)

    monkeypatch.setattr(module, "build_index", blocked)
    service = ViewerService(cache_bytes=1500)
    try:
        first = service.open(OpenRequest(path=str(root)))
        assert started.wait(3)
        second = service.open(OpenRequest(path=str(root)))
        assert cancelled.wait(3)
        for _ in range(300):
            if service.status(second.id).state == "ready":
                break
            time.sleep(0.01)
        assert service.status(second.id).state == "ready"
        assert first.id != second.id
        first_image = service.image(second.id, 0, True)
        assert service.image(second.id, 0, True) == first_image
        assert service._cache_size <= 1500
        service.open(OpenRequest(path=str(root)))
        assert service._cache_size == 0
    finally:
        service.close()


def test_import_is_independent_of_pipeline_and_gpu_libraries():
    command = (
        "import sys; from yolo_dataset_viewer.app import create_app; create_app(); "
        "assert not any(m.split('.')[0] in {'torch','ultralytics','cv2','video_dataset_pipeline'} for m in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", command],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    )
    assert result.returncode == 0, result.stderr
