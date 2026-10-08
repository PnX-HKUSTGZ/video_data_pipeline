"""Pose observation contracts, especially reference scope and full-image coordinates."""

import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from vlm_correct_pipeline.config import Config, PoseConfig
from vlm_correct_pipeline.models import KeypointState, PoseBox
from vlm_correct_pipeline.prompts import references, system_message
from vlm_correct_pipeline.render import POINT_COLORS, focus_region, observe, overlay


@pytest.fixture
def pose() -> PoseConfig:
    return PoseConfig.model_validate(
        {
            "kpt_shape": [3, 3],
            "kpt_names": ["TL", "TR", "ring"],
            "class_keypoint_indices": {0: [0, 1], 1: [2]},
            "point_definitions": {
                "TL": "Object-fixed top-left marker corner",
                "TR": "Object-fixed top-right marker corner",
                "ring": "Geometric reference, never infer its hidden location",
            },
            "skeleton": [[0, 1]],
        }
    )


def pose_box() -> PoseBox:
    return PoseBox(
        box_id="object1",
        class_id=0,
        bbox=[0.1, 0.1, 0.9, 0.9],
        keypoints=[
            KeypointState(x=0.25, y=0.5, visibility=2, reviewed=True),
            KeypointState(x=0.75, y=0.5, visibility=2, reviewed=True),
            KeypointState(x=0.0, y=0.0, visibility=0, reviewed=True),
        ],
    )


def test_points_only_pixels_never_depend_on_source_box(pose: PoseConfig) -> None:
    first = pose_box()
    second = first.model_copy(update={"bbox": [0.3, 0.6, 0.6, 0.9]})
    with (
        Image.new("RGB", (400, 400), "white") as image,
        overlay(image, [first], {0: "panel"}, pose=pose, show_boxes=False) as before,
        overlay(image, [second], {0: "panel"}, pose=pose, show_boxes=False) as after,
    ):
        assert before.tobytes() == after.tobytes()
        assert before.getpixel((40, 100)) == (255, 255, 255)
        assert before.getpixel((100, 200)) == POINT_COLORS[0]
        assert before.getpixel((200, 200)) == (190, 190, 190)


def test_missing_and_pending_points_do_not_form_visible_skeleton(pose: PoseConfig) -> None:
    box = pose_box()
    box.keypoints[1] = KeypointState(x=0.75, y=0.5, visibility=1, reviewed=False)
    with Image.new("RGB", (400, 400), "white") as image:
        with overlay(image, [box], {0: "panel"}, pose=pose, show_boxes=False) as rendered:
            assert rendered.getpixel((300, 200)) == (255, 255, 255)
            assert rendered.getpixel((200, 200)) == (255, 255, 255)
            assert rendered.getpixel((303, 200)) == (255, 165, 0)
        box.keypoints = [KeypointState(x=0.0, y=0.0, visibility=0, reviewed=True) for _ in range(3)]
        with overlay(image, [box], {0: "panel"}, pose=pose, show_boxes=False) as rendered:
            assert rendered.tobytes() == image.tobytes()


def test_skeleton_respects_class_slot_membership(pose: PoseConfig) -> None:
    box = pose_box().model_copy(update={"class_id": 1})
    with (
        Image.new("RGB", (400, 400), "white") as image,
        overlay(image, [box], {1: "other"}, pose=pose, show_boxes=False) as rendered,
    ):
        assert rendered.tobytes() == image.tobytes()


def test_focus_includes_visible_points_without_changing_geometry(pose: PoseConfig) -> None:
    box = pose_box().model_copy(update={"bbox": [0.4, 0.4, 0.6, 0.6]})
    box.keypoints[0] = KeypointState(x=0.0, y=0.9, visibility=2, reviewed=True)
    box.keypoints[1] = KeypointState(x=1.0, y=0.0, visibility=1, reviewed=False)
    before = box.model_dump()
    assert focus_region(box, pose) == pytest.approx([0.0, 0.35, 0.66, 0.95])
    assert box.model_dump() == before
    with Image.new("RGB", (301, 101), "white") as image:
        _, metadata = observe(
            image, [box], {0: "panel"}, 256, pose=pose, region=focus_region(box, pose)
        )
    assert metadata["crop_pixels"] == [0, 35, 199, 96]
    assert metadata["crop_bbox"] == pytest.approx([0.0, 35 / 101, 199 / 301, 96 / 101])


def test_crop_keeps_short_point_labels_within_viewport(pose: PoseConfig, monkeypatch) -> None:
    box = pose_box().model_copy(update={"box_id": "box_0001"})
    box.keypoints[0] = KeypointState(x=0.45, y=0.45, visibility=2, reviewed=False)
    box.keypoints[1] = KeypointState(x=0.55, y=0.55, visibility=2, reviewed=True)
    labels = []
    original = ImageDraw.ImageDraw.text

    def capture(draw, xy, text, *args, **kwargs):
        if text.startswith("b1/"):
            labels.append((text, draw.textbbox(xy, text, font=kwargs["font"])))
        return original(draw, xy, text, *args, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", capture)
    with Image.new("RGB", (400, 400), "white") as image:
        content, _ = observe(
            image,
            [box],
            {0: "panel"},
            256,
            pose=pose,
            show_boxes=False,
            region=[0.4, 0.4, 0.6, 0.6],
        )
    assert {text for text, _ in labels} == {"b1/0:TL?", "b1/1:TR"}
    for _, (x1, y1, x2, y2) in labels:
        assert 160 <= x1 < x2 <= 240
        assert 160 <= y1 < y2 <= 240
    assert "A ? means unreviewed" in str(content[2]["text"])


def reference_config(tmp_path: Path, pose: PoseConfig, scope: str) -> Config:
    image_path = tmp_path / "reference.png"
    with Image.new("RGB", (400, 400), "white") as image:
        image.save(image_path)
    return Config.model_validate(
        {
            "api": {"model": "test-model"},
            "dataset": {"source": ".", "input_task": "pose"},
            "pose": pose.model_dump(),
            "references": [
                {
                    "image": str(image_path),
                    "label": str(tmp_path / "reference.txt"),
                    "scope": scope,
                }
            ],
            "annotation_rules": "Use visible panel boundaries.",
            "output": {"directory": str(tmp_path / "output")},
            "image_format": "png",
        }
    )


def test_keypoint_reference_withholds_source_boxes_in_text_and_pixels(
    tmp_path: Path, pose: PoseConfig
) -> None:
    config = reference_config(tmp_path, pose, "keypoints")
    label = tmp_path / "reference.txt"
    points = "0.25 0.5 2 0.75 0.5 2 0 0 0\n"
    label.write_text("0 0.5 0.5 0.8 0.8 " + points)
    before, before_hashes = references(config, tmp_path, {0: "panel", 1: "other"})
    label.write_text("0 0.5 0.5 0.6 0.6 " + points)
    after, after_hashes = references(config, tmp_path, {0: "panel", 1: "other"})
    assert before == after
    assert before_hashes != after_hashes
    content = before[0]["content"]
    assert isinstance(content, list) and isinstance(content[0], dict)
    header = json.loads(str(content[0]["text"]))
    assert header["scope"] == "keypoints"
    assert "boxes" not in header and "confirmed_empty" not in header
    assert "bbox" not in header["instances"][0]
    assert header["crop_bbox"] == [0, 0, 1, 1]
    assert (
        len([part for part in content if isinstance(part, dict) and part["type"] == "input_image"])
        == 2
    )


def test_boxes_reference_remains_detection_only_in_pose_run(
    tmp_path: Path, pose: PoseConfig
) -> None:
    config = reference_config(tmp_path, pose, "boxes")
    (tmp_path / "reference.txt").write_text("0 .5 .5 .6 .6\n")
    messages, _ = references(config, tmp_path, {0: "panel", 1: "other"})
    content = messages[0]["content"]
    assert isinstance(content, list) and isinstance(content[0], dict)
    header = json.loads(str(content[0]["text"]))
    assert header["scope"] == "boxes" and "keypoints" not in header["boxes"][0]
    prompt = str(system_message(config, {0: "panel", 1: "other"})["content"])
    assert "YOLO pose dataset" in prompt
    assert "Object-fixed top-left marker corner" in prompt
    assert "Do not output v=1" in prompt
    assert "marker panel, excluding the protruding pole" in prompt
