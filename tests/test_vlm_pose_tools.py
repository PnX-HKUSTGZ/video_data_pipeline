"""Pose state and tool contracts, independent of network and image rendering."""

import copy
import json

import pytest
from pydantic import ValidationError

from vlm_correct_pipeline.config import Config, PoseConfig, identity_config
from vlm_correct_pipeline.models import (
    Box,
    KeypointState,
    PoseBox,
    SampleResult,
    initial_keypoints,
    label_text,
    validate_pose_boxes,
)
from vlm_correct_pipeline.tools import AnnotationTools, tool_definitions


@pytest.fixture
def pose():
    names = [f"point_{index}" for index in range(12)]
    return PoseConfig(
        kpt_shape=[12, 3],
        kpt_names=names,
        class_keypoint_indices={0: [0, 1, 2, 3], 1: list(range(12))},
        point_definitions={name: f"Visible landmark {name}" for name in names},
    )


def create_tools(pose):
    state = SampleResult(sample_id="sample", fingerprint="fp", status="running")
    tools = AnnotationTools(state, {0: "marker", 1: "housing"}, pose)
    result = tools.execute("add_box", json.dumps({"class_id": 0, "bbox": [0.2, 0.2, 0.8, 0.8]}))
    assert result.payload["ok"] and result.focus_box_id == "box_0001"
    return state, tools


def review_points(tools, *, visible=True):
    return tools.execute(
        "set_keypoints",
        json.dumps(
            {
                "box_id": "box_0001",
                "updates": [
                    {
                        "index": index,
                        "x": 0.3 + index * 0.1 if visible else 0,
                        "y": 0.4 if visible else 0,
                        "visibility": 2 if visible else 0,
                    }
                    for index in range(4)
                ],
            }
        ),
    )


@pytest.mark.parametrize(
    "change",
    [
        {"kpt_shape": [12, 2]},
        {"kpt_shape": [0, 3]},
        {"kpt_names": ["duplicate"] * 12},
        {"point_definitions": {"point_0": "only one"}},
        {"class_keypoint_indices": {0: [0, 0]}},
        {"class_keypoint_indices": {0: [12]}},
        {"class_keypoint_indices": {0: []}},
        {"flip_idx": [1, 2, 0, *range(3, 12)]},
        {"flip_idx": [0] * 12},
        {"skeleton": [[0, 12]]},
        {"skeleton": [[0, 0]]},
    ],
)
def test_pose_schema_rejects_ambiguous_metadata(pose, change):
    data = pose.model_dump()
    data.update(change)
    with pytest.raises(ValidationError):
        PoseConfig.model_validate(data)


def test_config_modes_and_legacy_identity(pose):
    values = {
        "api": {"model": "gpt-6-astra"},
        "dataset": {"source": "input/data.yaml"},
        "references": [{"image": "gold.jpg", "label": "gold.txt"}],
        "annotation_rules": "Use visible landmarks.",
        "output": {"directory": "output"},
    }
    config = Config.model_validate(values)
    identity = identity_config(config)
    expected = config.model_dump(mode="json")
    expected.pop("pose")
    expected["references"][0].pop("scope")
    assert identity == expected
    assert "pose" not in identity
    with pytest.raises(ValidationError, match="required exactly"):
        Config.model_validate({**values, "pose": pose.model_dump()})
    values["dataset"]["input_task"] = "pose"
    with pytest.raises(ValidationError, match="required exactly"):
        Config.model_validate(values)
    config = Config.model_validate({**values, "pose": pose.model_dump()})
    assert identity_config(config) == config.model_dump(mode="json")


def test_pending_is_not_confirmed_absence_and_no_hidden_point_inference(pose):
    state, tools = create_tools(pose)
    box = state.boxes[0]
    assert isinstance(box, PoseBox)
    assert not any(point.reviewed for point in box.keypoints[:4])
    assert all(point.reviewed for point in box.keypoints[4:])
    failure = tools.execute("finish_annotation", '{"reason":"done"}')
    assert not failure.payload["ok"] and state.status == "running"
    assert review_points(tools, visible=False).payload["ok"]
    failure = tools.execute("finish_annotation", '{"reason":"done"}')
    assert not failure.payload["ok"] and "visible" in str(failure.payload["error"])
    assert review_points(tools).payload["ok"]
    assert tools.execute("finish_annotation", '{"reason":"reviewed every point"}').payload["ok"]
    assert state.status == "completed"
    assert len(label_text(state.boxes, pose).split()) == 41


@pytest.mark.parametrize(
    "updates",
    [
        [{"index": 0, "x": 0.3, "y": 0.4, "visibility": 2}] * 2,
        [{"index": 4, "x": 0.3, "y": 0.4, "visibility": 2}],
        [{"index": 12, "x": 0.3, "y": 0.4, "visibility": 2}],
        [{"index": 0, "x": 0.3, "y": 0.4, "visibility": 1}],
        [{"index": 0, "x": 0.3, "y": 0.4, "visibility": 0}],
        [{"index": 0, "x": 0, "y": 0, "visibility": False}],
        [{"index": 0, "x": float("nan"), "y": 0.4, "visibility": 2}],
        [{"index": 0, "x": -0.1, "y": 0.4, "visibility": 2}],
        [],
    ],
)
def test_invalid_keypoint_batches_leave_draft_unchanged(pose, updates):
    state, tools = create_tools(pose)
    before = state.model_dump()
    result = tools.execute("set_keypoints", json.dumps({"box_id": "box_0001", "updates": updates}))
    assert not result.payload["ok"]
    assert state.model_dump() == before


def test_batch_rejection_is_atomic_after_a_valid_first_item(pose):
    state, tools = create_tools(pose)
    before = state.model_dump()
    result = tools.execute(
        "set_keypoints",
        json.dumps(
            {
                "box_id": "box_0001",
                "updates": [
                    {"index": 0, "x": 0.3, "y": 0.4, "visibility": 2},
                    {"index": 4, "x": 0, "y": 0, "visibility": 0},
                ],
            }
        ),
    )
    assert not result.payload["ok"] and state.model_dump() == before


def test_move_box_keeps_points_and_class_change_resets_applicability(pose):
    state, tools = create_tools(pose)
    assert review_points(tools).payload["ok"]
    before = copy.deepcopy(state.boxes[0].keypoints)
    result = tools.execute(
        "update_box",
        json.dumps({"box_id": "box_0001", "class_id": 0, "bbox": [0.6, 0.6, 0.9, 0.9]}),
    )
    assert result.focus_box_id == "box_0001"
    assert state.boxes[0].keypoints == before
    assert len(result.payload["warnings"]) == 4
    result = tools.execute(
        "update_box",
        json.dumps({"box_id": "box_0001", "class_id": 1, "bbox": [0.6, 0.6, 0.9, 0.9]}),
    )
    assert result.payload["ok"]
    assert all(not point.reviewed for point in state.boxes[0].keypoints)
    assert all(point.visibility == 0 for point in state.boxes[0].keypoints)


def test_input_visibility_one_requires_explicit_correction(pose):
    points = initial_keypoints(pose, 0)
    points[0] = KeypointState(x=0.3, y=0.4, visibility=1, reviewed=False)
    box = PoseBox(box_id="box_0001", class_id=0, bbox=[0.2, 0.2, 0.8, 0.8], keypoints=points)
    validate_pose_boxes([box], pose)
    with pytest.raises(ValueError, match="Unreviewed"):
        label_text([box], pose)
    for point in box.keypoints:
        point.reviewed = True
    with pytest.raises(ValueError, match="visibility-one"):
        label_text([box], pose)


def test_pose_and_detection_checkpoints_keep_their_exact_shapes(pose):
    state, tools = create_tools(pose)
    assert review_points(tools).payload["ok"]
    restored = SampleResult.model_validate_json(state.model_dump_json())
    assert restored == state and isinstance(restored.boxes[0], PoseBox)
    box = Box(box_id="box_0001", class_id=0, bbox=[0, 0, 1, 1])
    assert box.model_dump() == {"box_id": "box_0001", "class_id": 0, "bbox": [0, 0, 1, 1]}
    assert label_text([box]) == "0 0.5 0.5 1 1\n"
    state.boxes = [box]
    restored = SampleResult.model_validate_json(state.model_dump_json())
    assert type(restored.boxes[0]) is Box
    assert "set_keypoints" not in {tool["name"] for tool in tool_definitions()}
    assert "set_keypoints" in {tool["name"] for tool in tool_definitions(pose)}
    assert (
        AnnotationTools(state, {0: "target"}).execute("set_keypoints", "{}").payload["ok"] is False
    )
    for tool in tool_definitions(pose):
        parameters = tool["parameters"]
        assert parameters["additionalProperties"] is False
        assert set(parameters["properties"]) == set(parameters["required"])


def test_empty_is_an_explicit_terminal_decision_and_delete_removes_points(pose):
    state, tools = create_tools(pose)
    assert tools.execute("delete_box", '{"box_id":"box_0001"}').payload["ok"]
    assert state.boxes == [] and state.status == "running"
    assert not tools.execute("finish_annotation", '{"reason":"done"}').payload["ok"]
    assert tools.execute("mark_empty", '{"reason":"No target present"}').payload["ok"]
    assert state.status == "empty" and label_text(state.boxes, pose) == ""
