"""Offline pose ingestion, Responses feedback, recovery, and portable export checks."""

import json
from pathlib import Path
from threading import Event

import pytest
import yaml
from test_vlm_pipeline import ScriptedBackend, output_payload, picture

from vlm_correct_pipeline.config import Config
from vlm_correct_pipeline.data import discover, file_hash, load_boxes, object_hash
from vlm_correct_pipeline.models import PoseBox
from vlm_correct_pipeline.pipeline import run
from yolo_dataset_viewer.dataset import build_index
from yolo_dataset_viewer.models import OpenRequest

NAMES = {0: "marker", 1: "housing"}
ACTIVE = {0: list(range(4)), 1: list(range(4, 12))}


def pose_row(class_id, replacements=None):
    """Produce conventional source labels without using the implementation serializer."""
    replacements = replacements or {}
    fields = [str(class_id), "0.5", "0.5", "0.6", "0.6"]
    for index in range(12):
        point = replacements.get(
            index, (0.3 + index / 100, 0.4, 2) if index in ACTIVE[class_id] else (0, 0, 0)
        )
        fields.extend(map(str, point))
    return " ".join(fields) + "\n"


def review_action(class_id, *, overrides=None):
    overrides = overrides or {}
    return (
        "set_keypoints",
        {
            "box_id": f"box_{class_id + 1:04d}",
            "updates": [
                {"index": index, "x": 0.3 + index / 100, "y": 0.4, "visibility": 2}
                | overrides.get(index, {})
                for index in ACTIVE[class_id]
            ],
        },
    )


def completed_actions():
    return [
        review_action(0),
        review_action(1),
        ("finish_annotation", {"reason": "Checked all points"}),
    ]


def target_header(history):
    for message in history:
        content = message.get("content")
        if not isinstance(content, list) or not content or content[0].get("type") != "input_text":
            continue
        try:
            payload = json.loads(content[0]["text"])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("role") == "editable_target":
            return payload
    return None


@pytest.fixture
def pose_setup(tmp_path):
    source = tmp_path / "input"
    picture(source / "images/train/one.png", size=(400, 200))
    label = source / "labels/train/one.txt"
    label.parent.mkdir(parents=True)
    label.write_text(pose_row(0) + pose_row(1))
    point_names = [f"point_{index}" for index in range(12)]
    flip = [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10]
    metadata = {
        "path": ".",
        "train": "images/train",
        "names": NAMES,
        "kpt_shape": [12, 3],
        "kpt_names": {class_id: point_names for class_id in NAMES},
        "flip_idx": flip,
        "skeleton": [[0, 1], [4, 5]],
    }
    (source / "data.yaml").write_text(yaml.safe_dump(metadata))
    reference = picture(tmp_path / "references/gold.png", size=(400, 200))
    reference.with_suffix(".txt").write_text(pose_row(0) + pose_row(1))
    config = Config.model_validate(
        {
            "api": {"model": "gpt-6-astra"},
            "dataset": {"source": str(source / "data.yaml"), "input_task": "pose"},
            "pose": {
                "kpt_shape": [12, 3],
                "kpt_names": point_names,
                "class_keypoint_indices": ACTIVE,
                "point_definitions": {
                    name: f"Visible fixture landmark {name}" for name in point_names
                },
                "flip_idx": flip,
                "skeleton": [[0, 1], [4, 5]],
            },
            "references": [{"image": str(reference), "label": str(reference.with_suffix(".txt"))}],
            "annotation_rules": "Locate visible landmarks only; never infer hidden points.",
            "run": {"image_concurrency": 1, "max_retries": 0},
            "output": {"directory": str(tmp_path / "output")},
        }
    )
    return config, tmp_path, source


def test_source_points_are_drafts_and_normalization_is_auditable(pose_setup):
    config, base, source = pose_setup
    label = source / "labels/train/one.txt"
    label.write_text(pose_row(0, {1: (0.31, 0.4, 1), 2: (0.5, 0.6, 0)}) + pose_row(1))
    source_hash = file_hash(label)
    actions = [
        ("finish_annotation", {"reason": "Attempt to reuse source points"}),
        review_action(0, overrides={1: {"x": 0, "y": 0, "visibility": 0}}),
        ("finish_annotation", {"reason": "Second instance still pending"}),
        review_action(1),
        ("finish_annotation", {"reason": "Reviewed both instances"}),
    ]
    backend = ScriptedBackend(actions)
    report = run(config, base, backend, endpoint_identity="local")
    assert report["counts"] == {"completed": 1}
    initial = next(payload for history in backend.histories if (payload := target_header(history)))
    for box in initial["boxes"]:
        for index, point in enumerate(box["keypoints"]):
            assert point["reviewed"] is (index not in ACTIVE[box["class_id"]])
    assert initial["boxes"][0]["keypoints"][1]["visibility"] == 1
    assert initial["boxes"][0]["keypoints"][2] == {
        "x": 0,
        "y": 0,
        "visibility": 0,
        "reviewed": False,
    }
    errors = [
        output_payload(item)
        for item in backend.histories[-1]
        if item.get("type") == "function_call_output" and not output_payload(item)["ok"]
    ]
    assert len(errors) == 2 and all("Unreviewed" in error["error"]["message"] for error in errors)
    sample = report["samples"][0]
    assert sample["boxes"][0]["keypoints"][1]["visibility"] == 0
    events_path = Path(config.output.directory) / "events" / f"{sample['sample_id']}.jsonl"
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    source_event = next(event for event in events if event["event"] == "source_labels")
    assert source_event["rounding"] == [
        {
            "line": 1,
            "point_index": 2,
            "operation": "normalize_absent_point",
            "before": [0.5, 0.6, 0],
            "after": [0, 0, 0],
        }
    ]
    assert file_hash(label) == source_hash


@pytest.mark.parametrize("visibility", [0, 1, 2])
def test_nonzero_inactive_slots_are_rejected_without_target_calls(pose_setup, visibility):
    config, base, source = pose_setup
    (source / "labels/train/one.txt").write_text(pose_row(0, {4: (0.4, 0.4, visibility)}))
    backend = ScriptedBackend()
    report = run(config, base, backend, endpoint_identity="local")
    assert report["counts"] == {"invalid_input": 1}
    assert backend.target_calls == 0
    assert not (Path(config.output.directory) / "dataset").exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("kpt_shape", [11, 3]),
        ("kpt_names", {0: [f"wrong_{index}" for index in range(12)], 1: ["wrong"] * 12}),
        ("flip_idx", list(range(12))),
        ("skeleton", [[0, 2]]),
    ],
)
def test_conflicting_source_pose_metadata_blocks_all_api_calls(pose_setup, field, value):
    config, base, source = pose_setup
    path = source / "data.yaml"
    metadata = yaml.safe_load(path.read_text())
    metadata[field] = value
    path.write_text(yaml.safe_dump(metadata))
    backend = ScriptedBackend()
    with pytest.raises(ValueError, match="disagrees"):
        run(config, base, backend, endpoint_identity="local")
    assert backend.probe_calls == backend.target_calls == 0


def test_unlabeled_image_directory_uses_configured_pose_metadata(pose_setup):
    config, base, _ = pose_setup
    unlabelled = base / "unlabelled"
    picture(unlabelled / "one.png")
    config.dataset.source = str(unlabelled)
    config.dataset.names = NAMES
    data = discover(config, base)
    assert data.kpt_shape == (12, 3) and data.samples[0].splits == ["train"]
    assert (
        load_boxes(
            data.samples[0].label, NAMES, task="pose", kpt_shape=data.kpt_shape, pose=config.pose
        )
        == []
    )
    backend = ScriptedBackend(
        [
            ("add_box", {"class_id": 0, "bbox": [0.2, 0.2, 0.8, 0.8]}),
            review_action(0),
            ("finish_annotation", {"reason": "Annotated new target"}),
        ]
    )
    report = run(config, base, backend, endpoint_identity="local")
    assert report["counts"] == {"completed": 1}
    exported_label = Path(config.output.directory) / "dataset" / report["mapping"][0]["label"]
    assert len(exported_label.read_text().split()) == 41


def test_pose_edits_return_full_image_and_focus_crop_before_next_call(pose_setup):
    config, base, _ = pose_setup
    backend = ScriptedBackend(completed_actions())
    run(config, base, backend, endpoint_identity="local")
    outputs = [item for item in backend.histories[-1] if item.get("type") == "function_call_output"]
    assert len(outputs) == 2
    for output in outputs:
        payload = output_payload(output)
        images = [item for item in output["output"] if item["type"] == "input_image"]
        assert len(images) == 4
        assert payload["crop_bbox"] == [0, 0, 1, 1]
        assert payload["full_image_size"] == [400, 200]
        crop = payload["focus_region"]
        assert crop["full_image_size"] == [400, 200]
        assert 0 < crop["crop_bbox"][0] < 0.2
        assert 0.8 < crop["crop_bbox"][2] < 1
        assert crop["editing_coordinate_system"] == payload["editing_coordinate_system"]
        assert crop["crop_bbox"] == [
            crop["crop_pixels"][0] / 400,
            crop["crop_pixels"][1] / 200,
            crop["crop_pixels"][2] / 400,
            crop["crop_pixels"][3] / 200,
        ]
    first = output_payload(outputs[0])
    assert first["boxes"][0]["keypoints"][0]["reviewed"] is True
    assert first["boxes"][1]["keypoints"][4]["reviewed"] is False


def test_pose_export_viewer_roundtrip_and_network_free_resume(pose_setup):
    config, base, source = pose_setup
    metadata_path = source / "data.yaml"
    source_metadata = yaml.safe_load(metadata_path.read_text())
    source_metadata["val"] = source_metadata["train"]
    metadata_path.write_text(yaml.safe_dump(source_metadata))
    original_label = source / "labels/train/one.txt"
    original_image = source / "images/train/one.png"
    original_hashes = (file_hash(original_label), file_hash(original_image))
    report = run(config, base, ScriptedBackend(completed_actions()), endpoint_identity="local")
    root = Path(config.output.directory)
    output = root / "dataset"
    assert report["input_task"] == report["output_task"] == "pose"
    assert report["exported_entries"] == 2 and report["unique_images"] == 1
    assert {row["split"] for row in report["mapping"]} == {"train", "val"}
    for row in report["mapping"]:
        fields = [line.split() for line in (output / row["label"]).read_text().splitlines()]
        assert len(fields) == 2 and all(len(line) == 41 for line in fields)
        for class_id, line in enumerate(fields):
            for index in range(12):
                values = list(map(float, line[5 + index * 3 : 8 + index * 3]))
                assert values[2] == (2 if index in ACTIVE[class_id] else 0)
                if index not in ACTIVE[class_id]:
                    assert values == [0, 0, 0]
        copied = output / row["image"]
        assert file_hash(copied) == original_hashes[1]
        assert copied.stat().st_ino != original_image.stat().st_ino
    viewer = build_index(
        OpenRequest(path=str(output / "data.yaml"), task="pose"), Event(), lambda *_: None
    )
    assert len(viewer.samples) == 2 and viewer.spec.kpt_shape == (12, 3)
    assert viewer.spec.names == NAMES
    assert viewer.spec.config_data["kpt_names"] == source_metadata["kpt_names"]
    assert viewer.spec.config_data["flip_idx"] == source_metadata["flip_idx"]
    assert viewer.spec.config_data["skeleton"] == source_metadata["skeleton"]
    assert all(sample.state == "valid" for sample in viewer.samples)
    assert (file_hash(original_label), file_hash(original_image)) == original_hashes
    assert json.loads((root / "run.json").read_text())["format_version"] == 2
    backend = ScriptedBackend()
    resumed = run(config, base, backend, endpoint_identity="local")
    assert resumed["fingerprint"] == report["fingerprint"]
    assert resumed["mapping"] == report["mapping"]
    assert resumed["samples"] == report["samples"]
    assert backend.target_calls == backend.probe_calls == 0
    assert not list((root / ".tmp").iterdir())


def test_interrupted_pose_restarts_from_unreviewed_source_points(pose_setup):
    config, base, _ = pose_setup

    class InterruptBackend(ScriptedBackend):
        def respond(self, history, tools, *, forced_tool=None):
            if not forced_tool and self.target_calls:
                raise KeyboardInterrupt
            return super().respond(history, tools, forced_tool=forced_tool)

    interrupted = InterruptBackend(
        [
            (
                "set_keypoints",
                {
                    "box_id": "box_0001",
                    "updates": [{"index": 0, "x": 0.9, "y": 0.9, "visibility": 2}],
                },
            )
        ]
    )
    with pytest.raises(KeyboardInterrupt):
        run(config, base, interrupted, endpoint_identity="local")
    root = Path(config.output.directory)
    checkpoint = json.loads(next((root / "samples").glob("*.json")).read_text())
    assert checkpoint["status"] == "interrupted"
    assert checkpoint["boxes"][0]["keypoints"][0]["x"] == 0.9
    assert not (root / "dataset").exists()
    backend = ScriptedBackend(completed_actions())
    report = run(config, base, backend, endpoint_identity="local")
    initial = next(payload for history in backend.histories if (payload := target_header(history)))
    assert initial["boxes"][0]["keypoints"][0]["x"] == 0.3
    assert initial["boxes"][0]["keypoints"][0]["reviewed"] is False
    assert report["samples"][0]["attempt"] == 2
    assert report["counts"] == {"completed": 1}
    saved = load_boxes(
        root / "dataset" / report["mapping"][0]["label"],
        NAMES,
        task="pose",
        kpt_shape=(12, 3),
        pose=config.pose,
        reference=True,
    )
    assert all(isinstance(box, PoseBox) for box in saved)


def test_changed_point_semantics_cannot_resume_old_pose_results(pose_setup):
    config, base, _ = pose_setup
    report = run(config, base, ScriptedBackend(completed_actions()), endpoint_identity="local")
    config.pose.point_definitions["point_0"] = "Different physical feature"
    backend = ScriptedBackend()
    with pytest.raises(ValueError, match="Inputs/configuration changed"):
        run(config, base, backend, endpoint_identity="local")
    assert backend.target_calls == backend.probe_calls == 0
    saved = json.loads((Path(config.output.directory) / "run.json").read_text())
    assert saved["fingerprint"] == report["fingerprint"]


def test_legacy_detection_identity_and_checkpoint_resume_without_api(pose_setup):
    pose_config, base, source = pose_setup
    reference = pose_config.references[0]
    Path(reference.label).write_text("0 .5 .5 .6 .6\n1 .5 .5 .6 .6\n")
    (source / "labels/train/one.txt").write_text("0 .5 .5 .6 .6\n1 .5 .5 .6 .6\n")
    (source / "data.yaml").write_text(
        yaml.safe_dump({"path": ".", "train": "images/train", "names": NAMES})
    )
    # Historical persisted config: deliberately has neither pose nor reference scope.
    legacy_config = {
        "api": {
            "env_file": ".env",
            "base_url_env": "BASE_URL",
            "api_key_env": "API_KEY",
            "model": "gpt-6-astra",
            "timeout_seconds": 120.0,
            "stream": False,
            "max_output_tokens": 8192,
        },
        "dataset": {
            "source": str(source / "data.yaml"),
            "names": None,
            "input_task": "detect",
            "selection_file": None,
            "source_rounding_tolerance": 0.0,
        },
        "references": [{"image": reference.image, "label": reference.label, "description": ""}],
        "annotation_rules": "Review visible boxes.",
        "run": {
            "image_concurrency": 1,
            "max_model_calls": 30,
            "max_retries": 0,
            "max_consecutive_api_errors": 3,
        },
        "output": {"directory": str(base / "legacy-output")},
        "image_max_edge": 2048,
        "image_format": "jpeg",
        "jpeg_quality": 95,
        "coordinate_grid": False,
    }
    config = Config.model_validate(legacy_config)
    data = discover(config, base)
    sample = data.samples[0]
    identity = {
        "format_version": 1,
        "config": legacy_config,
        "base": str(base),
        "endpoint": "local",
        "names": {str(key): value for key, value in NAMES.items()},
        "references": {
            reference.image: file_hash(Path(reference.image)),
            reference.label: file_hash(Path(reference.label)),
        },
        "source_files": data.source_files,
        "samples": [[sample.sample_id, sample.fingerprint, sample.splits]],
    }
    old_fingerprint = object_hash(identity)
    root = Path(config.output.directory)
    (root / "samples").mkdir(parents=True)
    (root / "run.json").write_text(
        json.dumps({"status": "completed", "fingerprint": old_fingerprint, **identity})
    )
    old_checkpoint = {
        "sample_id": sample.sample_id,
        "fingerprint": sample.fingerprint,
        "attempt": 1,
        "status": "completed",
        "revision": 0,
        "boxes": [
            {
                "box_id": f"box_{class_id + 1:04d}",
                "class_id": class_id,
                "bbox": [0.2, 0.2, 0.8, 0.8],
            }
            for class_id in NAMES
        ],
        "reason": "Previously reviewed",
        "discard_category": None,
        "usage": {"requests": 1, "input_tokens": 10, "output_tokens": 5},
    }
    (root / "samples" / f"{sample.sample_id}.json").write_text(json.dumps(old_checkpoint))
    backend = ScriptedBackend()
    report = run(config, base, backend, endpoint_identity="local")
    assert report["fingerprint"] == old_fingerprint
    assert report["samples"] == [old_checkpoint]
    assert report["output_task"] == "detect" and report["exported_entries"] == 1
    assert backend.target_calls == backend.probe_calls == 0
    exported = root / "dataset" / report["mapping"][0]["label"]
    assert all(len(line.split()) == 5 for line in exported.read_text().splitlines())
