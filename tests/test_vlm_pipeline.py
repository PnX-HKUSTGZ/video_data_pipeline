"""Offline behavioral coverage for the VLM loop, recovery, and portable exports."""

import copy
import json
import subprocess
import sys
from pathlib import Path
from threading import Event

import httpx
import pytest
import yaml

pytest.importorskip("openai")
pytest.importorskip("dotenv")

from openai import OpenAI
from PIL import Image

from vlm_correct_pipeline.client import APIProblem, FunctionCall, ResponsesBackend, Turn
from vlm_correct_pipeline.config import APIConfig, Config, Credentials, credentials, load_config
from vlm_correct_pipeline.data import discover, load_boxes
from vlm_correct_pipeline.engine import Budget, CallLimit, annotate, probe
from vlm_correct_pipeline.models import Box, SampleResult, Usage, label_text
from vlm_correct_pipeline.pipeline import run
from vlm_correct_pipeline.render import load_image, observe
from vlm_correct_pipeline.tools import AnnotationTools, tool_definitions
from yolo_dataset_viewer.dataset import build_index
from yolo_dataset_viewer.models import OpenRequest


class ScriptedBackend:
    """Answer probes from their tool nonce and script target edits independently."""

    def __init__(self, actions=()):
        self.actions = iter(actions)
        self.histories = []
        self.probe_calls = 0
        self.target_calls = 0

    def respond(self, history, tools, *, forced_tool=None):
        self.histories.append(copy.deepcopy(history))
        if forced_tool:
            self.probe_calls += 1
            if forced_tool == "report_visual":
                args = {"first": 1, "second": 2}
            else:
                result = next(
                    x for x in reversed(history) if x.get("type") == "function_call_output"
                )
                args = {"nonce": output_payload(result)["nonce"], "count": 3}
            action = (forced_tool, args)
        else:
            self.target_calls += 1
            action = next(self.actions)
        if isinstance(action, Exception):
            raise action
        if action is None:
            return Turn([], [])
        if isinstance(action, list):
            calls = [
                FunctionCall(f"multi-{i}", name, json.dumps(args))
                for i, (name, args) in enumerate(action)
            ]
        else:
            name, args, *identifier = action
            calls = [
                FunctionCall(
                    identifier[0] if identifier else f"call-{len(self.histories)}",
                    name,
                    args if isinstance(args, str) else json.dumps(args),
                )
            ]
        output = [
            {
                "type": "function_call",
                "name": c.name,
                "call_id": c.call_id,
                "arguments": c.arguments,
            }
            for c in calls
        ]
        return Turn(output, calls, input_tokens=5, output_tokens=3)


def picture(path, color="white", size=(200, 100)):
    path.parent.mkdir(parents=True, exist_ok=True)
    with Image.new("RGB", size, color) as image:
        image.save(path)
    return path


def output_payload(item):
    output = item["output"]
    return json.loads(output[0]["text"] if isinstance(output, list) else output)


@pytest.fixture
def setup(tmp_path):
    root = tmp_path / "input"
    picture(root / "images/train/one.png")
    label = root / "labels/train/one.txt"
    label.parent.mkdir(parents=True)
    label.write_text("0 .5 .5 .4 .4\n")
    (root / "data.yaml").write_text(
        yaml.safe_dump({"path": ".", "train": "images/train", "names": {0: "target", 1: "second"}})
    )
    reference = picture(tmp_path / "references/gold.png")
    reference.with_suffix(".txt").write_text("0 .5 .5 .4 .4\n")
    config = Config.model_validate(
        {
            "api": {"model": "gpt-6-astra"},
            "dataset": {"source": str(root / "data.yaml")},
            "references": [{"image": str(reference), "label": str(reference.with_suffix(".txt"))}],
            "annotation_rules": "Label all visible targets; defer uncertain cases.",
            "output": {"directory": str(tmp_path / "output")},
        }
    )
    return config, tmp_path, root


def sample_state():
    return SampleResult(sample_id="sample", fingerprint="fp", status="running")


def loop(config, backend, state=None):
    state = state or sample_state()
    events = []
    with Image.new("RGB", (200, 100), "white") as image:
        annotate(
            backend, config, [], image, {0: "target", 1: "second"}, state, events.append, Event()
        )
    return state, events


def test_tools_ids_validation_and_explicit_empty():
    state = sample_state()
    tools = AnnotationTools(state, {0: "target", 1: "second"})
    assert not tools.execute("finish_annotation", '{"reason":"done"}').payload["ok"]
    for invalid in ([0, 0, 0, 1], [0, 0, 2, 1], [0, 0, float("nan"), 1]):
        assert not tools.execute("add_box", json.dumps({"class_id": 0, "bbox": invalid})).payload[
            "ok"
        ]
        assert state.revision == 0 and not state.boxes
    assert not tools.execute("add_box", '{"class_id":true,"bbox":[0,0,1,1]}').payload["ok"]
    assert not tools.execute("add_box", '{"class_id":9,"bbox":[0,0,1,1]}').payload["ok"]
    assert not tools.execute("delete_box", '{"box_id":"missing"}').payload["ok"]
    assert not tools.execute("add_box", '{"class_id":0,"bbox":[0,0,1,1],"extra":1}').payload["ok"]
    tools.execute("add_box", '{"class_id":0,"bbox":[0,0,1,1]}')
    tools.execute("update_box", '{"box_id":"box_0001","class_id":1,"bbox":[.1,.1,.9,.9]}')
    # The preceding JSON uses invalid leading-dot floats, so the valid prior box survives.
    assert state.boxes[0].class_id == 0
    tools.execute("update_box", '{"box_id":"box_0001","class_id":1,"bbox":[0.1,0.1,0.9,0.9]}')
    assert state.boxes[0].class_id == 1
    tools.execute("delete_box", '{"box_id":"box_0001"}')
    assert state.status == "running"
    tools.execute("add_box", '{"class_id":0,"bbox":[0,0,1,1]}')
    assert state.boxes[0].box_id == "box_0002"
    tools.execute("mark_empty", '{"reason":"No task objects"}')
    assert state.status == "empty" and not state.boxes
    assert not tools.execute("add_box", '{"class_id":0,"bbox":[0,0,1,1]}').payload["ok"]


def test_coordinate_roundtrip_and_crop(tmp_path):
    box = Box(box_id="one", class_id=0, bbox=[0.18, 0.28, 0.62, 0.72])
    path = tmp_path / "labels.txt"
    path.write_text(label_text([box]))
    assert load_boxes(path, {0: "target"})[0].bbox == pytest.approx(box.bbox)
    with Image.new("RGB", (301, 101), "white") as image:
        content, metadata = observe(image, [box], {0: "target"}, 256, region=[0.1, 0.2, 0.7, 0.8])
    assert metadata["crop_pixels"] == [30, 20, 211, 81]
    assert metadata["crop_bbox"] == pytest.approx([30 / 301, 20 / 101, 211 / 301, 81 / 101])
    assert metadata["editing_coordinate_system"] == "full_image_normalized_xyxy"
    assert len([x for x in content if x["type"] == "input_image"]) == 2


@pytest.mark.parametrize("orientation", range(1, 9))
def test_exif_coordinate_frame(tmp_path, orientation):
    path = tmp_path / "oriented.jpg"
    with Image.new("RGB", (80, 40)) as image:
        exif = Image.Exif()
        exif[274] = orientation
        image.save(path, exif=exif)
    with load_image(path) as image:
        assert image.size == ((40, 80) if orientation >= 5 else (80, 40))


def test_interaction_errors_feedback_class_changes_and_duplicate_calls(setup):
    config, _, _ = setup
    backend = ScriptedBackend(
        [
            ("add_box", {"class_id": 0, "bbox": [-1, 0, 1, 1]}),
            ("add_box", {"class_id": 0, "bbox": [0.1, 0.2, 0.6, 0.8]}, "repeat"),
            ("add_box", {"class_id": 0, "bbox": [0.1, 0.2, 0.6, 0.8]}, "repeat"),
            ("inspect_region", {"bbox": [0, 0, 0.8, 1]}),
            ("update_box", {"box_id": "box_0001", "class_id": 1, "bbox": [0.2, 0.2, 0.7, 0.8]}),
            ("finish_annotation", {"reason": "Reviewed latest rendering"}),
        ]
    )
    state, events = loop(config, backend)
    assert state.status == "completed" and state.revision == 2
    assert len(state.boxes) == 1 and state.boxes[0].class_id == 1
    assert state.usage.requests == 6
    outputs = [
        output_payload(x) for x in backend.histories[-1] if x.get("type") == "function_call_output"
    ]
    assert outputs[0]["ok"] is False and outputs[0]["revision"] == 0
    assert outputs[1]["revision"] == outputs[2]["revision"] == 1
    assert any(e.get("duplicate") for e in events)
    assert outputs[-1]["revision"] == 2 and outputs[-1]["boxes"][0]["class_id"] == 1
    raw = backend.histories[-1][0]["content"][2]["image_url"]
    final_overlay = backend.histories[-1][-1]["output"][-1]["image_url"]
    assert raw != final_overlay


@pytest.mark.parametrize(
    "action,status",
    [
        (("mark_empty", {"reason": "No objects"}), "empty"),
        (("defer_annotation", {"reason": "Ambiguous class"}), "deferred"),
        (APIProblem("network failed"), "api_error"),
    ],
)
def test_distinct_terminal_states(setup, action, status):
    state, _ = loop(setup[0], ScriptedBackend([action]))
    assert state.status == status


def test_limit_does_not_publish_unreviewed_edit_or_plain_text(setup):
    config = setup[0].model_copy(deep=True)
    config.run.max_model_calls = 1
    state, _ = loop(config, ScriptedBackend([("add_box", {"class_id": 0, "bbox": [0, 0, 1, 1]})]))
    assert state.status == "max_calls" and len(state.boxes) == 1
    state, _ = loop(config, ScriptedBackend([None]))
    assert state.status == "max_calls"


def test_multiple_calls_are_not_executed(setup):
    backend = ScriptedBackend(
        [
            [
                ("add_box", {"class_id": 0, "bbox": [0, 0, 1, 1]}),
                ("finish_annotation", {"reason": "Premature"}),
            ],
            ("mark_empty", {"reason": "No objects"}),
        ]
    )
    state, _ = loop(setup[0], backend)
    assert state.status == "empty" and state.revision == 1
    assert sum(x.get("type") == "function_call_output" for x in backend.histories[-1]) == 2


def test_retries_count_toward_real_request_budget(setup):
    class FastEvent(Event):
        def wait(self, timeout=None):
            return self.is_set()

    settings = setup[0].run.model_copy(update={"max_model_calls": 2, "max_retries": 3})
    budget = Budget(settings, Usage())
    backend = ScriptedBackend([APIProblem("timeout", retryable=True)] * 3)
    with pytest.raises(CallLimit):
        budget.request(backend, [], [], FastEvent(), lambda _: None)
    assert budget.usage.requests == backend.target_calls == 2


def test_probe_checks_nonce_and_new_image():
    backend = ScriptedBackend()
    assert probe(backend)["status"] == "passed"
    assert backend.probe_calls == 2
    assert any(x.get("type") == "function_call_output" for x in backend.histories[-1])


def test_pipeline_export_resume_and_viewer_compatibility(setup):
    config, base, source = setup
    original = (source / "labels/train/one.txt").read_bytes()
    backend = ScriptedBackend([("finish_annotation", {"reason": "Already correct"})])
    report = run(config, base, backend, endpoint_identity="https://test.invalid/v1")
    assert report["status"] == "completed" and report["counts"] == {"completed": 1}
    output = Path(config.output.directory)
    assert (output / "dataset").is_symlink()
    indexed = build_index(
        OpenRequest(path=str(output / "dataset/data.yaml")), Event(), lambda *_: None
    )
    assert len(indexed.samples) == 1 and indexed.samples[0].state == "valid"
    assert (source / "labels/train/one.txt").read_bytes() == original
    assert not list((output / ".tmp").iterdir())
    resumed = ScriptedBackend()
    assert (
        run(config, base, resumed, endpoint_identity="https://test.invalid/v1")["status"]
        == "completed"
    )
    assert resumed.probe_calls == resumed.target_calls == 0
    (source / "labels/train/one.txt").write_text("0 .5 .5 .2 .2\n")
    with pytest.raises(ValueError, match="changed"):
        run(config, base, ScriptedBackend(), endpoint_identity="https://test.invalid/v1")
    assert json.loads((output / "run.json").read_text())["status"] == "completed"


def test_negative_export_and_unfinished_recovery(setup):
    config, base, source = setup
    (source / "labels/train/one.txt").unlink()
    backend = ScriptedBackend([("defer_annotation", {"reason": "Need inspection"})])
    first = run(config, base, backend, endpoint_identity="local")
    assert first["status"] == "partial" and first["exported_entries"] == 0
    assert not (Path(config.output.directory) / "dataset").exists()
    second = run(
        config,
        base,
        ScriptedBackend([("mark_empty", {"reason": "No targets"})]),
        endpoint_identity="local",
    )
    assert second["counts"] == {"empty": 1} and second["samples"][0]["attempt"] == 2
    path = Path(config.output.directory) / "dataset" / second["mapping"][0]["label"]
    assert path.read_bytes() == b""
    assert second["usage"]["requests"] == 2


def test_invalid_reference_blocks_network_and_invalid_target_is_skipped(setup):
    config, base, source = setup
    reference_label = Path(config.references[0].label)
    reference_label.write_text("invalid\n")
    backend = ScriptedBackend()
    with pytest.raises(ValueError):
        run(config, base, backend, endpoint_identity="local")
    assert backend.probe_calls == backend.target_calls == 0
    reference_label.write_text("0 .5 .5 .4 .4\n")
    (source / "labels/train/one.txt").write_text("0 nan .5 .2 .2\n")
    report = run(config, base, backend, endpoint_identity="local")
    assert report["counts"] == {"invalid_input": 1}
    assert backend.target_calls == 0


def test_duplicate_split_membership_and_names_conflicts(setup):
    config, base, source = setup
    path = source / "data.yaml"
    data = yaml.safe_load(path.read_text())
    data["val"] = data["train"]
    path.write_text(yaml.safe_dump(data))
    report = run(
        config,
        base,
        ScriptedBackend([("finish_annotation", {"reason": "Valid"})]),
        endpoint_identity="local",
    )
    assert report["unique_images"] == 1 and report["exported_entries"] == 2
    config.dataset.names = {0: "different"}
    with pytest.raises(ValueError, match="disagree"):
        discover(config, base)


def test_output_overlap_and_modified_export_are_rejected(setup):
    config, base, source = setup
    bad = config.model_copy(deep=True)
    bad.output.directory = str(source / "output")
    with pytest.raises(ValueError, match="overlaps"):
        run(bad, base, ScriptedBackend(), endpoint_identity="local")
    report = run(
        config,
        base,
        ScriptedBackend([("mark_empty", {"reason": "No target"})]),
        endpoint_identity="local",
    )
    exported = Path(config.output.directory) / "dataset" / report["mapping"][0]["label"]
    exported.write_text("0 .5 .5 .2 .2\n")
    with pytest.raises(ValueError, match="modified"):
        run(config, base, ScriptedBackend(), endpoint_identity="local")


def test_env_loading_and_config_paths(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text(
        'export BASE_URL="https://test.invalid/v1"\nexport API_KEY="test-only"\n'
    )
    monkeypatch.delenv("BASE_URL", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)
    auth = credentials(APIConfig(model="gpt-6-astra"), tmp_path)
    assert auth.base_url.endswith("/v1") and "test-only" not in repr(auth)
    monkeypatch.setenv("API_KEY", "process-value")
    assert (
        credentials(APIConfig(model="gpt-6-astra"), tmp_path).api_key.get_secret_value()
        == "process-value"
    )
    path = Path(__file__).parents[1] / "src/vlm_correct_pipeline/config.example.yaml"
    assert load_config(path).api.model == "gpt-6-astra"


def test_sdk_wire_contract_and_reasoning_roundtrip():
    captured = []

    def transport(request):
        body = json.loads(request.content)
        captured.append(body)
        return httpx.Response(
            200,
            json={
                "id": "resp-test",
                "object": "response",
                "created_at": 0,
                "model": "gpt-6-astra",
                "status": "completed",
                "output": [
                    {
                        "type": "reasoning",
                        "id": "reasoning-test",
                        "summary": [],
                        "encrypted_content": "opaque-context",
                    },
                    {
                        "type": "function_call",
                        "id": "fc-test",
                        "call_id": "call-test",
                        "name": "mark_empty",
                        "arguments": '{"reason":"No targets"}',
                        "status": "completed",
                    },
                ],
                "usage": {"input_tokens": 10, "output_tokens": 3, "total_tokens": 13},
            },
        )

    backend = ResponsesBackend(
        APIConfig(model="gpt-6-astra"),
        Credentials(base_url="https://test.invalid/v1", api_key="fake-key"),
    )
    backend.client.close()
    backend.client = OpenAI(
        api_key="fake-key",
        base_url="https://test.invalid/v1",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(transport)),
    )
    try:
        first = backend.respond([], tool_definitions())
        backend.respond(first.output, tool_definitions())
    finally:
        backend.close()
    assert captured[0]["model"] == "gpt-6-astra"
    assert captured[0]["store"] is False and captured[0]["parallel_tool_calls"] is False
    assert captured[0]["include"] == ["reasoning.encrypted_content"]
    assert captured[1]["input"][0]["encrypted_content"] == "opaque-context"
    assert all(tool["strict"] for tool in captured[0]["tools"])


def test_imports_do_not_initialize_training_or_gpu_libraries():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import vlm_correct_pipeline.cli; "
                "assert not ({'torch', 'ultralytics', 'cv2'} & sys.modules.keys())"
            ),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_interrupted_draft_restarts_from_source_labels(setup):
    config, base, _ = setup
    backend = ScriptedBackend(
        [
            ("add_box", {"class_id": 1, "bbox": [0.1, 0.1, 0.2, 0.2]}),
            RuntimeError("simulated host interruption"),
        ]
    )
    with pytest.raises(RuntimeError, match="interruption"):
        run(config, base, backend, endpoint_identity="local")
    checkpoint = next((Path(config.output.directory) / "samples").glob("*.json"))
    draft = json.loads(checkpoint.read_text())
    assert draft["status"] == "interrupted" and len(draft["boxes"]) == 2
    report = run(
        config,
        base,
        ScriptedBackend([("finish_annotation", {"reason": "Original correct"})]),
        endpoint_identity="local",
    )
    assert report["status"] == "completed"
    assert len(report["samples"][0]["boxes"]) == 1 and report["samples"][0]["attempt"] == 2


def test_publication_failure_recovers_without_repeating_model_calls(setup, monkeypatch):
    from vlm_correct_pipeline import storage

    config, base, _ = setup
    target = Path(config.output.directory) / "dataset"
    replace = storage.os.replace

    def fail_publication(source, destination):
        if Path(destination) == target:
            raise OSError("simulated publication failure")
        replace(source, destination)

    with monkeypatch.context() as patch:
        patch.setattr(storage.os, "replace", fail_publication)
        with pytest.raises(OSError, match="publication failure"):
            run(
                config,
                base,
                ScriptedBackend([("mark_empty", {"reason": "No objects"})]),
                endpoint_identity="local",
            )
    assert not target.exists()
    assert not list((target.parent / ".tmp").iterdir())
    backend = ScriptedBackend()
    result = run(config, base, backend, endpoint_identity="local")
    assert result["status"] == "completed" and backend.target_calls == backend.probe_calls == 0
    assert len(list((target.parent / ".exports").iterdir())) == 1


def test_concurrent_sessions_and_same_stem_images_remain_separate(setup):
    config, base, source = setup
    picture(source / "images/train/one.jpg")
    config.run.image_concurrency = 2
    backend = ScriptedBackend([("mark_empty", {"reason": "No targets"})] * 2)
    result = run(config, base, backend, endpoint_identity="local")
    assert result["status"] == "completed" and result["counts"] == {"empty": 2}
    assert len({row["label"] for row in result["mapping"]}) == 2
    assert backend.target_calls == 2


def test_source_change_during_annotation_prevents_publication(setup):
    config, base, source = setup

    class ChangingBackend(ScriptedBackend):
        def respond(self, history, tools, *, forced_tool=None):
            response = super().respond(history, tools, forced_tool=forced_tool)
            if not forced_tool:
                (source / "labels/train/one.txt").write_text("0 .5 .5 .1 .1\n")
            return response

    result = run(
        config,
        base,
        ChangingBackend([("finish_annotation", {"reason": "Correct"})]),
        endpoint_identity="local",
    )
    assert result["counts"] == {"invalid_input": 1} and result["exported_entries"] == 0


def test_pose_box_conversion_is_explicit_and_source_remains_intact(setup):
    config, base, root = setup
    path = root / "data.yaml"
    metadata = yaml.safe_load(path.read_text())
    metadata["kpt_shape"] = [1, 3]
    path.write_text(yaml.safe_dump(metadata))
    label = root / "labels/train/one.txt"
    original = "0 .5 .5 .4 .4 .5 .5 2\n"
    label.write_text(original)
    with pytest.raises(ValueError, match="pose_boxes"):
        discover(config, base)
    config.dataset.input_task = "pose_boxes"
    result = run(
        config,
        base,
        ScriptedBackend([("finish_annotation", {"reason": "Correct"})]),
        endpoint_identity="local",
    )
    output = Path(config.output.directory) / "dataset"
    assert len((output / result["mapping"][0]["label"]).read_text().split()) == 5
    assert "kpt_shape" not in yaml.safe_load((output / "data.yaml").read_text())
    assert label.read_text() == original
    assert result["input_task"] == "pose_boxes" and result["output_task"] == "detect"


def test_source_rounding_is_explicit_logged_and_does_not_relax_model_tools(tmp_path):
    path = tmp_path / "source.txt"
    path.write_text("0 0.100000000 0.5 0.200000001 0.5\n")
    with pytest.raises(ValueError):
        load_boxes(path, {0: "target"})
    adjustments = []
    boxes = load_boxes(path, {0: "target"}, rounding_tolerance=1e-6, adjustments=adjustments)
    assert boxes[0].bbox[0] == 0 and adjustments[0]["before"][0] < 0
    state = sample_state()
    reply = AnnotationTools(state, {0: "target"}).execute(
        "add_box", json.dumps({"class_id": 0, "bbox": [-5e-10, 0, 1, 1]})
    )
    assert reply.payload["ok"] is False


def test_selection_preserves_split_and_rejects_unknown_or_duplicate_images(setup):
    config, base, source = setup
    picture(source / "images/train/two.png")
    selection = base / "selection.json"
    config.dataset.selection_file = str(selection)
    selection.write_text(json.dumps(["images/train/one.png"]))
    data = discover(config, base)
    assert len(data.samples) == 1 and data.samples[0].splits == ["train"]
    assert str(selection) in data.source_files
    selection.write_text(json.dumps(["images/train/one.png"] * 2))
    with pytest.raises(ValueError, match="duplicate"):
        discover(config, base)
    selection.write_text(json.dumps(["images/train/absent.png", "images/train/one.png"]))
    with pytest.raises(ValueError, match="outside"):
        discover(config, base)


def test_discard_is_durable_exclusion_and_never_a_negative(setup):
    config, base, source = setup
    backend = ScriptedBackend(
        [
            (
                "discard_image",
                {"category": "poor_lighting", "reason": "Target boundaries obscured by glare"},
            )
        ]
    )
    result = run(config, base, backend, endpoint_identity="local")
    assert result["status"] == "completed" and result["counts"] == {"discarded": 1}
    assert (
        result["exported_entries"] == 0 and not (Path(config.output.directory) / "dataset").exists()
    )
    assert result["samples"][0]["discard_category"] == "poor_lighting"
    assert len(result["samples"][0]["boxes"]) == 1
    resumed = ScriptedBackend()
    again = run(config, base, resumed, endpoint_identity="local")
    assert again["counts"] == {"discarded": 1} and resumed.target_calls == resumed.probe_calls == 0
    assert (source / "labels/train/one.txt").read_text() == "0 .5 .5 .4 .4\n"


def test_api_circuit_stops_scheduling_new_images(setup):
    config, base, source = setup
    for i in range(4):
        picture(source / f"images/train/extra{i}.png")
    config.run.max_retries = 0
    config.run.max_consecutive_api_errors = 2
    backend = ScriptedBackend([APIProblem("temporary", retryable=True)] * 2)
    report = run(config, base, backend, endpoint_identity="local")
    assert backend.target_calls == 2 and report["counts"] == {"api_error": 2}
    assert report["status"] == "failed" and len(report["not_processed"]) == 3
    assert not (Path(config.output.directory) / "dataset").exists()


def test_lighting_audit_uses_target_visibility_not_just_background(tmp_path):
    from PIL import ImageDraw

    from vlm_correct_pipeline.quality import audit_lighting

    source = tmp_path / "source"
    for name, background, target in [
        ("dark_background", 0, 110),
        ("dark_target", 110, 0),
        ("glare", 110, 255),
    ]:
        path = picture(source / f"images/train/{name}.png", size=(200, 100))
        with Image.new("L", (200, 100), background) as im:
            ImageDraw.Draw(im).rectangle((50, 25, 149, 74), fill=target)
            im.save(path)
        label = source / f"labels/train/{name}.txt"
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text("0 .5 .5 .5 .5\n")
    (source / "data.yaml").write_text(
        yaml.safe_dump({"path": ".", "train": "images/train", "names": {0: "target"}})
    )
    output = tmp_path / "audit"
    summary = audit_lighting(source, output)
    assert summary["images"] == 3 and summary["candidate_counts"] == {
        "normal": 1,
        "underexposed_candidate": 1,
        "overexposed_candidate": 1,
    }
    rows = {
        Path(r["image"]).stem: r
        for r in map(json.loads, (output / "records.jsonl").read_text().splitlines())
    }
    assert rows["dark_background"]["candidate"] == "normal"
    assert rows["dark_target"]["candidate"] == "underexposed_candidate"
    assert (output / "COMPLETE.json").exists() and not list(output.glob("*.tmp"))
    assert len(list((source / "images/train").glob("*.png"))) == 3
    with pytest.raises(ValueError, match="empty"):
        audit_lighting(source, output)


def test_non_responses_payload_is_sanitized_protocol_error():
    backend = ResponsesBackend(
        APIConfig(model="gpt-6-astra"),
        Credentials(base_url="https://test.invalid/v1", api_key="fake-key"),
    )
    backend.client.close()
    backend.client = OpenAI(
        api_key="fake-key",
        base_url="https://test.invalid/v1",
        http_client=httpx.Client(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(200, text="gateway error with private details")
            )
        ),
    )
    try:
        with pytest.raises(APIProblem, match="non-Responses") as error:
            backend.respond([], tool_definitions())
        assert error.value.fatal and "private" not in str(error.value)
    finally:
        backend.close()


def test_streaming_accumulates_complete_response_before_exposing_tools():
    captured = []
    response = {
        "id": "resp-stream",
        "object": "response",
        "created_at": 0,
        "model": "gpt-6-astra",
        "status": "completed",
        "output": [
            {
                "id": "fc-stream",
                "type": "function_call",
                "call_id": "call-stream",
                "name": "mark_empty",
                "arguments": '{"reason":"No targets"}',
                "status": "completed",
            }
        ],
        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    }

    def transport(request):
        captured.append(json.loads(request.content))
        events = [
            {
                "type": "response.created",
                "sequence_number": 0,
                "response": {**response, "status": "in_progress", "output": []},
            },
            {"type": "response.completed", "sequence_number": 1, "response": response},
        ]
        body = "".join("data: " + json.dumps(event) + "\n\n" for event in events)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=body)

    backend = ResponsesBackend(
        APIConfig(model="gpt-6-astra", stream=True),
        Credentials(base_url="https://test.invalid/v1", api_key="fake-key"),
    )
    backend.client.close()
    backend.client = OpenAI(
        api_key="fake-key",
        base_url="https://test.invalid/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(transport)),
    )
    try:
        result = backend.respond([], tool_definitions())
        assert captured[0]["stream"] is True
        assert result.calls[0].name == "mark_empty" and result.input_tokens == 10
        backend.respond(result.output, tool_definitions())
        assert captured[1]["input"] == response["output"]
    finally:
        backend.close()


def test_failed_probe_persists_evidence_without_creating_sample_labels(setup):
    config, base, _ = setup

    class BadProbe(ScriptedBackend):
        def respond(self, history, tools, *, forced_tool=None):
            return Turn([], [], input_tokens=10, output_tokens=2)

    with pytest.raises(APIProblem, match="report_visual"):
        run(config, base, BadProbe(), endpoint_identity="local")
    output = Path(config.output.directory)
    events = json.loads((output / "capability_probe.json").read_text())
    report = json.loads((output / "report.json").read_text())
    assert events["status"] == "failed"
    assert any(e["event"] == "probe_tools" and e["call_count"] == 0 for e in events["events"])
    assert (
        report["counts"] == {} and report["unique_images"] == 1 and not report["published_dataset"]
    )
    assert not list((output / "samples").iterdir())
