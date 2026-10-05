from dataclasses import replace

import av
import numpy as np
import pytest

from video_dataset_pipeline.dataset import annotate_frame
from video_dataset_pipeline.inference_video import InferenceVideoWriter, validate_inference_video
from video_dataset_pipeline.records import Detection, FrameRecord


@pytest.mark.parametrize("start,width,height", [(0, 160, 96), (100, 159, 95)])
def test_video_preserves_irregular_timestamps_and_last_frame_duration(
    tmp_path, start, width, height
):
    image = np.zeros((height, width, 3), dtype=np.uint8)
    records = [
        FrameRecord(i, start + t, width, height, None, "no_detection")
        for i, t in enumerate((0, 0.52, 1.0))
    ]
    with InferenceVideoWriter(tmp_path, width, height, 2, start + 1.4) as video:
        for record in records:
            video.write(image, record.timestamp)
    validate_inference_video(tmp_path, records, start + 1.4)
    assert not (tmp_path / "inference.mp4.tmp").exists()
    with av.open(str(tmp_path / "inference.mp4")) as container:
        assert not container.streams.audio
        stream = container.streams.video[0]
        assert stream.codec_context.name == "h264"
        frames = list(container.decode(stream))
        assert len(frames) == 3
        assert [float(f.pts * f.time_base) for f in frames] == pytest.approx([0, 0.52, 1.0])
        assert float(frames[-1].duration * frames[-1].time_base) == pytest.approx(0.4)
        assert (stream.width, stream.height) == (160, 96)
    with pytest.raises(ValueError, match="duration"):
        validate_inference_video(tmp_path, records, start + 2.0)
    changed = [records[0], replace(records[1], timestamp=start + 0.6), records[2]]
    with pytest.raises(ValueError, match="timestamps"):
        validate_inference_video(tmp_path, changed, start + 1.4)
    with pytest.raises(ValueError, match="frame count"):
        validate_inference_video(tmp_path, records[:-1], start + 1.4)


@pytest.mark.parametrize("failure", ["inference", "encoding", "empty"])
def test_video_failure_cleans_partial_file(tmp_path, monkeypatch, failure):
    image = np.zeros((96, 160, 3), dtype=np.uint8)

    def fail_encoding(self, frame):
        raise RuntimeError("encoding failed")

    if failure == "encoding":
        monkeypatch.setattr(InferenceVideoWriter, "_encode", fail_encoding)
    with (
        pytest.raises((RuntimeError, ValueError)),
        InferenceVideoWriter(tmp_path, 160, 96, 2, 1.0) as video,
    ):
        if failure != "empty":
            video.write(image, 0.0)
            video.write(image, 0.5)
        if failure == "inference":
            raise RuntimeError("inference failed")
    assert not (tmp_path / "inference.mp4").exists()
    assert not (tmp_path / "inference.mp4.tmp").exists()


def test_annotations_draw_boxes_without_modifying_source_image():
    image = np.zeros((96, 160, 3), dtype=np.uint8)
    box = (20.0, 40.0, 80.0, 70.0)
    record = FrameRecord(
        0,
        0.0,
        160,
        96,
        None,
        "candidate",
        [Detection(0, 0.9, box, box, (0.3125, 0.572916667, 0.375, 0.3125))],
    )
    annotated = annotate_frame(image, record, {0: "pillar"})
    assert not image.any()
    assert tuple(annotated[60, 20]) == (40, 220, 40)
    assert np.any(annotated[10:35])  # Timestamp/status text is also drawn.
