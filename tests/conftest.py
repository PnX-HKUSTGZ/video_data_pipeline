import av
import numpy as np
import pytest
import yaml

from video_dataset_pipeline import pipeline
from video_dataset_pipeline.config import Config
from video_dataset_pipeline.dataset import validate_dataset
from video_dataset_pipeline.records import RawDetection


@pytest.fixture
def synthetic_run(tmp_path, monkeypatch):
    video = tmp_path / "中文视频.mp4"
    with av.open(str(video), "w") as container:
        stream = container.add_stream("mpeg4", rate=10)
        stream.width, stream.height, stream.pix_fmt = 160, 96, "yuv420p"
        for i in range(120):
            image = np.full((96, 160, 3), i * 2, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(image, format="bgr24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    model = tmp_path / "model"
    model.mkdir()
    (model / "metadata.yaml").write_text(yaml.safe_dump({"names": {0: "pillar", 1: "exchange"}}))
    (model / "custom_weights.pt").write_bytes(b"synthetic weights")
    monkeypatch.setattr(pipeline, "validate_dataset", lambda root: validate_dataset(root, False))

    class FakeDetector:
        fail_at = None
        calls = 0

        def __init__(self, *args):
            self.names = {0: "pillar", 1: "exchange"}
            self.task = "detect"
            self.pose = None

        def predict(self, image):
            FakeDetector.calls += 1
            if FakeDetector.calls == FakeDetector.fail_at:
                raise RuntimeError("simulated interruption")
            if image.mean() < 20:
                return []
            return [RawDetection(0, 0.9, (20, 10, 80, 70))]

    monkeypatch.setattr(pipeline, "Detector", FakeDetector)
    config = Config(
        input=str(video),
        output=str(tmp_path / "out"),
        model=str(model / "custom_weights.pt"),
        model_yaml=str(model / "metadata.yaml"),
        device="cpu",
        split_gap_seconds=0.5,
    )
    return config, FakeDetector
