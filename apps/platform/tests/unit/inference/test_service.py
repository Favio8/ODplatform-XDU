from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import cv2
import numpy as np

from odp_platform.common.logging_utils import reset_logging, setup_logging
from odp_platform.inference.service import InferenceService
from odp_platform.runtime_config.base import ConfigTrace, FieldTrace, InferConfig, SourceOverride


class _FakeTensor:
    def __init__(self, value) -> None:
        self._value = np.asarray(value)

    def cpu(self):
        return self

    def int(self):
        return _FakeTensor(self._value.astype(int))

    def numpy(self):
        return self._value

    def tolist(self):
        return self._value.tolist()


class _FakeBoxes:
    def __init__(self) -> None:
        self.xyxy = _FakeTensor([[4, 5, 26, 20]])
        self.conf = _FakeTensor([0.93])
        self.cls = _FakeTensor([0])

    def __len__(self) -> int:
        return 1


class _FakeResult:
    def __init__(self) -> None:
        self.names = {0: "ship"}
        self.boxes = _FakeBoxes()
        self.masks = None
        self.speed = {"preprocess": 1.2, "inference": 12.3, "postprocess": 0.8}

    def save_txt(self, path: str, save_conf: bool = False) -> None:
        Path(path).write_text("0 0.5 0.5 0.4 0.3\n", encoding="utf-8")

    def plot(self, labels: bool = True, conf: bool = True, boxes: bool = True, line_width=None):
        del labels, conf, boxes, line_width
        return np.full((24, 32, 3), 120, dtype=np.uint8)


class _FakeYOLO:
    def __init__(self, _model_path: str) -> None:
        self.names = {0: "ship"}

    def __call__(self, images, **_kwargs):
        return [_FakeResult() for _ in images]


def _fake_trace(config: InferConfig) -> ConfigTrace:
    return ConfigTrace(
        by_field={
            field_name: FieldTrace(
                field_name=field_name,
                final_value=value,
                final_source="default",
                history=(SourceOverride(source_name="default", value=value),),
            )
            for field_name, value in config.to_runtime_dict().items()
        }
    )


def _write_image(path: Path) -> None:
    image = np.full((24, 32, 3), 80, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def test_inference_service_runs_and_saves_outputs(monkeypatch, tmp_path: Path) -> None:
    reset_logging()
    setup_logging(base_path=tmp_path / "logs", log_type="infer", model_name="demo")

    image_path = tmp_path / "demo.jpg"
    _write_image(image_path)
    model_path = tmp_path / "demo.pt"
    model_path.write_text("weights", encoding="utf-8")
    output_root = tmp_path / "runs"
    config = InferConfig(
        model=str(model_path),
        source=str(image_path),
        task_type="detect",
        project=str(output_root),
        name="demo_infer",
        save=True,
        save_txt=True,
        show=False,
    )
    trace = _fake_trace(config)

    monkeypatch.setattr(
        "odp_platform.inference.service.build_infer_config",
        lambda **_kwargs: (config, trace),
    )
    monkeypatch.setattr("odp_platform.inference.service.resolve_model_path", lambda _model, search_dirs=None: model_path)
    monkeypatch.setattr("odp_platform.inference.service._load_yolo_class", lambda: _FakeYOLO)

    service = InferenceService()
    result = service.infer(yaml_path="infer.yaml", cli_args=Namespace())

    assert result.success is True
    assert result.output_dir == output_root / "demo_infer"
    assert result.audit_path is not None
    assert (result.output_dir / "demo.jpg").exists()
    assert (result.output_dir / "labels" / "demo.txt").exists()
    assert result.summary is not None
    assert result.summary.frames_processed == 1
    assert result.summary.detections_total == 1
    reset_logging()
