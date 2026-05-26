from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from odp_platform.common.logging_utils import reset_logging, setup_logging
from odp_platform.data_validation.report import ValidationReport
from odp_platform.runtime_config.base import ConfigTrace, FieldTrace, SourceOverride, TrainConfig
from odp_platform.training.service import TrainService


class _FakeResults:
    task = "detect"
    save_dir = ""
    speed = {"preprocess": 0.1, "inference": 0.2, "loss": 0.0, "postprocess": 0.3}
    results_dict = {"metrics/precision(B)": 0.9}
    fitness = 0.8
    names = {0: "ship"}
    maps = [0.5]


class _FakeYOLO:
    def __init__(self, _model_path: str) -> None:
        self.trainer = None

    def train(self, **_kwargs):
        results = _FakeResults()
        project = Path(_kwargs["project"])
        name = _kwargs.get("name") or "train"
        results.save_dir = str(project / name)
        return results


def _fake_trace(config: TrainConfig) -> ConfigTrace:
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


def test_train_service_happy_path(monkeypatch, tmp_path: Path) -> None:
    reset_logging()
    setup_logging(base_path=tmp_path / "logs", log_type="train", model_name="demo")

    output_dir = tmp_path / "runs" / "detect_train" / "train1"
    output_dir.mkdir(parents=True)
    (output_dir / "weights").mkdir(parents=True)
    (output_dir / "weights" / "best.pt").write_text("best", encoding="utf-8")
    (output_dir / "weights" / "last.pt").write_text("last", encoding="utf-8")
    dataset_yaml = tmp_path / "demo.yaml"
    dataset_yaml.write_text("path: .\ntrain: train/images\nval: val/images\nnc: 1\nnames: [ship]\n", encoding="utf-8")
    model_file = tmp_path / "yolo11n.pt"
    model_file.write_text("weights", encoding="utf-8")

    config = TrainConfig(model="yolo11n.pt", data="demo.yaml", epochs=1, project=str(output_dir.parent), name="train1")
    trace = _fake_trace(config)

    monkeypatch.setattr(
        "odp_platform.training.service.build_train_config",
        lambda **_kwargs: (config, trace),
    )
    monkeypatch.setattr("odp_platform.training.service.resolve_dataset_path", lambda _data: dataset_yaml)
    monkeypatch.setattr("odp_platform.training.service.resolve_model_path", lambda _model: model_file)
    monkeypatch.setattr("odp_platform.training.service._load_yolo_class", lambda: _FakeYOLO)
    monkeypatch.setattr("odp_platform.training.service.render_to_logger", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "odp_platform.training.service.validate_dataset",
        lambda *_args, **_kwargs: ValidationReport(
            run_id="demo",
            yaml_path=dataset_yaml,
            snapshot=type("Snap", (), {"yaml_path": dataset_yaml})(),
            results=[],
            duration_seconds=0.1,
            started_at_iso="2026-01-01T00:00:00",
        ),
    )

    service = TrainService()
    cli_args = Namespace(epochs=1)
    result = service.train(
        yaml_path="train.yaml",
        cli_args=cli_args,
        pre_validate=False,
        archive=False,
        rename_log=False,
    )

    assert result.success is True
    assert result.output_dir == output_dir
    assert result.audit_path is not None
    reset_logging()
