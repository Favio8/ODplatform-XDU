from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

import odp_platform.data_validation.service as validation_service
from odp_platform.cli.validate_data import main
from odp_platform.data_validation.service import validate_dataset


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color=(255, 255, 255)).save(path)


def _write_label(path: Path, *, lines: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines or []), encoding="utf-8")


def _build_dataset(
    root: Path,
    *,
    train_stems: list[str],
    val_stems: list[str],
    test_stems: list[str],
    missing_labels: set[str] | None = None,
) -> tuple[Path, Path]:
    missing_labels = missing_labels or set()
    data_root = root / "prepared"
    for split, stems in {
        "train": train_stems,
        "val": val_stems,
        "test": test_stems,
    }.items():
        for stem in stems:
            _write_image(data_root / split / "images" / f"{stem}.jpg")
            if stem not in missing_labels:
                _write_label(data_root / split / "labels" / f"{stem}.txt", lines=["0 0.5 0.5 0.5 0.5"])

    yaml_path = root / "dataset.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {data_root}",
                "train: train/images",
                "val: val/images",
                "test: test/images",
                "nc: 1",
                "names:",
                "  0: ship",
                "odp_meta:",
                "  task: detect",
            ]
        ),
        encoding="utf-8",
    )
    return data_root, yaml_path


def test_validate_cli_happy_path_writes_json_report(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _, yaml_path = _build_dataset(
        tmp_path,
        train_stems=["train_001", "train_002"],
        val_stems=["val_001"],
        test_stems=["test_001"],
    )
    run_dir = tmp_path / "runs" / "healthy"
    monkeypatch.setattr(validation_service, "validation_run_dir", lambda run_id: run_dir)

    exit_code = main(["--yaml", str(yaml_path)])

    assert exit_code == 0
    report_path = run_dir / "report.json"
    assert report_path.exists()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["overall_severity"] == "PASS"
    assert payload["exit_code"] == 0
    assert payload["counts_by_severity"]["PASS"] == 4


def test_validate_cli_returns_warning_exit_code(tmp_path: Path, monkeypatch) -> None:
    _, yaml_path = _build_dataset(
        tmp_path,
        train_stems=[f"train_{index:03d}" for index in range(10)],
        val_stems=[],
        test_stems=[],
        missing_labels={"train_009"},
    )
    monkeypatch.setattr(validation_service, "validation_run_dir", lambda run_id: tmp_path / "runs" / "warning")

    exit_code = main(["--yaml", str(yaml_path)])

    assert exit_code == 1


def test_validate_cli_returns_error_exit_code_for_data_leak(tmp_path: Path, monkeypatch) -> None:
    _, yaml_path = _build_dataset(
        tmp_path,
        train_stems=["duplicate_001", "train_002"],
        val_stems=["duplicate_001"],
        test_stems=[],
    )
    monkeypatch.setattr(validation_service, "validation_run_dir", lambda run_id: tmp_path / "runs" / "error")

    exit_code = main(["--yaml", str(yaml_path)])

    assert exit_code == 2


def test_validate_cli_no_report_skips_json_file(tmp_path: Path, monkeypatch) -> None:
    _, yaml_path = _build_dataset(
        tmp_path,
        train_stems=["sample_001"],
        val_stems=[],
        test_stems=[],
    )
    run_dir = tmp_path / "runs" / "no-report"
    monkeypatch.setattr(validation_service, "validation_run_dir", lambda run_id: run_dir)

    exit_code = main(["--yaml", str(yaml_path), "--no-report"])

    assert exit_code == 0
    assert not (run_dir / "report.json").exists()


def test_validate_dataset_report_path_and_payload_are_stable(tmp_path: Path) -> None:
    _, yaml_path = _build_dataset(
        tmp_path,
        train_stems=["sample_001"],
        val_stems=["sample_002"],
        test_stems=[],
    )
    run_dir = tmp_path / "runs" / "direct"

    report = validate_dataset(yaml_path=yaml_path, run_dir=run_dir)

    assert report.report_path == run_dir / "report.json"
    assert report.report_path.exists()
    payload = json.loads(report.report_path.read_text(encoding="utf-8"))
    assert payload["snapshot"]["task_type"] == "detect"
    assert payload["results"][0]["name"] == "yaml_schema"
