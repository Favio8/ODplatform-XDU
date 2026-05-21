from __future__ import annotations

from pathlib import Path

from odp_platform.common.constants import TASK_DETECT
from odp_platform.data_validation.checks.yaml_schema import validate_yaml_schema
from odp_platform.data_validation.registry import CheckContext, CheckSeverity
from odp_platform.data_validation.snapshot import DatasetSnapshot


def _make_ctx(yaml_path: Path) -> CheckContext:
    snapshot = DatasetSnapshot(
        yaml_path=yaml_path,
        yaml_data={},
        yaml_load_error=None,
        data_root=yaml_path.parent,
        nc=None,
        class_names=(),
        task_type=TASK_DETECT,
        images_per_split={},
        labels_per_split={},
        stats_per_split={},
        scan_warnings=(),
    )
    return CheckContext(yaml_path=yaml_path, snapshot=snapshot)


def test_yaml_schema_reports_missing_file(tmp_path: Path) -> None:
    result = validate_yaml_schema(_make_ctx(tmp_path / "missing.yaml"))
    assert result.severity == CheckSeverity.ERROR
    assert "does not exist" in result.details["problems"][0]


def test_yaml_schema_reports_parse_error(tmp_path: Path) -> None:
    yaml_path = tmp_path / "broken.yaml"
    yaml_path.write_text("names: [a,\n", encoding="utf-8")

    result = validate_yaml_schema(_make_ctx(yaml_path))
    assert result.severity == CheckSeverity.ERROR
    assert result.summary == "YAML file cannot be parsed."


def test_yaml_schema_reports_non_mapping_root(tmp_path: Path) -> None:
    yaml_path = tmp_path / "list.yaml"
    yaml_path.write_text("- a\n- b\n", encoding="utf-8")

    result = validate_yaml_schema(_make_ctx(yaml_path))
    assert result.severity == CheckSeverity.ERROR
    assert "top-level" in result.details["problems"][0]


def test_yaml_schema_collects_multiple_field_problems(tmp_path: Path) -> None:
    yaml_path = tmp_path / "fields.yaml"
    yaml_path.write_text("train: train/images\n", encoding="utf-8")

    result = validate_yaml_schema(_make_ctx(yaml_path))
    assert result.severity == CheckSeverity.ERROR
    assert "nc is required and must be a positive integer" in result.details["problems"]
    assert "names is required" in result.details["problems"]


def test_yaml_schema_reports_names_length_mismatch(tmp_path: Path) -> None:
    yaml_path = tmp_path / "mismatch.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                "nc: 3",
                "names:",
                "  - a",
                "  - b",
            ]
        ),
        encoding="utf-8",
    )

    result = validate_yaml_schema(_make_ctx(yaml_path))
    assert result.severity == CheckSeverity.ERROR
    assert "nc (3) does not match names length (2)" in result.details["problems"]


def test_yaml_schema_accepts_list_and_dict_names_forms(tmp_path: Path) -> None:
    list_yaml = tmp_path / "list_names.yaml"
    list_yaml.write_text(
        "\n".join(
            [
                "nc: 2",
                "names:",
                "  - aircraft",
                "  - ship",
            ]
        ),
        encoding="utf-8",
    )
    dict_yaml = tmp_path / "dict_names.yaml"
    dict_yaml.write_text(
        "\n".join(
            [
                "nc: 2",
                "names:",
                "  0: aircraft",
                "  1: ship",
            ]
        ),
        encoding="utf-8",
    )

    list_result = validate_yaml_schema(_make_ctx(list_yaml))
    dict_result = validate_yaml_schema(_make_ctx(dict_yaml))

    assert list_result.severity == CheckSeverity.PASS
    assert dict_result.severity == CheckSeverity.PASS
