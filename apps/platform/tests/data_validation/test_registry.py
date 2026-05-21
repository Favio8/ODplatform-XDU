from __future__ import annotations

from pathlib import Path

import pytest

from odp_platform.common.constants import TASK_DETECT
from odp_platform.data_validation import list_checks
from odp_platform.data_validation.registry import (
    CheckContext,
    CheckEntry,
    CheckResult,
    CheckSeverity,
    check,
)
from odp_platform.data_validation.service import run_all_checks
from odp_platform.data_validation.snapshot import DatasetSnapshot, SplitStats, build_snapshot


def _make_snapshot(tmp_path: Path) -> DatasetSnapshot:
    return DatasetSnapshot(
        yaml_path=tmp_path / "dataset.yaml",
        yaml_data={},
        yaml_load_error=None,
        data_root=tmp_path,
        nc=1,
        class_names=("ship",),
        task_type=TASK_DETECT,
        images_per_split={},
        labels_per_split={},
        stats_per_split={},
        scan_warnings=(),
    )


def test_check_severity_rank_order_and_passed_property() -> None:
    assert CheckSeverity.rank(CheckSeverity.PASS) < CheckSeverity.rank(CheckSeverity.INFO)
    assert CheckSeverity.rank(CheckSeverity.INFO) < CheckSeverity.rank(CheckSeverity.WARNING)
    assert CheckSeverity.rank(CheckSeverity.WARNING) < CheckSeverity.rank(CheckSeverity.ERROR)

    assert CheckResult("demo", CheckSeverity.PASS, "ok", {}).passed is True
    assert CheckResult("demo", CheckSeverity.INFO, "ok", {}).passed is True
    assert CheckResult("demo", CheckSeverity.WARNING, "warn", {}).passed is False


def test_duplicate_registration_raises_value_error() -> None:
    check_name = "__duplicate_registration_test__"

    @check(check_name)
    def first(_: CheckContext) -> CheckResult:
        return CheckResult(check_name, CheckSeverity.PASS, "ok", {})

    try:
        with pytest.raises(ValueError):
            @check(check_name)
            def second(_: CheckContext) -> CheckResult:
                return CheckResult(check_name, CheckSeverity.PASS, "ok", {})
    finally:
        from odp_platform.data_validation import registry as registry_module

        registry_module._REGISTRY.pop(check_name, None)


def test_list_checks_includes_builtin_checks() -> None:
    names = list_checks()
    assert "yaml_schema" in names
    assert "pair_existence" in names
    assert "label_format" in names
    assert "split_uniqueness" in names


def test_run_all_checks_wraps_one_crashing_check_without_blocking_others(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import odp_platform.data_validation.service as service_module

    def ok_check(_: CheckContext) -> CheckResult:
        return CheckResult("ok_check", CheckSeverity.PASS, "ok", {})

    def crash_check(_: CheckContext) -> CheckResult:
        raise KeyError("boom")

    fake_registry = {
        "ok_check": CheckEntry(name="ok_check", func=ok_check),
        "crash_check": CheckEntry(name="crash_check", func=crash_check),
    }
    monkeypatch.setattr(service_module, "autodiscover_checks", lambda: None)
    monkeypatch.setattr(service_module, "get_registered_checks", lambda: fake_registry)

    ctx = CheckContext(yaml_path=tmp_path / "demo.yaml", snapshot=_make_snapshot(tmp_path))
    results = run_all_checks(ctx)

    assert [result.name for result in results] == ["ok_check", "crash_check"]
    assert results[0].severity == CheckSeverity.PASS
    assert results[1].severity == CheckSeverity.ERROR
    assert results[1].details["exception_type"] == "KeyError"


def test_build_snapshot_is_best_effort_and_split_tuples_are_immutable(tmp_path: Path) -> None:
    yaml_path = tmp_path / "broken.yaml"
    yaml_path.write_text(":\n", encoding="utf-8")

    snapshot = build_snapshot(yaml_path)
    assert snapshot.yaml_load_error is not None
    assert snapshot.task_type == TASK_DETECT

    healthy_yaml = tmp_path / "healthy.yaml"
    data_root = tmp_path / "prepared"
    train_images = data_root / "train" / "images"
    train_labels = data_root / "train" / "labels"
    train_images.mkdir(parents=True)
    train_labels.mkdir(parents=True)
    (train_images / "sample.jpg").write_bytes(b"fake")
    (train_labels / "sample.txt").write_text("", encoding="utf-8")
    healthy_yaml.write_text(
        "\n".join(
            [
                f"path: {data_root}",
                "train: train/images",
                "nc: 1",
                "names:",
                "  0: ship",
            ]
        ),
        encoding="utf-8",
    )

    healthy_snapshot = build_snapshot(healthy_yaml)
    with pytest.raises(AttributeError):
        healthy_snapshot.images_per_split["train"].append(Path("another.jpg"))  # type: ignore[attr-defined]

    assert isinstance(healthy_snapshot.stats_per_split["train"], SplitStats)
