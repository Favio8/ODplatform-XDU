from odp_platform.common import constants
from odp_platform.data_pipeline import ConvertOptions, get_converter, list_capabilities
from odp_platform.data_pipeline.service import DataPipelineService


def test_convert_options_defaults() -> None:
    options = ConvertOptions(dataset_name="rsod", source_format=constants.FORMAT_PASCAL_VOC)
    assert options.task == constants.TASK_DETECT
    assert options.train_rate == constants.DEFAULT_TRAIN_RATE
    assert options.val_rate == constants.DEFAULT_VAL_RATE
    assert options.test_rate == constants.DEFAULT_TEST_RATE
    assert options.classes is None


def test_convert_options_normalizes_classes() -> None:
    options = ConvertOptions(
        dataset_name="rsod",
        source_format=constants.FORMAT_PASCAL_VOC,
        classes=[" aircraft ", "", "playground"],
    )
    assert options.classes == ["aircraft", "playground"]


def test_convert_options_rejects_unknown_format() -> None:
    try:
        ConvertOptions(dataset_name="rsod", source_format="unknown")
    except ValueError as exc:
        assert "Unsupported source format" in str(exc)
    else:
        raise AssertionError("ConvertOptions should reject unknown formats")


def test_convert_options_rejects_bad_rates() -> None:
    try:
        ConvertOptions(
            dataset_name="rsod",
            source_format=constants.FORMAT_PASCAL_VOC,
            train_rate=0.7,
            val_rate=0.2,
            test_rate=0.2,
        )
    except ValueError as exc:
        assert "must equal 1.0" in str(exc)
    else:
        raise AssertionError("ConvertOptions should reject invalid split rates")


def test_list_capabilities_matches_expected() -> None:
    assert list_capabilities() == {
        "pascal_voc": ("detect",),
        "coco": ("detect", "segment"),
        "yolo": ("detect",),
    }


def test_get_converter_returns_pascal_voc_module() -> None:
    converter = get_converter(constants.FORMAT_PASCAL_VOC)
    assert converter.SUPPORTED_SOURCE_FORMAT == constants.FORMAT_PASCAL_VOC


def test_service_rejects_unsupported_task() -> None:
    service = DataPipelineService()
    options = ConvertOptions(
        dataset_name="demo",
        source_format=constants.FORMAT_PASCAL_VOC,
        task=constants.TASK_SEGMENT,
    )

    try:
        service.convert(options, source_root="unused", output_labels_dir="unused")
    except ValueError as exc:
        assert "does not support task" in str(exc)
    else:
        raise AssertionError("unsupported format-task pair should raise")
