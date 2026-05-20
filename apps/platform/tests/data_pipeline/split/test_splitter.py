from pathlib import Path

from odp_platform.data_pipeline.split.manifest import PreparedSample
from odp_platform.data_pipeline.split.splitter import split_pairs


def _sample(index: int) -> PreparedSample:
    stem = f"sample_{index:03d}"
    return PreparedSample(
        stem=stem,
        image_path=Path(f"/tmp/{stem}.jpg"),
        label_path=Path(f"/tmp/{stem}.txt"),
        class_names=("aircraft",),
    )


def _samples(count: int) -> list[PreparedSample]:
    return [_sample(index) for index in range(count)]


def test_split_pairs_rejects_empty_samples() -> None:
    try:
        split_pairs([], train_rate=0.8, val_rate=0.1, test_rate=0.1, random_state=42)
    except ValueError as exc:
        assert "must not be empty" in str(exc)
    else:
        raise AssertionError("empty samples should raise")


def test_split_pairs_rejects_non_positive_train_rate() -> None:
    try:
        split_pairs(_samples(4), train_rate=0.0, val_rate=0.5, test_rate=0.5, random_state=42)
    except ValueError as exc:
        assert "train_rate" in str(exc)
    else:
        raise AssertionError("bad train rate should raise")


def test_split_pairs_rejects_negative_val_rate() -> None:
    try:
        split_pairs(_samples(4), train_rate=0.8, val_rate=-0.1, test_rate=0.3, random_state=42)
    except ValueError as exc:
        assert "val_rate" in str(exc)
    else:
        raise AssertionError("negative val rate should raise")


def test_split_pairs_rejects_invalid_total_rate() -> None:
    try:
        split_pairs(_samples(4), train_rate=0.6, val_rate=0.2, test_rate=0.3, random_state=42)
    except ValueError as exc:
        assert "must equal 1.0" in str(exc)
    else:
        raise AssertionError("invalid total rate should raise")


def test_split_pairs_rejects_single_sample_with_holdout() -> None:
    try:
        split_pairs(_samples(1), train_rate=0.8, val_rate=0.1, test_rate=0.1, random_state=42)
    except ValueError as exc:
        assert "At least two samples" in str(exc)
    else:
        raise AssertionError("single sample with holdout should raise")


def test_split_pairs_returns_single_train_sample_when_no_holdout() -> None:
    split_map = split_pairs(_samples(1), train_rate=1.0 - 1e-9, val_rate=0.0, test_rate=0.0, random_state=42)
    assert len(split_map["train"]) == 1
    assert split_map["val"] == []
    assert split_map["test"] == []


def test_split_pairs_is_deterministic() -> None:
    left = split_pairs(_samples(10), train_rate=0.8, val_rate=0.1, test_rate=0.1, random_state=42)
    right = split_pairs(_samples(10), train_rate=0.8, val_rate=0.1, test_rate=0.1, random_state=42)
    assert [sample.stem for sample in left["train"]] == [sample.stem for sample in right["train"]]
    assert [sample.stem for sample in left["val"]] == [sample.stem for sample in right["val"]]
    assert [sample.stem for sample in left["test"]] == [sample.stem for sample in right["test"]]


def test_split_pairs_preserves_total_count() -> None:
    split_map = split_pairs(_samples(11), train_rate=0.8, val_rate=0.1, test_rate=0.1, random_state=7)
    total = len(split_map["train"]) + len(split_map["val"]) + len(split_map["test"])
    assert total == 11
