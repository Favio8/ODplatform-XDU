"""Dataset splitting helpers."""

from __future__ import annotations

from random import Random

from odp_platform.common.constants import (
    RATE_EPSILON,
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VAL,
)
from odp_platform.data_pipeline.split.manifest import PreparedSample


def _validate_rates(train_rate: float, val_rate: float, test_rate: float) -> None:
    if train_rate <= 0.0 or train_rate >= 1.0:
        raise ValueError("train_rate must be between 0 and 1")
    if val_rate < 0.0 or val_rate >= 1.0:
        raise ValueError("val_rate must be between 0 and 1")
    if test_rate < 0.0 or test_rate >= 1.0:
        raise ValueError("test_rate must be between 0 and 1")
    if abs((train_rate + val_rate + test_rate) - 1.0) > RATE_EPSILON:
        raise ValueError("train_rate + val_rate + test_rate must equal 1.0")


def _fallback_split(
    samples: list[PreparedSample],
    *,
    train_rate: float,
    val_rate: float,
    test_rate: float,
    random_state: int,
) -> dict[str, list[PreparedSample]]:
    shuffled = list(samples)
    Random(random_state).shuffle(shuffled)
    total = len(shuffled)

    train_count = max(1, int(round(total * train_rate)))
    remaining = total - train_count
    val_count = int(round(total * val_rate))
    val_count = min(val_count, remaining)
    test_count = total - train_count - val_count

    train_samples = shuffled[:train_count]
    val_samples = shuffled[train_count : train_count + val_count]
    test_samples = shuffled[train_count + val_count : train_count + val_count + test_count]
    return {
        SPLIT_TRAIN: train_samples,
        SPLIT_VAL: val_samples,
        SPLIT_TEST: test_samples,
    }


def split_pairs(
    samples: list[PreparedSample],
    *,
    train_rate: float,
    val_rate: float,
    test_rate: float,
    random_state: int,
) -> dict[str, list[PreparedSample]]:
    """Split converted samples into train/val/test buckets."""

    if not samples:
        raise ValueError("samples must not be empty")

    _validate_rates(train_rate, val_rate, test_rate)
    ordered_samples = sorted(samples, key=lambda item: item.stem)

    if len(ordered_samples) == 1:
        if val_rate > RATE_EPSILON or test_rate > RATE_EPSILON:
            raise ValueError("At least two samples are required when val/test splits are requested")
        return {
            SPLIT_TRAIN: ordered_samples,
            SPLIT_VAL: [],
            SPLIT_TEST: [],
        }

    try:
        from sklearn.model_selection import train_test_split
    except ModuleNotFoundError:
        return _fallback_split(
            ordered_samples,
            train_rate=train_rate,
            val_rate=val_rate,
            test_rate=test_rate,
            random_state=random_state,
        )

    if val_rate <= RATE_EPSILON and test_rate <= RATE_EPSILON:
        return {
            SPLIT_TRAIN: ordered_samples,
            SPLIT_VAL: [],
            SPLIT_TEST: [],
        }

    train_samples, holdout_samples = train_test_split(
        ordered_samples,
        train_size=train_rate,
        random_state=random_state,
        shuffle=True,
    )

    if not holdout_samples:
        return {
            SPLIT_TRAIN: sorted(train_samples, key=lambda item: item.stem),
            SPLIT_VAL: [],
            SPLIT_TEST: [],
        }

    if val_rate <= RATE_EPSILON:
        val_samples = []
        test_samples = holdout_samples
    elif test_rate <= RATE_EPSILON:
        val_samples = holdout_samples
        test_samples = []
    else:
        holdout_train_ratio = val_rate / (val_rate + test_rate)
        val_samples, test_samples = train_test_split(
            holdout_samples,
            train_size=holdout_train_ratio,
            random_state=random_state,
            shuffle=True,
        )

    return {
        SPLIT_TRAIN: sorted(train_samples, key=lambda item: item.stem),
        SPLIT_VAL: sorted(val_samples, key=lambda item: item.stem),
        SPLIT_TEST: sorted(test_samples, key=lambda item: item.stem),
    }


__all__ = [
    "split_pairs",
]
