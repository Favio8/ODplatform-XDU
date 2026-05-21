"""Cross-split image uniqueness check."""

from __future__ import annotations

from itertools import combinations
from typing import Final

from odp_platform.data_validation.registry import CheckContext, CheckResult, CheckSeverity, check


DETAILS_PREVIEW_LIMIT: Final[int] = 20


@check("split_uniqueness")
def validate_split_uniqueness(ctx: CheckContext) -> CheckResult:
    snapshot = ctx.snapshot
    overlaps: dict[str, dict[str, object]] = {}
    total_duplicates = 0

    split_to_stems = {
        split: {path.stem for path in snapshot.images_per_split.get(split, ())}
        for split in snapshot.splits
    }

    for left_split, right_split in combinations(snapshot.splits, 2):
        duplicates = sorted(split_to_stems[left_split] & split_to_stems[right_split])
        pair_key = f"{left_split}|{right_split}"
        overlaps[pair_key] = {
            "left_split": left_split,
            "right_split": right_split,
            "duplicates": duplicates,
            "preview": duplicates[:DETAILS_PREVIEW_LIMIT],
            "count": len(duplicates),
        }
        total_duplicates += len(duplicates)

    if total_duplicates == 0:
        return CheckResult(
            name="split_uniqueness",
            severity=CheckSeverity.PASS,
            summary="No cross-split image stem overlaps found.",
            details={
                "splits": list(snapshot.splits),
                "total_duplicates": 0,
                "overlaps": overlaps,
            },
        )

    return CheckResult(
        name="split_uniqueness",
        severity=CheckSeverity.ERROR,
        summary=f"Found {total_duplicates} cross-split duplicate stem(s).",
        details={
            "splits": list(snapshot.splits),
            "total_duplicates": total_duplicates,
            "overlaps": overlaps,
        },
    )
