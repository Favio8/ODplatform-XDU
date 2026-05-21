"""Image-label pair existence check."""

from __future__ import annotations

from typing import Final

from odp_platform.common.constants import PAIR_MISSING_ERROR_RATIO, PAIR_MISSING_WARN_RATIO
from odp_platform.data_validation.registry import CheckContext, CheckResult, CheckSeverity, check


DETAILS_PREVIEW_LIMIT: Final[int] = 20


@check("pair_existence")
def validate_pair_existence(ctx: CheckContext) -> CheckResult:
    snapshot = ctx.snapshot
    missing_per_split: dict[str, int] = {}
    missing_examples: dict[str, list[str]] = {}
    total_missing = 0

    for split in snapshot.splits:
        label_stems = {path.stem for path in snapshot.labels_per_split.get(split, ())}
        split_missing = [
            f"{image_path.stem}.txt"
            for image_path in snapshot.images_per_split.get(split, ())
            if image_path.stem not in label_stems
        ]
        missing_per_split[split] = len(split_missing)
        missing_examples[split] = split_missing[:DETAILS_PREVIEW_LIMIT]
        total_missing += len(split_missing)

    total_images = snapshot.total_images
    missing_ratio = (total_missing / total_images) if total_images > 0 else 0.0

    if missing_ratio == 0.0:
        severity = CheckSeverity.PASS
    elif missing_ratio < PAIR_MISSING_WARN_RATIO:
        severity = CheckSeverity.INFO
    elif missing_ratio < PAIR_MISSING_ERROR_RATIO:
        severity = CheckSeverity.WARNING
    else:
        severity = CheckSeverity.ERROR

    summary = f"Missing labels for {total_missing}/{total_images} images ({missing_ratio:.2%})."
    return CheckResult(
        name="pair_existence",
        severity=severity,
        summary=summary,
        details={
            "total_images": total_images,
            "total_missing": total_missing,
            "missing_ratio": missing_ratio,
            "thresholds": {
                "warn_ratio": PAIR_MISSING_WARN_RATIO,
                "error_ratio": PAIR_MISSING_ERROR_RATIO,
            },
            "missing_per_split": missing_per_split,
            "missing_examples": missing_examples,
        },
    )
