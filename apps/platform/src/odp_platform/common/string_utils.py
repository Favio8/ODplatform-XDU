"""String helpers."""

from __future__ import annotations

import re
import unicodedata
from typing import Final, Literal, Sequence


Alignment = Literal["left", "right", "center"]
_WIDE_EAST_ASIAN_WIDTHS: Final[frozenset[str]] = frozenset({"W", "F"})


def slugify(value: str) -> str:
    """Convert text into a filesystem- and URL-friendly slug."""

    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower())
    return normalized.strip("-")


def get_display_width(text: str) -> int:
    """Return the terminal display width of a string."""

    width = 0
    for char in text:
        if unicodedata.combining(char):
            continue
        width += 2 if unicodedata.east_asian_width(char) in _WIDE_EAST_ASIAN_WIDTHS else 1
    return width


def pad_to_width(text: str, width: int, align: Alignment = "left") -> str:
    """Pad text to a target display width with CJK-aware width handling."""

    if width < 0:
        raise ValueError("width must be non-negative")
    if align not in {"left", "right", "center"}:
        raise ValueError(f"Unsupported align value: {align!r}")

    current_width = get_display_width(text)
    padding = width - current_width
    if padding <= 0:
        return text

    if align == "right":
        return f"{' ' * padding}{text}"
    if align == "center":
        left_padding = padding // 2
        right_padding = padding - left_padding
        return f"{' ' * left_padding}{text}{' ' * right_padding}"
    return f"{text}{' ' * padding}"


def format_table_row(
    columns: Sequence[object],
    widths: Sequence[int],
    aligns: Sequence[Alignment] | None = None,
) -> str:
    """Format one plain-text table row with display-width-aware padding."""

    if aligns is None:
        aligns = ["left"] * len(columns)
    if not (len(columns) == len(widths) == len(aligns)):
        raise ValueError("columns, widths, and aligns must have the same length")

    parts = [pad_to_width(str(column), width, align) for column, width, align in zip(columns, widths, aligns)]
    return " ".join(parts)


def format_table_separator(widths: Sequence[int], char: str = "-") -> str:
    """Create a separator line that matches the formatted table width."""

    if not char:
        raise ValueError("char must not be empty")
    if any(width < 0 for width in widths):
        raise ValueError("widths must be non-negative")

    total_width = sum(widths) + max(len(widths) - 1, 0)
    return char[0] * total_width


__all__ = [
    "Alignment",
    "format_table_row",
    "format_table_separator",
    "get_display_width",
    "pad_to_width",
    "slugify",
]
