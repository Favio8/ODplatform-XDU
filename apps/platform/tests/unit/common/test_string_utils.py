from __future__ import annotations

import pytest

from odp_platform.common.string_utils import (
    format_table_row,
    format_table_separator,
    get_display_width,
    pad_to_width,
    slugify,
)


def test_slugify_normalizes_basic_text() -> None:
    assert slugify("  Hello, ODPlatform  ") == "hello-odplatform"


def test_get_display_width_handles_cjk_and_combining_characters() -> None:
    assert get_display_width("hello") == 5
    assert get_display_width("你好") == 4
    assert get_display_width("hi你好") == 6
    assert get_display_width("e\u0301") == 1


def test_pad_to_width_supports_left_right_and_center_alignment() -> None:
    assert pad_to_width("你好", 6) == "你好  "
    assert pad_to_width("42", 4, align="right") == "  42"
    assert pad_to_width("A", 5, align="center") == "  A  "


def test_pad_to_width_rejects_invalid_arguments() -> None:
    with pytest.raises(ValueError):
        pad_to_width("x", -1)

    with pytest.raises(ValueError):
        pad_to_width("x", 3, align="middle")  # type: ignore[arg-type]


def test_format_table_row_aligns_mixed_language_columns() -> None:
    row = format_table_row(["ID", "反光衣", 116], [4, 10, 5], ["left", "left", "right"])

    assert row == "ID   反光衣       116"


def test_format_table_row_requires_matching_lengths() -> None:
    with pytest.raises(ValueError):
        format_table_row(["A", "B"], [4], ["left"])


def test_format_table_separator_uses_widths_and_validates_input() -> None:
    assert format_table_separator([4, 10, 5]) == "-" * 21

    with pytest.raises(ValueError):
        format_table_separator([4, -1], char="-")

    with pytest.raises(ValueError):
        format_table_separator([4, 2], char="")
