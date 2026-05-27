from __future__ import annotations

import numpy as np
import pytest

from odp_platform.visualization import (
    BeautifyVisualizer,
    Detection,
    DrawStyle,
    LabelPosition,
    LayoutCalculator,
    TextSizeCache,
)


def test_draw_style_validates_text_color() -> None:
    with pytest.raises(ValueError, match="text_color"):
        DrawStyle(text_color=(256, 0, 0))


def test_draw_style_scales_from_image_size() -> None:
    style = DrawStyle.from_image_size(1080, 1920)
    assert style.font_size >= 10
    assert style.line_width >= 1
    assert style.radius >= 3


def test_layout_calculator_positions_label_above_when_space_exists() -> None:
    style = DrawStyle(font_size=16, padding_x=4, padding_y=4, line_width=1, radius=3)
    layout = LayoutCalculator.compute((20, 40, 120, 140), (60, 18), (200, 200), style)
    assert layout.position is LabelPosition.ABOVE


def test_text_size_cache_returns_scaled_fallback_for_unknown_font_size() -> None:
    cache = TextSizeCache(labels=["person"], font_sizes=(12, 16))
    width, height = cache.get_size("person", 20)
    assert width > 0
    assert height > 0


def test_visualizer_draw_returns_modified_copy() -> None:
    image = np.zeros((160, 200, 3), dtype=np.uint8)
    visualizer = BeautifyVisualizer(
        labels=["person"],
        label_mapping={"person": "人员"},
        color_mapping={"person": (0, 255, 0)},
    )
    detections = [
        Detection(box=(20, 30, 120, 130), confidence=0.95, label="person"),
    ]

    annotated = visualizer.draw(image, detections, use_label_mapping=True)

    assert annotated.shape == image.shape
    assert not np.shares_memory(annotated, image)
    assert np.any(annotated != image)


def test_visualizer_returns_copy_when_no_detections() -> None:
    image = np.zeros((40, 60, 3), dtype=np.uint8)
    visualizer = BeautifyVisualizer(labels=["person"])

    annotated = visualizer.draw(image, [])

    assert np.array_equal(annotated, image)
    assert annotated is not image


def test_from_yolo_results_builds_detection_values() -> None:
    boxes = np.array([[1, 2, 30, 40]], dtype=float)
    confidences = np.array([0.8], dtype=float)
    labels = ["door"]

    detections = BeautifyVisualizer.from_yolo_results(boxes, confidences, labels)

    assert detections == [
        Detection(box=(1, 2, 30, 40), confidence=0.8, label="door", color=(0, 255, 0))
    ]
