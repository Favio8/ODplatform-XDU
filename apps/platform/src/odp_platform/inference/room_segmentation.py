"""Room segmentation inference helpers shared by CLI and Web backend."""

from __future__ import annotations

import base64
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RoomSegmentationOptions:
    """Rendering and filtering options for room segmentation inference."""

    device: str | None = None
    alpha: float = 0.4
    label_threshold: float = 0.02
    min_area_ratio: float = 0.01
    render: bool = True


@dataclass(frozen=True)
class RoomSegmentationResult:
    """Structured room segmentation result."""

    payload: dict[str, Any]
    rooms: list[dict[str, Any]]
    image_bgr: np.ndarray
    rendered_bgr: np.ndarray | None

    @property
    def rendered_base64(self) -> str:
        image = self.rendered_bgr if self.rendered_bgr is not None else self.image_bgr
        ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise RuntimeError("Failed to encode rendered segmentation image.")
        return base64.b64encode(buffer).decode("utf-8")


def run_room_segmentation(
    *,
    model: Any,
    source: str | Path | bytes | np.ndarray,
    source_name: str | None = None,
    options: RoomSegmentationOptions | None = None,
) -> RoomSegmentationResult:
    """Run YOLO segmentation and return JSON-compatible data plus rendered image."""

    options = options or RoomSegmentationOptions()
    image_bgr = _load_source_image(source)
    yolo_source = str(source) if isinstance(source, (str, Path)) else image_bgr
    predict_kwargs: dict[str, Any] = {}
    if options.device is not None:
        predict_kwargs["device"] = options.device

    results = model(yolo_source, **predict_kwargs)
    if not results:
        raise RuntimeError("YOLO returned no inference result.")
    result = results[0]
    image_path = str(source_name or getattr(result, "path", "") or "")
    height, width = image_bgr.shape[:2]

    output: dict[str, Any] = {
        "image_path": image_path,
        "image_size": {"width": int(width), "height": int(height)},
        "detections": [],
    }

    boxes_obj = getattr(result, "boxes", None)
    if boxes_obj is None or len(boxes_obj) == 0:
        logger.warning("No room detected.")
        rendered = _render_beautiful(image_bgr, output, [], None, options) if options.render else None
        return RoomSegmentationResult(payload=output, rooms=[], image_bgr=image_bgr, rendered_bgr=rendered)

    boxes_all = _to_numpy(boxes_obj.xyxy)
    confs_all = _to_numpy(boxes_obj.conf)
    cls_ids_all = _to_numpy(boxes_obj.cls).astype(int, copy=False)
    masks_obj = getattr(result, "masks", None)
    masks_xy_all = getattr(masks_obj, "xy", None) if masks_obj is not None else None
    masks_data_all = getattr(masks_obj, "data", None) if masks_obj is not None else None
    names = getattr(model, "names", None) or getattr(result, "names", {}) or {}

    areas_all = _mask_areas(masks_data_all, len(boxes_all))
    total_area_all = float(sum(areas_all))
    if total_area_all <= 0:
        logger.warning("No valid mask area found; returning empty segmentation result.")
        rendered = _render_beautiful(image_bgr, output, [], None, options) if options.render else None
        return RoomSegmentationResult(payload=output, rooms=[], image_bgr=image_bgr, rendered_bgr=rendered)

    valid_indices = [
        index
        for index, area in enumerate(areas_all)
        if area / total_area_all >= options.min_area_ratio
    ]
    if not valid_indices:
        logger.warning("All detected rooms were filtered by area ratio threshold.")
        rendered = _render_beautiful(image_bgr, output, [], None, options) if options.render else None
        return RoomSegmentationResult(payload=output, rooms=[], image_bgr=image_bgr, rendered_bgr=rendered)

    boxes = boxes_all[valid_indices]
    confs = confs_all[valid_indices]
    cls_ids = cls_ids_all[valid_indices]
    areas = [areas_all[index] for index in valid_indices]
    total_area = float(sum(areas))
    masks_data = _slice_tensor_or_array(masks_data_all, valid_indices)
    mask_array = _to_numpy(masks_data) if masks_data is not None else None
    masks_xy = [masks_xy_all[index] for index in valid_indices] if masks_xy_all is not None else None

    rooms: list[dict[str, Any]] = []
    for index, box in enumerate(boxes):
        ratio = float(areas[index] / total_area) if total_area > 0 else 0.0
        class_id = int(cls_ids[index])
        class_name = _class_name(names, class_id)
        segmentation = _mask_contours(mask_array[index]) if mask_array is not None and index < len(mask_array) else []
        polygon = _polygon_from_mask_xy(masks_xy[index]) if masks_xy is not None and index < len(masks_xy) else []
        detection = {
            "class_id": class_id,
            "class_name": class_name,
            "confidence": float(round(float(confs[index]), 4)),
            "bbox": {
                "x1": float(round(float(box[0]), 2)),
                "y1": float(round(float(box[1]), 2)),
                "x2": float(round(float(box[2]), 2)),
                "y2": float(round(float(box[3]), 2)),
            },
            "area": float(round(float(areas[index]), 4)),
            "area_ratio": float(round(ratio, 4)),
            "segmentation": segmentation,
        }
        output["detections"].append(detection)
        room = {
            "id": index + 1,
            "class_id": class_id,
            "class_name": class_name,
            "confidence": detection["confidence"],
            "bbox": detection["bbox"],
            "area": detection["area"],
            "area_ratio": detection["area_ratio"],
            "segmentation": segmentation,
        }
        if polygon:
            room["polygon"] = polygon
        rooms.append(room)

    rendered = _render_beautiful(image_bgr, output, boxes, mask_array, options) if options.render else None
    return RoomSegmentationResult(payload=output, rooms=rooms, image_bgr=image_bgr, rendered_bgr=rendered)


def write_room_segmentation_outputs(
    result: RoomSegmentationResult,
    *,
    output_json: str | Path,
    output_img: str | Path | None = None,
) -> None:
    """Persist room segmentation JSON and optional rendered image."""

    json_path = Path(output_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result.payload, indent=2, ensure_ascii=False), encoding="utf-8")

    if output_img is not None and result.rendered_bgr is not None:
        img_path = Path(output_img)
        img_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(img_path), result.rendered_bgr):
            raise RuntimeError(f"Failed to save rendered image: {img_path}")


def _load_source_image(source: str | Path | bytes | np.ndarray) -> np.ndarray:
    if isinstance(source, np.ndarray):
        return source.copy()
    if isinstance(source, bytes):
        arr = np.frombuffer(source, np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Input bytes are not a valid image.")
        return image
    image = cv2.imread(str(source))
    if image is None:
        raise ValueError(f"Cannot read source image: {source}")
    return image


def _to_numpy(value: Any) -> np.ndarray:
    if value is None:
        return np.array([])
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _slice_tensor_or_array(value: Any, indices: list[int]) -> Any:
    if value is None:
        return None
    try:
        return value[indices]
    except Exception:
        array = _to_numpy(value)
        return array[indices]


def _mask_areas(masks_data: Any, expected_count: int) -> list[float]:
    if masks_data is None:
        return [0.0] * expected_count
    mask_array = _to_numpy(masks_data)
    if mask_array.size == 0:
        return [0.0] * expected_count
    return [float((mask_array[index] > 0.5).sum().item()) for index in range(mask_array.shape[0])]


def _mask_contours(mask: np.ndarray) -> list[list[int]]:
    mask_uint8 = (mask > 0.5).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: list[list[int]] = []
    for contour in contours:
        contour = contour.squeeze(1)
        if contour.ndim == 2 and contour.shape[0] >= 3:
            polygons.append(contour.flatten().astype(int).tolist())
    return polygons


def _polygon_from_mask_xy(mask_xy: Any) -> list[list[float]]:
    points = np.asarray(mask_xy)
    if points.ndim != 2 or points.shape[0] < 3:
        return []
    return [[round(float(x), 2), round(float(y), 2)] for x, y in points[:, :2]]


def _class_name(names: Any, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, list) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def _render_beautiful(
    image_bgr: np.ndarray,
    output: dict[str, Any],
    boxes: np.ndarray | list[Any],
    mask_array: np.ndarray | None,
    options: RoomSegmentationOptions,
) -> np.ndarray:
    img = image_bgr.copy()
    if mask_array is None or len(output["detections"]) == 0:
        return img

    num_rooms = len(output["detections"])
    cmap = plt.get_cmap("Set1", max(num_rooms, 1))
    colors = [
        (
            int(cmap(index)[2] * 255),
            int(cmap(index)[1] * 255),
            int(cmap(index)[0] * 255),
        )
        for index in range(num_rooms)
    ]

    overlay = np.zeros_like(img, dtype=np.uint8)
    mask_sum = np.zeros(img.shape[:2], dtype=np.uint8)
    for index in range(num_rooms):
        color = colors[index % len(colors)]
        instance_mask = (mask_array[index] > 0.5).astype(np.uint8) * 255
        if instance_mask.shape[:2] != img.shape[:2]:
            instance_mask = cv2.resize(
                instance_mask,
                (img.shape[1], img.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        if instance_mask.sum() == 0:
            continue

        new_mask = instance_mask & (mask_sum == 0)
        if new_mask.sum() > 0:
            colored_layer = np.zeros_like(img, dtype=np.uint8)
            colored_layer[:, :] = color
            colored_layer = cv2.bitwise_and(colored_layer, colored_layer, mask=new_mask)
            overlay = cv2.add(overlay, colored_layer)
            mask_sum = cv2.bitwise_or(mask_sum, new_mask)

        contours, _ = cv2.findContours(instance_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, color, 2)

        ratio = float(output["detections"][index]["area_ratio"])
        if ratio >= options.label_threshold and len(boxes) > index:
            _draw_room_label(img, boxes[index], f"R{index + 1} ({ratio:.1%})")

    return cv2.addWeighted(overlay, options.alpha, img, 1.0, 0)


def _draw_room_label(img: np.ndarray, box: Any, label: str) -> None:
    x1, y1, _, _ = [float(value) for value in box]
    font_scale = 0.5
    thickness = 2
    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    img_h, img_w = img.shape[:2]
    x_pos = int(x1)
    y_pos = int(y1) - 5
    if x_pos + text_w > img_w:
        x_pos = img_w - text_w - 2
    if x_pos < 0:
        x_pos = 2
    if y_pos - text_h < 0:
        y_pos = text_h + 5
    if y_pos > img_h - 2:
        y_pos = img_h - 2
    text_pos = (x_pos, y_pos)
    cv2.putText(img, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 4)
    cv2.putText(img, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (30, 30, 30), thickness)


__all__ = [
    "RoomSegmentationOptions",
    "RoomSegmentationResult",
    "run_room_segmentation",
    "write_room_segmentation_outputs",
]
