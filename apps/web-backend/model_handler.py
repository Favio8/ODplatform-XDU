import base64
import io
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO


class ModelHandler:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def predict(self, image_bytes: bytes) -> dict:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        results = self.model(img)
        result = results[0]
        h, w = result.orig_shape[:2]

        rooms = []
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            masks = result.masks.xy if result.masks is not None else None

            for i in range(len(boxes)):
                room = {
                    "id": i + 1,
                    "confidence": round(float(confs[i]), 4),
                    "bbox": {
                        "x1": round(float(boxes[i][0]), 2),
                        "y1": round(float(boxes[i][1]), 2),
                        "x2": round(float(boxes[i][2]), 2),
                        "y2": round(float(boxes[i][3]), 2),
                    },
                    "area_ratio": round(
                        float((boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]))
                        / (w * h),
                        4,
                    ),
                }
                if masks is not None and i < len(masks):
                    room["polygon"] = [
                        [round(float(pt), 2) for pt in poly] for poly in masks[i]
                    ]
                rooms.append(room)

        vis_b64 = self._render_visualization(img, result)

        return {
            "image_size": {"width": w, "height": h},
            "rooms": rooms,
            "room_count": len(rooms),
            "visualization": vis_b64,
        }

    def _render_visualization(self, img: np.ndarray, result) -> str:
        overlay = img.copy()

        if result.boxes is not None:
            n = len(result.boxes)
            cmap = plt.get_cmap("Pastel1", max(n, 1))
            colors = []
            for i in range(n):
                rgba = cmap(i)
                bgr = (
                    int(rgba[2] * 255),
                    int(rgba[1] * 255),
                    int(rgba[0] * 255),
                )
                colors.append(bgr)

            if result.masks is not None:
                masks_xy = result.masks.xy
                for i, mask in enumerate(masks_xy):
                    color = colors[i % len(colors)]
                    pts = np.array(mask, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.fillPoly(overlay, [pts], color)
                    cv2.polylines(
                        img, [pts], isClosed=True, color=color, thickness=2
                    )
                    cv2.putText(
                        img,
                        f"Room {i + 1}",
                        (pts[0][0][0], pts[0][0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                    )

            alpha = 0.35
            img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

        _, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return base64.b64encode(buffer).decode("utf-8")
