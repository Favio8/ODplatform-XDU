from __future__ import annotations

from odp_platform.inference.room_segmentation import RoomSegmentationOptions, run_room_segmentation


class ModelHandler:
    def __init__(self, model_path: str):
        from ultralytics import YOLO

        self.model = YOLO(model_path)

    def predict(self, image_bytes: bytes) -> dict:
        result = run_room_segmentation(
            model=self.model,
            source=image_bytes,
            source_name="uploaded_floorplan",
            options=RoomSegmentationOptions(device=None, alpha=0.4, label_threshold=0.02),
        )
        return {
            "image_size": result.payload["image_size"],
            "rooms": result.rooms,
            "room_count": len(result.rooms),
            "visualization": result.rendered_base64,
            "raw_result": result.payload,
        }
