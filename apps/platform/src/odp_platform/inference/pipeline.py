"""Frame-source driven inference pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from odp_platform.common.constants import TASK_SEGMENT
from odp_platform.frame_source import Frame, FrameSource
from odp_platform.runtime_config.base import InferConfig
from odp_platform.visualization import BeautifyVisualizer, DrawStyle

from .components import FramePrediction, InferenceArtifact, InferenceSummary


def _tensor_to_numpy(value: Any) -> np.ndarray:
    if value is None:
        return np.array([])
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


@dataclass
class _VideoSink:
    path: Path
    writer: cv2.VideoWriter

    @classmethod
    def create(cls, path: Path, frame: Frame, fps: float | None) -> "_VideoSink":
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps if fps and fps > 0 else 25.0,
            (frame.width, frame.height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"无法创建视频输出: {path}")
        return cls(path=path, writer=writer)

    def write(self, image: np.ndarray) -> None:
        self.writer.write(image)

    def close(self) -> None:
        self.writer.release()


class InferencePipeline:
    """Run prediction frame by frame and render beautified outputs."""

    def __init__(
        self,
        *,
        model: Any,
        config: InferConfig,
        source: FrameSource,
        output_dir: Path,
    ) -> None:
        self._model = model
        self._config = config
        self._source = source
        self._output_dir = output_dir
        self._visualizer: BeautifyVisualizer | None = None
        self._video_sink: _VideoSink | None = None
        self._frame_outputs_dir = output_dir / "frames"
        self._labels_dir = output_dir / "labels"
        self._crop_dir = output_dir / "crops"

    def run(self) -> InferenceSummary:
        if not self._source.open():
            raise RuntimeError(f"无法打开输入源: {self._source.source_path}")

        artifacts: list[InferenceArtifact] = []
        frames_processed = 0
        detections_total = 0

        try:
            for frame in self._source:
                result = self._predict_one(frame)
                annotated = self._render_one(frame, result)
                prediction = self._persist_one(frame, result, annotated)
                frames_processed += 1
                detections_total += prediction.detections
                if prediction.output_path is not None:
                    kind = "video" if self._video_sink is not None else "image"
                    artifacts.append(
                        InferenceArtifact(
                            input_name=prediction.input_name,
                            output_path=prediction.output_path,
                            kind=kind,
                        )
                    )

                if self._config.show:
                    cv2.imshow("ODPlatform Inference", annotated)
                    key = cv2.waitKey(1 if self._source.get_source_type().value in {"camera", "video"} else 0)
                    if key in (27, ord("q"), ord("Q")):
                        break
        finally:
            self._source.close()
            if self._video_sink is not None:
                self._video_sink.close()
            if self._config.show:
                cv2.destroyAllWindows()

        deduped_artifacts = self._dedupe_artifacts(artifacts)
        return InferenceSummary(
            frames_processed=frames_processed,
            detections_total=detections_total,
            source=self._source.source_path,
            artifacts=tuple(deduped_artifacts),
        )

    def _predict_one(self, frame: Frame) -> Any:
        kwargs = {
            "conf": self._config.conf,
            "iou": self._config.iou,
            "max_det": self._config.max_det,
            "agnostic_nms": self._config.agnostic_nms,
            "augment": self._config.augment,
            "device": self._config.device,
            "imgsz": self._config.imgsz,
            "retina_masks": self._config.retina_masks,
            "visualize": self._config.visualize,
            "verbose": self._config.verbose,
        }
        if self._config.classes is not None:
            kwargs["classes"] = self._config.classes
        if self._config.embed is not None:
            kwargs["embed"] = self._config.embed

        results = self._model.predict(frame.image, **kwargs)
        if not results:
            raise RuntimeError("模型未返回任何推理结果。")
        return results[0]

    def _render_one(self, frame: Frame, result: Any) -> np.ndarray:
        names = getattr(result, "names", None) or getattr(self._model, "names", {}) or {}
        if isinstance(names, list):
            labels = [str(name) for name in names]
            names_dict = {index: label for index, label in enumerate(labels)}
        else:
            names_dict = {int(index): str(name) for index, name in dict(names).items()}
            labels = list(names_dict.values())

        if self._visualizer is None:
            self._visualizer = BeautifyVisualizer(labels=labels)

        if self._config.task_type == TASK_SEGMENT and getattr(result, "masks", None) is not None:
            rendered = result.plot(
                labels=False,
                conf=False,
                boxes=False,
                line_width=self._config.line_width,
            )
            base_image = np.asarray(rendered).copy()
        else:
            base_image = frame.image.copy()

        boxes = getattr(result, "boxes", None)
        xyxy = _tensor_to_numpy(getattr(boxes, "xyxy", None))
        if xyxy.size == 0:
            return base_image

        confidences = _tensor_to_numpy(getattr(boxes, "conf", None))
        class_ids = _tensor_to_numpy(getattr(boxes, "cls", None)).astype(int, copy=False)
        label_names = [names_dict.get(int(class_id), str(class_id)) for class_id in class_ids.tolist()]

        style_kwargs: dict[str, object] = {}
        if self._config.line_width is not None:
            style_kwargs["line_width"] = self._config.line_width
        style = DrawStyle.from_image_size(frame.height, frame.width, **style_kwargs)

        detections = BeautifyVisualizer.from_yolo_results(
            boxes=xyxy,
            confidences=confidences,
            labels=label_names,
        )
        return self._visualizer.draw(
            base_image,
            detections,
            style=style,
            use_label_mapping=False,
            draw_boxes=self._config.show_boxes,
            show_labels=self._config.show_labels,
            show_conf=self._config.show_conf,
        )

    def _persist_one(self, frame: Frame, result: Any, annotated: np.ndarray) -> FramePrediction:
        detections = int(_tensor_to_numpy(getattr(getattr(result, "boxes", None), "xyxy", None)).shape[0])
        inference_ms = None
        speed = getattr(result, "speed", None) or {}
        if speed and speed.get("inference") is not None:
            inference_ms = float(speed["inference"])

        output_path: Path | None = None
        source_type = self._source.get_source_type().value
        if self._config.save:
            if source_type in {"image", "image_folder"}:
                output_path = self._save_image(frame, annotated)
            else:
                output_path = self._save_video_frame(frame, annotated)

        if self._config.save_frames and source_type in {"video", "camera"}:
            self._save_frame_image(frame, annotated)
        if self._config.save_txt:
            self._save_txt(result, frame)
        if self._config.save_crop:
            self._save_crop(result, frame)

        return FramePrediction(
            frame_index=frame.info.frame_index,
            input_name=frame.info.filename or f"frame_{frame.info.frame_index:06d}",
            detections=detections,
            output_path=output_path,
            inference_ms=inference_ms,
        )

    def _save_image(self, frame: Frame, annotated: np.ndarray) -> Path:
        output_path = self._output_dir / (frame.info.filename or f"frame_{frame.info.frame_index:06d}.jpg")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), annotated):
            raise RuntimeError(f"保存图片失败: {output_path}")
        return output_path

    def _save_video_frame(self, frame: Frame, annotated: np.ndarray) -> Path:
        if self._video_sink is None:
            stem = Path(frame.info.filename or "stream").stem
            output_path = self._output_dir / f"{stem or 'stream'}_annotated.mp4"
            self._video_sink = _VideoSink.create(output_path, frame, frame.info.fps)
        self._video_sink.write(annotated)
        return self._video_sink.path

    def _save_frame_image(self, frame: Frame, annotated: np.ndarray) -> Path:
        self._frame_outputs_dir.mkdir(parents=True, exist_ok=True)
        output_path = self._frame_outputs_dir / f"frame_{frame.info.frame_index:06d}.jpg"
        if not cv2.imwrite(str(output_path), annotated):
            raise RuntimeError(f"保存帧图失败: {output_path}")
        return output_path

    def _save_txt(self, result: Any, frame: Frame) -> Path | None:
        save_txt = getattr(result, "save_txt", None)
        if callable(save_txt):
            self._labels_dir.mkdir(parents=True, exist_ok=True)
            output_path = self._labels_dir / f"{Path(frame.info.filename or f'frame_{frame.info.frame_index:06d}').stem}.txt"
            save_txt(str(output_path), save_conf=self._config.save_conf)
            return output_path
        return None

    def _save_crop(self, result: Any, frame: Frame) -> Path | None:
        save_crop = getattr(result, "save_crop", None)
        if callable(save_crop):
            target_dir = self._crop_dir / Path(frame.info.filename or f"frame_{frame.info.frame_index:06d}").stem
            target_dir.mkdir(parents=True, exist_ok=True)
            save_crop(save_dir=str(target_dir), file_name=Path(frame.info.filename or "frame").stem)
            return target_dir
        return None

    @staticmethod
    def _dedupe_artifacts(artifacts: list[InferenceArtifact]) -> list[InferenceArtifact]:
        seen: set[tuple[str, str]] = set()
        deduped: list[InferenceArtifact] = []
        for artifact in artifacts:
            key = (artifact.kind, str(artifact.output_path))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(artifact)
        return deduped


__all__ = ["InferencePipeline"]
