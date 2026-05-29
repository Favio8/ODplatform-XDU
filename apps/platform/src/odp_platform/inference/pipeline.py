#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""Threaded inference pipeline aligned with the teacher version."""
from __future__ import annotations

import logging
import time
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
from typing import Any

import numpy as np

from odp_platform.frame_source import create_frame_source

from .cancel import CancelToken
from .hooks import FrameEvent, InferHooks, ProgressEvent
from .overlay import Metrics, draw_hud, draw_pause
from .sinks import NullSink, OutputSink


logger = logging.getLogger(__name__)

_SENTINEL = object()


def _put_latest(queue: Queue, item: object) -> None:
    try:
        queue.put_nowait(item)
    except Full:
        try:
            queue.get_nowait()
        except Empty:
            pass
        try:
            queue.put_nowait(item)
        except Full:
            pass


def _put_block(queue: Queue, item: object) -> None:
    while True:
        try:
            queue.put(item, timeout=1.0)
            return
        except Full:
            continue


def _tensor_to_numpy(value: Any) -> np.ndarray:
    if value is None:
        return np.array([])
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _serialize_detections(result: Any, labels: list[str]) -> list[dict[str, Any]]:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []
    xyxy = _tensor_to_numpy(getattr(boxes, "xyxy", None))
    confs = _tensor_to_numpy(getattr(boxes, "conf", None))
    if xyxy.size == 0:
        return []

    detections: list[dict[str, Any]] = []
    for index, box in enumerate(xyxy):
        label = labels[index] if index < len(labels) else str(index)
        conf = float(confs[index]) if index < len(confs) else None
        detections.append(
            {
                "label": label,
                "conf": conf,
                "xyxy": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
            }
        )
    return detections


class _Controller:
    def __init__(self) -> None:
        self._paused = Event()

    def toggle(self) -> None:
        self._paused.clear() if self._paused.is_set() else self._paused.set()

    def is_paused(self) -> bool:
        return self._paused.is_set()


class _Reader(Thread):
    """Read frames from the source in a dedicated thread."""

    def __init__(self, source, camera_config, *, live: bool, capacity: int, capture_fps) -> None:
        super().__init__(daemon=True)
        self._source = source
        self._camera_config = camera_config
        self._live = live
        self._capture_fps = capture_fps
        self.q: Queue = Queue(maxsize=1 if live else capacity)
        self._stop_event = Event()
        self.source_type = None
        self.error: Exception | None = None

    def run(self) -> None:
        try:
            with create_frame_source(self._source, camera_config=self._camera_config) as source:
                self.source_type = source.get_source_type()
                previous = time.perf_counter()
                for frame in source:
                    if self._stop_event.is_set():
                        break
                    now = time.perf_counter()
                    self._capture_fps.update((now - previous) * 1000.0)
                    previous = now
                    if self._live:
                        _put_latest(self.q, frame)
                    else:
                        _put_block(self.q, frame)
        except Exception as exc:  # pragma: no cover - surfaced through main thread
            self.error = exc
        finally:
            _put_block(self.q, _SENTINEL)

    def get(self, timeout: float):
        try:
            return self.q.get(timeout=timeout)
        except Empty:
            return None

    def get_nowait(self):
        try:
            return self.q.get_nowait()
        except Empty:
            return None

    def stop(self) -> None:
        self._stop_event.set()


class _Renderer(Thread):
    """Render frames, write outputs, and notify hooks."""

    def __init__(
        self,
        processor,
        in_q: Queue,
        out_q: Queue,
        *,
        drop: bool,
        output_sink: OutputSink,
        show: bool,
        show_info: bool,
        recording: bool,
        metrics: Metrics,
        hooks: InferHooks,
        output_dir: Path,
        save_frames: bool,
        save_txt: bool,
        save_conf: bool,
        save_crop: bool,
    ) -> None:
        super().__init__(daemon=True)
        self._processor = processor
        self._in_q = in_q
        self._out_q = out_q
        self._drop = drop
        self._sink = output_sink
        self._show = show
        self._show_info = show_info
        self._recording = recording
        self._metrics = metrics
        self._hooks = hooks
        self._output_dir = output_dir
        self._save_frames = save_frames
        self._save_txt = save_txt
        self._save_conf = save_conf
        self._save_crop = save_crop
        self._frame_outputs_dir = output_dir / "frames"
        self._labels_dir = output_dir / "labels"
        self._crop_dir = output_dir / "crops"
        self._stop_event = Event()
        self._frame_idx = 0

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                item = self._in_q.get(timeout=0.1)
            except Empty:
                continue

            if item is _SENTINEL:
                _put_block(self._out_q, _SENTINEL)
                break

            frame, result, labels, n = item
            started = time.perf_counter()
            try:
                annotated = self._processor.draw(frame.image, result, labels, n)
            except Exception as exc:
                logger.warning("渲染单帧失败, 跳过: %s", exc)
                continue
            self._metrics.render.update((time.perf_counter() - started) * 1000.0)

            self._sink.write(frame, annotated)
            self._persist_auxiliary_outputs(frame, result, annotated)

            if self._hooks.on_frame is not None:
                self._hooks.fire_frame(
                    FrameEvent(
                        frame_idx=self._frame_idx,
                        image=frame.image,
                        annotated=annotated,
                        n_detections=n,
                        detections=_serialize_detections(result, labels),
                    )
                )
            self._frame_idx += 1

            if self._show:
                display_frame = annotated.copy()
                draw_hud(
                    display_frame,
                    self._metrics,
                    n_dets=n,
                    recording=self._recording,
                    show_info=self._show_info,
                )
                _put_latest(self._out_q, display_frame) if self._drop else _put_block(self._out_q, display_frame)

    def _persist_auxiliary_outputs(self, frame, result: Any, annotated: np.ndarray) -> None:
        if self._save_frames:
            self._save_frame_image(frame, annotated)
        if self._save_txt:
            self._save_txt_result(result, frame)
        if self._save_crop:
            self._save_crop_result(result, frame)

    def _save_frame_image(self, frame, annotated: np.ndarray) -> None:
        import cv2

        self._frame_outputs_dir.mkdir(parents=True, exist_ok=True)
        output_path = self._frame_outputs_dir / f"frame_{frame.info.frame_index:06d}.jpg"
        if not cv2.imwrite(str(output_path), annotated):
            logger.warning("保存帧图失败: %s", output_path)

    def _save_txt_result(self, result: Any, frame) -> None:
        save_txt = getattr(result, "save_txt", None)
        if not callable(save_txt):
            return
        self._labels_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(frame.info.filename or f"frame_{frame.info.frame_index:06d}").stem
        output_path = self._labels_dir / f"{stem}.txt"
        try:
            save_txt(str(output_path), save_conf=self._save_conf)
        except Exception as exc:
            logger.warning("保存 txt 结果失败: %s", exc)

    def _save_crop_result(self, result: Any, frame) -> None:
        save_crop = getattr(result, "save_crop", None)
        if not callable(save_crop):
            return
        stem = Path(frame.info.filename or f"frame_{frame.info.frame_index:06d}").stem
        target_dir = self._crop_dir / stem
        target_dir.mkdir(parents=True, exist_ok=True)
        try:
            save_crop(save_dir=str(target_dir), file_name=stem)
        except Exception as exc:
            logger.warning("保存 crop 结果失败: %s", exc)


class _Display(Thread):
    """Display thread that owns imshow and non-blocking key polling."""

    def __init__(self, out_q: Queue, window_name: str, controller: _Controller) -> None:
        super().__init__(daemon=True)
        self._out_q = out_q
        self._window_name = window_name
        self._controller = controller
        self._stop_event = Event()
        self._key_lock = Lock()
        self._key = -1
        self._last = None

    def stop(self) -> None:
        self._stop_event.set()

    def get_key(self) -> int:
        with self._key_lock:
            key, self._key = self._key, -1
            return key

    def run(self) -> None:
        import cv2

        poll = cv2.pollKey if hasattr(cv2, "pollKey") else (lambda: cv2.waitKey(1))
        while not self._stop_event.is_set():
            frame = None
            try:
                item = self._out_q.get(timeout=0.03)
                if item is not _SENTINEL:
                    frame = item
                    self._last = frame
            except Empty:
                if self._controller.is_paused() and self._last is not None:
                    frame = self._last.copy()
                    draw_pause(frame)
            if frame is not None:
                cv2.imshow(self._window_name, frame)
            key = poll() & 0xFF
            if key != 255:
                with self._key_lock:
                    self._key = key
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


class ThreadedPipeline:
    """Teacher-aligned multi-threaded inference pipeline."""

    def __init__(
        self,
        *,
        processor,
        source,
        camera_config,
        output_dir,
        output_sink: OutputSink,
        batch_size: int,
        save: bool,
        show: bool,
        show_info: bool,
        window_name: str,
        warmup_frames: int,
        hooks: InferHooks | None = None,
        cancel_token: CancelToken | None = None,
        save_frames: bool = False,
        save_txt: bool = False,
        save_conf: bool = False,
        save_crop: bool = False,
    ) -> None:
        self.processor = processor
        self.source = str(source)
        self.camera_config = camera_config
        self.output_dir = Path(output_dir)
        self.sink = output_sink
        self.batch_size = max(1, batch_size)
        self.save = save
        self.show = show
        self.show_info = show_info
        self.window_name = window_name
        self.warmup_frames = warmup_frames
        self.hooks = hooks if hooks is not None else InferHooks()
        self.cancel_token = cancel_token
        self.save_frames = save_frames
        self.save_txt = save_txt
        self.save_conf = save_conf
        self.save_crop = save_crop

    def _is_cancelled(self) -> bool:
        return self.cancel_token is not None and self.cancel_token.is_cancelled()

    def run(self, stats) -> bool:
        metrics = Metrics()
        source_value = self.source
        live = source_value.isdigit() or source_value.lower().startswith(("rtsp://", "rtmp://"))
        effective_batch = 1 if live else self.batch_size
        render_drop = not self.save

        reader = _Reader(
            source_value,
            self.camera_config,
            live=live,
            capacity=max(effective_batch * 2, 8),
            capture_fps=metrics.capture,
        )
        in_q: Queue = Queue(maxsize=max(effective_batch * 2, 4))
        out_q: Queue = Queue(maxsize=2)
        controller = _Controller()

        renderer = None
        display = None
        interrupted = False
        warmed = 0
        last_batch_end = None
        start_time = time.perf_counter()
        sink_opened = False

        reader.start()

        try:
            while True:
                if controller.is_paused():
                    if self._is_cancelled():
                        logger.info("收到取消信号 (暂停状态), 退出.")
                        interrupted = True
                        break
                    if self._handle_key(display, controller):
                        interrupted = True
                        break
                    time.sleep(0.02)
                    continue

                if self._is_cancelled():
                    logger.info("收到取消信号, 退出主循环.")
                    interrupted = True
                    break

                first = reader.get(timeout=2.0)
                if first is None:
                    if reader.error:
                        raise reader.error
                    continue
                if first is _SENTINEL:
                    break

                batch = [first]
                ended = False
                for _ in range(effective_batch - 1):
                    nxt = reader.get_nowait()
                    if nxt is None:
                        break
                    if nxt is _SENTINEL:
                        ended = True
                        break
                    batch.append(nxt)

                if warmed < self.warmup_frames:
                    warmed += len(batch)
                    if ended:
                        break
                    continue

                if renderer is None:
                    try:
                        self.sink.open(self.output_dir, reader.source_type)
                        sink_opened = True
                    except Exception as exc:
                        logger.error("sink.open 失败, 退化用 NullSink: %s", exc)
                        self.sink = NullSink()
                        self.sink.open(self.output_dir, reader.source_type)
                        sink_opened = True

                    renderer = _Renderer(
                        self.processor,
                        in_q,
                        out_q,
                        drop=render_drop,
                        output_sink=self.sink,
                        show=self.show,
                        show_info=self.show_info,
                        recording=self.save,
                        metrics=metrics,
                        hooks=self.hooks,
                        output_dir=self.output_dir,
                        save_frames=self.save_frames,
                        save_txt=self.save_txt,
                        save_conf=self.save_conf,
                        save_crop=self.save_crop,
                    )
                    renderer.start()
                    if self.show:
                        display = _Display(out_q, self.window_name, controller)
                        display.start()

                images = [frame.image for frame in batch]
                results, labels_list, n_list, batch_dt = self.processor.infer_batch(images)
                stats.infer_time_sec += batch_dt

                total_frames = batch[0].info.total_frames if batch else None
                for frame, result, labels, n in zip(batch, results, labels_list, n_list):
                    stats.frames += 1
                    stats.detections += n
                    for label in labels:
                        stats.per_class[label] = stats.per_class.get(label, 0) + 1
                    metrics.add_speed(getattr(result, "speed", None))
                    if render_drop:
                        _put_latest(in_q, (frame, result, labels, n))
                    else:
                        _put_block(in_q, (frame, result, labels, n))

                    if self.hooks.on_progress is not None and stats.frames % self.hooks.progress_interval_frames == 0:
                        self.hooks.fire_progress(
                            ProgressEvent(
                                frame_idx=stats.frames,
                                total_frames=total_frames,
                                elapsed_sec=time.perf_counter() - start_time,
                                fps_loop=metrics.loop.fps,
                                fps_infer=metrics.infer.fps,
                                detections_total=stats.detections,
                            )
                        )

                batch_end = time.perf_counter()
                if last_batch_end is not None:
                    per_frame_loop_ms = (batch_end - last_batch_end) * 1000.0 / len(batch)
                    for _ in batch:
                        metrics.loop.update(per_frame_loop_ms)
                last_batch_end = batch_end

                if self._is_cancelled():
                    logger.info("收到取消信号 (派发后), 退出.")
                    interrupted = True
                    break

                if self._handle_key(display, controller):
                    interrupted = True
                    break
                if ended:
                    break
        finally:
            reader.stop()
            _put_block(in_q, _SENTINEL)
            if renderer is not None:
                renderer.join(timeout=3.0)
                renderer.stop()
            if display is not None:
                time.sleep(0.05)
                display.stop()
                display.join(timeout=1.0)
            if sink_opened:
                try:
                    self.sink.close()
                except Exception as exc:
                    logger.warning("sink.close 异常 (已吞): %s", exc)

        _write_fps(stats, metrics)
        logger.info(
            "流水线收尾: 捕获 %.1f | 推理 %.1f | 渲染 %.1f | loop %.1f FPS",
            metrics.capture.fps,
            metrics.infer.fps,
            metrics.render.fps,
            metrics.loop.fps,
        )
        return interrupted

    def _handle_key(self, display, controller: _Controller) -> bool:
        if display is None:
            return False
        key = display.get_key()
        if key in (ord("q"), 27):
            logger.info("用户请求退出 (q/Esc).")
            return True
        if key == ord(" "):
            controller.toggle()
            logger.info("已暂停 (空格恢复)" if controller.is_paused() else "已恢复")
        return False


def _write_fps(stats, metrics: Metrics) -> None:
    snapshot = metrics.snapshot()
    stats.capture_fps = snapshot["capture_fps"]
    stats.infer_fps = snapshot["infer_fps"]
    stats.render_fps = snapshot["render_fps"]
    stats.loop_fps = snapshot["loop_fps"]
    stats.current_fps = snapshot["current_fps"]
    stats.speed_ms = snapshot["speed_ms"]


InferencePipeline = ThreadedPipeline


__all__ = ["InferencePipeline", "ThreadedPipeline"]
