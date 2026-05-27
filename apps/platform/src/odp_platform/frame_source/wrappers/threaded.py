#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""Threaded capture wrapper for frame sources."""
from __future__ import annotations

import logging
import threading
from queue import Empty, Full, Queue
from typing import Literal, Optional

from ..core.base import FrameSource
from ..core.types import Frame, SourceType

logger = logging.getLogger(__name__)

BufferStrategy = Literal["latest", "bounded"]


class ThreadedSource(FrameSource):
    """Background-thread capture plus buffered consumption."""

    _EOS = object()

    def __init__(
        self,
        inner: FrameSource,
        buffer: BufferStrategy = "latest",
        buffer_size: int = 1,
        warmup_frames: int = 0,
        read_timeout: float = 5.0,
    ):
        if buffer not in ("latest", "bounded"):
            raise ValueError(f"buffer 取值必须是 'latest' 或 'bounded',收到: {buffer!r}")
        if buffer_size < 1:
            raise ValueError(f"buffer_size 必须 ≥ 1,收到: {buffer_size}")
        if warmup_frames < 0:
            raise ValueError(f"warmup_frames 必须 ≥ 0,收到: {warmup_frames}")
        if read_timeout <= 0:
            raise ValueError(f"read_timeout 必须 > 0,收到: {read_timeout}")

        super().__init__(inner.source_path)
        self._inner = inner
        self._buffer_strategy = buffer
        self._capacity = 1 if buffer == "latest" else buffer_size
        self._warmup_frames = warmup_frames
        self._read_timeout = read_timeout
        self._queue: Optional[Queue] = None
        self._capture_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._eos = False

    def open(self) -> bool:
        if not self._inner.open():
            return False

        self._queue = Queue(maxsize=self._capacity)
        self._stop_event.clear()
        self._eos = False
        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            name=f"ThreadedSource-{self._inner.__class__.__name__}",
            daemon=True,
        )
        self._capture_thread.start()
        logger.info(
            "采集线程已启动 (buffer=%s, capacity=%s, warmup_frames=%s)",
            self._buffer_strategy,
            self._capacity,
            self._warmup_frames,
        )
        return True

    def _capture_loop(self) -> None:
        warmup_left = self._warmup_frames
        try:
            while not self._stop_event.is_set():
                frame = self._inner.read()
                if frame is None:
                    self._push(self._EOS)
                    logger.debug("采集线程: inner source 耗尽")
                    return
                if warmup_left > 0:
                    warmup_left -= 1
                    if warmup_left == 0:
                        logger.debug("采集线程: 预热完成(%s 帧)", self._warmup_frames)
                    continue
                self._push(frame)
        except Exception as exc:
            logger.error("采集线程异常: %s", exc, exc_info=True)
            self._push(self._EOS)

    def _push(self, item) -> None:
        try:
            self._queue.put_nowait(item)
        except Full:
            if item is self._EOS:
                return
            try:
                self._queue.get_nowait()
            except Empty:
                pass
            try:
                self._queue.put_nowait(item)
            except Full:
                pass

    def read(self) -> Optional[Frame]:
        if self._eos or self._queue is None:
            return None

        try:
            item = self._queue.get(timeout=self._read_timeout)
        except Empty:
            if self._capture_thread is not None and not self._capture_thread.is_alive():
                self._eos = True
                return None
            logger.warning("采集超时:%ss 内无新帧抵达缓冲", self._read_timeout)
            return None

        if item is self._EOS:
            self._eos = True
            return None
        return item

    def close(self) -> None:
        self._stop_event.set()
        if self._capture_thread is not None and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=2.0)
            if self._capture_thread.is_alive():
                logger.warning("采集线程未在 2 秒内退出(可能 inner.read() 阻塞中)")
        self._capture_thread = None
        self._inner.close()
        logger.info("采集线程已停止,inner source 已关闭")

    def get_source_type(self) -> SourceType:
        return self._inner.get_source_type()

    @property
    def seekable(self) -> bool:
        return False

    @property
    def inner(self) -> FrameSource:
        return self._inner
