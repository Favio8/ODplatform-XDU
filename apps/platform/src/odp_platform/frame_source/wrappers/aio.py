#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""Async wrapper for frame sources."""
from __future__ import annotations

import asyncio
from typing import AsyncIterator, Optional

from ..core.base import FrameSource
from ..core.types import Frame


class AsyncSource:
    """Async protocol wrapper around a synchronous FrameSource."""

    def __init__(self, inner: FrameSource):
        self._inner = inner

    async def open(self) -> bool:
        return await asyncio.to_thread(self._inner.open)

    async def read(self) -> Optional[Frame]:
        return await asyncio.to_thread(self._inner.read)

    async def close(self) -> None:
        await asyncio.to_thread(self._inner.close)

    async def seek(
        self,
        frame: Optional[int] = None,
        time_sec: Optional[float] = None,
    ) -> bool:
        return await asyncio.to_thread(self._inner.seek, frame, time_sec)

    @property
    def seekable(self) -> bool:
        return self._inner.seekable

    @property
    def source_path(self) -> str:
        return self._inner.source_path

    @property
    def inner(self) -> FrameSource:
        return self._inner

    async def __aenter__(self) -> "AsyncSource":
        await self.open()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        await self.close()
        return False

    def __aiter__(self) -> AsyncIterator[Frame]:
        return self

    async def __anext__(self) -> Frame:
        frame = await self.read()
        if frame is None:
            raise StopAsyncIteration
        return frame
