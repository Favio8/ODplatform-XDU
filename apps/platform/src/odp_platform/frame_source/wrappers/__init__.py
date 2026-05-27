#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""Frame source wrappers."""
from __future__ import annotations

from .aio import AsyncSource
from .threaded import BufferStrategy, ThreadedSource

__all__ = [
    "ThreadedSource",
    "BufferStrategy",
    "AsyncSource",
]
