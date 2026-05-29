#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""Thread-safe cancellation token for inference pipelines."""
from __future__ import annotations

import threading


class CancelToken:
    """Shared cancellation signal checked by the pipeline loop."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def wait(self, timeout: float | None = None) -> bool:
        return self._event.wait(timeout)


class InferenceCancelled(Exception):
    """Raised when inference is terminated intentionally by the caller."""

