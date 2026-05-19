"""Performance measurement helpers."""

from __future__ import annotations

import logging
from functools import wraps
from time import perf_counter
from typing import Callable, ParamSpec, TypeVar


P = ParamSpec("P")
R = TypeVar("R")

logger = logging.getLogger(__name__)


def format_duration(seconds: float) -> str:
    """Format a duration using a human-friendly time unit."""

    if seconds < 0:
        raise ValueError("seconds must be non-negative")
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.2f} us"
    if seconds < 1.0:
        return f"{seconds * 1_000:.2f} ms"
    if seconds < 60.0:
        return f"{seconds:.2f} s"
    if seconds < 3600.0:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes} min {remaining_seconds:.2f} s"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    remaining_seconds = seconds % 60
    return f"{hours} h {minutes} min {remaining_seconds:.2f} s"


def time_it(
    iterations: int = 1,
    name: str | None = None,
    logger_instance: logging.Logger | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Measure execution time and emit a performance log entry."""

    if iterations < 1:
        raise ValueError("iterations must be greater than or equal to 1")

    log = logger_instance if logger_instance is not None else logger

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            display_name = name if name is not None else func.__name__
            total_duration = 0.0
            result: R | None = None

            for _ in range(iterations):
                start = perf_counter()
                result = func(*args, **kwargs)
                total_duration += perf_counter() - start

            average_duration = total_duration / iterations
            if iterations == 1:
                log.info("性能报告: %s 执行时间: %s", display_name, format_duration(total_duration))
            else:
                log.info(
                    "性能报告: %s 执行了 %s 次，总耗时: %s | 平均耗时: %s",
                    display_name,
                    iterations,
                    format_duration(total_duration),
                    format_duration(average_duration),
                )

            return result

        return wrapper

    return decorator


__all__ = ["format_duration", "time_it"]
