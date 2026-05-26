"""Matplotlib plot style helpers shared by train/val workflows."""

from __future__ import annotations

import logging


logger = logging.getLogger(__name__)


def apply_academic_style(*, logger_instance: logging.Logger | None = None) -> bool:
    """Apply a lightweight academic plotting style in a best-effort manner."""

    log = logger_instance or logger
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        log.warning("无法应用学术绘图风格: %s", exc)
        return False

    plt.rcParams.update(
        {
            "figure.figsize": (8, 6),
            "figure.dpi": 140,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.frameon": False,
            "savefig.bbox": "tight",
        }
    )
    log.info("已应用学术绘图风格。")
    return True


__all__ = ["apply_academic_style"]
