from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[5]
SERVING_MODELS_DIR = ROOT / "models" / "serving"
CHECKPOINTS_DIR = ROOT / "models" / "checkpoints"
DEFAULT_SERVING_MODEL_NAME = "yolo26m_seg_best.pt"


def list_serving_models() -> list[dict[str, Any]]:
    SERVING_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    models = [_model_payload(path) for path in sorted(SERVING_MODELS_DIR.glob("*.pt"))]
    return sorted(models, key=lambda item: (not item["is_default"], item["name"]))


def default_serving_model_path() -> Path | None:
    default_path = SERVING_MODELS_DIR / DEFAULT_SERVING_MODEL_NAME
    if default_path.exists():
        return default_path.resolve()
    candidates = sorted(
        SERVING_MODELS_DIR.glob("*seg*.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0].resolve()
    checkpoint_candidates = sorted(
        CHECKPOINTS_DIR.glob("*seg-best.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if checkpoint_candidates:
        return checkpoint_candidates[0].resolve()
    return None


def resolve_serving_model(model_name: str | None = None) -> Path:
    SERVING_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if not model_name:
        default_path = default_serving_model_path()
        if default_path is None:
            raise FileNotFoundError(
                f"No serving segmentation model found. Put {DEFAULT_SERVING_MODEL_NAME} under {SERVING_MODELS_DIR}."
            )
        return default_path

    clean_name = Path(model_name).name
    if clean_name != model_name or not clean_name.endswith(".pt"):
        raise ValueError("model_name must be a .pt file name under models/serving.")

    model_path = (SERVING_MODELS_DIR / clean_name).resolve()
    serving_root = SERVING_MODELS_DIR.resolve()
    if serving_root not in model_path.parents or not model_path.is_file():
        raise FileNotFoundError(f"Serving model not found: {clean_name}")
    return model_path


def _model_payload(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "name": path.name,
        "path": str(path),
        "size_bytes": stat.st_size,
        "updated_at": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
        "is_default": path.name == DEFAULT_SERVING_MODEL_NAME,
        "label": _model_label(path.name),
    }


def _model_label(name: str) -> str:
    if name == DEFAULT_SERVING_MODEL_NAME:
        return "YOLO26m Seg Best · 推荐"
    if "26n" in name:
        return "YOLO26n Seg Best · 轻量"
    if "11m" in name:
        return "YOLO11m Seg Best · 高精度"
    return name.replace(".pt", "").replace("_", " ").replace("-", " ")
