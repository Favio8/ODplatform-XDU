from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[5]
SERVING_MODELS_DIR = ROOT / "models" / "serving"
PRETRAINED_MODELS_DIR = ROOT / "models" / "pretrained"
CHECKPOINTS_DIR = ROOT / "models" / "checkpoints"
DEFAULT_SERVING_MODEL_NAME = "yolo26m_seg_best.pt"
DEFAULT_PRETRAINED_MODEL_NAMES = (
    "yolov8n-seg.pt",
    "yolo26n-seg.pt",
    "yolo11m-seg.pt",
)


def list_serving_models() -> list[dict[str, Any]]:
    SERVING_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    models = [
        _model_payload(path, default_name=DEFAULT_SERVING_MODEL_NAME, label_builder=_serving_model_label)
        for path in sorted(SERVING_MODELS_DIR.glob("*.pt"))
    ]
    return sorted(models, key=lambda item: (not item["is_default"], item["name"]))


def list_pretrained_models() -> list[dict[str, Any]]:
    PRETRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    default_name = _default_pretrained_model_name()
    models = [
        _model_payload(path, default_name=default_name, label_builder=_pretrained_model_label)
        for path in sorted(PRETRAINED_MODELS_DIR.glob("*.pt"))
    ]
    return sorted(
        models,
        key=lambda item: (
            not item["is_default"],
            not _is_segment_model_name(item["name"]),
            item["name"].lower(),
        ),
    )


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


def _default_pretrained_model_name() -> str | None:
    for name in DEFAULT_PRETRAINED_MODEL_NAMES:
        if (PRETRAINED_MODELS_DIR / name).exists():
            return name
    segment_candidates = sorted(
        PRETRAINED_MODELS_DIR.glob("*seg*.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if segment_candidates:
        return segment_candidates[0].name
    first_candidate = sorted(PRETRAINED_MODELS_DIR.glob("*.pt"))
    return first_candidate[0].name if first_candidate else None


def _model_payload(
    path: Path,
    *,
    default_name: str | None,
    label_builder,
) -> dict[str, Any]:
    stat = path.stat()
    return {
        "name": path.name,
        "path": str(path),
        "size_bytes": stat.st_size,
        "updated_at": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
        "is_default": default_name is not None and path.name == default_name,
        "label": label_builder(path.name),
    }


def _serving_model_label(name: str) -> str:
    if name == DEFAULT_SERVING_MODEL_NAME:
        return "YOLO26m Seg Best · 推荐"
    if "26n" in name:
        return "YOLO26n Seg Best · 轻量"
    if "11m" in name:
        return "YOLO11m Seg Best · 高精度"
    return name.replace(".pt", "").replace("_", " ").replace("-", " ")


def _pretrained_model_label(name: str) -> str:
    known_labels = {
        "yolov8n-seg.pt": "YOLOv8n-seg · 轻量分割",
        "yolo26n-seg.pt": "YOLO26n-seg · 轻量分割",
        "yolo11m-seg.pt": "YOLO11m-seg · 高精度分割",
        "yolov8n.pt": "YOLOv8n · 通用预训练",
    }
    lower_name = name.lower()
    if lower_name in known_labels:
        return known_labels[lower_name]
    stem = name.removesuffix(".pt")
    suffix = "分割预训练" if _is_segment_model_name(name) else "通用预训练"
    return f"{stem} · {suffix}"


def _is_segment_model_name(name: str) -> bool:
    return "seg" in name.lower()
