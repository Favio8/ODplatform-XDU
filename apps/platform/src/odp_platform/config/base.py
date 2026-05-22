"""Metadata-driven runtime configuration models."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from odp_platform.common.constants import (
    RUNTIME_TASK_INFER,
    RUNTIME_TASK_TRAIN,
    RUNTIME_TASK_VAL,
    SUPPORTED_RUNTIME_TASKS,
    SUPPORTED_TASKS,
    TASK_DETECT,
)


RuntimeTaskKind = Literal["train", "val", "infer"]
SemanticsTaskType = Literal["detect", "segment"]


@dataclass(frozen=True)
class FieldSpec:
    """Single-source metadata for one configurable field."""

    description: str
    default: Any
    examples: tuple[Any, ...] = ()
    tuning_tips: tuple[str, ...] = ()
    group: str = "general"
    sensitive: bool = False
    internal: bool = False
    cli_name: str | None = None
    yaml_key: str | None = None


@dataclass(frozen=True)
class SourceOverride:
    """One value provided by one config source before merge."""

    source_name: str
    value: Any

    @property
    def source_label(self) -> str:
        prefix = self.source_name.split(":", 1)[0].strip().upper()
        if prefix in {"DEFAULT", "DEFAULTS"}:
            return "DEFAULT"
        return prefix or "UNKNOWN"


@dataclass(frozen=True)
class FieldTrace:
    """Merged provenance chain for one field."""

    field_name: str
    final_value: Any
    final_source: str
    history: tuple[SourceOverride, ...]
    sensitive: bool = False

    @property
    def final_source_label(self) -> str:
        return self.history[-1].source_label if self.history else "UNKNOWN"

    def _render_value(self, value: Any, *, redact_sensitive: bool = True) -> str:
        if redact_sensitive and self.sensitive:
            return "***"
        return str(value)

    def to_effective_line(self, *, redact_sensitive: bool = True) -> str:
        rendered_value = self._render_value(self.final_value, redact_sensitive=redact_sensitive)
        return f"{self.field_name}: {rendered_value}  (来源: {self.final_source_label})"

    def to_override_chain(self, *, redact_sensitive: bool = True) -> str:
        rendered_history = [
            f"{self._render_value(override.value, redact_sensitive=redact_sensitive)}({override.source_label})"
            for override in self.history
        ]
        return f"{self.field_name}: " + " <- ".join(rendered_history)

    def to_human_readable(self, *, redact_sensitive: bool = True) -> str:
        return self.to_override_chain(redact_sensitive=redact_sensitive)

    def to_dict(self, *, redact_sensitive: bool = True) -> dict[str, Any]:
        return {
            "field_name": self.field_name,
            "final_value": "***" if redact_sensitive and self.sensitive else self.final_value,
            "final_source": self.final_source,
            "final_source_label": self.final_source_label,
            "history": [
                {
                    "source_name": override.source_name,
                    "source_label": override.source_label,
                    "value": "***" if redact_sensitive and self.sensitive else override.value,
                }
                for override in self.history
            ],
        }


@dataclass(frozen=True)
class ConfigTrace:
    """Full provenance for one built configuration."""

    by_field: dict[str, FieldTrace]

    def get(self, field_name: str) -> FieldTrace:
        return self.by_field[field_name]

    def to_human_readable(self, *, redact_sensitive: bool = True) -> str:
        lines = [trace.to_human_readable(redact_sensitive=redact_sensitive) for trace in self.by_field.values()]
        return "\n".join(lines)

    def to_effective_lines(
        self,
        *,
        redact_sensitive: bool = True,
        field_names: tuple[str, ...] | list[str] | None = None,
    ) -> list[str]:
        names = field_names or list(self.by_field)
        return [
            self.by_field[name].to_effective_line(redact_sensitive=redact_sensitive)
            for name in names
            if name in self.by_field
        ]

    def to_override_lines(
        self,
        *,
        redact_sensitive: bool = True,
        field_names: tuple[str, ...] | list[str] | None = None,
    ) -> list[str]:
        names = field_names or list(self.by_field)
        return [
            self.by_field[name].to_override_chain(redact_sensitive=redact_sensitive)
            for name in names
            if name in self.by_field
        ]

    def to_dict(self, *, redact_sensitive: bool = True) -> dict[str, dict[str, Any]]:
        return {
            name: trace.to_dict(redact_sensitive=redact_sensitive)
            for name, trace in self.by_field.items()
        }


class RuntimeConfigBase(BaseModel):
    """Base runtime configuration shared by train/val/infer tasks."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    task_kind: RuntimeTaskKind = Field(default=RUNTIME_TASK_TRAIN)
    task_type: SemanticsTaskType = Field(default=TASK_DETECT)
    experiment_name: str = Field(default="")
    data: str = Field(default="")
    model: str = Field(default="")
    project: str = Field(default="runs")
    name: str = Field(default="")
    device: str = Field(default="")
    verbose: bool = Field(default=False)

    __field_specs__: Final[dict[str, FieldSpec]] = {
        "task_kind": FieldSpec(
            description="Runtime task kind used by ODPlatform.",
            default=RUNTIME_TASK_TRAIN,
            examples=(RUNTIME_TASK_TRAIN, RUNTIME_TASK_VAL, RUNTIME_TASK_INFER),
            tuning_tips=("Usually inferred by the builder and not edited manually.",),
            group="runtime",
            internal=True,
        ),
        "task_type": FieldSpec(
            description="Algorithm semantics for the run.",
            default=TASK_DETECT,
            examples=SUPPORTED_TASKS,
            tuning_tips=("Use detect for bounding-box models.", "Use segment for polygon or mask outputs."),
            group="runtime",
            yaml_key="task",
        ),
        "experiment_name": FieldSpec(
            description="Human-friendly experiment label used only by ODPlatform.",
            default="",
            examples=("rsod-train-baseline",),
            tuning_tips=("Safe to leave empty when you do not need custom naming.",),
            group="runtime",
            internal=True,
        ),
        "data": FieldSpec(
            description="Dataset yaml path or dataset identifier consumed by Ultralytics.",
            default="",
            examples=("apps/platform/configs/datasets/rsod.yaml",),
            tuning_tips=("Prefer the dataset yaml generated by the data pipeline.",),
            group="input",
        ),
        "model": FieldSpec(
            description="Model weights or model identifier.",
            default="",
            examples=("models/pretrained/yolo11n.pt",),
            tuning_tips=("Use a pretrained checkpoint for most training runs.",),
            group="model",
        ),
        "project": FieldSpec(
            description="Output root directory used by Ultralytics.",
            default="runs",
            examples=("runs", "runs/train"),
            tuning_tips=("Keep results under the repository runs/ tree for easy cleanup.",),
            group="output",
        ),
        "name": FieldSpec(
            description="Ultralytics run name.",
            default="",
            examples=("exp", "baseline_v1"),
            tuning_tips=("Leave empty to let the runner pick a default name.",),
            group="output",
        ),
        "device": FieldSpec(
            description="Execution device expression.",
            default="",
            examples=("cpu", "0", "0,1"),
            tuning_tips=("Use cpu when no CUDA device is available.",),
            group="runtime",
        ),
        "verbose": FieldSpec(
            description="Whether Ultralytics should emit verbose logs.",
            default=False,
            examples=(False, True),
            tuning_tips=("Enable when debugging configuration or data issues.",),
            group="runtime",
        ),
    }

    __internal_fields__: Final[tuple[str, ...]] = ("task_kind", "experiment_name")

    @classmethod
    def field_specs(cls) -> dict[str, FieldSpec]:
        return dict(cls.__field_specs__)

    @classmethod
    def internal_fields(cls) -> tuple[str, ...]:
        return tuple(cls.__internal_fields__)

    @classmethod
    def task_kind_name(cls) -> RuntimeTaskKind:
        return cls.field_specs()["task_kind"].default

    def to_runtime_dict(self) -> dict[str, Any]:
        return self.model_dump()

    @classmethod
    def external_field_name(cls, field_name: str) -> str:
        spec = cls.field_specs().get(field_name)
        if spec is None:
            return field_name
        return spec.yaml_key or field_name

    @classmethod
    def field_name_aliases(cls) -> dict[str, str]:
        aliases: dict[str, str] = {}
        for field_name, spec in cls.field_specs().items():
            aliases[field_name] = field_name
            if spec.yaml_key:
                aliases[spec.yaml_key] = field_name
            if spec.cli_name:
                aliases[spec.cli_name] = field_name
        return aliases

    def to_ultralytics_kwargs(self) -> dict[str, Any]:
        payload = self.model_dump()
        return {
            self.external_field_name(key): value
            for key, value in payload.items()
            if key not in self.internal_fields() and value is not None and value != ""
        }

    def to_snapshot(self) -> dict[str, Any]:
        return {
            "config_type": type(self).__name__,
            "task_kind": self.task_kind,
            "values": self.model_dump(),
        }

    @classmethod
    def from_snapshot(cls, snapshot: Mapping[str, Any]) -> RuntimeConfigBase:
        values = snapshot.get("values")
        if not isinstance(values, Mapping):
            raise ValueError("snapshot.values must be a mapping")
        return cls.model_validate(dict(values))

    @classmethod
    def examples_by_group(cls) -> dict[str, list[tuple[str, FieldSpec]]]:
        grouped: dict[str, list[tuple[str, FieldSpec]]] = {}
        for field_name, spec in cls.field_specs().items():
            grouped.setdefault(spec.group, []).append((field_name, spec))
        return grouped


class TrainConfig(RuntimeConfigBase):
    task_kind: RuntimeTaskKind = Field(default=RUNTIME_TASK_TRAIN)
    epochs: int = Field(default=100, ge=1)
    imgsz: int = Field(default=640, ge=32)
    batch: int = Field(default=16, ge=0)
    workers: int = Field(default=8, ge=0)
    patience: int = Field(default=50, ge=0)
    lr0: float = Field(default=0.01, gt=0.0)
    lrf: float = Field(default=0.01, ge=0.0)
    momentum: float = Field(default=0.937, ge=0.0)
    weight_decay: float = Field(default=0.0005, ge=0.0)
    save: bool = Field(default=True)
    save_period: int = Field(default=-1)
    cache: bool = Field(default=False)
    rect: bool = Field(default=False)
    amp: bool = Field(default=True)
    exist_ok: bool = Field(default=False)
    seed: int = Field(default=0)
    deterministic: bool = Field(default=True)
    time: float | None = Field(default=None)
    resume: bool = Field(default=False)
    close_mosaic: int = Field(default=10, ge=0)
    multi_scale: bool = Field(default=False)
    fraction: float = Field(default=1.0, gt=0.0, le=1.0)
    freeze: int | list[int] | None = Field(default=None)
    optimizer: str = Field(default="auto")
    cos_lr: bool = Field(default=False)
    warmup_epochs: float = Field(default=3.0, ge=0.0)
    warmup_momentum: float = Field(default=0.8, ge=0.0)
    warmup_bias_lr: float = Field(default=0.1, ge=0.0)
    box: float = Field(default=7.5, ge=0.0)
    cls: float = Field(default=0.5, ge=0.0)
    dfl: float = Field(default=1.5, ge=0.0)
    pose: float = Field(default=12.0, ge=0.0)
    kobj: float = Field(default=2.0, ge=0.0)
    nbs: int = Field(default=64, ge=1)
    hsv_h: float = Field(default=0.015, ge=0.0)
    hsv_s: float = Field(default=0.7, ge=0.0)
    hsv_v: float = Field(default=0.4, ge=0.0)
    bgr: float = Field(default=0.0, ge=0.0, le=1.0)
    degrees: float = Field(default=0.0)
    translate: float = Field(default=0.1, ge=0.0)
    scale: float = Field(default=0.5, ge=0.0)
    shear: float = Field(default=0.0)
    perspective: float = Field(default=0.0, ge=0.0)
    flipud: float = Field(default=0.0, ge=0.0, le=1.0)
    fliplr: float = Field(default=0.5, ge=0.0, le=1.0)
    mosaic: float = Field(default=1.0, ge=0.0, le=1.0)
    mixup: float = Field(default=0.0, ge=0.0, le=1.0)
    copy_paste: float = Field(default=0.0, ge=0.0, le=1.0)
    val: bool = Field(default=True)
    plots: bool = Field(default=True)
    overlap_mask: bool = Field(default=True)
    mask_ratio: int = Field(default=4, ge=1)
    dropout: float = Field(default=0.0, ge=0.0, le=1.0)
    copy_paste_mode: str = Field(default="flip")
    auto_augment: str = Field(default="randaugment")
    erasing: float = Field(default=0.4, ge=0.0, le=1.0)
    pretrained: bool | str = Field(default=True)
    single_cls: bool = Field(default=False)
    classes: list[int] | None = Field(default=None)
    compile: bool | str = Field(default=False)
    profile: bool = Field(default=False)
    augmentations: list[object] | None = Field(default=None)

    __field_specs__: Final[dict[str, FieldSpec]] = RuntimeConfigBase.field_specs() | {
        "epochs": FieldSpec(
            description="Total number of training epochs.",
            default=100,
            examples=(100, 300),
            tuning_tips=("Increase for larger datasets if validation still improves.",),
            group="train",
        ),
        "imgsz": FieldSpec(
            description="Training image size in pixels.",
            default=640,
            examples=(640, 1024),
            tuning_tips=("Keep it divisible by the model stride; larger sizes need more VRAM.",),
            group="train",
        ),
        "batch": FieldSpec(
            description="Batch size per iteration.",
            default=16,
            examples=(8, 16, 32),
            tuning_tips=("Use 0 only if the downstream runner supports auto-batch.",),
            group="train",
        ),
        "workers": FieldSpec(
            description="Number of dataloader workers.",
            default=8,
            examples=(0, 4, 8),
            tuning_tips=("Windows users can reduce this when multiprocessing is unstable.",),
            group="performance",
        ),
        "patience": FieldSpec(
            description="Early-stopping patience.",
            default=50,
            examples=(20, 50),
            tuning_tips=("Lower values stop faster when validation no longer improves.",),
            group="train",
        ),
        "lr0": FieldSpec(
            description="Initial learning rate.",
            default=0.01,
            examples=(0.01, 0.001),
            tuning_tips=("Reduce when training is unstable or diverges early.",),
            group="train",
        ),
        "lrf": FieldSpec(description="Final learning-rate ratio.", default=0.01, group="train"),
        "momentum": FieldSpec(description="Optimizer momentum or beta1.", default=0.937, group="optimizer"),
        "weight_decay": FieldSpec(description="Weight decay used by the optimizer.", default=0.0005, group="optimizer"),
        "save": FieldSpec(
            description="Whether to save checkpoints.",
            default=True,
            examples=(True, False),
            tuning_tips=("Disable only for throwaway experiments.",),
            group="output",
        ),
        "save_period": FieldSpec(
            description="Checkpoint save interval in epochs.",
            default=-1,
            examples=(-1, 10),
            tuning_tips=("Use -1 to disable periodic saves and keep only final checkpoints.",),
            group="output",
        ),
        "cache": FieldSpec(
            description="Whether to cache images for faster training.",
            default=False,
            examples=(False, True),
            tuning_tips=("Enable only when disk and memory pressure are acceptable.",),
            group="performance",
        ),
        "rect": FieldSpec(description="Whether to enable rectangular training.", default=False, group="input"),
        "amp": FieldSpec(description="Whether to enable AMP mixed precision.", default=True, group="runtime"),
        "exist_ok": FieldSpec(description="Allow reusing an existing output directory.", default=False, group="output"),
        "seed": FieldSpec(description="Random seed for reproducibility.", default=0, group="runtime"),
        "deterministic": FieldSpec(description="Force deterministic training algorithms.", default=True, group="runtime"),
        "time": FieldSpec(description="Maximum training time in hours.", default=None, group="train"),
        "resume": FieldSpec(description="Resume from the latest checkpoint.", default=False, group="train"),
        "close_mosaic": FieldSpec(description="Disable mosaic in the last N epochs.", default=10, group="augmentation"),
        "multi_scale": FieldSpec(description="Enable multiscale training.", default=False, group="augmentation"),
        "fraction": FieldSpec(description="Fraction of training data to use.", default=1.0, group="train"),
        "freeze": FieldSpec(description="Layer freezing strategy.", default=None, group="model"),
        "optimizer": FieldSpec(description="Optimizer selection.", default="auto", group="optimizer"),
        "cos_lr": FieldSpec(description="Use cosine learning-rate scheduling.", default=False, group="optimizer"),
        "warmup_epochs": FieldSpec(description="Warmup duration in epochs.", default=3.0, group="optimizer"),
        "warmup_momentum": FieldSpec(description="Warmup starting momentum.", default=0.8, group="optimizer"),
        "warmup_bias_lr": FieldSpec(description="Warmup learning rate for bias terms.", default=0.1, group="optimizer"),
        "box": FieldSpec(description="Box loss gain.", default=7.5, group="loss"),
        "cls": FieldSpec(description="Classification loss gain.", default=0.5, group="loss"),
        "dfl": FieldSpec(description="Distribution focal loss gain.", default=1.5, group="loss"),
        "pose": FieldSpec(description="Pose loss gain.", default=12.0, group="loss"),
        "kobj": FieldSpec(description="Keypoint objectness loss gain.", default=2.0, group="loss"),
        "nbs": FieldSpec(description="Nominal batch size used for normalization.", default=64, group="loss"),
        "hsv_h": FieldSpec(description="HSV hue augmentation.", default=0.015, group="augmentation"),
        "hsv_s": FieldSpec(description="HSV saturation augmentation.", default=0.7, group="augmentation"),
        "hsv_v": FieldSpec(description="HSV value augmentation.", default=0.4, group="augmentation"),
        "bgr": FieldSpec(description="Probability of RGB/BGR channel reversal.", default=0.0, group="augmentation"),
        "degrees": FieldSpec(description="Rotation augmentation in degrees.", default=0.0, group="augmentation"),
        "translate": FieldSpec(description="Translation augmentation ratio.", default=0.1, group="augmentation"),
        "scale": FieldSpec(description="Scale augmentation gain.", default=0.5, group="augmentation"),
        "shear": FieldSpec(description="Shear augmentation angle.", default=0.0, group="augmentation"),
        "perspective": FieldSpec(description="Perspective augmentation amount.", default=0.0, group="augmentation"),
        "flipud": FieldSpec(description="Vertical flip probability.", default=0.0, group="augmentation"),
        "fliplr": FieldSpec(description="Horizontal flip probability.", default=0.5, group="augmentation"),
        "mosaic": FieldSpec(description="Mosaic augmentation probability.", default=1.0, group="augmentation"),
        "mixup": FieldSpec(description="MixUp augmentation probability.", default=0.0, group="augmentation"),
        "copy_paste": FieldSpec(description="Copy-paste augmentation probability.", default=0.0, group="augmentation"),
        "val": FieldSpec(description="Run validation during training.", default=True, group="validation"),
        "plots": FieldSpec(description="Generate training plots.", default=True, group="output"),
        "overlap_mask": FieldSpec(description="Merge overlapping masks for segmentation tasks.", default=True, group="task"),
        "mask_ratio": FieldSpec(description="Mask downsample ratio.", default=4, group="task"),
        "dropout": FieldSpec(description="Dropout ratio for classification tasks.", default=0.0, group="task"),
        "copy_paste_mode": FieldSpec(description="Copy-paste mode for segmentation.", default="flip", group="task"),
        "auto_augment": FieldSpec(description="Auto augmentation policy.", default="randaugment", group="task"),
        "erasing": FieldSpec(description="Random erasing probability.", default=0.4, group="task"),
        "pretrained": FieldSpec(description="Pretrained weight strategy.", default=True, group="model"),
        "single_cls": FieldSpec(description="Treat all labels as a single class.", default=False, group="task"),
        "classes": FieldSpec(description="Subset of class IDs to train.", default=None, group="task"),
        "compile": FieldSpec(description="Compile the model graph when supported.", default=False, group="runtime"),
        "profile": FieldSpec(description="Enable speed profiling.", default=False, group="runtime"),
        "augmentations": FieldSpec(description="Custom augmentation configuration.", default=None, group="augmentation"),
    }


class ValConfig(RuntimeConfigBase):
    task_kind: RuntimeTaskKind = Field(default=RUNTIME_TASK_VAL)
    batch: int = Field(default=16, ge=1)
    conf: float = Field(default=0.001, ge=0.0, le=1.0)
    iou: float = Field(default=0.6, ge=0.0, le=1.0)
    plots: bool = Field(default=False)

    __field_specs__: Final[dict[str, FieldSpec]] = RuntimeConfigBase.field_specs() | {
        "batch": FieldSpec(
            description="Validation batch size.",
            default=16,
            examples=(8, 16, 32),
            tuning_tips=("Reduce when validation runs out of memory.",),
            group="validation",
        ),
        "conf": FieldSpec(
            description="Confidence threshold used during validation.",
            default=0.001,
            examples=(0.001, 0.25),
            tuning_tips=("Keep low for benchmark-style evaluation.",),
            group="validation",
        ),
        "iou": FieldSpec(
            description="IoU threshold used during evaluation.",
            default=0.6,
            examples=(0.5, 0.6, 0.7),
            tuning_tips=("Increase only when you intentionally want stricter overlap filtering.",),
            group="validation",
        ),
        "plots": FieldSpec(
            description="Whether to emit validation plots.",
            default=False,
            examples=(False, True),
            tuning_tips=("Enable while debugging model behavior or reporting results.",),
            group="output",
        ),
    }

    __internal_fields__: Final[tuple[str, ...]] = RuntimeConfigBase.internal_fields() + ("task_type",)


class InferConfig(RuntimeConfigBase):
    task_kind: RuntimeTaskKind = Field(default=RUNTIME_TASK_INFER)
    conf: float = Field(default=0.25, ge=0.0, le=1.0)
    iou: float = Field(default=0.7, ge=0.0, le=1.0)
    save_txt: bool = Field(default=False)
    save_conf: bool = Field(default=False)
    source: str = Field(default="")

    __field_specs__: Final[dict[str, FieldSpec]] = RuntimeConfigBase.field_specs() | {
        "conf": FieldSpec(
            description="Confidence threshold used during inference.",
            default=0.25,
            examples=(0.25, 0.5),
            tuning_tips=("Raise it when false positives are too noisy.",),
            group="inference",
        ),
        "iou": FieldSpec(
            description="IoU threshold for NMS during inference.",
            default=0.7,
            examples=(0.5, 0.7),
            tuning_tips=("Lower it when overlapping predictions are being suppressed too aggressively.",),
            group="inference",
        ),
        "save_txt": FieldSpec(
            description="Whether to save txt prediction outputs.",
            default=False,
            examples=(False, True),
            tuning_tips=("Enable when you need raw prediction files for downstream analysis.",),
            group="output",
        ),
        "save_conf": FieldSpec(
            description="Whether to include confidence scores in saved outputs.",
            default=False,
            examples=(False, True),
            tuning_tips=("Useful only together with txt or custom output persistence.",),
            group="output",
        ),
        "source": FieldSpec(
            description="Input source for inference.",
            default="",
            examples=("data/images/demo.jpg", "0"),
            tuning_tips=("Use file paths for images or a camera index for webcam demos.",),
            group="input",
        ),
    }

    __internal_fields__: Final[tuple[str, ...]] = RuntimeConfigBase.internal_fields() + ("task_type",)


CONFIG_CLASS_BY_TASK: Final[dict[str, type[RuntimeConfigBase]]] = {
    RUNTIME_TASK_TRAIN: TrainConfig,
    RUNTIME_TASK_VAL: ValConfig,
    RUNTIME_TASK_INFER: InferConfig,
}


def get_config_class(task_kind: str) -> type[RuntimeConfigBase]:
    normalized = task_kind.strip().lower()
    if normalized not in CONFIG_CLASS_BY_TASK:
        raise ValueError(
            f"Unsupported runtime task kind {task_kind!r}; expected one of {SUPPORTED_RUNTIME_TASKS}"
        )
    return CONFIG_CLASS_BY_TASK[normalized]


def is_valid_task_type(task_type: str) -> bool:
    return task_type in SUPPORTED_TASKS


__all__ = [
    "CONFIG_CLASS_BY_TASK",
    "ConfigTrace",
    "FieldSpec",
    "FieldTrace",
    "InferConfig",
    "RuntimeConfigBase",
    "SemanticsTaskType",
    "SourceOverride",
    "TrainConfig",
    "ValConfig",
    "get_config_class",
    "is_valid_task_type",
]
