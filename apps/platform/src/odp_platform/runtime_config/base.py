"""Metadata-driven runtime configuration models."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any, ClassVar, Final, Literal
import warnings

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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

    def chain_str(self, *, redact_sensitive: bool = True) -> str:
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
    created_at: str = ""

    def get(self, field_name: str) -> FieldTrace:
        return self.by_field[field_name]

    def get_metadata(self, field_name: str) -> FieldTrace | None:
        return self.by_field.get(field_name)

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

    def get_source_report(self, *, redact_sensitive: bool = True) -> str:
        grouped: dict[str, list[FieldTrace]] = defaultdict(list)
        for trace in self.by_field.values():
            grouped[trace.final_source_label].append(trace)

        lines = ["=" * 70, "配置来源报告".center(70), "=" * 70]
        for source_label in sorted(grouped):
            traces = sorted(grouped[source_label], key=lambda item: item.field_name)
            lines.append(f"[{source_label}] ({len(traces)} 项)")
            for trace in traces:
                lines.append(
                    f"  - {trace.field_name} = "
                    f"{trace._render_value(trace.final_value, redact_sensitive=redact_sensitive)}"
                )
        return "\n".join(lines)

    def get_conflict_report(self, *, redact_sensitive: bool = True) -> str:
        overridden = [
            trace
            for trace in self.by_field.values()
            if len(trace.history) > 1
        ]
        if not overridden:
            return "\n".join(["=" * 70, "配置覆盖报告".center(70), "=" * 70, "没有字段发生覆盖。"])

        lines = ["=" * 70, "配置覆盖报告".center(70), "=" * 70]
        for trace in sorted(overridden, key=lambda item: item.field_name):
            lines.append(f"- {trace.to_override_chain(redact_sensitive=redact_sensitive)}")
        return "\n".join(lines)

    def to_audit_log(self) -> dict[str, Any]:
        by_source: dict[str, int] = defaultdict(int)
        overridden_fields: list[str] = []
        field_sources: dict[str, dict[str, Any]] = {}

        for field_name, trace in self.by_field.items():
            by_source[trace.final_source_label] += 1
            if len(trace.history) > 1:
                overridden_fields.append(field_name)
            field_sources[field_name] = {
                "final_source": trace.final_source,
                "final_source_label": trace.final_source_label,
                "history_sources": [override.source_name for override in trace.history],
                "history_labels": [override.source_label for override in trace.history],
            }

        return {
            "merger_completed_at": self.created_at or datetime.now().isoformat(timespec="seconds"),
            "fields_count_total": len(self.by_field),
            "fields_count_overridden": len(overridden_fields),
            "fields_by_final_source": dict(sorted(by_source.items())),
            "overridden_fields": sorted(overridden_fields),
            "field_sources": field_sources,
        }


class RuntimeConfigBase(BaseModel):
    """Base runtime configuration shared by train/val/infer tasks."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    FRAMEWORK_ONLY_FIELDS: ClassVar[set[str]] = {"verbose", "task_kind", "experiment_name"}
    SENSITIVE_MASK: ClassVar[str] = "***"

    task_kind: RuntimeTaskKind = Field(default=RUNTIME_TASK_TRAIN)
    task_type: SemanticsTaskType = Field(default=TASK_DETECT)
    experiment_name: str = Field(default="")
    data: str = Field(default="")
    model: str = Field(default="")
    batch: int | float = Field(default=16)
    imgsz: int = Field(default=640, ge=32)
    workers: int = Field(default=8, ge=0)
    cache: bool | str = Field(default=False)
    rect: bool = Field(default=False)
    device: int | str | list[int | str] | None = Field(default=None)
    amp: bool = Field(default=True)
    project: str = Field(default="runs")
    name: str = Field(default="")
    exist_ok: bool = Field(default=False)
    save: bool = Field(default=True)
    verbose: bool = Field(default=False)
    seed: int = Field(default=0)
    deterministic: bool = Field(default=True)

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
        "batch": FieldSpec(
            description="Batch size or auto-batch hint.",
            default=16,
            examples=(8, 16, 32, 0, -1, 0.7),
            tuning_tips=(
                "Use a positive integer for fixed batch size.",
                "Use 0 or -1 only when the downstream runner supports auto-batch.",
                "Float values between 0 and 1 represent a memory utilization ratio.",
            ),
            group="input",
        ),
        "imgsz": FieldSpec(
            description="Input image size in pixels.",
            default=640,
            examples=(640, 1024),
            tuning_tips=("Keep it divisible by the model stride; 32-aligned sizes are recommended.",),
            group="input",
        ),
        "workers": FieldSpec(
            description="Number of dataloader workers.",
            default=8,
            examples=(0, 4, 8),
            tuning_tips=("Windows users can reduce this when multiprocessing is unstable.",),
            group="input",
        ),
        "cache": FieldSpec(
            description="Dataset cache policy.",
            default=False,
            examples=(False, True, "ram", "disk"),
            tuning_tips=("Enable only when disk and memory pressure are acceptable.",),
            group="input",
        ),
        "rect": FieldSpec(
            description="Whether to enable rectangular batching.",
            default=False,
            examples=(False, True),
            tuning_tips=("Useful when image aspect ratios vary a lot and downstream supports it.",),
            group="input",
        ),
        "device": FieldSpec(
            description="Execution device expression.",
            default=None,
            examples=("cpu", "0", "0,1"),
            tuning_tips=("Use cpu when no CUDA device is available.",),
            group="runtime",
        ),
        "amp": FieldSpec(
            description="Whether to enable AMP mixed precision.",
            default=True,
            examples=(True, False),
            tuning_tips=("Disable when numerical stability matters more than speed.",),
            group="runtime",
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
        "exist_ok": FieldSpec(
            description="Allow reusing an existing output directory.",
            default=False,
            examples=(False, True),
            tuning_tips=("Keep disabled to avoid accidentally overwriting earlier runs.",),
            group="output",
        ),
        "save": FieldSpec(
            description="Whether to save run artifacts such as checkpoints or outputs.",
            default=True,
            examples=(True, False),
            tuning_tips=("Disable only for throwaway experiments or dry runs.",),
            group="output",
        ),
        "verbose": FieldSpec(
            description="Whether Ultralytics should emit verbose logs.",
            default=False,
            examples=(False, True),
            tuning_tips=("Enable when debugging configuration or data issues.",),
            group="runtime",
        ),
        "seed": FieldSpec(
            description="Random seed for reproducibility.",
            default=0,
            examples=(0, 42, 3407),
            tuning_tips=("Use a fixed seed when you need stable experiment reproduction.",),
            group="runtime",
        ),
        "deterministic": FieldSpec(
            description="Force deterministic algorithms when supported.",
            default=True,
            examples=(True, False),
            tuning_tips=("Disable only when you intentionally prefer performance over determinism.",),
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
            if key not in set(self.internal_fields()) | set(self.FRAMEWORK_ONLY_FIELDS)
            and value is not None
            and value != ""
        }

    def to_snapshot(self) -> dict[str, Any]:
        return {
            "config_type": type(self).__name__,
            "task_kind": self.task_kind,
            "values": self.model_dump(),
        }

    def to_audit_snapshot(self) -> dict[str, Any]:
        return {
            "_config_class": type(self).__name__,
            "_timestamp": datetime.now().isoformat(timespec="seconds"),
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
    def from_audit_snapshot(cls, snapshot: Mapping[str, Any]) -> RuntimeConfigBase:
        return cls.from_snapshot(snapshot)

    @classmethod
    def examples_by_group(cls) -> dict[str, list[tuple[str, FieldSpec]]]:
        grouped: dict[str, list[tuple[str, FieldSpec]]] = {}
        for field_name, spec in cls.field_specs().items():
            grouped.setdefault(spec.group, []).append((field_name, spec))
        return grouped

    def get_field_groups(self) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = {}
        for field_name, spec in self.field_specs().items():
            groups.setdefault(spec.group, []).append(field_name)
        return groups

    def get_field_metadata(self, field_name: str) -> dict[str, Any]:
        spec = self.field_specs().get(field_name)
        if spec is None:
            raise ValueError(f"字段 '{field_name}' 不存在")
        return {
            "description": spec.description,
            "default": spec.default,
            "examples": list(spec.examples),
            "tips": list(spec.tuning_tips),
            "yaml_comment": spec.description,
            "group": spec.group,
            "sensitive": spec.sensitive,
            "internal": spec.internal,
            "cli_name": spec.cli_name,
            "yaml_key": spec.yaml_key,
        }

    def mask_sensitive_dump(self) -> dict[str, Any]:
        masked = self.model_dump()
        for field_name, spec in self.field_specs().items():
            if spec.sensitive and masked.get(field_name) is not None:
                masked[field_name] = self.SENSITIVE_MASK
        return masked

    @classmethod
    def sensitive_field_names(cls) -> set[str]:
        return {
            field_name
            for field_name, spec in cls.field_specs().items()
            if spec.sensitive
        }

    @field_validator("imgsz")
    @classmethod
    def _validate_imgsz(cls, value: int) -> int:
        if value % 32 != 0:
            warnings.warn(
                f"imgsz={value} 不是 32 的倍数，虽然可以运行，但建议使用 32 对齐尺寸以获得更稳定的性能。",
                UserWarning,
            )
        return value

    @field_validator("batch", mode="before")
    @classmethod
    def _validate_batch(cls, value: Any) -> int | float:
        if isinstance(value, bool):
            raise ValueError("batch 不能是 bool 类型")
        if isinstance(value, int):
            if value < -1:
                raise ValueError("batch 为 int 时必须为 -1、0 或 >= 1")
            return value

        try:
            batch_float = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("batch 必须是 int 或 0~1 之间的 float") from exc

        if 0.0 <= batch_float <= 1.0:
            return batch_float
        if batch_float.is_integer() and batch_float >= 0:
            return int(batch_float)
        raise ValueError("batch 为 float 时必须落在 0~1 之间")

    @field_validator("device")
    @classmethod
    def _validate_device(cls, value: Any) -> Any:
        if value is None:
            return value
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, int):
            return value
        if isinstance(value, list) and all(isinstance(item, (int, str)) for item in value):
            return value
        raise ValueError("device 必须是 None、str、int 或由 str/int 组成的 list")

    @field_validator("cache")
    @classmethod
    def _validate_cache(cls, value: bool | str) -> bool | str:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            cache_value = value.lower().strip()
            if cache_value in {"ram", "disk"}:
                return cache_value
            raise ValueError("cache 为 str 时必须是 'ram' 或 'disk'")
        raise ValueError("cache 必须是 bool 或 str")

    @model_validator(mode="after")
    def _cross_field_validation(self) -> "RuntimeConfigBase":
        if isinstance(self.batch, int) and self.batch > 0 and self.workers > self.batch * 2:
            warnings.warn(
                f"workers={self.workers} 远大于 batch={self.batch}，可能造成资源浪费，建议 workers <= batch * 2。",
                UserWarning,
            )
        return self


class TrainConfig(RuntimeConfigBase):
    task_kind: RuntimeTaskKind = Field(default=RUNTIME_TASK_TRAIN)
    epochs: int = Field(default=100, ge=1)
    patience: int = Field(default=50, ge=0)
    lr0: float = Field(default=0.01, gt=0.0)
    lrf: float = Field(default=0.01, ge=0.0)
    momentum: float = Field(default=0.937, ge=0.0)
    weight_decay: float = Field(default=0.0005, ge=0.0)
    save_period: int = Field(default=-1)
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
        "save_period": FieldSpec(
            description="Checkpoint save interval in epochs.",
            default=-1,
            examples=(-1, 10),
            tuning_tips=("Use -1 to disable periodic saves and keep only final checkpoints.",),
            group="output",
        ),
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
    split: Literal["train", "val", "test"] = Field(default="val")
    conf: float = Field(default=0.001, ge=0.0, le=1.0)
    iou: float = Field(default=0.6, ge=0.0, le=1.0)
    max_det: int = Field(default=300, ge=1)
    half: bool = Field(default=True)
    plots: bool = Field(default=True)
    save_json: bool = Field(default=True)
    save_hybrid: bool = Field(default=False)
    mask_ratio: int = Field(default=4, ge=1)
    overlap_mask: bool = Field(default=True)
    dnn: bool = Field(default=False)

    __field_specs__: Final[dict[str, FieldSpec]] = RuntimeConfigBase.field_specs() | {
        "split": FieldSpec(
            description="Dataset split used during validation.",
            default="val",
            examples=("val", "test", "train"),
            tuning_tips=("Use val for standard evaluation and test for held-out scoring.",),
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
        "max_det": FieldSpec(
            description="Maximum detections kept per image.",
            default=300,
            examples=(100, 300, 1000),
            tuning_tips=("Increase only for dense scenes.",),
            group="validation",
        ),
        "half": FieldSpec(
            description="Whether to prefer FP16 validation when supported.",
            default=True,
            examples=(True, False),
            tuning_tips=("Disable on CPU or when numerical stability is a concern.",),
            group="runtime",
        ),
        "plots": FieldSpec(
            description="Whether to emit validation plots.",
            default=True,
            examples=(False, True),
            tuning_tips=("Enable while debugging model behavior or reporting results.",),
            group="output",
        ),
        "save_json": FieldSpec(
            description="Whether to save COCO-style json predictions.",
            default=True,
            examples=(True, False),
            tuning_tips=("Disable only when you do not need external evaluation tooling.",),
            group="output",
        ),
        "save_hybrid": FieldSpec(
            description="Whether to save hybrid label outputs for analysis.",
            default=False,
            examples=(False, True),
            tuning_tips=("Useful when diagnosing difficult samples.",),
            group="output",
        ),
        "mask_ratio": FieldSpec(
            description="Mask downsample ratio for segmentation validation.",
            default=4,
            group="task",
        ),
        "overlap_mask": FieldSpec(
            description="Whether overlapping masks should be merged for segment validation.",
            default=True,
            group="task",
        ),
        "dnn": FieldSpec(
            description="Whether to use the OpenCV DNN backend when available.",
            default=False,
            examples=(False, True),
            tuning_tips=("Keep disabled for normal PyTorch validation.",),
            group="runtime",
        ),
    }

    __internal_fields__: Final[tuple[str, ...]] = RuntimeConfigBase.internal_fields() + ("task_type",)
    FRAMEWORK_ONLY_FIELDS: ClassVar[set[str]] = RuntimeConfigBase.FRAMEWORK_ONLY_FIELDS | {"task_type"}


class InferConfig(RuntimeConfigBase):
    task_kind: RuntimeTaskKind = Field(default=RUNTIME_TASK_INFER)
    conf: float = Field(default=0.25, ge=0.0, le=1.0)
    iou: float = Field(default=0.7, ge=0.0, le=1.0)
    max_det: int = Field(default=300, ge=1)
    classes: list[int] | None = Field(default=None)
    agnostic_nms: bool = Field(default=False)
    augment: bool = Field(default=False)
    vid_stride: int = Field(default=1, ge=1)
    stream: bool = Field(default=False)
    stream_buffer: bool = Field(default=False)
    save_txt: bool = Field(default=False)
    save_conf: bool = Field(default=False)
    save_crop: bool = Field(default=False)
    save_frames: bool = Field(default=False)
    show: bool = Field(default=False)
    show_labels: bool = Field(default=True)
    show_conf: bool = Field(default=True)
    show_boxes: bool = Field(default=True)
    line_width: int | None = Field(default=None, ge=1)
    retina_masks: bool = Field(default=False)
    visualize: bool = Field(default=False)
    embed: list[int] | None = Field(default=None)
    source: str | None = Field(default=None)

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
        "max_det": FieldSpec(
            description="Maximum detections kept per image.",
            default=300,
            examples=(100, 300, 1000),
            tuning_tips=("Increase only for dense scenes such as crowds or cluttered floorplans.",),
            group="inference",
        ),
        "classes": FieldSpec(
            description="Optional subset of class ids to keep during inference.",
            default=None,
            examples=(None, [0], [0, 1, 2]),
            tuning_tips=("Leave empty to keep all predicted classes.",),
            group="inference",
        ),
        "agnostic_nms": FieldSpec(
            description="Whether to perform class-agnostic NMS.",
            default=False,
            examples=(False, True),
            tuning_tips=("Enable only when classes are mutually exclusive in the same area.",),
            group="inference",
        ),
        "augment": FieldSpec(
            description="Whether to enable test-time augmentation.",
            default=False,
            examples=(False, True),
            tuning_tips=("Useful for benchmarking; usually disabled in production inference.",),
            group="inference",
        ),
        "vid_stride": FieldSpec(
            description="Video frame stride during streaming inference.",
            default=1,
            examples=(1, 2, 5),
            tuning_tips=("Higher values skip frames and improve throughput.",),
            group="stream",
        ),
        "stream": FieldSpec(
            description="Whether to keep inference in streaming mode.",
            default=False,
            examples=(False, True),
            tuning_tips=("Enable for webcams, RTSP streams, or long-running video feeds.",),
            group="stream",
        ),
        "stream_buffer": FieldSpec(
            description="Whether to buffer all frames during streaming inference.",
            default=False,
            examples=(False, True),
            tuning_tips=("Buffering avoids dropped frames but increases latency.",),
            group="stream",
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
        "save_crop": FieldSpec(
            description="Whether to save cropped detections as standalone images.",
            default=False,
            examples=(False, True),
            tuning_tips=("Useful for review sets or downstream retrieval pipelines.",),
            group="output",
        ),
        "save_frames": FieldSpec(
            description="Whether to save every processed frame when handling video inputs.",
            default=False,
            examples=(False, True),
            tuning_tips=("Enable only for debugging or offline analysis because it can consume disk quickly.",),
            group="output",
        ),
        "show": FieldSpec(
            description="Whether to display inference windows in real time.",
            default=False,
            examples=(False, True),
            tuning_tips=("Keep disabled on servers or headless environments.",),
            group="output",
        ),
        "show_labels": FieldSpec(
            description="Whether rendered results should include class labels.",
            default=True,
            examples=(True, False),
            tuning_tips=("Disable when cleaner visuals are preferred for demos.",),
            group="output",
        ),
        "show_conf": FieldSpec(
            description="Whether rendered results should include confidence scores.",
            default=True,
            examples=(True, False),
            tuning_tips=("Disable when labels should stay visually compact.",),
            group="output",
        ),
        "show_boxes": FieldSpec(
            description="Whether rendered results should draw boxes.",
            default=True,
            examples=(True, False),
            tuning_tips=("Disable only for special segmentation-style overlays.",),
            group="output",
        ),
        "line_width": FieldSpec(
            description="Fixed line width for rendered predictions.",
            default=None,
            examples=(None, 2, 4),
            tuning_tips=("Leave empty to let the runtime choose automatically.",),
            group="output",
        ),
        "retina_masks": FieldSpec(
            description="Whether segmentation masks should be rendered at original resolution.",
            default=False,
            examples=(False, True),
            tuning_tips=("Enable only for segmentation tasks that need higher-fidelity masks.",),
            group="task",
        ),
        "visualize": FieldSpec(
            description="Whether to export internal feature visualizations.",
            default=False,
            examples=(False, True),
            tuning_tips=("Useful for debugging but expensive in normal runs.",),
            group="task",
        ),
        "embed": FieldSpec(
            description="Optional list of feature-layer indices to export embeddings from.",
            default=None,
            examples=(None, [10], [10, 20]),
            tuning_tips=("Leave empty unless a downstream retrieval or analysis workflow needs embeddings.",),
            group="task",
        ),
        "source": FieldSpec(
            description="Input source for inference.",
            default=None,
            examples=("data/images/demo.jpg", "0", "rtsp://camera/live"),
            tuning_tips=("Use file paths, URLs, or camera indices depending on the demo source.",),
            group="input",
        ),
    }

    __internal_fields__: Final[tuple[str, ...]] = RuntimeConfigBase.internal_fields() + ("task_type",)
    FRAMEWORK_ONLY_FIELDS: ClassVar[set[str]] = RuntimeConfigBase.FRAMEWORK_ONLY_FIELDS | {"task_type"}


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


BaseConfig = RuntimeConfigBase
YOLOTrainConfig = TrainConfig
YOLOValConfig = ValConfig
YOLOInferConfig = InferConfig


__all__ = [
    "BaseConfig",
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
    "YOLOInferConfig",
    "YOLOTrainConfig",
    "YOLOValConfig",
    "get_config_class",
    "is_valid_task_type",
]
