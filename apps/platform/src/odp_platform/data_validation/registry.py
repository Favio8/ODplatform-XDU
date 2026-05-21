"""Registry and core data contracts for dataset validation checks."""

from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TYPE_CHECKING, Final


if TYPE_CHECKING:
    from odp_platform.data_validation.snapshot import DatasetSnapshot


CheckFunc = Callable[["CheckContext"], "CheckResult"]

_DISCOVERY_PRIORITY: Final[dict[str, int]] = {
    "yaml_schema": 0,
    "pair_existence": 1,
    "label_format": 2,
    "split_uniqueness": 3,
}


class CheckSeverity:
    """String severity levels ordered for report aggregation."""

    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"
    PASS = "PASS"

    _RANKS: Final[dict[str, int]] = {
        PASS: 0,
        INFO: 1,
        WARNING: 2,
        ERROR: 3,
    }

    @classmethod
    def rank(cls, level: str) -> int:
        if level not in cls._RANKS:
            raise ValueError(f"Unsupported check severity: {level!r}")
        return cls._RANKS[level]


@dataclass(frozen=True)
class CheckEntry:
    name: str
    func: CheckFunc


@dataclass
class CheckResult:
    name: str
    severity: str
    summary: str
    details: dict[str, Any]

    @property
    def passed(self) -> bool:
        return self.severity in (CheckSeverity.PASS, CheckSeverity.INFO)


@dataclass(frozen=True)
class CheckContext:
    yaml_path: Path
    snapshot: "DatasetSnapshot"


_REGISTRY: dict[str, CheckEntry] = {}
_CHECKS_DISCOVERED = False


def check(name: str) -> Callable[[CheckFunc], CheckFunc]:
    """Register one validation check under a stable name."""

    normalized_name = name.strip()
    if not normalized_name:
        raise ValueError("check name must not be empty")

    def decorator(func: CheckFunc) -> CheckFunc:
        if normalized_name in _REGISTRY:
            raise ValueError(f"Check {normalized_name!r} is already registered")
        _REGISTRY[normalized_name] = CheckEntry(name=normalized_name, func=func)
        return func

    return decorator


def _iter_check_module_names() -> list[str]:
    checks_dir = Path(__file__).resolve().parent / "checks"
    module_names = [
        module_info.name
        for module_info in pkgutil.iter_modules([str(checks_dir)])
        if not module_info.name.startswith("_")
    ]
    module_names.sort(key=lambda name: (_DISCOVERY_PRIORITY.get(name, 100), name))
    return module_names


def autodiscover_checks() -> None:
    """Import all builtin check modules once so decorators can register them."""

    global _CHECKS_DISCOVERED
    if _CHECKS_DISCOVERED:
        return

    for module_name in _iter_check_module_names():
        importlib.import_module(f"{__package__}.checks.{module_name}")
    _CHECKS_DISCOVERED = True


def get_registered_checks() -> dict[str, CheckEntry]:
    autodiscover_checks()
    return dict(_REGISTRY)


def list_checks() -> tuple[str, ...]:
    autodiscover_checks()
    return tuple(_REGISTRY.keys())


__all__ = [
    "CheckContext",
    "CheckEntry",
    "CheckResult",
    "CheckSeverity",
    "autodiscover_checks",
    "check",
    "get_registered_checks",
    "list_checks",
]
