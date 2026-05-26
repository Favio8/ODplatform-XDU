"""Compatibility CLI shim for runtime-config template generation."""

from odp_platform.runtime_config.generator import main


__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
