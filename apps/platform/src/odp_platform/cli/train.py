"""Compatibility shim for the D6 real training CLI."""

from odp_platform.cli.train_model import build_parser, main


__all__ = ["build_parser", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
