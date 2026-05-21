"""Compatibility shim for the dataset validation CLI."""

from odp_platform.cli.validate_data import main


if __name__ == "__main__":
    raise SystemExit(main())
