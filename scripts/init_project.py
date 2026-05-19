#!/usr/bin/env python
"""Development wrapper for the platform init-project CLI."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
PLATFORM_SRC_DIR = REPO_ROOT / "apps" / "platform" / "src"

if str(PLATFORM_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(PLATFORM_SRC_DIR))

from odp_platform.cli.init_project import main


if __name__ == "__main__":
    raise SystemExit(main())
