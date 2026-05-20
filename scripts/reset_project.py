import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PLATFORM_SRC_DIR = REPO_ROOT / "apps" / "platform" / "src"


sys.path.insert(0, str(PLATFORM_SRC_DIR))


from odp_platform.cli.reset_project import main

if __name__ == "__main__":
    sys.exit(main())