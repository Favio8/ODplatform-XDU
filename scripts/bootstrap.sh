#!/usr/bin/env bash
set -euo pipefail

python scripts/init_workspace.py
echo "Workspace skeleton is ready. Activate the odplatform-gpu Conda environment before installing runtime dependencies."
