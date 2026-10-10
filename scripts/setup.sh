#!/usr/bin/env bash
# One-time setup on the supercomputer LOGIN node (needs internet): venv + data caches.
# Usage: bash scripts/setup.sh

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

uv venv --python 3.12
uv pip install -r requirements.txt
.venv/bin/python fetch_data.py
mkdir -p slurm_logs

echo "Setup done. Submit with: sbatch scripts/sbatch_train.sh"
