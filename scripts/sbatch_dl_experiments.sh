#!/usr/bin/env bash
#SBATCH --job-name=stock_dl_exp
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --qos=dw87

# Deep-learning experiment sweep. Results append to results/dl_experiments.csv.
# Usage: sbatch scripts/sbatch_dl_experiments.sh [grid]   (grid: baseline|cells|size|lookback|features|regularization|all; default all)

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")/..}"
mkdir -p slurm_logs

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
source .venv/bin/activate

python -c "import torch; print('Torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"

python -u dl_experiments.py --grid "${1:-all}"
