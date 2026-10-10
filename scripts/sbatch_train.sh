#!/usr/bin/env bash
#SBATCH --job-name=stock_train
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --qos=dw87

# Full pipeline for all tickers in one job (results_all.csv and charts_data.json cover every ticker).
# Usage: sbatch scripts/sbatch_train.sh

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")/..}"
mkdir -p slurm_logs

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MPLBACKEND=Agg
source .venv/bin/activate

nvidia-smi || true
python -c "import torch; print('Torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"

python -u run_experiment.py
