#!/bin/bash
# Layer-wise probe sweep on Compute Canada / Narval.
#
# The repo's own scripts/internal_evals/run_eval.sbatch targets MedARC's
# internal Slurm (--partition=main --account=training), which does not exist
# on Narval. This uses Narval's actual conventions instead.
#
# Usage:
#   sbatch experiments/layer_sweep/narval_layer_sweep.sbatch <ckpt_path> [task]
#
# Before submitting:
#   - replace def-ssfels below if your account string differs
#   - any gated dataset (e.g. medarc/adni_eval) must already be downloaded on
#     the login node -- compute nodes have no internet
#SBATCH --account=def-ssfels
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --job-name=layer_sweep
#SBATCH --output=slurms/%x-%j.out
#SBATCH --error=slurms/%x-%j.err

set -euo pipefail

CKPT="${1:?Usage: sbatch experiments/layer_sweep/narval_layer_sweep.sbatch <ckpt_path> [task]}"
TASK="${2:-dlbs_age}"

mkdir -p slurms
module load StdEnv/2023 gcc/12.3 cuda/12.2 cudnn/8.9 arrow/21.0.0 python/3.11

export HF_HOME="${HF_HOME:-$PROJECT/hf_cache}"
export HF_HUB_DISABLE_XET=1
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"

echo "Host:       $(hostname)"
echo "Job ID:     ${SLURM_JOB_ID:-not_set}"
echo "Checkpoint: $CKPT"
echo "Task:       $TASK"
echo "==============================="

uv run python experiments/layer_sweep/run_layer_sweep.py \
  --ckpt-path "$CKPT" \
  --task "$TASK" \
  --device cuda
