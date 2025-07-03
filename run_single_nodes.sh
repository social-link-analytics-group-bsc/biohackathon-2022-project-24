#!/bin/bash
#SBATCH --job-name=vllm_single
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm_single_%j.out
#SBATCH --error=logs/slurm_single_%j.err

# ----------------------------
# Load environment and activate venv
# ----------------------------
cd /gpfs/projects/bsc02/sla_projects/vllm_server || {
  echo "[VLLM] ERROR: Failed to cd into project directory"
  exit 1
}

module purge && module load mkl intel python/3.12
unset PYTHONPATH
source venv_mn5/bin/activate

# ----------------------------
# Detect GPU count
# ----------------------------
gpus=$(scontrol show job "$SLURM_JOB_ID" | grep -oP 'gres/gpu=\K[0-9]+' | head -n1)
gpus=${gpus:-1} # fallback to 1 if detection fails

echo "[VLLM] Detected $gpus GPU(s) for launch"

# ----------------------------
# Run vLLM launcher
# ----------------------------
vllm-launch --tensor-parallel-size "$gpus" "$@"
