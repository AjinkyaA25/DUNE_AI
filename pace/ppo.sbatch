#!/bin/bash
# PPO training on one 36-core GPU node:   sbatch pace/ppo.sbatch
# Resumes from models_ppo/checkpoint.pt, so resubmit after the walltime ends.
# Add  #SBATCH -A <account>  and  #SBATCH -q <qos>  if your PACE cluster needs them.
#SBATCH -J dune-ppo
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=36
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2G
#SBATCH -t 24:00:00
#SBATCH -o pace/logs/ppo_%j.log

cd "$SLURM_SUBMIT_DIR"
module load anaconda3
eval "$(conda shell.bash hook)"
conda activate dune
export PYTHONHASHSEED=0
python -u -m src.selfplay.ppo_train --out models_ppo --games 700 --workers 34 \
    --T 1.0 --eval-every 10 --eval-games 2000
