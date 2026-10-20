#!/bin/bash
# Wider PPO net (512,256,256,32), started from models_ppo/checkpoint.pt with the same play:   sbatch pace/ppo_wide.sbatch
# Resumes from models_ppo_wide/checkpoint.pt (the latest iteration) and queues its own
# successor, so training continues job after job; stop with  scancel -n dune-ppo-wide
# Add  #SBATCH -A <account>  and  #SBATCH -q <qos>  if your PACE cluster needs them.
#SBATCH -J dune-ppo-wide
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=36
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2G
#SBATCH -t 5:00:00     # longer trips the QOS CPU-minutes-per-job cap at 36 cores
#SBATCH -o pace/logs/ppo_wide_%j.log
#SBATCH --signal=B:USR1@300

cd "$SLURM_SUBMIT_DIR"
sbatch --dependency=afterany:$SLURM_JOB_ID pace/ppo_wide.sbatch
module load anaconda3
eval "$(conda shell.bash hook)"
conda activate dune
export PYTHONHASHSEED=0
MODEL_DIR=models_ppo_wide bash pace/autopush.sh 3600 &
PUSH_PID=$!
# 5 min before the walltime: stop training and push the latest state once more
trap 'kill $TRAIN_PID; wait $TRAIN_PID; kill $PUSH_PID; MODEL_DIR=models_ppo_wide bash pace/autopush.sh 0 once' USR1
python -u -m src.selfplay.ppo_train --out models_ppo_wide --init models_ppo/checkpoint.pt --widths 512,256,256,32 --games 700 --workers 34 \
    --T 1.0 --eval-every 10 --eval-games 2000 &
TRAIN_PID=$!
wait $TRAIN_PID
