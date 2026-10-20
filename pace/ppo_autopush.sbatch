#!/bin/bash
# PPO training + hourly git push of models_ppo:   sbatch pace/ppo_autopush.sbatch
# Resumes from models_ppo/checkpoint.pt (the latest iteration) and queues its own
# successor, so training continues job after job; stop with  scancel -n dune-ppo
# Add  #SBATCH -A <account>  and  #SBATCH -q <qos>  if your PACE cluster needs them.
#SBATCH -J dune-ppo
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=36
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2G
#SBATCH -t 5:00:00     # longer trips the QOS CPU-minutes-per-job cap at 36 cores
#SBATCH -o pace/logs/ppo_%j.log
#SBATCH --signal=B:USR1@300

cd "$SLURM_SUBMIT_DIR"
sbatch --dependency=afterany:$SLURM_JOB_ID pace/ppo_autopush.sbatch
module load anaconda3
eval "$(conda shell.bash hook)"
conda activate dune
export PYTHONHASHSEED=0
bash pace/autopush.sh 3600 &
PUSH_PID=$!
# 5 min before the walltime: stop training and push the latest state once more
trap 'kill $TRAIN_PID; wait $TRAIN_PID; kill $PUSH_PID; bash pace/autopush.sh 0 once' USR1
python -u -m src.selfplay.ppo_train --out models_ppo --games 700 --workers 34 \
    --T 1.0 --eval-every 10 --eval-games 2000 &
TRAIN_PID=$!
wait $TRAIN_PID
