#!/bin/bash
# PPO self-play league (half the opponents = the current policy, the rest older PPO nets / Best AI),
# started from models_ppo/checkpoint.pt; progress = win share vs 3 x frozen iteration 1490:   sbatch pace/ppo_self.sbatch
# Resumes from models_ppo_self/checkpoint.pt (the latest iteration) and queues its own
# successor, so training continues job after job; stop with  scancel -n dune-ppo-self
# Add  #SBATCH -A <account>  and  #SBATCH -q <qos>  if your PACE cluster needs them.
#SBATCH -J dune-ppo-self
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=36
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2G
#SBATCH -t 5:00:00     # longer trips the QOS CPU-minutes-per-job cap at 36 cores
#SBATCH -o pace/logs/ppo_self_%j.log
#SBATCH --signal=B:USR1@300

cd "$SLURM_SUBMIT_DIR"
sbatch --dependency=afterany:$SLURM_JOB_ID pace/ppo_self.sbatch
module load anaconda3
eval "$(conda shell.bash hook)"
conda activate dune
export PYTHONHASHSEED=0
MODEL_DIR=models_ppo_self bash pace/autopush.sh 3600 &
PUSH_PID=$!
# 5 min before the walltime: stop training and push the latest state once more
trap 'kill $TRAIN_PID; wait $TRAIN_PID; kill $PUSH_PID; MODEL_DIR=models_ppo_self bash pace/autopush.sh 0 once' USR1
P=tuned=config/ppo/base.json:greedy
python -u -m src.selfplay.ppo_train --out models_ppo_self --init models_ppo/checkpoint.pt --games 700 --workers 34 \
    --T 1.0 --cur-frac 0.5 --pool "models_ppo_self/pool/*.npz" --self-frac 0.6 --league-frac 0.1 \
    --eval-every 10 --eval-games 1000 \
    --eval-opp "ppo:net=models_ppo_self/pool/iter1490.npz:$P" --eval-base "ppo:net=models_ppo_self/pool/iter1555.npz:$P" &
TRAIN_PID=$!
wait $TRAIN_PID
