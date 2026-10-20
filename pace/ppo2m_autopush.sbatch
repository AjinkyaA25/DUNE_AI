#!/bin/bash
# 2M-parameter PPO (1024/768/768), pure self-play vs its last 20 snapshots, hourly
# eval vs Best AI and vs the previous hour's snapshot, hourly git push of models_ppo_2m:   sbatch pace/ppo2m_autopush.sbatch
# Resumes from models_ppo_2m/checkpoint.pt (seeded once from models_ppo) and queues its own
# successor, so training continues job after job; stop with  scancel -n dune-ppo2m
# Add  #SBATCH -A <account>  and  #SBATCH -q <qos>  if your PACE cluster needs them.
#SBATCH -J dune-ppo2m
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=36
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2G
#SBATCH -t 5:00:00     # longer trips the QOS CPU-minutes-per-job cap at 36 cores
#SBATCH -o pace/logs/ppo2m_%j.log
#SBATCH --signal=B:USR1@300

cd "$SLURM_SUBMIT_DIR"
sbatch --dependency=afterany:$SLURM_JOB_ID pace/ppo2m_autopush.sbatch
module load anaconda3
eval "$(conda shell.bash hook)"
conda activate dune
export PYTHONHASHSEED=0
# one BLAS thread per rollout worker: 34 workers x 36 OpenBLAS threads each
# oversubscribe the node (2x slower with the small net, ~75x with the 2M net)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PPO_DIR=models_ppo_2m
# first run: start from the small run's latest checkpoint (widened on load, same play),
# its 5 newest snapshots as self-play opponents and the same fixed eval baseline
if [ ! -f $PPO_DIR/checkpoint.pt ]; then
    mkdir -p $PPO_DIR
    cp models_ppo/checkpoint.pt models_ppo/eval_baseline.json $PPO_DIR/
    cp $(ls models_ppo/snap_*.npz | tail -5) $PPO_DIR/
fi
bash pace/autopush.sh 3600 &
PUSH_PID=$!
# 5 min before the walltime: stop training and push the latest state once more
trap 'kill $TRAIN_PID; wait $TRAIN_PID; kill $PUSH_PID; bash pace/autopush.sh 0 once' USR1
python -u -m src.selfplay.ppo_train --out $PPO_DIR --widths 1024,768,768,16 --games 700 --workers 34 \
    --T 1.0 --eval-every 10 --eval-secs 3600 --eval-games 2000 --self-eval-games 1000 \
    --self-frac 1.0 --league-frac 0 --snap-pool 20 &
TRAIN_PID=$!
wait $TRAIN_PID
