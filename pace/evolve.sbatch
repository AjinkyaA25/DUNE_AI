#!/bin/bash
# Evolution loop (heuristic knob tuning) on one 36-core node, no GPU:   sbatch pace/evolve.sbatch
#SBATCH -J dune-evolve
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=36
#SBATCH --mem-per-cpu=2G
#SBATCH -t 24:00:00
#SBATCH -o pace/logs/evolve_%j.log

cd "$SLURM_SUBMIT_DIR"
module load anaconda3
eval "$(conda shell.bash hook)"
conda activate dune
export PYTHONHASHSEED=0
python -u -m src.selfplay.evolve_loop --recalib 1500 --paired 3 \
    --league "config/evolve_plan/champion.json,config/evolve_fixed/champion.json,config/gap/best_ref2.json" \
    --league-frac 0.25 --cross 0.15 --dead 0.15 --games 4000 --gens 20 --confirm 10000 --workers 34
