#!/bin/bash
# One-time setup on a PACE login node (or Interactive Desktop terminal):
#   bash pace/setup_env.sh
# Creates a conda env "dune" with NumPy + CUDA PyTorch.
set -e
module load anaconda3
if ! conda env list | grep -q '^dune '; then
    conda create -n dune python=3.11 -y
fi
eval "$(conda shell.bash hook)"
conda activate dune
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu121
python -c "import torch, numpy; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
