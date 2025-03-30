#!/bin/bash
#SBATCH --job-name=train_sentence_encoder        # Job name
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks-per-node=1           # Number of tasks per node
#SBATCH --cpus-per-task=32            # Number of CPU cores per task
#SBATCH --gres=gpu:1                  # Number of GPUs per node
#SBATCH --mem=128000M
#SBATCH --time=0-8:00:00               # Maximum execution time (HH:MM:SS)
#SBATCH --output=./slurm_out/fine-tune-%j.out            
#SBATCH --error=./slurm_out/fine-tune-%j.err
#SBATCH --account=rrg-zhu2048

module load StdEnv/2023
ml cuda python/3.11 arrow/17.0.0

pip install --upgrade pip --no-index

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install --no-index sentence-transformers
pip install --no-index torch scikit_learn tqdm nltk torchtext transformers>=4.43.1 spacy triton accelerate datasets scipy matplotlib numpy huggingface_hub ipython
pip install -r requirements.txt

# Environment variables
export MASTER_ADDR=$(hostname)
export MASTER_PORT="$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1])')"
export RDVZ_ID=$RANDOM
echo "RDZV Endpoint $MASTER_ADDR:$MASTER_PORT"
echo "r$SLURM_NODEID master: $MASTER_ADDR"

# Enable for A100
export FI_PROVIDER="efa"

# Debugging flags
export LOGLEVEL=INFO
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=WARN
export TORCH_NCCL_BLOCKING_WAIT=1 
export PYTHONFAULTHANDLER=1
export LD_LIBRARY_PATH=/opt/amazon/efa/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH
export CUDA_LAUNCH_BLOCKING=0
export NCCL_SOCKET_IFNAME="eno8303,ib0"
export WANDB_MODE=offline
export HF_HUB_OFFLINE=1

# Launch the training
srun -c 8 \
    -N 1 \
    --mem=128000 \
    --gres=gpu:1 \
    bash -c '
    python ./finetune.py
    '