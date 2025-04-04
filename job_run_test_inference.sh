#!/bin/bash
#SBATCH --job-name=inference_sentence_encoder
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=128000M
#SBATCH --time=0-8:00:00
#SBATCH --output=./slurm_out/infer-%j.out
#SBATCH --error=./slurm_out/infer-%j.err
#SBATCH --account=rrg-zhu2048

module load StdEnv/2023
ml cuda python/3.11 arrow/17.0.0

pip install --upgrade pip --no-index

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install --no-index --find-links=./offline_packages sentence-transformers torch transformers pddl datasets tqdm tabulate
pip install --no-index accelerate scikit_learn

# Verify installation
python -c "from sentence_transformers import SentenceTransformer; print('Success!')"

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

echo "Running inference..."
python ./run_inference.py
