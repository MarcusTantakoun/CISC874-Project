from Sem2Plan.pipelines.finetuning_sentence_encoder.nodes import train_sentence_encoder
import os
import torch.distributed as dist
import torch

def setup_distributed():
    """ Initializes distributed training and assigns correct GPU. """
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl')
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # Explicitly set CUDA device
    torch.cuda.set_device(local_rank)

    # Synchronize processes
    dist.barrier()

    if rank == 0:
        print(f"Distributed training initialized. World size: {world_size}, Local rank: {local_rank}")

    return local_rank

if __name__=="__main__":
    
    local_rank = setup_distributed()

    setup_sentence_encoder_cfg = {
        "model_name": "/home/tant2002/scratch/codebert-base",
        "model_type": "bi_encoder",
        "is_evaluated": False,
        "local_rank": local_rank  # Pass to training function
    }

    finetuning_encoder_cfg = {
        "train_batch_size": 64,
        "training_epoch": 40,
        "is_finetune_complete": False
    }
    
    if local_rank == 0:
        print(f"Starting training on rank {local_rank} with config: {setup_sentence_encoder_cfg}")

    train_sentence_encoder(setup_sentence_encoder_cfg=setup_sentence_encoder_cfg, finetuning_encoder_cfg=finetuning_encoder_cfg)