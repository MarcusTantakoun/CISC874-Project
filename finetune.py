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

    # explicitly set CUDA device
    torch.cuda.set_device(local_rank)

    # ensure process group is initialized
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=world_size)

    # explicitly perform barrier with assigned GPU
    dist.barrier(device_ids=[local_rank])

    if rank == 0:
        print(f"Distributed training initialized. World size: {world_size}, Local rank: {local_rank}, Rank: {rank}")


if __name__=="__main__":
    
    # set up distributed training
    local_rank = setup_distributed()

    # setup configuration for encoder
    setup_sentence_encoder_cfg = {
        "model_name": "/home/tant2002/scratch/all-roberta-large-v1",
        "model_type": "bi_encoder",
        "is_evaluated": False,
        "local_rank": local_rank  # Pass to training function
    }

    # setup finetuning configuration
    finetuning_encoder_cfg = {
        "train_batch_size": 32,
        "training_epoch": 40,
        "is_finetune_complete": False
    }
    
    if local_rank == 0:
        print(f"Starting training on rank {local_rank} with config: {setup_sentence_encoder_cfg}")

    # call training function
    train_sentence_encoder(setup_sentence_encoder_cfg=setup_sentence_encoder_cfg, finetuning_encoder_cfg=finetuning_encoder_cfg)