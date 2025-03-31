"""
This module is dedicated to training the selected sentence model.
"""

import os
from datetime import datetime
from pathlib import Path
from sentence_transformers import losses
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import BatchSamplers, SentenceTransformerTrainingArguments
from transformers import TrainerCallback
from .finetune_dataset import create_train_dataset
from ..setup_sentence_encoder.nodes import create_sentence_encoder_helper
import time
import torch

SLURM_JOB_TIME_LIMIT = 1200

class TimeLimitCallback(TrainerCallback):
    """Saves checkpoint when <10 minutes remain in SLURM job"""
    def __init__(self, time_limit_seconds, output_dir):
        self.start_time = time.time()
        self.time_limit = time_limit_seconds
        self.output_dir = output_dir
        self.warning_triggered = False

    def on_step_begin(self, args, state, control, **kwargs):
        if not self.warning_triggered:
            elapsed = time.time() - self.start_time
            remaining = self.time_limit - elapsed
            
            if remaining < 600:  # 10 minutes = 600 seconds
                self.warning_triggered = True
                if torch.distributed.get_rank() == 0:  # Only rank 0 saves
                    checkpoint_dir = os.path.join(self.output_dir, "emergency_checkpoint")
                    print(f"\n⚠️ Less than 10 minutes remaining! Saving checkpoint to {checkpoint_dir}")
                    kwargs['model'].save(checkpoint_dir)
                    # Write metadata file
                    with open(os.path.join(checkpoint_dir, "checkpoint_info.txt"), "w") as f:
                        f.write(f"Emergency save at {datetime.now().isoformat()}\n")
                        f.write(f"Remaining time: {remaining//60}m {remaining%60:.0f}s\n")
                        f.write(f"Training step: {state.global_step}\n")


class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, early_stopping_patience: int, early_stopping_threshold: float):
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        self.best_score = None
        self.patience_counter = 0

    def on_evaluate(self, args, state, control, **kwargs):
        eval_metric = kwargs.get("metrics", {}).get("eval_loss", None) # replace eval_loss with evaluation metric
        if eval_metric is not None:
            if self.best_score is None or eval_metric < self.best_score + self.early_stopping_threshold:
                self.best_score = eval_metric
                self.patience_counter = 0
                print(f"Best score: {self.best_score}")
            else:
                self.patience_counter += 1
                print(f"Patience counter: {self.patience_counter}")
                if self.patience_counter >= self.early_stopping_patience:
                    print("Early stopping triggered")
                    control.should_training_stop = True     


def train_sentence_encoder(setup_sentence_encoder_cfg, finetuning_encoder_cfg):
    
    # initialize sentence encoder cfg
    train_batch_size = finetuning_encoder_cfg['train_batch_size']
    training_epoch = finetuning_encoder_cfg['training_epoch']
    
    # initialize distributed training
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    # set device
    torch.cuda.set_device(local_rank)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        
    # clear GPU cache
    torch.cuda.empty_cache()
    
    # initalize model
    sentence_model = create_sentence_encoder_helper(setup_sentence_encoder_cfg)
    sentence_model = sentence_model.to(f"cuda:{local_rank}")
    
    # Verify device placement
    if torch.distributed.is_initialized():
        assert all(p.device == torch.device(f"cuda:{local_rank}") 
                for p in sentence_model.parameters())
    
    # only rank 0 should handle output directory
    if rank == 0:
        output_dir = os.path.join("data/03_models", f"finetuned_sentence_encoder_batch_{train_batch_size}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
        print(f"Model will be saved at: {output_dir}")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
    # load dataset with distributed sampler
    train_dataset = create_train_dataset()
    
    # set train loss function
    train_loss = losses.MultipleNegativesRankingLoss(model=sentence_model)
    
    args = SentenceTransformerTrainingArguments(
        output_dir=output_dir if rank == 0 else None,
        per_device_train_batch_size=train_batch_size,
        warmup_ratio=0.1,
        fp16=True,
        bf16=False,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        # Optional tracking/debugging parameters:
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=50,
        eval_steps=50,
        num_train_epochs=training_epoch,
        max_steps=len(train_dataset) * training_epoch // (train_batch_size * world_size),
        save_total_limit=10,
        logging_steps=10,
        logging_first_step=True,
        run_name=f"batch_{train_batch_size}_finetune_sentence_encoder_on_{setup_sentence_encoder_cfg['model_name'].split('/')[-1]}",
        # add distributed training settings
        local_rank=local_rank,
        world_size=world_size,
        dataloader_pin_memory=True
    )
    
    # Prepare callbacks - keep EarlyStopping and add TimeLimit
    callbacks = [
        EarlyStoppingCallback(early_stopping_patience=8, early_stopping_threshold=0.05)
    ]
    
    # Add time limit callback if running under SLURM
    if 'SLURM_JOB_TIME_LIMIT' in os.environ and rank == 0:
        try:
            # Parse SLURM time format (DD-HH:MM:SS)
            time_str = os.environ['SLURM_JOB_TIME_LIMIT']
            if '-' in time_str:
                days, time_str = time_str.split('-')
                days = int(days)
            else:
                days = 0
            hours, minutes, seconds = map(int, time_str.split(':'))
            total_seconds = days*86400 + hours*3600 + minutes*60 + seconds
            
            # Subtract 5 minutes as safety buffer
            callbacks.append(TimeLimitCallback(total_seconds - 300, output_dir))
        except Exception as e:
            print(f"⚠️ Failed to parse SLURM time limit: {e}")
            
    
    trainer = SentenceTransformerTrainer(
        model=sentence_model,
        args=args,
        train_dataset=train_dataset,
        loss=train_loss,
        callbacks=callbacks
    )
    
    trainer.train()
    
    if rank == 0:
        final_output_dir = f"{output_dir}/final"
        Path(final_output_dir).mkdir(parents=True, exist_ok=True)
        sentence_model.save(final_output_dir)