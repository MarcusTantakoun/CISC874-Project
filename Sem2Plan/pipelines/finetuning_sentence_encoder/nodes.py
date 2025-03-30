"""
This module is dedicated to training the selected sentence model.
"""

import os
from datetime import datetime
from pathlib import Path
from sentence_transformers import losses
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from transformers import TrainerCallback
from .finetune_dataset import create_train_dataset
from ..setup_sentence_encoder.nodes import create_sentence_encoder_helper
import time

SLURM_JOB_TIME_LIMIT = 8 * 3600


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
    
    train_batch_size = finetuning_encoder_cfg['train_batch_size']
    training_epoch = finetuning_encoder_cfg['training_epoch']
    is_finetune_complete = finetuning_encoder_cfg['is_finetune_complete']
    
    # define SentenceTransformer model
    if not is_finetune_complete:
        sentence_model = create_sentence_encoder_helper(setup_sentence_encoder_cfg)
        
        # save path of model
        output_dir = os.path.join("data/03_models", f"finetuned_sentence_encoder_batch_{train_batch_size}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
        print(f"Model will be saved at: {output_dir}")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # load dataset
        train_dataset = create_train_dataset()

        train_loss = losses.MultipleNegativesRankingLoss(model=sentence_model)
        
        num_examples = len(train_dataset)
        args = SentenceTransformerTrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=train_batch_size,
            warmup_ratio=0.1,
            fp16=False,
            bf16=False,
            num_train_epochs=training_epoch,
            max_steps=num_examples * training_epoch // train_batch_size,
            save_total_limit=10,
            logging_steps=10,
            logging_first_step=True,
            run_name=f"batch_{train_batch_size}_finetune_sentence_encoder_on_{setup_sentence_encoder_cfg['model_name'].split('/'[-1])}"
        )

        # set up the trainer
        trainer = SentenceTransformerTrainer(
            model=sentence_model,
            args=args,
            train_dataset=train_dataset,
            loss=train_loss
        )

        start_time = time.time()
        # train model and periodically save the model if near the timeout
        for epoch in range(training_epoch):
            trainer.train()
            # Check for remaining time in the SLURM job
            elapsed_time = time.time() - start_time
            remaining_time = max(0, (SLURM_JOB_TIME_LIMIT - elapsed_time))
            print(f"Remaining time: {remaining_time} seconds")

            if remaining_time < 600:  # 10 minutes remaining
                print("Time is running out, saving the model checkpoint...")
                final_output_dir = f"{output_dir}/checkpoint_epoch_{epoch+1}"
                Path(final_output_dir).mkdir(parents=True, exist_ok=True)
                sentence_model.save(final_output_dir)

        final_output_dir = f"{output_dir}/final"
        Path(final_output_dir).mkdir(parents=True, exist_ok=True)
        sentence_model.save(final_output_dir)


if __name__ == "__main__":
    
    setup_sentence_encoder_cfg = {
        "model_name": "microsoft/codebert-base",
        "model_type": "bi_encoder",
        "device": "cpu",  # Change to "cuda" if using GPU
        "is_evaluated": False
    }

    finetuning_encoder_cfg = {
        "train_batch_size": 256,
        "training_epoch": 40,
        "is_finetune_complete": False
    }

    train_sentence_encoder(setup_sentence_encoder_cfg=setup_sentence_encoder_cfg, finetuning_encoder_cfg=finetuning_encoder_cfg)