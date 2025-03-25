import random
import torch
import os
import glob
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, losses, InputExample
from sentence_transformers.evaluation import TripletEvaluator
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import BatchSamplers, SentenceTransformerTrainingArguments
from transformers import TrainerCallback
from datasets import load_dataset
import tqdm
from pddl.parser.problem import ProblemParser

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

def train_sentence_encoder(
    setup_sentence_encoder_cfg, 
    finetuning_encoder_cfg, 
    cosine_sim_comparison_data
    ):
    
    train_batch_size = finetuning_encoder_cfg['train_batch_size']
    eval_negative_weights = finetuning_encoder_cfg['eval_negative_weights']
    train_negative_weights = finetuning_encoder_cfg['train_negative_weights']
    training_epoch = finetuning_encoder_cfg['training_epoch']
    is_finetune_complete = finetuning_encoder_cfg['is_finetune_complete']
    
    # define SentenceTransformer model
    if not is_finetune_complete:
        model_name = setup_sentence_encoder_cfg['model_name']
        sentence_model = SentenceTransformer(model_name)
        
        # save path of model
        output_dir = os.path.join(os.environ['WORKING_DIR'], "data/02_models", f"finetuned_sentence_encoder_batch_{train_batch_size}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
        print(f"Model will be saved at: {output_dir}")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # load dataset
        train_dataset = create_train_dataset()
        eval_dataset = create_eval_dataset()
        
        # convert dataset to InputExample format
        train_examples = [InputExample(texts=[d['anchor'], d['positive'], d['negative']]) for d in train_dataset]
        eval_examples = [InputExample(texts=[d['anchor'], d['positive'], d['negative']]) for d in eval_dataset]
        
        # Create DataLoader
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=train_batch_size)
        eval_dataloader = DataLoader(eval_examples, shuffle=False, batch_size=train_batch_size)

        # Define training loss function
        # train_loss = losses.BatchHardTripletLoss(sentence_model)
        train_loss = losses.MultipleNegativesRankingLoss(sentence_model)

        # Define evaluator (optional)
        evaluator = TripletEvaluator.from_input_examples(eval_examples, name="eval")

        # Train model
        sentence_model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            evaluator=evaluator,
            epochs=training_epoch,
            warmup_steps=100,
            evaluation_steps=1000,  # Optional, adjust based on dataset size
            output_path=output_dir,
            show_progress_bar=True
        )
        
        # save final model
        final_output_dir = f"{output_dir}/final"
        Path(final_output_dir).mkdir(parents=True, exist_ok=True)
        sentence_model.save(final_output_dir)