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

class TorchDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        neg_weights,
        dir_path,
        expand_size = False,
        estimate_batch_size = 32
    ):
        self.neg_weights = neg_weights # controls probability of sampling easy / semi-hard / hard negatives
        self.estimate_batch_size = estimate_batch_size
        self.expand_size = expand_size
        
        # retrieve problem filepaths
        problem_filepaths = glob(os.path.join(dir_path, f"**/problem.pddl"), recursive=True)
        self.data = pd.DataFrame(columns=["problem_name", "model"])
        
        self.manipulated_problem_model_dict = dict() # key is f'{problem_name}_{problem_model}'
        
        for problem_filepath in tqdm(problem_filepaths, desc="Setting up dataset"):
            problem_name = os.path.basename(os.path.dirname(problem_filepath))
            with open(problem_filepath, 'r') as f:
                problem_str = f.read()
            
            probem_model = ProblemParser(problem_str)
            
            # TO-DO
        
        


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
        train_loss = losses.BatchHardTripletLoss(sentence_model)

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
        

def create_train_dataset():
    """loads in training dataset"""
    data_dir = os.path.join(os.environ['WORKING_DIR'], "data/LOCATION")
    data_paths = glob(os.path.join(data_dir, "*jsonl"))
    train_dataset = load_dataset("json", data_files=data_paths, split="train")
    print("Length of train dataset: ", len(train_dataset))
    return train_dataset
    

def create_eval_dataset():
    pass

def create_test_dataset():
    pass
    
    
def generate_training_dataset(
    train_neg_weights, total_num_examples = 2.0e5, chunksize=10000
    ):
    """
    generates training dataset by sampling from a TorchDataset object and saving it 
    to JSONL files"""
    
    data_dir = os.path.join(os.environ['WORKING_DIR'], "data/01_raw")
    train_dataset = TorchDataset(neg_weights=train_neg_weights, dir_path=data_dir)
    save_dir = os.path.join(os.environ['WORKING_DIR'], "data/02_intermediate")
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    file_id = 0
    num_count = 0
    output_list = []
    pbar = tqdm(total=total_num_examples, desc="Generating training dataset")
    
    while num_count < total_num_examples:
        for i in range(len(train_dataset)):
            output_list.append(json.dumps(train_dataset[i]))
            num_count += 1
            pbar.update(1)
            if len(output_list) == chunksize:
                # save jsonl file
                with open(os.path.join(save_dir, f"train_data_{file_id}.jsonl"), 'w') as f:
                    f.write("\n".join(output_list))
                file_id += 1
                output_list = []
                
        # shuffle the dataset after each epoch
        train_dataset.shuffle()
    # save last chunk
    with open(os.path.join(save_dir, f"train_data_{file_id}.jsonl"), 'w') as f:
        f.write("\n".join(output_list))
    pbar.close()