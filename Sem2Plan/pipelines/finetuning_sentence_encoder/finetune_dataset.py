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

from utils.pddl_manipulation import get_manipulated_problem_list

"""
This module sets up the training / testing dataset
"""

class TorchDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        negative_weights,
        dir_path,
        expand_size = False,
        estimate_batch_size = 32
    ):
        self.negative_weights = negative_weights
        self.estimate_batch_size = estimate_batch_size
        self.expand_size = expand_size
        
        # retrieve problem filepaths
        problem_filepaths = glob(os.path.join(dir_path, f"**/problem.pddl"), recursive=True)
        self.data = pd.DataFrame(columns=["problem_name", "model"])
        
        self.manipulated_problem_model_dict = dict() # key is f'{problem_name}_{problem_model}'
        
        for problem_filepath in tqdm(problem_filepaths, desc="Setting up dataset"):

            # retrieve problem name (problem_name)
            with open(problem_filepath, 'r') as f:
                problem_str = f.read()
            problem_model = ProblemParser()(problem_str)
            problem_name = problem_model.name

            problem_entry = "" # FIX THIS - retrieve path number
            query_content = "" # FIX THIS - retrieve anchor query
            positive_content = "" # FIX THIS - retrieve positive content

            # TO DO - create manipulate problem file function
            manipulated_problem_list, _ = get_manipulated_problem_list(problem_model, self.estimate_batch_size)
            self.manipulated_problem_model_dict[f"{problem_name}_{problem_entry}"] = manipulated_problem_list
            
            # problem_name: domain of problem
            # problem_entry: specific problem entry
            self.data.loc[len(self.data)] = {
                "problem_name": problem_name,
                "problem_entry": problem_entry,
                "query_content": query_content,         # anchor
                "positive_content": positive_content    # positive sample
            }

    def __len__(self):
        if self.expand_size:
            return len(self.data) * 50
        else:
            return len(self.data)
        
    def __getitem__(self, idx):
        idx = idx % len(self.data)
        idx_problem_name = self.data.iloc[idx]["problem_name"]
        idx_problem_idx = self.data.iloc[idx]["problem_idx"]
        output_dict = {
            "anchor": self.data.iloc[idx]["query_content"],
            "positive": self.data.iloc[idx]["positive_content"]
        }

        # extract 10 hard-negative samples
        for range in (10):
            # manipulate current problem file
            pass

        return output_dict
    
    def shuffle(self):
        self.data = self.data.sample(frac=1).reset_index(drop=True)
        

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
    Generates training dataset by sampling from a TorchDataset object and saving it 
    to JSONL files
    """
    
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