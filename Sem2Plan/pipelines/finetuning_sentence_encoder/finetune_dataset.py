import io
import torch
import os
from glob import glob
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
from tqdm import tqdm
from pddl.parser.problem import ProblemParser
from pddl.core import Problem

from ...utils.pddl_manipulation import get_manipulated_problem_list

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
        
        # /*positive.pddl
        
        # retrieve problem filepaths
        problem_filepaths = glob(os.path.join(dir_path, "*/p*/p*"), recursive=True)
        self.data = pd.DataFrame(columns=["problem_name", "problem_entry", "query_content", "positive_content"])
        
        self.manipulated_problem_model_dict = dict() # key is f'{problem_name}_{problem_model}'
        
        for problem_filepath in tqdm(problem_filepaths, desc="Setting up dataset"):
            
            print(f"Processing: {problem_filepath}")

            anchor_path = f"{problem_filepath}/anchor.nl"
            positive_path = f"{problem_filepath}/positive.pddl"
            
            with open(anchor_path, 'r') as f:
                query_str = f.read()

            # retrieve problem name (problem_name)
            with open(positive_path, 'r') as f:
                problem_str = f.read()
            problem_model = ProblemParser()(problem_str)
            problem_name = problem_model.name

            problem_entry = "" # FIX THIS - retrieve path number
            query_content = query_str
            positive_content = problem_str

            # TO DO - create manipulate problem file function
            manipulated_problem_list, _ = get_manipulated_problem_list(problem_model, self.estimate_batch_size, 3)
            
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
        return len(self.data)
        
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        anchor = row["query_content"]
        positive = row["positive_content"]
        
        # retrieve negative samples
        problem_key = f"{row['problem_name']}_{row['problem_entry']}"
        negatives = self.manipulated_problem_model_dict.get(problem_key, [])
        
        negative_strings = [Problem.__str__(neg) for neg in negatives]
        
        output_dict = {
            "anchor": anchor,
            "positive": positive,
            "negatives": negative_strings
        }
        
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
    
if __name__ == "__main__":
    
    dir_path = "data/01_model_datasets/training/"

    # Instantiate Dataset
    dataset = TorchDataset(
        negative_weights=0.5,
        dir_path=dir_path,
        expand_size=True,
        estimate_batch_size=4
    )
    
    print(f"Dataset size: {len(dataset)}")  # Expected: 1 (or more if multiple problems)
    

    print("Anchor:\n", dataset[0]["anchor"])
    print("Positive:\n", dataset[0]["positive"])
    print()
    print("Negative samples:")
    for i in dataset[0]["negatives"]:
        print(i)


