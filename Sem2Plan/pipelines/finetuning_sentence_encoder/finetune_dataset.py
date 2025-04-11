"""
This module sets up the training / testing dataset for training the model.
"""

import torch
import os
import json
import re
import pandas as pd
from glob import glob
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader
from datasets import load_dataset
from pddl.parser.problem import ProblemParser
from pddl.core import Problem
from ...utils.pddl_manipulation import get_manipulated_problem_list


class TorchDataset(torch.utils.data.Dataset):
    """
    A PyTorch-compatible dataset class for semantic similarity learning between
    natural language problem descriptions and PDDL problem files.

    This dataset generates (anchor, positive, negatives) triplets:
        - anchor: natural language description (from `anchor.nl`)
        - positive: corresponding correct PDDL problem (from `positive.pddl`)
        - negatives: 10 manipulated PDDL problems per entry

    Each entry is created by slicing a list of 1000 manipulated problems into 100 groups.

    Attributes:
        data (pd.DataFrame): Stores metadata including problem name, entry ID,
                             and content of anchor and positive samples.
        manipulated_problem_model_dict (dict): Maps unique keys to lists of 10 negative samples.
    """
    
    def __init__(self, dir_path, expand_size = False, estimate_batch_size = 32):
        """
        Initializes the TorchDataset object by loading and preparing the dataset.

        Args:
            dir_path (str): Root directory containing all problem subdirectories.
            expand_size (bool, optional): Placeholder for future data expansion. Defaults to False.
            estimate_batch_size (int, optional): Estimated number of manipulated problems to generate. Defaults to 32.
        """
        self.estimate_batch_size = estimate_batch_size # number of problems to make from a single problem file
        self.expand_size = expand_size
        
        # retrieve problem filepaths
        problem_filepaths = glob(os.path.join(dir_path, "*/p*/p*"), recursive=True)
        self.data = pd.DataFrame(columns=["problem_name", "problem_entry", "query_content", "positive_content"])
        
        self.manipulated_problem_model_dict = dict() # key is f'{problem_name}_{problem_model}'
        
        # iterate through each problem...
        for problem_filepath in tqdm(problem_filepaths, desc="Setting up dataset"):

            anchor_path = f"{problem_filepath}/anchor.nl"
            positive_path = f"{problem_filepath}/positive.pddl"
            
            match = re.search(r"p\d+", os.path.basename(problem_filepath))  # Extracts "p01", "p123", etc.
            if match:
                problem_entry = match.group()
            
            with open(anchor_path, 'r') as f:
                query_str = f.read()

            # retrieve problem name (problem_name)
            with open(positive_path, 'r') as f:
                problem_str = f.read()
            problem_model = ProblemParser()(problem_str)
            problem_name = problem_model.name
            query_content = query_str
            positive_content = problem_str

            # retrieve list of negative samples (1000) from a single problem set
            manipulated_problem_list, _ = get_manipulated_problem_list(problem_model, self.estimate_batch_size, 4)
            
            assert len(manipulated_problem_list) == 1000, f"Expected 1000 problems per problem file, got {len(manipulated_problem_list)}"
            
            # split into 100 groups, each with 10 manipulated problems
            num_entries = 100       # number of data entries
            problems_per_entry = 10 # each data entry gets 10 problems
            
            for i in range(num_entries):
                start_idx = i * problems_per_entry
                end_idx = start_idx + problems_per_entry
                
                negative_samples = manipulated_problem_list[start_idx:end_idx] # get 10 problems
            
                self.manipulated_problem_model_dict[f"{problem_name}_{problem_entry}_{i}"] = negative_samples

                self.data.loc[len(self.data)] = {
                    "problem_name": problem_name,
                    "problem_entry": f"{problem_entry}_{i}",
                    "query_content": query_content,         # anchor
                    "positive_content": positive_content    # positive sample
                }

    def __len__(self):
        """Returns the number of entries in the dataset."""
        return len(self.data)
        
    def __getitem__(self, idx):
        """
        Retrieves the dataset entry at the given index.

        Args:
            idx (int): Index of the desired entry.

        Returns:
            dict: {
                'anchor': str,          # Natural language description
                'positive': str,        # Correct PDDL problem
                'negatives': List[str]  # 10 manipulated PDDL problems
            }
        """
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
        """Shuffles the dataset entries randomly in place."""
        self.data = self.data.sample(frac=1).reset_index(drop=True)



class TorchTestDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe):
        self.data = dataframe
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data.iloc[idx].to_dict()

        

def create_train_dataset():
    """loads in training dataset"""
    data_dir = "data/02_intermediate_dataset/training"
    data_paths = glob(os.path.join(data_dir, "*jsonl"))
    train_dataset = load_dataset("json", data_files=data_paths, split="train")
    return train_dataset
    

def create_test_dataset():
    data_dir = "data/02_intermediate_dataset/testing"
    data_paths = glob(os.path.join(data_dir, "*.jsonl"))  # Ensure correct file extension

    # Load all JSONL files and specify "test" split
    test_dataset = load_dataset("json", data_files={"test": data_paths})["test"]

    test_dataset = test_dataset.shuffle(seed=42)

    return test_dataset
    
    
    
def generate_dataset(data_path, save_path, total_num_examples = 1.0e5, chunksize=5000):
    """
    Generates training dataset by sampling from a TorchDataset object and saving it 
    to JSONL files by certain chunk sizes.
    """
    
    data_dir = data_path
    train_dataset = TorchDataset(dir_path=data_dir, expand_size=False, estimate_batch_size=1000)
    train_dataset_length = len(train_dataset)
        
    save_dir = save_path
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    file_id = 0
    num_count = 0
    output_list = []
    pbar = tqdm(total=train_dataset_length, desc="Generating dataset")
    
    while num_count < train_dataset_length:
        for i in range(len(train_dataset)):
            output_list.append(json.dumps(train_dataset[i]))
            num_count += 1
            pbar.update(1)
            if len(output_list) == chunksize:
                # save jsonl file
                with open(os.path.join(save_dir, f"test_data_{file_id}.jsonl"), 'w') as f:
                    f.write("\n".join(output_list))
                file_id += 1
                output_list = []
                
        # shuffle the dataset after each epoch
        train_dataset.shuffle()
    # save last chunk
    with open(os.path.join(save_dir, f"test_data_{file_id}.jsonl"), 'w') as f:
        f.write("\n".join(output_list))
    pbar.close()
    
    
# if __name__ == "__main__":
    # generate_dataset(data_path="data/01_raw_dataset/training/", save_path="data/02_intermediate_dataset/training/")
    # generate_dataset(data_path="data/01_raw_dataset/testing/", save_path="data/02_intermediate_dataset/testing/")


