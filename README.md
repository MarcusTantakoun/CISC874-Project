# Sem2Plan: Bridging Natural Language and PDDL Code Using Sentence Encoders
This is the project code for CISC 874. Specifically, it trains sentence encoders, namely, CodeBert-base and all-roberta-large-v1 (but any encoder supporting SentenceTransformers can be used), on self-produced PDDL training dataset, as an exploration to see accuracy performance of compute semantic similarity of PDDL problems to its original natural language code. 

## Usage
To generate PDDL problems for Blocksworld domain:
```python
import argparse
from Sem2Plan.pipelines.generate_dataset.generate_pddl import Blocksworld

parser = argparse.ArgumentParser(description="Blocksworld Problem Generator")
parser.add_argument("--name", type=str, default="blocksworld")
parser.add_argument("--ops", type=int, default=4)
parser.add_argument("--blocks", type=int, default=10)
parser.add_argument("--max_iterations", type=int, default=3)
args = parser.parse_args()

b = Blocksworld()
b.generate_problem(dataset_dir="data/05_demonstration/blocksworld", args=args)
```

To convert those PDDL problems to natural language:
```python
from Sem2Plan.pipelines.generate_dataset.convert_pddl import Blocksworld

b = Blocksworld()
b.convert_pddl_to_nl("data/05_demonstration/blocksworld")
```

Here is how you test out a sentence encoder on the test data:
```python
from sentence_transformers import SentenceTransformer
from Sem2Plan.pipelines.compare_cos_sim.nodes import compute_similarity_01, evaluate_model
from Sem2Plan.pipelines.finetuning_sentence_encoder.finetune_dataset import create_test_dataset

model_path = "path/to/model"
test_data = create_test_dataset()

model = SentenceTransformer(model_path)
results = compute_similarity_01(test_data=test_data, model=model, num_samples=10)
metrics = evaluate_model(results)

print(metrics)
```

The data format to follow is a triplet, which contains *q* query/anchor (natural language), *p* positive PDDL code, and *n_k* negative samples up to *k* in a list. Specifically, a test set should be formatted into a Python dictionary:
```
{
    "anchor": "There are 3 blocks...", 
    "positive": "(define (problem BW-rand-10) (:domain blocksworld-4ops)...", 
    "negatives": ["negative 1", "negative 2", "negative k"]
}
```

## Requirements
To use this project, create a virtual Python environment and do the following:
```
pip install -r requirements.txt
```

To generate PDDL files, you must download the submodule Github repository **pddl-generators** into project: [Github](https://github.com/AI-Planning/pddl-generators)

