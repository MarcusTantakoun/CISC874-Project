import glob
import os
from collections import namedtuple
from pddl import parse_problem

directory = "data/01_model_datasets/training/blocksworld"
os.makedirs(directory, exist_ok=True)

Domain = namedtuple('Domain', ['name', 'requirements', 'types', 'type_dict', 'constants',
                               'predicates', 'predicate_dict', 'functions', 'actions', 'axioms'])
Problem = namedtuple('Problem', ['task_name', 'task_domain_name', 'task_requirements',
                                 'objects', 'init', 'goal', 'use_metric'])

domain_file = "domain.pddl"
problem_path = os.path.join(directory, 'problems/p*')
problem_files = glob.glob(problem_path)

for problem_file in problem_files:

    problem_dir = problem_file + "/positive.pddl"

    task = parse_problem(problem_dir)
    count = {}
    description = ""
    object_count = len(task.objects)
    description += f"You have {object_count} blocks. \n"
    for atom in task.init:
        if "on" == atom.name:
            description += f"{atom.terms[0].name} is on top of {atom.terms[1].name}. \n"
    for atom in task.init:
        if "on-table" == atom.name:
            description += f"{atom.terms[0].name} is on the table. \n"
    for atom in task.init:
        if "clear" == atom.name:
            description += f"{atom.terms[0].name} is clear. \n"
    for atom in task.init:
        if "arm-empty" == atom.name:
            description += f"Your arm is empty. \n" 
    description += f"Your goal is to move the blocks. \n"
       
    
    for goal in task.goal.operands:
        description += f"{goal.terms[0].name} should be on top of {goal.terms[1].name}. \n"            

    anchor_dir = problem_file + "/anchor.nl"

    nl_file = os.path.splitext(anchor_dir)[0] + ".nl"
    with open(nl_file, 'w') as f:
        f.write(description)