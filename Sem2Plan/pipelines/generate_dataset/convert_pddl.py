"""
This module converts PDDL to natural language
"""

import glob, os
from pddl import parse_problem
from abc import ABC, abstractmethod


class Domain(ABC):
    def __init__(self, train_dir: str | None = None) -> None:
        self.train_dir = train_dir

    @abstractmethod
    def convert_pddl_to_nl(self):
        """
        Abstract method to convert PDDL problem files into 
        natural language components.
        """

class Blocksworld(Domain):
    def convert_pddl_to_nl(self, train_dir):

        # "data/01_model_datasets/training/blocksworld"

        os.makedirs(train_dir, exist_ok=True)

        problem_path = os.path.join(train_dir, 'problems/p*')
        problem_files = glob.glob(problem_path)

        for problem_file in problem_files:

            problem_dir = problem_file + "/positive.pddl"

            task = parse_problem(problem_dir)
            description = ""
            object_count = len(task.objects)
            description += f"You have {object_count} blocks. "
            for atom in task.init:
                if "on" == atom.name:
                    description += f"{atom.terms[0].name} is on top of {atom.terms[1].name}. "
            for atom in task.init:
                if "on-table" == atom.name:
                    description += f"{atom.terms[0].name} is on the table. "
            for atom in task.init:
                if "clear" == atom.name:
                    description += f"{atom.terms[0].name} is clear. "
            for atom in task.init:
                if "arm-empty" == atom.name:
                    description += f"Your arm is empty. " 
            description += f"Your goal is to move the blocks. "
            
            
            for goal in task.goal.operands:
                description += f"{goal.terms[0].name} should be on top of {goal.terms[1].name}. "            

            anchor_dir = problem_file + "/anchor.nl"

            nl_file = os.path.splitext(anchor_dir)[0] + ".nl"
            with open(nl_file, 'w') as f:
                f.write(description)

class Barman(Domain):
    pass

class Floortile(Domain):
    pass

class Grippers(Domain):
    pass

class Manipulation(Domain):
    pass

class Storage(Domain):
    pass

class Termes(Domain):
    pass

class Tyreworld(Domain):
    pass