"""
This module modifies original PDDL files by removing / adding states
"""

import random

def manipulate_file(problem_dir: str):
    """
    Randomizer to manipulate problem file (remove/add states)
    """
    pass


def add_state(
        problem_dir: str, 
        state: str = "initial", 
        num: int = 1
        ) -> str:
    """
    Args:
        problem_dir (str): problem file path to change
        state (str): either "initial" or "goal" state to change
        num (int): number of predicates to add

    Returns:
        modified_pddl (str): modified pddl as string format
    """
    pass


def remove_state(
        problem_dir: str, 
        state: str = "initial", 
        num: int = 1
        ) -> str:
    """
    Args:
        problem_dir (str): problem file path to change
        state (str): either "initial" or "goal" state to change
        num (int): number of predicates to add

    Returns:
        modified_pddl (str): modified pddl as string format
    """
    pass