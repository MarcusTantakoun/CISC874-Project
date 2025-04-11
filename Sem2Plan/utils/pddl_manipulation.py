# This file will manipulate pddl problems to get hard negatives examples.
import os
from copy import deepcopy
import numpy as np
from pddl.logic.base import And, Not
from pddl.parser.problem import ProblemParser
from pddl.core import Problem


MANIPULATION_TYPE_CONSTANT_LST = ["swap", "negate", "remove"]


def get_manipulated_problem_list(problem, manipulated_problem_num, pollution_cap=2):
    """
    Mutates a PDDL problem by manipulating its initial and goal states.
    - Swap: Swaps predicates between the initial and goal states.
    - Negate: Negates predicates in either the initial or goal state.
    - Remove: Removes predicates from either the initial or goal state.
    
    Parameters:
    - problem: The original PDDL problem object.
    - manipulated_problem_num: Number of mutated problems to generate.
    - pollution_cap: Maximum number of manipulations per problem.
    
    Returns:
    - manipulated_problem_lst: List of manipulated problem objects.
    - manipulation_details_lst: List of strings detailing each manipulation.
    """

    # randomly select number of manipulations for each action
    manipulated_problem_lst = []
    manipulation_details_lst = [] 
    
    for _ in range(manipulated_problem_num):
        # randomly select the number of manipulations
        num_manipulations = np.random.randint(1, pollution_cap + 1)

        # get initial and goal state predicates
        init_state = list(problem.init)
        goal_state = list(problem.goal.operands if isinstance(problem.goal, And) else [problem.goal])
        
        updated_init_state = []
        updated_goal_state = []
        
        avail_manip_type_lst = deepcopy(MANIPULATION_TYPE_CONSTANT_LST)
        cur_manip_count = 0
        manip_detail_str = ""
        
        # go through manipulations
        while (
            cur_manip_count < num_manipulations
            and len(avail_manip_type_lst) > 0
            and (len(init_state) > 0 or len(goal_state) > 0)
        ):
            manip_type = np.random.choice(avail_manip_type_lst)
            
            # SWAP
            if manip_type == "swap":
                if len(init_state) == 0 or len(goal_state) == 0:
                    avail_manip_type_lst.remove("swap")
                    continue
                
                swap_idx = np.random.randint(0, len(init_state))
                swap_pred = init_state[swap_idx]
                swap_with_idx = np.random.randint(0, len(goal_state))
                swap_with_pred = goal_state[swap_with_idx]
                
                updated_init_state.append(swap_with_pred)
                updated_goal_state.append(swap_pred)
                
                init_state.pop(swap_idx)
                goal_state.pop(swap_with_idx)
                
                cur_manip_count += 1
                manip_detail_str += f"swap {swap_pred} with {swap_with_pred}\n"
                
            # NEGATE
            elif manip_type == "negate":
                neg_idx = np.random.randint(0,2)
                if len(init_state) == 0:
                    neg_idx = 1
                elif len(goal_state) == 0:
                    neg_idx = 0
                
                if neg_idx == 0 and len(init_state) > 0:
                    neg_idx = np.random.randint(0, len(init_state))
                    neg_pred = init_state[neg_idx]
                    
                    if isinstance(neg_pred, Not):
                        updated_init_state.append(neg_pred.argument)
                    else:
                        updated_init_state.append(Not(neg_pred))
                    
                    init_state.pop(neg_idx)
                    cur_manip_count += 1
                    manip_detail_str += f"negate {neg_pred} in initial state\n"
                    
                elif neg_idx == 1 and len(goal_state) > 0:
                    neg_idx = np.random.randint(0, len(goal_state))
                    neg_pred = goal_state[neg_idx]
                    
                    if isinstance(neg_pred, Not):
                        updated_goal_state.append(neg_pred.argument)
                    else:
                        updated_goal_state.append(Not(neg_pred))
                    
                    goal_state.pop(neg_idx)
                    cur_manip_count += 1
                    manip_detail_str += f"negate {neg_pred} in goal state\n"
                else:
                    avail_manip_type_lst.remove("negate")
                    continue
            
            # REMOVE
            elif manip_type == "remove":
                rem_idx = np.random.randint(0,2)
                if len(init_state) <= 1:
                    rem_idx = 1
                elif len(goal_state) <= 1:
                    rem_idx = 0
                
                if rem_idx == 0 and len(init_state) > 1:
                    rem_idx = np.random.randint(0, len(init_state))
                    manip_detail_str += f"remove {init_state[rem_idx]} from initial state\n"
                    init_state.pop(rem_idx)
                    cur_manip_count += 1
                    
                elif rem_idx == 1 and len(goal_state) > 1:
                    rem_idx = np.random.randint(0, len(goal_state))
                    manip_detail_str += f"remove {goal_state[rem_idx]} from goal state\n"
                    goal_state.pop(rem_idx)
                    cur_manip_count += 1
                else:
                    avail_manip_type_lst.remove("remove")
                    continue
        
        updated_init_state.extend(init_state)
        updated_goal_state.extend(goal_state)
        
        manip_problem = Problem(
            name=problem.name,
            domain_name=problem.domain_name,
            objects=problem.objects,
            init=updated_init_state,  # New mutated initial state
            goal=And(*updated_goal_state) if len(updated_goal_state) > 1 else updated_goal_state[0],
        )
        
        manipulated_problem_lst.append(manip_problem)
        manipulation_details_lst.append(manip_detail_str)
        
    return manipulated_problem_lst, manipulation_details_lst
                


if __name__ == "__main__":
    # Load PDDL problem file
    problem_file_path = "data/05_demonstration/blocksworld/problems/p00/positive.pddl"
    
    if not os.path.exists(problem_file_path):
        raise FileNotFoundError(f"Problem file not found: {problem_file_path}")
    
    with open(problem_file_path, 'r') as f:
        problem_str = f.read()
    
    problem = ProblemParser()(problem_str)

    # generate manipulated problems
    manipulated_problem_lst, manipulation_details_lst = get_manipulated_problem_list(problem, 10, 3)

    # print results
    for poll_problem, manipulate_detail in zip(manipulated_problem_lst, manipulation_details_lst):
        print("Manipulation Detail")
        print(manipulate_detail)
        print("Polluted Problem")
        print(poll_problem)
        print("\n\n")

    print("Original Problem")
    print(problem)