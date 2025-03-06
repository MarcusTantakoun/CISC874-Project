"""
This module converts PDDL to natural language.

Training should be separated for each training:
    - [10:30:60]
        - Easy: predicates from different domain injected
        - Medium: three predicatess from either init and goal removed/added
        - Hard: single predicate from either init and goal removed/added
"""

import glob, os
from pddl import parse_problem
from pddl.logic.predicates import Predicate
from abc import ABC, abstractmethod


def retrieve_problem_files(dataset_dir: str) -> list[str]:
    os.makedirs(dataset_dir, exist_ok=True)
    problem_path = os.path.join(dataset_dir, 'problems/p*')
    problem_files = glob.glob(problem_path)
    return problem_files


def write_anchor_files(problem_file: str, description: str):
    anchor_dir = problem_file + "/anchor.nl"

    nl_file = os.path.splitext(anchor_dir)[0] + ".nl"
    with open(nl_file, 'w') as f:
        f.write(description)


def count_types(task):
    count = {}

    for obj in task.objects:
        if obj.type_tag not in count.keys():
            count[obj.type_tag] = 0
        count[obj.type_tag] += 1

    return count


def get_goals(task):
    if len(task.goal.operands) > 0:
        goals = task.goal.operands
    else:
        goals = [task.goals]

    return goals


class Domain(ABC):

    @abstractmethod
    def convert_pddl_to_nl(self, dataset_dir: str):
        """
        Abstract method to convert PDDL problem files into 
        natural language components.
        """


class Blocksworld(Domain):

    def convert_pddl_to_nl(self, dataset_dir: str):

        problem_files = retrieve_problem_files(dataset_dir)

        for problem_file in problem_files:
            problem_dir = problem_file + "/positive.pddl"
            task = parse_problem(problem_dir)
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

            goals = get_goals(task)
            for goal in goals:
                description += f"{goal.terms[0].name} should be on top of {goal.terms[1].name}. \n"            

            write_anchor_files(problem_file, description)


class Barman(Domain):
    
    def convert_pddl_to_nl(self, dataset_dir: str):

        problem_files = retrieve_problem_files(dataset_dir)

        for problem_file in problem_files:
            problem_dir = problem_file + "/positive.pddl"
            task = parse_problem(problem_dir)
            description = ""
            count = count_types(task)

            description += f'You have {count["shaker"]} shaker with {count["level"]} levels, {count["shot"]} shot glasses, {count["dispenser"]} dispensers for {count["ingredient"]} ingredients. \n'
            description += "The shaker and shot glasses are clean, empty, and on the table. Your left and right hands are empty. \n"
            cocktails = {obj.name: [0,0] for obj in task.objects if obj.type_tag == "cocktail"}

            for atom in task.init:
                if "cocktail-part1" in atom.name:
                    cocktails[atom.terms[0].name][0] = atom.terms[1]
                if "cocktail-part2" in atom.name:
                    cocktails[atom.terms[0].name][1] = atom.terms[1]

            for cocktail in cocktails.keys():
                description += f"The first ingredient of {cocktail} is {cocktails[cocktail][0]}. "
                description += f"The second ingredient of {cocktail} is {cocktails[cocktail][1]}. \n"

            goals = get_goals(task)
            description += f"Your goal is to make {len(goals)} cocktails. \n"
            for goal in goals:
                description += f"{goal.terms[0].name} contains {goal.terms[1].name}. "

            write_anchor_files(problem_file, description)
        

class Floortile(Domain):
    def convert_pddl_to_nl(self, dataset_dir: str):

        problem_files = retrieve_problem_files(dataset_dir)

        for problem_file in problem_files:
            problem_dir = problem_file + "/positive.pddl"
            task = parse_problem(problem_dir)
            description = ""

            row = 1
            column = 0
            robots = 0
            colors = []

            for obj in task.objects:
                if obj.type_tag == "robot":
                    robots += 1
                elif obj.type_tag == "color":
                    colors.append(obj.name)
                elif obj.type_tag == "tile":
                    row = max(int(obj.name.split('_')[1].split('-')[0]) + 1, row)
                    column = max(int(obj.name.split('-')[1]), column)

            description += f'You have {row} rows and {column} columns of unpainted floor tiles. \n'

            tiles = []
            for obj in task.objects:
                if obj.type_tag == "tile":
                    tiles.append(obj)

            tiles.sort()
            for tile in tiles:
                description += tile.name + " "

                if int(tile.name.split('-')[1]) == column:
                        description += "\n"

            description += f'You have {robots} robot'
            description += 's. \n' if robots > 1 else '. \n'
            description += f'Each robot can paint in color '
            for color in colors[:-1]:
                description += color + ' or '
            description += colors[-1] + '. \n'

            for atom in task.init:

                if type(atom) is Predicate and atom.name == "robot-at":
                    description += f'{atom.terms[0].name} is at {atom.terms[1].name}. \n'

            description += f'Your goal is to paint the grid in the following pattern: \n'
            for goal in task.goal.operands[:-1]:
                description += f'{goal.terms[0].name} is {goal.terms[1].name}; '
            description += f'{task.goal.operands[-1].terms[0].name} is {task.goal.operands[-1].terms[1].name}. \n'

            write_anchor_files(problem_file, description)


class Grippers(Domain):
    def convert_pddl_to_nl(self, dataset_dir: str):

        problem_files = retrieve_problem_files(dataset_dir)

        for problem_file in problem_files:
            problem_dir = problem_file + "/positive.pddl"
            task = parse_problem(problem_dir)
            description = ""
            count = count_types(task)

            description += f"You control {count["robot"]} robots, each robot has a left gripper and a right gripper. \n"
            description += f"There are {count["room"]} rooms and {count["object"]} balls. \n"

            robot_loc = {}
            object_loc = {}
            for atom in task.init:
                if "at-robby" == atom.name:
                    robot_loc[atom.terms[0].name] = atom.terms[1].name
                if "at" == atom.name:
                    object_loc[atom.terms[0].name] = atom.terms[1].name

            for k, v in robot_loc.items():
                description += f"{k} is in {v}. "
            description += "\n"

            for k, v in object_loc.items():
                description += f"{k} is in {v}. "
            description += "\nThe robots' grippers are free. \n"
            description += "Your goal is to transport the balls to their destinations. \n"
            
            goals = get_goals(task)
            for goal in goals:
                description += f"{goal.terms[0].name} should be in {goal.terms[1].name}. \n"   

            write_anchor_files(problem_file, description)


class Storage(Domain):
    def convert_pddl_to_nl(self, dataset_dir: str):

        problem_files = retrieve_problem_files(dataset_dir)

        for problem_file in problem_files:
            problem_dir = problem_file + "/positive.pddl"
            task = parse_problem(problem_dir)
            description = ""

            count = count_types(task)
            count["depotarea"] = 0
            count["containerarea"] = 0
            depot_names = []
            container_names = []

            for obj in task.objects:
                if "depot48-" in obj.name:
                    count["depotarea"] += 1
                    depot_names.append(obj.name)
                if "container-" in obj.name:
                    count["containerarea"] += 1
                    container_names.append(obj.name)
            
            description += f"You have {count["depotarea"]} depot storeareas, {count["containerarea"]} container storeareas, {count["hoist"]} hoists, {count["crate"]} crates, 1 container0, 1 depot48, 1 loadarea. \n"
            description += f"Depot storeareas are: "

            for depot_name in depot_names:
                description += f"{depot_name} "
            description += f"\nContainer storeareas are: "
            for container_name in container_names:
                description += f"{container_name} "
            description += f"\n"
            description += f"Here is a map of depot storeareas: \n"
            description += f"\n"    

            if count["depotarea"]/2 < 2:
                row = 1
                col = count["depotarea"]
            else:
                row = 2
                col = int(count["depotarea"]/2)
            for r in range(1, row+1):
                for c in range(1, col+1):
                    description += f"depot48-{r}-{c} "
                description += f"\n"
            description += f"\n"
            description += f"According to the map, adjacent depot storeareas are connected. \n"
            description += f"All depot storeareas are in depot48. \n"

            for atom in task.init:
                if "on" == atom.name:
                    description += f"{atom.terms[0].name} is on {atom.terms[1].name}. \n"
            description += f"All crates and container storeareas are in container0. \n"
            description += f"All container storeareas are connected to loadarea. \n"
            for atom in task.init:
                if "connected" == atom.name and "depot" in atom.terms[0].name and "loadarea" in atom.terms[1].name:
                    description += f"{atom.terms[0].name} and {atom.terms[1].name} are connected. \n"
            clear_depot = []
            hoist_loc = {}
            for atom in task.init:
                if "clear" == atom.name:
                    clear_depot.append(atom.terms[0].name)
                if "at" == atom.name and "hoist" in atom.terms[0].name:
                    hoist_loc[atom.terms[0].name] = atom.terms[1].name
            for item in clear_depot:
                description += f"{item} "
            if clear_depot:
                description += f"are clear. \n"
            for k,v in hoist_loc.items():
                description += f"{k} is in {v}\n"
            description += f"All hoists are available. \n"
            description += f"Your goal is to move all crates to depot48."            

            write_anchor_files(problem_file, description)  


class Termes(Domain):
    def convert_pddl_to_nl(self, dataset_dir: str):

        problem_files = retrieve_problem_files(dataset_dir)

        for problem_file in problem_files:
            problem_dir = problem_file + "/positive.pddl"
            task = parse_problem(problem_dir)
            description = ""
            stacks = []
            row = 1
            column = 0
            numb = 0
            for obj in task.objects:
                if obj.type_tag == "numb":
                    numb += 1
                elif obj.type_tag == "position":
                    row = max(int(obj.name.split('-')[1]) + 1, row)
                    column = max(int(obj.name.split('-')[2]) + 1, column)

            description += f'The robot is on a grid with {row} rows and {column} columns. \n'

            positions = []
            for obj in task.objects:
                if obj.type_tag == "position":
                    positions.append(obj)

            positions.sort()
            for obj in positions:
                description += obj.name + " "
                if int(obj.name.split('-')[2]) == column - 1:
                    description += "\n"

            for atom in task.init:
                if type(atom) is Predicate and atom.name == "at":
                    description += f'The robot is at {atom.terms[0].name}. \n'
                if type(atom) is Predicate and atom.name == "IS-DEPOT":
                    description += f'The depot for new blocks is at {atom.terms[0].name}. \n'

            description += f'The maximum height of blocks is {numb - 1}. \n'
            description += f'Your goal is to build blocks so that '

            for goal in task.goal.operands:

                if type(goal) is Predicate and goal.name == "height" and int(goal.terms[1].name[-1]) > 0:

                    stacks.append((goal.terms[1].name[-1], goal.terms[0].name))
            for stack in stacks[:-1]:
                description += f'the height at {stack[1]} is {stack[0]}, '
            description += f'the height at {stacks[-1][1]} is {stacks[-1][0]}. \n'
            description += f'You cannot have an unplaced block at the end.'

            write_anchor_files(problem_file, description)


class Tyreworld(Domain):
    def convert_pddl_to_nl(self, dataset_dir: str):

        problem_files = retrieve_problem_files(dataset_dir)

        for problem_file in problem_files:
            problem_dir = problem_file + "/positive.pddl"
            task = parse_problem(problem_dir)
            description = ""

            count = count_types(task)
            wheel_count = int(count["wheel"]/2)
            
            description += f"You have a jack, a pump, a wrench, a boot, {count["hub"]} hubs, {count["nut"]} nuts, {wheel_count} flat tyres, and {wheel_count} intact tyres. \n"
            description += f"The jack, pump, wrench, and intact tyres are in the boot. \n"
            description += f"The boot is unlocked but is closed. \n"
            description += f"The intact tyres are not inflated. \n"
            description += f"The flat tyres are on the hubs. \n"
            description += f"The hubs are on the ground. \n"
            description += f"The nuts are tight on the hubs. \n"
            description += f"The hubs are fastened. \n"                     
            description += f"Your goal is to replace flat tyres with intact tyres on the hubs. Intact tyres should be inflated. The nuts should be tight on the hubs. The flat tyres, wrench, jack, and pump should be in the boot. The boot should be closed. \n"

            write_anchor_files(problem_file, description)


class Logistics(Domain):
    pass


class Movie(Domain):
    pass


class MiniGrid(Domain):
    pass