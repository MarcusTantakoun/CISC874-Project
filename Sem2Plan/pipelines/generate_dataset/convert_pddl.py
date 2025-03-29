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
    if hasattr(task.goal, 'operands') and len(task.goal.operands) > 0:
        goals = task.goal.operands
    else:
        goals = [getattr(task, 'goals', [])]  # Fallback to task.goals or empty list

    return goals




""" TRAINING DOMAINS """
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

            description += f"You have {count['shaker']} shaker with {count['level']} levels, {count['shot']} shot glasses, {count['dispenser']} dispensers for {count['ingredient']} ingredients. \n"
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

            description += f"You have {row} rows and {column} columns of unpainted floor tiles. \n"

            tiles = []
            for obj in task.objects:
                if obj.type_tag == "tile":
                    tiles.append(obj)

            tiles.sort()
            for tile in tiles:
                description += tile.name + " "

                if int(tile.name.split('-')[1]) == column:
                        description += "\n"

            description += f"You have {robots} robot"
            description += 's. \n' if robots > 1 else '. \n'
            description += f"Each robot can paint in color "
            for color in colors[:-1]:
                description += color + ' or '
            description += colors[-1] + '. \n'

            for atom in task.init:
                if type(atom) is Predicate and atom.name == "robot-has":
                    description += f"{atom.terms[0].name} start with the color {atom.terms[1].name}. \n"

            for atom in task.init:
                if type(atom) is Predicate and atom.name == "robot-at":
                    description += f"{atom.terms[0].name} is at {atom.terms[1].name}. \n"

            description += f"Your goal is to paint the grid in the following pattern: \n"
            for goal in task.goal.operands[:-1]:
                description += f"{goal.terms[0].name} is {goal.terms[1].name}; "
            description += f"{task.goal.operands[-1].terms[0].name} is {task.goal.operands[-1].terms[1].name}. \n"

            write_anchor_files(problem_file, description)


class Grippers(Domain):
    def convert_pddl_to_nl(self, dataset_dir: str):

        problem_files = retrieve_problem_files(dataset_dir)

        for problem_file in problem_files:
            problem_dir = problem_file + "/positive.pddl"
            task = parse_problem(problem_dir)
            description = ""
            count = count_types(task)

            description += f"You control {count['robot']} robots, each robot has a left gripper and a right gripper. \n"
            description += f"There are {count['room']} rooms and {count['object']} balls. \n"

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
            count['depotarea'] = 0
            count['containerarea'] = 0
            depot_names = []
            container_names = []

            for obj in task.objects:
                if "depot48-" in obj.name:
                    count['depotarea'] += 1
                    depot_names.append(obj.name)
                if "container-" in obj.name:
                    count['containerarea'] += 1
                    container_names.append(obj.name)
            
            description += f"You have {count['depotarea']} depot storeareas, {count['containerarea']} container storeareas, {count['hoist']} hoists, {count['crate']} crates, 1 container0, 1 depot48, 1 loadarea. \n"
            description += f"Depot storeareas are: "

            for depot_name in depot_names:
                description += f"{depot_name} "
            description += f"\nContainer storeareas are: "
            for container_name in container_names:
                description += f"{container_name} "
            description += f"\n"
            description += f"Here is a map of depot storeareas: \n"
            description += f"\n"    

            if count['depotarea']/2 < 2:
                row = 1
                col = count['depotarea']
            else:
                row = 2
                col = int(count['depotarea']/2)
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

            description += f"The robot is on a grid with {row} rows and {column} columns. \n"

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
                    description += f"The robot is at {atom.terms[0].name}. \n"
                if type(atom) is Predicate and atom.name == "IS-DEPOT":
                    description += f"The depot for new blocks is at {atom.terms[0].name}. \n"

            description += f"The maximum height of blocks is {numb - 1}. \n"
            description += f"Your goal is to build blocks so that "

            for goal in task.goal.operands:

                if type(goal) is Predicate and goal.name == "height" and int(goal.terms[1].name[-1]) > 0:

                    stacks.append((goal.terms[1].name[-1], goal.terms[0].name))
            for stack in stacks[:-1]:
                description += f"the height at {stack[1]} is {stack[0]}, "
            description += f"the height at {stacks[-1][1]} is {stacks[-1][0]}. \n"
            description += f"You cannot have an unplaced block at the end."

            write_anchor_files(problem_file, description)


class Logistics(Domain):
    def convert_pddl_to_nl(self, dataset_dir: str):

        problem_files = retrieve_problem_files(dataset_dir)

        for problem_file in problem_files:
            problem_dir = problem_file + "/positive.pddl"
            task = parse_problem(problem_dir)
            description = ""
            airplanes, trucks, cities, locs, pkgs = [], [], [], [], []
            
            for i in task.objects:
                if i.name.startswith("a"):
                    airplanes.append(i)
                elif i.name.startswith("t"):
                    trucks.append(i)
                elif i.name.startswith("c"):
                    cities.append(i)
                elif i.name.startswith("l"):
                    locs.append(i)
                elif i.name.startswith("p"):
                    pkgs.append(i)
            
            description += f"You have {len(airplanes)} airplanes, {len(trucks)} trucks, {len(cities)} cities, {len(locs)} locations, and {len(pkgs)} packages. \n"
            
            truck_description = ""
            airplane_description = ""
            package_description = ""

            for atom in task.init:
                if type(atom) is Predicate and atom.name == "in-city":
                    description += f"Location {atom.terms[0].name} is in city {atom.terms[1].name}. \n"
                
                if type(atom) is Predicate and atom.name == "at":
                    if atom.terms[0].name.startswith("t"):
                        truck_description += f"Truck {atom.terms[0].name} is at location {atom.terms[1]}. \n"
                    elif atom.terms[0].name.startswith("a"):
                        airplane_description += f"Airplane {atom.terms[0].name} is at location {atom.terms[1]}. \n"
                    elif atom.terms[0].name.startswith("p"):
                        package_description += f"Package {atom.terms[0].name} is at location {atom.terms[1]}. \n"
                        
            description += truck_description + package_description + airplane_description
            description += "\nYour goal is for: \n"
            
            for atom in task.goal.operands:
                if type(atom) is Predicate and atom.name == "at":
                    description += f"Package {atom.terms[0].name} to be at location {atom.terms[1].name}. \n"
                    
            write_anchor_files(problem_file, description)

            
class Rovers(Domain):
    """This function is heavily inefficient. I was too lazy to switch things around :)"""
    def convert_pddl_to_nl(self, dataset_dir: str):

        problem_files = retrieve_problem_files(dataset_dir)

        for problem_file in problem_files:
            problem_dir = problem_file + "/positive.pddl"
            task = parse_problem(problem_dir)
            description = ""
            
            count = count_types(task)
            count['Camera'] , count['Rover'], count['Waypoint'], count['Objective'] = 0, 0, 0, 0
            camera_names, rover_names, waypoint_names, objective_names = [], [], [], []

            for obj in task.objects:
                if "Camera" in obj.type_tag:
                    count['Camera'] += 1
                    camera_names.append(obj.name)
                if "Rover" in obj.type_tag:
                    count['Rover'] += 1
                    rover_names.append(obj.name)
                if "Waypoint" in obj.type_tag:
                    count['Waypoint'] += 1
                    waypoint_names.append(obj.name)
                if "Objective" in obj.type_tag:
                    count['Objective'] += 1
                    objective_names.append(obj.name)
                    
            description += f"You have {count['Camera']} cameras, {count['Rover']} rovers, {count['Waypoint']} waypoints, and {count['Objective']} objectives. \n"
            description += f"You also have a general, 2 modes of high and low resolution, and 2 rover stores. \n"
            description += f"Inititally, the general channel is free and you have the following: \n"
            
            for atom in task.init:
                if "visible" == atom.name:
                    description += f"Waypoint {atom.terms[0].name} is visible to waypoint {atom.terms[1].name}. \n"
            
            for atom in task.init:
                if "at_soil_sample" == atom.name:
                    description += f"Soil sample is at {atom.terms[0].name}. \n"
                elif "at_rock_sample" == atom.name:
                    description += f"Rock sample is at {atom.terms[0].name}. \n"
                elif "at_lander" == atom.name:
                    description += f"The general is at the lander at {atom.terms[1].name}. \n"

            for atom in task.init:
                if "at" == atom.name:
                    description += f"Rover {atom.terms[0].name} is at {atom.terms[1].name}. \n"
                    
            for atom in task.init:
                if "available" == atom.name:
                    description += f"Rover {atom.terms[0].name} is available. \n"
                if "store_of" == atom.name:
                    description += f"{atom.terms[0].name} is the store of {atom.terms[1].name}. \n"
                if "empty" == atom.name:
                    description += f"{atom.terms[0].name} is empty. \n"
                
            for atom in task.init:
                if "equipped_for_rock_analysis" == atom.name:
                    description += f"Rover {atom.terms[0].name} is equipped for rock analysis. \n"
                elif "equipped_for_soil_analysis" == atom.name:
                    description += f"Rover {atom.terms[0].name} is equipped for soil analysis. \n"
                elif "equipped_for_imaging" == atom.name:
                    description += f"Rover {atom.terms[0].name} is equipped for imaging. \n"
                    
            for atom in task.init:
                if "can_traverse" == atom.name:
                    description += f"Rover {atom.terms[0].name} can traverse from {atom.terms[1].name} to {atom.terms[2].name}. \n"
                    
            for atom in task.init:
                if "on_board" == atom.name:
                    description += f"Camera {atom.terms[0].name} is on board with rover {atom.terms[1].name}. \n"
                if "calibration_target" == atom.name:
                    description += f"Camera {atom.terms[0].name}'s calibration target is objective {atom.terms[1].name}. \n"
                if "supports" == atom.name:
                    description += f"Camera {atom.terms[0].name} supports {atom.terms[1].name}. \n"
                    
            for atom in task.init:
                if "visible_from" == atom.name:
                    description += f"Objective {atom.terms[0].name} is visible from {atom.terms[1].name}. \n"
            
            description += "\nYour goal is the following: \n"
            
            goals = get_goals(task)
            if len(goals) > 1:
                for goal in goals:
                    if goal.name == "communicated_rock_data":
                        description += f"Communicated rock data should be at {goal.terms[0].name}. \n"
                    elif goal.name == "communicated_soil_data":
                        description += f"Communicated soil data should be at {goal.terms[0].name}. \n"
                    elif goal.name == "communicated_image_data":
                        description += f"Communicated image data should be at {goal.terms[0].name} with {goal.terms[1].name} resolution. \n"
            else:
                if task.goal.name == "communicated_rock_data":
                    description += f"Communicated rock data should be at {goal.terms[0].name}. \n"
                elif task.goal.name == "communicated_soil_data":
                    description += f"Communicated soil data should be at {goal.terms[0].name}. \n"
                elif task.goal.name == "communicated_image_data":
                    description += f"Communicated image data should be at {goal.terms[0].name} with {goal.terms[1].name} resolution. \n"
                
            write_anchor_files(problem_file, description)


""" TESTING DOMAINS """
class Hiking(Domain):
    def convert_pddl_to_nl(self, dataset_dir: str):

        problem_files = retrieve_problem_files(dataset_dir)

        for problem_file in problem_files:
            problem_dir = problem_file + "/positive.pddl"
            task = parse_problem(problem_dir)
            description = ""
            
            count = count_types(task)
            count['car'] , count['couple'], count['person'], count['place'], count['tent'] = 0, 0, 0, 0, 0

            for obj in task.objects:
                if "car" in obj.type_tag:
                    count['car'] += 1
                if "couple" in obj.type_tag:
                    count['couple'] += 1
                if "person" in obj.type_tag:
                    count['person'] += 1
                if "place" in obj.type_tag:
                    count['place'] += 1
                if "tent" in obj.type_tag:
                    count['tent'] += 1
                    
            description += f"You have {count['car']} cars, {count['couple']} couples, and thus {count['person']} people ({count['person']//2} guys and {count['person']//2} girls), {count['place']} places, and {count['tent']} tents. \n"
            description += f"Inititally, you have the following: \n"
            
            at_car_desc = ""
            at_person_desc = ""
            at_tent_desc = ""
            down_desc = ""
            up_desc = ""
            next_desc = ""
            partners_desc = ""
            walked_desc = ""
            
            for atom in task.init:
                if "at_car" == atom.name:
                    at_car_desc += f"Car {atom.terms[0].name} is at place {atom.terms[1].name}. \n"
                elif "at_person" == atom.name:
                    at_person_desc += f"Person {atom.terms[0].name} is at place {atom.terms[1].name}. \n"
                elif "at_tent" == atom.name:
                    at_tent_desc += f"Tent {atom.terms[0].name} is at place {atom.terms[1].name}. \n"
                elif "down" == atom.name:
                    down_desc += f"Tent {atom.terms[0].name} is down. \n"
                elif "up" == atom.name:
                    up_desc += f"Tent {atom.terms[0].name} is up. \n"
                elif "next" == atom.name:
                    next_desc += f"Place {atom.terms[0].name} is next to place {atom.terms[1].name}. \n"
                elif "partners" == atom.name:
                    partners_desc += f"Couple {atom.terms[0].name} consists of partners {atom.terms[1].name} and {atom.terms[2].name}. \n"
                elif "walked" == atom.name:
                    walked_desc += f"Couple {atom.terms[0].name} walked to place {atom.terms[1].name}. \n"
            
            description += at_car_desc + at_person_desc + at_tent_desc + down_desc + up_desc + next_desc + partners_desc + walked_desc
            description += "\nThe goal is the following: \n"
            
            goals = get_goals(task)
            if len(goals) > 1:
                for goal in goals:
                    if goal.name == "walked":
                        description += f"Couple {goal.terms[0].name} walked to place {goal.terms[1].name}. \n"
            else:
                if goals[0] == "walked":
                    description += f"Couple {goal.terms[0].name} walked to place {goal.terms[1].name}. \n"
                
            write_anchor_files(problem_file, description)
                    


class MiniGrid(Domain):
    def convert_pddl_to_nl(self, dataset_dir: str):

        problem_files = retrieve_problem_files(dataset_dir)

        for problem_file in problem_files:
            problem_dir = problem_file + "/positive.pddl"
            task = parse_problem(problem_dir)
            description = ""
            
            keys, cells, shapes = [], [], []
            
            for i in task.objects:
                if i.name.startswith("key"):
                    keys.append(i)
                elif i.name.startswith("p"):
                    cells.append(i)
                elif i.name.startswith("shape"):
                    shapes.append(i)
            
            description += f"You have {len(keys)} keys, {len(cells)} places, and {len(shapes)} shapes. \n"
            description += f"Inititally, the robot arm is empty and you have the following: \n"
            
            at_robot_desc = ""
            at_desc = ""
            conn_desc = ""
            key_shape_desc = ""
            lock_shape_desc = ""
            locked_desc = ""
            open_desc = ""
            
            for atom in task.init:
                if "at-robot" == atom.name:
                    at_robot_desc += f"Robot is at place {atom.terms[0].name}. \n"
                elif "at" == atom.name:
                    at_desc += f"Key {atom.terms[0].name} is at place {atom.terms[1].name}. \n"
                elif "conn" == atom.name:
                    conn_desc += f"Place {atom.terms[0].name} is connected to place {atom.terms[1].name}. \n"
                elif "key-shape" == atom.name:
                    key_shape_desc += f"Key {atom.terms[0].name} is shaped {atom.terms[1].name}. \n"
                elif "lock-shape" == atom.name:
                    lock_shape_desc += f"Lock {atom.terms[0].name} is shaped {atom.terms[1].name}. \n"
                elif "locked" == atom.name:
                    locked_desc += f"Place {atom.terms[0].name} is locked. \n"
                elif "open" == atom.name:
                    open_desc += f"Place {atom.terms[0].name} is open. \n"
            
            description += at_robot_desc + at_desc + conn_desc + key_shape_desc + lock_shape_desc + locked_desc + open_desc
            description += "\nThe goal is the following: \n"
            
            goals = get_goals(task)
            if len(goals) > 1:
                for goal in goals:
                    if goal.name == "at-robot":
                        description += f"Robot should be at place {goal.terms[0].name}. \n"
            else:
                if task.goal.name == "at-robot":
                    description += f"Robot should be at place {task.goal.terms[0].name}. \n"
                
            write_anchor_files(problem_file, description)


if __name__ == "__main__":
    # b = Blocksworld()
    # b.convert_pddl_to_nl("data/01_raw_dataset/training/blocksworld")
    
    # b = Barman()
    # b.convert_pddl_to_nl("data/01_raw_dataset/training/barman")
    
    # f = Floortile()
    # f.convert_pddl_to_nl("data/01_raw_dataset/training/floortile")
    
    # g = Grippers()
    # g.convert_pddl_to_nl("data/01_raw_dataset/training/grippers")
    
    # l = Logistics()
    # l.convert_pddl_to_nl("data/01_raw_dataset/training/logistics")
    
    # t = Termes()
    # t.convert_pddl_to_nl("data/01_raw_dataset/training/termes")
    
    # s = Storage()
    # s.convert_pddl_to_nl("data/01_raw_dataset/training/storage")
    
    # r = Rovers()
    # r.convert_pddl_to_nl("data/01_raw_dataset/training/rovers")
    
    # # TESTING DATASET
    # h = Hiking()
    # h.convert_pddl_to_nl("data/01_raw_dataset/testing/hiking")
    
    m = MiniGrid()
    m.convert_pddl_to_nl("data/01_raw_dataset/testing/minigrid")
    