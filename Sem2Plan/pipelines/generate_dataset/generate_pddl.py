"""
This module generates the PDDL task files
"""

import os, argparse, hashlib, tempfile
from abc import ABC, abstractmethod
import subprocess
import numpy
from pddl.parser.problem import ProblemParser
from pddl.core import Problem
import random


def write_file(problem_dir: str, iteration, desc: str):
    
    # Create problem subdirectory
    problem_subdir = os.path.join(problem_dir, f"p{iteration:02d}")
    os.makedirs(problem_subdir, exist_ok=True)

    # Write the PDDL problem file
    problem_path = os.path.join(problem_subdir, "positive.pddl")
    with open(problem_path, "w") as f:
        f.write(desc)
        
    return problem_path
        
def parse_problem_file(problem_file_path):
        with open(problem_file_path, "r") as f:
            problem_str = f.read()
        parsed_problem = ProblemParser()(problem_str)
        return Problem.__str__(parsed_problem)


class Domain(ABC):

    @abstractmethod
    def generate_problem(self, pddl_generator_dir: str, dataset_dir: str, args):
        """
        Abstract method to convert PDDL problem files into 
        natural language components.
        """


class Blocksworld(Domain):
    def generate_problem(self, dataset_dir: str, args):
        ops = args.ops
        num_blocks = args.blocks
        max_iters = args.max_iterations
        
        seen_problems = set()  # Store unique problems
        iteration = 0
        
        problem_dir = os.path.join(dataset_dir, "problems")
        os.makedirs(problem_dir, exist_ok=True)
        
        while len(seen_problems) < max_iters:

            command = ['pddl-generators/blocksworld/blocksworld', str(ops), str(num_blocks)]

            try:
                result = subprocess.run(command, check=True, text=True, capture_output=True)
                desc = result.stdout  # Capture the standard output as a string
            except subprocess.CalledProcessError as e:
                print(f"Error running the command: {e}")
            
            problem_hash = hashlib.md5(desc.encode()).hexdigest()  # Generate a unique hash for the problem
            
            if problem_hash not in seen_problems:
                seen_problems.add(problem_hash)
                problem_path = write_file(problem_dir, iteration, desc)
                parsed_problem = parse_problem_file(problem_path)
                
                with open(problem_path, 'w') as f:
                    f.write(parsed_problem)
                    
                iteration += 1

            if iteration >= max_iters:
                break  # Stop if reaching limit


class Barman(Domain):
    def generate_problem(self, dataset_dir: str, args):
        num_cocktails = args.cocktails
        num_ingredients = args.ingredients
        num_shots = args.shots
        max_iters = args.max_iterations
        
        seen_problems = set()  # Store unique problems
        iteration = 0
        
        problem_dir = os.path.join(dataset_dir, "problems")
        os.makedirs(problem_dir, exist_ok=True)
        
        while len(seen_problems) < max_iters:

            command = ['pddl-generators/barman/barman-generator.py', str(num_cocktails), str(num_ingredients), str(num_shots)]

            try:
                result = subprocess.run(command, check=True, text=True, capture_output=True)
                desc = result.stdout  # Capture the standard output as a string
            except subprocess.CalledProcessError as e:
                print(f"Error running the command: {e}")
            
            problem_hash = hashlib.md5(desc.encode()).hexdigest()  # Generate a unique hash for the problem
            
            if problem_hash not in seen_problems:
                seen_problems.add(problem_hash)
                problem_path = write_file(problem_dir, iteration, desc)
                parsed_problem = parse_problem_file(problem_path)
                
                with open(problem_path, 'w') as f:
                    f.write(parsed_problem)
                    
                iteration += 1

            if iteration >= max_iters:
                break  # Stop if reaching limit
        

class Floortile(Domain):
    def generate_problem(self, dataset_dir: str, args):
        name = args.name
        num_rows = args.rows
        num_columns = args.columns
        num_robots = args.robots
        mode_flag = args.mode_flag
        max_iters = args.max_iterations
        
        seen_problems = set()  # Store unique problems
        iteration = 0
        
        problem_dir = os.path.join(dataset_dir, "problems")
        os.makedirs(problem_dir, exist_ok=True)
        
        while len(seen_problems) < max_iters:

            command = ['pddl-generators/floortile/floortile-generator.py', 
                       name, str(num_rows), str(num_columns), str(num_robots), mode_flag]

            try:
                result = subprocess.run(command, check=True, text=True, capture_output=True)
                desc = result.stdout  # Capture the standard output as a string
            except subprocess.CalledProcessError as e:
                print(f"Error running the command: {e}")
                
            problem_hash = hashlib.md5(desc.encode()).hexdigest()  # Generate a unique hash for the problem
            
            if problem_hash not in seen_problems:
                seen_problems.add(problem_hash)
                problem_path = write_file(problem_dir, iteration, desc)
                parsed_problem = parse_problem_file(problem_path)
                
                with open(problem_path, 'w') as f:
                    f.write(parsed_problem)
                    
                iteration += 1

            if iteration >= max_iters:
                break  # Stop if reaching limit


class Grippers(Domain):
    def generate_problem(self, dataset_dir: str, args):
        max_iters = args.max_iterations
        
        seen_problems = set()  # Store unique problems
        iteration = 0
        
        problem_dir = os.path.join(dataset_dir, "problems")
        os.makedirs(problem_dir, exist_ok=True)
        
        while len(seen_problems) < max_iters:
            
            num_robots = random.randint(3, args.robots)
            num_rooms = random.randint(3, args.rooms)
            num_balls = random.randint(3, args.balls)

            command = ['pddl-generators/grippers/grippers', "-n", str(num_robots), "-r", str(num_rooms), "-o", str(num_balls)]

            try:
                result = subprocess.run(command, check=True, text=True, capture_output=True)
                desc = result.stdout  # Capture the standard output as a string
            except subprocess.CalledProcessError as e:
                print(f"Error running the command: {e}")
                
            problem_hash = hashlib.md5(desc.encode()).hexdigest()  # Generate a unique hash for the problem
            
            if problem_hash not in seen_problems:
                seen_problems.add(problem_hash)
                problem_path = write_file(problem_dir, iteration, desc)
                parsed_problem = parse_problem_file(problem_path)
                
                with open(problem_path, 'w') as f:
                    f.write(parsed_problem)
                    
                iteration += 1

            if iteration >= max_iters:
                break  # Stop if reaching limit


class Storage(Domain):
    def generate_problem(self, dataset_dir: str, args):
        name = args.name
        num_containers = args.containers
        num_crates = args.crates
        max_iters = args.max_iterations
        
        seen_problems = set()  # Store unique problems
        iteration = 0
        
        problem_dir = os.path.join(dataset_dir, "problems")
        os.makedirs(problem_dir, exist_ok=True)
        
        while len(seen_problems) < max_iters:
            
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file_name = temp_file.name  # Temporary file for the command's argument

            num_hoists = random.randint(3, args.hoists)
            num_depots = random.randint(1, args.depots)
            num_store_areas = random.randint(3, args.store_areas)
            
            dynamic_seed = random.randint(1, 10000)
            command = ['pddl-generators/storage/storage',
                '-p', str(name),                   # Added the problem header
                '-n', str(num_hoists),             # Number of hoists
                '-d', str(num_depots),             # Number of depots
                '-o', str(num_containers),         # Number of containers
                '-s', str(num_store_areas),        # Number of store-areas
                '-c', str(num_crates),             # Number of crates
                '-e', str(dynamic_seed), temp_file_name]
            
            try:
                result = subprocess.run(command, check=True, text=True, capture_output=True)
            
                with open(temp_file_name, 'r') as temp_file:
                    desc = temp_file.read()
                    
                if not desc:
                    continue
            except subprocess.CalledProcessError as e:
                print(f"Error running the command: {e}")
                continue
            
            problem_hash = hashlib.md5(desc.encode()).hexdigest()  # Generate a unique hash for the problem
            
            if problem_hash not in seen_problems:
                seen_problems.add(problem_hash)
                problem_path = write_file(problem_dir, iteration, desc)
                parsed_problem = parse_problem_file(problem_path)
                
                with open(problem_path, 'w') as f:
                    f.write(parsed_problem)
                    
                iteration += 1

            if iteration >= max_iters:
                break  # Stop if reaching limit


class Termes(Domain):
    def generate_problem(self, dataset_dir, args):
        
        problem_dir = os.path.join(dataset_dir, "problems")
        os.makedirs(problem_dir, exist_ok=True)
        
        (size_x, size_y) = (args.size_x, args.size_y)
        height = args.height
        num_towers = args.towers
        seed = args.seed
        max_iters = args.max_iterations

        numpy.random.seed(seed)

        iteration = 0
        
        for range_num_towers, range_height in [
            ([1, 2], range(3, 8)),
            ([3, 4], range(3, 6)),
            ([5, 6], range(3, 5)),
        ]:
            for height in range_height:
                for num_towers in range_num_towers:
                    for i in range(4):
                        board_name, board_str = self.gen_board(
                            size_x, size_y, height, num_towers, seed, dataset_dir, iteration
                        )
                        
                        problem_subdir = os.path.join(problem_dir, f"p{iteration:02d}")
                        os.makedirs(problem_subdir, exist_ok=True)
                        
                        problem_path = os.path.join(problem_subdir, board_name)
                        with open(problem_path, "w") as f:
                            f.write(board_str)
                            
                        command = ['pddl-generators/termes/generate.py', 
                                   'pddl-generators/termes/boards/empty.txt', problem_path, 'pddl', '--dont_remove_slack']
                        
                        try:
                            result = subprocess.run(command, check=True, text=True, capture_output=True)
                            desc = result.stdout  # Capture the standard output as a string
                        except subprocess.CalledProcessError as e:
                            print(f"Error running the command: {e}")
                            
                        problem_path = write_file(problem_dir, iteration, desc)
                        parsed_problem = parse_problem_file(problem_path)
                        
                        with open(problem_path, 'w') as f:
                            f.write(parsed_problem)
                        
                        seed += 1
                        iteration += 1
                    
                        if iteration >= max_iters:
                            break  # Stop if reaching limit
        
                        
    def gen_board(self, size_x, size_y, height, num_towers, seed, dataset_dir, iteration):
        numpy.random.seed(seed)

        board = [[0 for x in range(size_x)] for y in range(size_y)]
        cells = [(x, y) for x in range(size_x) for y in range(size_y)]

        first = True
        for i in numpy.random.choice(range(len(cells)), num_towers, replace=False):
            (x, y) = cells[i]
            col_height = height if first else numpy.random.choice(range(2, height + 1))
            first = False
            board[y][x] = col_height
            
        board_name = "random_towers_{}x{}_{}_{}_{}.txt".format(
            size_x, size_y, height, num_towers, seed
        )
        
        board_str = ""
        
        for row in board:
            board_str += " ".join(map(str, row)) + "\n"
            
        return board_name, board_str


class Logistics(Domain):
    def generate_problem(self, dataset_dir: str, args):
        city_size = args.city_size
        max_iters = args.max_iterations
        
        seen_problems = set()  # Store unique problems
        iteration = 0
        
        problem_dir = os.path.join(dataset_dir, "problems")
        os.makedirs(problem_dir, exist_ok=True)
        
        while len(seen_problems) < max_iters:
            
            num_airplanes = random.randint(3, args.airplanes)
            num_cities = random.randint(3, args.cities)
            num_packages = random.randint(3, args.packages)
            num_trucks = random.randint(3, args.trucks)

            command = ['pddl-generators/logistics/logistics', "-a", str(num_airplanes), "-c", 
                       str(num_cities), "-s", str(city_size), "-p", str(num_packages), "-t", str(num_trucks)]

            try:
                result = subprocess.run(command, check=True, text=True, capture_output=True)
                desc = result.stdout  # Capture the standard output as a string
                print(desc)
            except subprocess.CalledProcessError as e:
                print(f"Error running the command: {e}")
                
            problem_hash = hashlib.md5(desc.encode()).hexdigest()  # Generate a unique hash for the problem
            
            if problem_hash not in seen_problems:
                seen_problems.add(problem_hash)
                problem_path = write_file(problem_dir, iteration, desc)
                parsed_problem = parse_problem_file(problem_path)
                
                with open(problem_path, 'w') as f:
                    f.write(parsed_problem)
                    
                iteration += 1

            if iteration >= max_iters:
                break  # Stop if reaching limit


class TidyBot(Domain):
    pass


class Movie(Domain):
    def generate_problem(self, dataset_dir, args):
        pass
    
    def get_objects():
        pass
    
    def get_init():
        pass
    
    def get_goals():
        pass


class MiniGrid(Domain):
    def generate_problem(self, dataset_dir, args):
        pass
    
    def get_objects():
        pass
    
    def get_init():
        pass
    
    def get_goals():
        pass
    


# if __name__ == "__main__":
    
    # # BARMAN
    # parser = argparse.ArgumentParser(description="Barman Problem Generator")
    # parser.add_argument("--name", type=str, default="prob")
    # parser.add_argument("--levels", type=int, default=4)
    # parser.add_argument("--ingredients", type=int, default=3)
    # parser.add_argument("--shots", type=int, default=4)
    # parser.add_argument("--cocktails", type=int, default=3)
    # parser.add_argument("--max_iterations", type=int, default=50)
    # args = parser.parse_args()

    # b = Barman()
    # b.generate_problem(dataset_dir="data/01_raw_dataset/training/barman", args=args)
    
    # # BLOCKSWORLD
    # parser = argparse.ArgumentParser(description="Blocksworld Problem Generator")
    # parser.add_argument("--name", type=str, default="blocksworld")
    # parser.add_argument("--ops", type=int, default=4)
    # parser.add_argument("--blocks", type=int, default=10)
    # parser.add_argument("--max_iterations", type=int, default=50)
    # args = parser.parse_args()

    # b = Blocksworld()
    # b.generate_problem(dataset_dir="data/01_raw_dataset/training/blocksworld", args=args)
        
    # # FLOORTILE
    # parser = argparse.ArgumentParser(description="Floortile Problem Generator")
    # parser.add_argument("--name", type=str, default="floor-tile")
    # parser.add_argument("--rows", type=int, default=5)
    # parser.add_argument("--columns", type=int, default=3)
    # parser.add_argument("--robots", type=int, default=2)
    # parser.add_argument("--mode_flag", type=str, default="time")
    # parser.add_argument("--max_iterations", type=int, default=50)
    # args = parser.parse_args()

    # f = Floortile()
    # f.generate_problem(dataset_dir="data/01_raw_dataset/training/floortile", args=args)
    
    # # GRIPPERS
    # parser = argparse.ArgumentParser(description="Grippers Problem Generator")
    # parser.add_argument("--name", type=str, default="gripper-strips")
    # parser.add_argument("--robots", type=int, default=4)
    # parser.add_argument("--rooms", type=int, default=8)
    # parser.add_argument("--balls", type=int, default=15)
    # parser.add_argument("--seed", type=int, default=-1)
    # parser.add_argument("--max_iterations", type=int, default=50)
    # args = parser.parse_args()

    # f = Grippers()
    # f.generate_problem(dataset_dir="data/01_raw_dataset/training/grippers", args=args)
    
    # # LOGISTICS
    # parser = argparse.ArgumentParser(description="Logistics Problem Generator")
    # parser.add_argument("--name", type=str, default="logistics-strips")
    # parser.add_argument("--airplanes", type=int, default=5)
    # parser.add_argument("--cities", type=int, default=5)
    # parser.add_argument("--city_size", type=int, default=1)
    # parser.add_argument("--packages", type=int, default=10)
    # parser.add_argument("--trucks", type=int, default=5)
    # parser.add_argument("--max_iterations", type=int, default=50)
    # args = parser.parse_args()
    
    # l = Logistics()
    # l.generate_problem(dataset_dir="data/01_raw_dataset/training/logistics", args=args)
    
    # # STORAGE
    # parser = argparse.ArgumentParser(description="Storage Problem Generator")
    # parser.add_argument("--name", type=str, default="storage")
    # parser.add_argument("--hoists", type=int, default=10)
    # parser.add_argument("--depots", type=int, default=3)
    # parser.add_argument("--containers", type=int, default=1)
    # parser.add_argument("--store_areas", type=int, default=9)
    # parser.add_argument("--crates", type=int, default=4)
    # parser.add_argument("--max_iterations", type=int, default=50)
    # args = parser.parse_args()

    # s = Storage()
    # s.generate_problem(dataset_dir="data/01_raw_dataset/training/storage", args=args)
    
    # # TERMES
    # parser = argparse.ArgumentParser(description="Termes Problem Generator")
    # parser.add_argument("--name", type=str, default="termes")
    # parser.add_argument("--size_x", type=int, default=4)
    # parser.add_argument("--size_y", type=int, default=3)
    # parser.add_argument("--height", type=int, default=3)
    # parser.add_argument("--towers", type=int, default=4)
    # parser.add_argument("--seed", type=int, default=123)
    # parser.add_argument("--max_iterations", type=int, default=50)
    # args = parser.parse_args()

    # f = Termes()
    # f.generate_problem(dataset_dir="data/01_raw_dataset/training/termes", args=args)