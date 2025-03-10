"""
This module generates the PDDL task files
"""

import os, argparse, hashlib
from abc import ABC, abstractmethod
import subprocess
import numpy


def write_file(problem_dir: str, iteration, desc: str):
    
    # Create problem subdirectory
    problem_subdir = os.path.join(problem_dir, f"p{iteration:02d}")
    os.makedirs(problem_subdir, exist_ok=True)

    # Write the PDDL problem file
    problem_path = os.path.join(problem_subdir, "positive.pddl")
    with open(problem_path, "w") as f:
        f.write(desc)


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
                write_file(problem_dir, iteration, desc)
                    
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
                write_file(problem_dir, iteration, desc)
                    
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
                write_file(problem_dir, iteration, desc)
                    
                iteration += 1

            if iteration >= max_iters:
                break  # Stop if reaching limit


class Grippers(Domain):
    def generate_problem(self, dataset_dir: str, args):
        num_robots = args.robots
        num_rooms = args.rooms
        num_balls = args.balls
        max_iters = args.max_iterations
        
        seen_problems = set()  # Store unique problems
        iteration = 0
        
        problem_dir = os.path.join(dataset_dir, "problems")
        os.makedirs(problem_dir, exist_ok=True)
        
        while len(seen_problems) < max_iters:

            command = ['pddl-generators/grippers/grippers', "-n", str(num_robots), "-r", str(num_rooms), "-o", str(num_balls)]

            try:
                result = subprocess.run(command, check=True, text=True, capture_output=True)
                desc = result.stdout  # Capture the standard output as a string
            except subprocess.CalledProcessError as e:
                print(f"Error running the command: {e}")
                
            problem_hash = hashlib.md5(desc.encode()).hexdigest()  # Generate a unique hash for the problem
            
            if problem_hash not in seen_problems:
                seen_problems.add(problem_hash)
                write_file(problem_dir, iteration, desc)
                    
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

        numpy.random.seed(seed)

        seed = 1
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
                            
                        write_file(problem_dir, iteration, desc)
                        
                        seed += 1
                        iteration += 1
        
                        
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
        num_airplanes = args.airplanes
        num_cities = args.cities
        city_size = args.city_size
        num_packages = args.packages
        num_trucks = args.trucks
        max_iters = args.max_iterations
        
        seen_problems = set()  # Store unique problems
        iteration = 0
        
        problem_dir = os.path.join(dataset_dir, "problems")
        os.makedirs(problem_dir, exist_ok=True)
        
        while len(seen_problems) < max_iters:

            command = ['pddl-generators/logistics/logistics', "-a", str(num_airplanes), "-c", 
                       str(num_cities), "-s", str(city_size), "-p", str(num_packages), "-t", str(num_trucks)]

            try:
                result = subprocess.run(command, check=True, text=True, capture_output=True)
                desc = result.stdout  # Capture the standard output as a string
            except subprocess.CalledProcessError as e:
                print(f"Error running the command: {e}")
                
            problem_hash = hashlib.md5(desc.encode()).hexdigest()  # Generate a unique hash for the problem
            
            if problem_hash not in seen_problems:
                seen_problems.add(problem_hash)
                write_file(problem_dir, iteration, desc)
                    
                iteration += 1

            if iteration >= max_iters:
                break  # Stop if reaching limit


class Tyreworld(Domain):
    def generate_problem(self, dataset_dir: str, args):
        num_tyres = args.tyres
        max_iters = args.max_iterations
        
        seen_problems = set()  # Store unique problems
        iteration = 0
        
        problem_dir = os.path.join(dataset_dir, "problems")
        os.makedirs(problem_dir, exist_ok=True)
        
        while len(seen_problems) < max_iters:

            command = ['pddl-generators/tyreworld/tyreworld', "-n", str(num_tyres)]

            try:
                result = subprocess.run(command, check=True, text=True, capture_output=True)
                desc = result.stdout  # Capture the standard output as a string
                print(desc)
            except subprocess.CalledProcessError as e:
                print(f"Error running the command: {e}")
                
            problem_hash = hashlib.md5(desc.encode()).hexdigest()  # Generate a unique hash for the problem
            
            if problem_hash not in seen_problems:
                seen_problems.add(problem_hash)
                write_file(problem_dir, iteration, desc)
                    
                iteration += 1

            if iteration >= max_iters:
                break  # Stop if reaching limit


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
    


if __name__ == "__main__":

    # parser = argparse.ArgumentParser(description="Barman Problem Generator")
    # parser.add_argument("--name", type=str, default="prob")
    # parser.add_argument("--levels", type=int, default=4)
    # parser.add_argument("--ingredients", type=int, default=3)
    # parser.add_argument("--shots", type=int, default=4)
    # parser.add_argument("--cocktails", type=int, default=3)
    # parser.add_argument("--max_iterations", type=int, default=3)
    # args = parser.parse_args()

    # b = Barman()
    # b.generate_problem(dataset_dir="data/01_model_datasets/training/barman", args=args)
        
    # parser = argparse.ArgumentParser(description="Floortile Problem Generator")
    # parser.add_argument("--name", type=str, default="floor-tile")
    # parser.add_argument("--rows", type=int, default=5)
    # parser.add_argument("--columns", type=int, default=3)
    # parser.add_argument("--robots", type=int, default=2)
    # parser.add_argument("--mode_flag", type=str, default="time")
    # parser.add_argument("--max_iterations", type=int, default=3)
    # args = parser.parse_args()

    # f = Floortile()
    # f.generate_problem(dataset_dir="data/01_model_datasets/training/floortile", args=args)
    
    # parser = argparse.ArgumentParser(description="Grippers Problem Generator")
    # parser.add_argument("--name", type=str, default="gripper-strips")
    # parser.add_argument("--robots", type=int, default=2)
    # parser.add_argument("--rooms", type=int, default=4)
    # parser.add_argument("--balls", type=int, default=10)
    # parser.add_argument("--seed", type=int, default=-1)
    # parser.add_argument("--max_iterations", type=int, default=3)
    # args = parser.parse_args()

    # f = Grippers()
    # f.generate_problem(dataset_dir="data/01_model_datasets/training/grippers", args=args)
    
    # parser = argparse.ArgumentParser(description="Logistics Problem Generator")
    # parser.add_argument("--name", type=str, default="gripper-strips")
    # parser.add_argument("--airplanes", type=int, default=2)
    # parser.add_argument("--cities", type=int, default=4)
    # parser.add_argument("--city_size", type=int, default=1)
    # parser.add_argument("--packages", type=int, default=5)
    # parser.add_argument("--trucks", type=int, default=5)
    # parser.add_argument("--max_iterations", type=int, default=3)
    # args = parser.parse_args()

    # f = Logistics()
    # f.generate_problem(dataset_dir="data/01_model_datasets/training/logistics", args=args)

    
    parser = argparse.ArgumentParser(description="Termes Problem Generator")
    parser.add_argument("--name", type=str, default="termes")
    parser.add_argument("--size_x", type=int, default=4)
    parser.add_argument("--size_y", type=int, default=3)
    parser.add_argument("--height", type=int, default=3)
    parser.add_argument("--towers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max_iterations", type=int, default=1)
    args = parser.parse_args()

    f = Termes()
    f.generate_problem(dataset_dir="data/01_model_datasets/training/termes", args=args)