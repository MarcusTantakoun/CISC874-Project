"""
RUNS PDDL GENERATION
"""
import os
import sys

def run_generate_pddl_example():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

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


if __name__ == "__main__":
    run_generate_pddl_example()

    # example: python demonstrations/run_generate_pddl.py --name blocksworld --ops 4 --blocks 12 --max_iterations 10
