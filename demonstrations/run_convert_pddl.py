import os
import sys

def run_convert_pddl_example():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    import argparse
    from Sem2Plan.pipelines.generate_dataset.convert_pddl import Blocksworld

    parser = argparse.ArgumentParser(description="Blocksworld Problem Converter")
    parser.add_argument("--name", type=str, default="blocksworld")
    parser.add_argument("--dir_path", type=str, default="data/05_demonstration/blocksworld")
    args = parser.parse_args()

    b = Blocksworld()
    b.convert_pddl_to_nl(args.dir_path)


if __name__ == "__main__":
    run_convert_pddl_example()

    # python demonstrations/run_convert_pddl.py --name blocksworld --dir_path data/05_demonstration/blocksworld