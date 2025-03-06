"""
This module generates the PDDL task files
"""

import glob, os, argparse, random
from pddl import parse_problem
from pddl.logic.predicates import Predicate
from abc import ABC, abstractmethod

class Domain(ABC):

    @abstractmethod
    def generate_problem(self, pddl_generator_dir: str, dataset_dir: str, args):
        """
        Abstract method to convert PDDL problem files into 
        natural language components.
        """

    @abstractmethod
    def get_objects():
        pass
    
    @abstractmethod
    def get_init():
        pass

    @abstractmethod
    def get_goals():
        pass

class Blocksworld(Domain):
    def generate_problem(self, pddl_generator_dir, dataset_dir: str, args):
        pass


class Barman(Domain):
    def generate_problem(self, pddl_generator_dir, dataset_dir, args):

        name = args.name
        max_levels = args.levels
        num_ingredients = args.ingredients
        num_cocktails = args.cocktails
        num_shots = args.shots

        desc = "(define (problem " + name + ")\n"
        desc += "   (:domain barman)"
        desc += f"  (:objects {self.get_objects(max_levels, num_shots, num_ingredients, num_cocktails)})"
        desc += f"  (:init {self.get_init(max_levels, num_shots, num_ingredients, num_cocktails)})"
        desc += f"  (:goal {self.get_goals(num_shots, num_ingredients, num_cocktails)}))"

        print(desc)

    def get_objects(self, max_levels, num_shots, num_ingredients, num_cocktails):
        str_objects = "\n"
        str_objects = str_objects + "      shaker1 - shaker\n"
        str_objects = str_objects + "      left right - hand\n     "

        for i in range(1, num_shots + 1):
            str_objects = str_objects + " shot" + str(i)
        str_objects = str_objects + " - shot\n     "

        for i in range(1, num_ingredients + 1):
            str_objects = str_objects + " ingredient" + str(i)
        str_objects = str_objects + " - ingredient\n     "

        for i in range(1, num_cocktails + 1):
            str_objects = str_objects + " cocktail" + str(i)
        str_objects = str_objects + " - cocktail\n     "

        for i in range(1, num_ingredients + 1):
            str_objects = str_objects + " dispenser" + str(i)
        str_objects = str_objects + " - dispenser\n     "

        for i in range(max_levels + 1):
            str_objects = str_objects + " l" + str(i)
        str_objects = str_objects + " - level\n"

        return str_objects
    
    def get_init(self, max_levels, num_shots, num_ingredients, num_cocktails):
        str_init = "\n"
        str_init = str_init + "  (ontable shaker1)\n"

        for i in range(1, num_shots + 1):
            str_init = str_init + "  (ontable shot" + str(i) + ")\n"

        for i in range(1, num_ingredients + 1):
            str_init = (
                str_init
                + "  (dispenses dispenser"
                + str(i)
                + " ingredient"
                + str(i)
                + ")\n"
            )

        str_init = str_init + "  (clean shaker1)\n"

        for i in range(1, num_shots + 1):
            str_init = str_init + "  (clean shot" + str(i) + ")\n"

        str_init = str_init + "  (empty shaker1)\n"

        for i in range(1, num_shots + 1):
            str_init = str_init + "  (empty shot" + str(i) + ")\n"

        str_init = str_init + "  (handempty left)\n  (handempty right)\n"

        str_init = str_init + "  (shaker-empty-level shaker1 l0)\n"
        str_init = str_init + "  (shaker-level shaker1 l0)\n"
        for i in range(max_levels):
            str_init = str_init + "  (next l" + str(i) + " l" + str(i + 1) + ")\n"

        for i in range(1, num_cocktails + 1):
            parts = random.sample(list(range(1, num_ingredients + 1)), 2)
            str_init = (
                str_init
                + "  (cocktail-part1 cocktail"
                + str(i)
                + " ingredient"
                + str(parts[0])
                + ")\n"
            )
            str_init = (
                str_init
                + "  (cocktail-part2 cocktail"
                + str(i)
                + " ingredient"
                + str(parts[1])
                + ")\n"
            )
        return str_init

    def get_goals(self, num_shots, num_ingredients, num_cocktails):
        str_goal = ""
        str_goal = str_goal + "\n  (and\n"

        serving = random.sample(list(range(1, num_cocktails + 1)), num_cocktails)
        for i in range(1, num_cocktails + 1):
            str_goal = (
                str_goal
                + "      (contains shot"
                + str(i)
                + " cocktail"
                + str(serving[i - 1])
                + ")\n"
            )

        # there is at least one shot for not serving
        for j in range(i + 1, num_shots):
            flip = random.randint(0, 1)
            if flip == 1:
                str_goal = (
                    str_goal
                    + "      (contains shot"
                    + str(j)
                    + " cocktail"
                    + str(random.randint(1, num_cocktails))
                    + ")\n"
                )
            else:
                str_goal = (
                    str_goal
                    + "      (contains shot"
                    + str(j)
                    + " ingredient"
                    + str(random.randint(1, num_ingredients))
                    + ")\n"
                )

        str_goal = str_goal + ")"
        return str_goal
        

class Floortile(Domain):
    pass


class Grippers(Domain):
    pass


class Storage(Domain):
    pass


class Termes(Domain):
    pass


class Logistics(Domain):
    pass


class Movie(Domain):
    pass


class MiniGrid(Domain):
    pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Blocksworld Problem Generator")
    parser.add_argument("--name", type=str, default="prob")
    parser.add_argument("--levels", type=int, default=4)
    parser.add_argument("--ingredients", type=int, default=3)
    parser.add_argument("--shots", type=int, default=4)
    parser.add_argument("--cocktails", type=int, default=3)
    args = parser.parse_args()

    b = Barman()
    b.generate_problem(pddl_generator_dir="pddl-generators/barman/barman-generator.py", dataset_dir="", args=args)
