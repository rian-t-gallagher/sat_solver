import argparse

parser = argparse.ArgumentParser(description="SAT Solver CLI")
parser.add_argument("--solver", choices=["dpll", "walksat", "genetic"], required=True)
parser.add_argument("--input", required=True, help="Path to CNF file")
parser.add_argument("--output", help="Path to output file")
parser.add_argument("--seed", type=int, default=None)
arguments = parser.parse_args()
print(f"Running {arguments.solver} on {arguments.input} with seed {arguments.seed}")
