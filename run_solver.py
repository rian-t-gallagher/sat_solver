#!/usr/bin/env python3
"""
SAT Solver CLI - CS 463G Program 3

Main command-line interface for running SAT solving algorithms.
Currently supports DIMACS CNF file parsing.
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from cnf_parser import parse_dimacs, parse_raw_cnf
from solvers import DPLLSolver, WalkSATSolver, GeneticSATSolver, SATResult


def parse_cnf_file(cnf_file_path: str):
    """
    Parse and display information about a CNF file using appropriate parser.
    
    This function determines the correct parser based on file extension and
    provides detailed information about the parsed formula.
    
    Args:
        cnf_file_path: Path to the CNF file (.cnf or .rcnf format)
        
    Returns:
        CNFFormula object containing the parsed Boolean satisfiability formula
        
    Raises:
        SystemExit: If parsing fails or file is not found
    """
    try:
        if cnf_file_path.endswith('.rcnf'):
            parsed_cnf_formula = parse_raw_cnf(cnf_file_path)
            print(f"Successfully parsed raw CNF file: {cnf_file_path}")
        else:
            parsed_cnf_formula = parse_dimacs(cnf_file_path)
            print(f"Successfully parsed DIMACS CNF file: {cnf_file_path}")
            
        print(f"  Number of Variables: {parsed_cnf_formula.num_vars}")
        print(f"  Number of Clauses: {parsed_cnf_formula.num_clauses}")
        print(f"  Formula Validation: {parsed_cnf_formula.validate()}")
        
        # Show first few clauses for debugging
        first_three_clauses = parsed_cnf_formula.clauses[:3]
        print(f"  First 3 clauses: {first_three_clauses}")
        
        return parsed_cnf_formula
        
    except Exception as parsing_error:
        print(f"Error parsing {cnf_file_path}: {parsing_error}")
        sys.exit(1)

parser = argparse.ArgumentParser(
	description="SAT Solver - CS 463G Program 3",
	formatter_class=argparse.RawDescriptionHelpFormatter,
	epilog="""
Examples:
python run_solver.py --input benchmarks/CNF\\ Formulas/uf20-0156.cnf
python run_solver.py --solver dpll --input uf20-0156.cnf --output results.txt
python run_solver.py --solver walksat --input uf50-01.cnf --seed 42
	"""
)

parser.add_argument(
	'--solver', 
	choices=['dpll', 'walksat', 'genetic'],
	default='dpll',
	help='SAT solver algorithm to use (default: dpll)'
)

parser.add_argument(
	'--input', 
	required=True,
	help='Path to CNF input file (.cnf or .rcnf)'
)

parser.add_argument(
	'--output',
	help='Output file for results (default: stdout)'
)

parser.add_argument(
	'--seed',
	type=int,
	default=42,
	help='Random seed for randomized algorithms (default: 42)'
)

parser.add_argument(
	'--validate-only',
	action='store_true',
	help='Only parse and validate the CNF file, don\'t solve'
)

# Parse the command line arguments
command_line_args = parser.parse_args()

print(f"SAT Solver - Phase 3: Heuristic Algorithm Implementation")
print(f"Selected Solver Algorithm: {command_line_args.solver}")
print(f"Input CNF File: {command_line_args.input}")
if command_line_args.output:
	print(f"Output Results File: {command_line_args.output}")
print(f"Random Seed: {command_line_args.seed}")
print()

# Parse the CNF input file to create formula object
parsed_cnf_formula = parse_cnf_file(command_line_args.input)

# If user only wants validation, exit after successful parsing
if command_line_args.validate_only:
	print("CNF file validation completed successfully.")
	sys.exit(0)

# Initialize solver statistics dictionary for all algorithms
solving_result = None
satisfying_assignment = None
solver_statistics = {}

# Phase 3: Implement different SAT solving algorithms
print(f"\n=== Phase 3: Heuristic SAT Solver Implementation ===")
print(f"Selected Algorithm: {command_line_args.solver}")

if command_line_args.solver == 'dpll':
    # Russell & Norvig Chapter 7: Complete DPLL algorithm
    print(f"Textbook Reference: Chapter 7, Section 7.5.2, Figure 7.17")
    print(f"Algorithm Type: Complete (can prove UNSAT)")
    
    # Map CLI solver names to DPLL variable selection heuristics
    dpll_heuristic_mapping = {
        'dpll': 'first_available'
    }
    
    chosen_dpll_heuristic = dpll_heuristic_mapping.get(command_line_args.solver, 'first_available')
    dpll_solver_instance = DPLLSolver(heuristic=chosen_dpll_heuristic)
    
    # Solve using DPLL algorithm
    solving_result, satisfying_assignment, solver_statistics = dpll_solver_instance.solve(parsed_cnf_formula)
    
elif command_line_args.solver == 'walksat':
    # WalkSAT: Local search with random walk
    print(f"Algorithm Reference: Selman, Kautz & Cohen (1994)")
    print(f"Algorithm Type: Heuristic (cannot prove UNSAT)")
    
    walksat_solver_instance = WalkSATSolver(
        max_flips_per_try=1000,
        max_restart_attempts=10,
        random_walk_probability=0.5,
        random_seed=command_line_args.seed
    )
    
    # Solve using WalkSAT algorithm
    solving_result, satisfying_assignment, solver_statistics = walksat_solver_instance.solve(parsed_cnf_formula)
    
elif command_line_args.solver == 'genetic':
    # Genetic Algorithm: Evolutionary approach
    print(f"Algorithm Reference: Mitchell (1996) - Genetic Algorithms")
    print(f"Algorithm Type: Heuristic (cannot prove UNSAT)")
    
    genetic_solver_instance = GeneticSATSolver(
        population_size=100,
        max_generations=500,
        crossover_probability=0.8,
        mutation_probability=0.1,
        random_seed=command_line_args.seed
    )
    
    # Solve using Genetic Algorithm
    solving_result, satisfying_assignment, solver_statistics = genetic_solver_instance.solve(parsed_cnf_formula)
    
else:
    print(f"ERROR: Unsupported solver algorithm: {command_line_args.solver}")
    sys.exit(1)

print(f"\nSolver Result: {solving_result.value}")
if solving_result == SATResult.SATISFIABLE and satisfying_assignment is not None:
    print(f"Satisfying assignment found: {len(satisfying_assignment)} variables assigned")
    if command_line_args.output:
        # Write the satisfying assignment to output file in DIMACS format
        with open(command_line_args.output, 'w') as output_file_handle:
            output_file_handle.write(f"s SATISFIABLE\n")
            for variable_number, truth_value in sorted(satisfying_assignment.items()):
                dimacs_literal = variable_number if truth_value else -variable_number
                output_file_handle.write(f"v {dimacs_literal}\n")
            output_file_handle.write("v 0\n")
        print(f"Satisfying assignment written to {command_line_args.output}")
    else:
        # Show first few variable assignments for debugging
        first_ten_assignments = sorted(satisfying_assignment.items())[:10]
        print(f"Sample variable assignments: {first_ten_assignments}")
        if len(satisfying_assignment) > 10:
            remaining_assignments_count = len(satisfying_assignment) - 10
            print(f"... and {remaining_assignments_count} more variable assignments")
else:
    print("Formula is UNSATISFIABLE - no satisfying assignment exists")
    if command_line_args.output:
        with open(command_line_args.output, 'w') as unsatisfiable_output_file:
            unsatisfiable_output_file.write("s UNSATISFIABLE\n")
        print(f"Unsatisfiable result written to {command_line_args.output}")

print(f"\nAlgorithm Performance Statistics:")

if command_line_args.solver == 'dpll':
    print(f"  Total Decisions Made: {solver_statistics['decisions']}")
    print(f"  Unit Propagations: {solver_statistics['propagations']}")
    print(f"  Conflicts Encountered: {solver_statistics['conflicts']}")  
    print(f"  Backtrack Operations: {solver_statistics['backtracks']}")
    print(f"  Total Solving Time: {solver_statistics['time']:.4f} seconds")
    
elif command_line_args.solver == 'walksat':
    print(f"  Total Variable Flips: {solver_statistics['total_flips']}")
    print(f"  Restart Attempts: {solver_statistics['restart_attempts']}")
    print(f"  Random Walk Flips: {solver_statistics['random_flips']}")
    print(f"  Greedy Flips: {solver_statistics['greedy_flips']}")
    print(f"  Best Fitness Score: {solver_statistics['best_satisfied_clauses']}/{parsed_cnf_formula.num_clauses}")
    print(f"  Total Solving Time: {solver_statistics['solving_time']:.4f} seconds")
    
elif command_line_args.solver == 'genetic':
    print(f"  Generations Evolved: {solver_statistics['generations_evolved']}")
    print(f"  Fitness Evaluations: {solver_statistics['fitness_evaluations']}")
    print(f"  Crossover Operations: {solver_statistics['crossover_operations']}")
    print(f"  Mutation Operations: {solver_statistics['mutation_operations']}")
    print(f"  Best Fitness Score: {solver_statistics['best_fitness_achieved']}/{parsed_cnf_formula.num_clauses}")
    print(f"  Final Population Avg: {solver_statistics['final_population_fitness']:.2f}")
    print(f"  Total Evolution Time: {solver_statistics['evolution_time']:.4f} seconds")

# Verify the solution correctness if formula was satisfiable
if solving_result == SATResult.SATISFIABLE and satisfying_assignment is not None:
    # Choose appropriate verification method based on solver
    if command_line_args.solver == 'dpll':
        solution_is_verified = dpll_solver_instance.verify_solution(parsed_cnf_formula, satisfying_assignment)
    elif command_line_args.solver == 'walksat':
        solution_is_verified = walksat_solver_instance.verify_solution(parsed_cnf_formula, satisfying_assignment)
    elif command_line_args.solver == 'genetic':
        solution_is_verified = genetic_solver_instance.verify_solution(parsed_cnf_formula, satisfying_assignment)
    
    print(f"  Solution Verification: {solution_is_verified}")
    if not solution_is_verified:
        print("  WARNING: Solution verification failed - this indicates a bug!")
