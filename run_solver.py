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
from solvers import DPLLSolver, SATResult


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

print(f"SAT Solver - Phase 2: DPLL Implementation")
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
    
# Russell & Norvig Chapter 7: Implement DPLL solving algorithm
print(f"\n=== Russell & Norvig DPLL Solver Implementation ===")
print(f"Textbook Reference: Chapter 7, Section 7.5.2, Figure 7.17")
print(f"Selected Heuristic Strategy: {command_line_args.solver}")

# Map CLI solver names to DPLL variable selection heuristics
solver_name_to_heuristic_mapping = {
    'dpll': 'first_available',
    'walksat': 'most_constrained',  # Will be replaced with actual WalkSAT in Phase 3
    'genetic': 'least_constraining'  # Will be replaced with actual GA in Phase 3
}

chosen_variable_selection_heuristic = solver_name_to_heuristic_mapping.get(command_line_args.solver, 'first_available')
dpll_solver_instance = DPLLSolver(heuristic=chosen_variable_selection_heuristic)

# Solve the CNF formula using DPLL algorithm
solving_result, satisfying_assignment, solver_statistics = dpll_solver_instance.solve(parsed_cnf_formula)

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

print(f"\nDPLL Solver Performance Statistics:")
print(f"  Total Decisions Made: {solver_statistics['decisions']}")
print(f"  Unit Propagations: {solver_statistics['propagations']}")
print(f"  Conflicts Encountered: {solver_statistics['conflicts']}")  
print(f"  Backtrack Operations: {solver_statistics['backtracks']}")
print(f"  Total Solving Time: {solver_statistics['time']:.4f} seconds")

# Verify the solution correctness if formula was satisfiable
if solving_result == SATResult.SATISFIABLE and satisfying_assignment is not None:
    solution_is_verified = dpll_solver_instance.verify_solution(parsed_cnf_formula, satisfying_assignment)
    print(f"  Solution Verification: {solution_is_verified}")
    if not solution_is_verified:
        print("  WARNING: Solution verification failed - this indicates a bug!")
