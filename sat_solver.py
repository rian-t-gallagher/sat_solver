"""
SAT Solver - Main driver program.

This program implements three SAT solving algorithms:
1. DPLL - Complete algorithm
2. WalkSAT - Incomplete local search algorithm
3. Genetic Algorithm - Evolutionary approach

Fitness equals the number of satisfied clauses.
"""

import argparse
import sys
from cnf_parser import parse_dimacs_file
from dpll import solve_dpll
from walksat import solve_walksat
from genetic import solve_genetic


def print_solution(algorithm, satisfiable, assignment, num_satisfied, total_clauses):
    """Print the solution in a readable format."""
    print(f"\n{'='*60}")
    print(f"Algorithm: {algorithm}")
    print(f"{'='*60}")
    print(f"Satisfiable: {satisfiable}")
    print(f"Satisfied clauses: {num_satisfied}/{total_clauses}")
    
    if assignment:
        print(f"\nAssignment:")
        # Sort variables for readable output
        sorted_vars = sorted(assignment.keys())
        for i in range(0, len(sorted_vars), 10):
            line_vars = sorted_vars[i:i+10]
            print("  ", end="")
            for var in line_vars:
                value = "T" if assignment[var] else "F"
                print(f"x{var}={value}", end=" ")
            print()
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='SAT Solver with multiple algorithms',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -f formula.cnf -a dpll
  %(prog)s -f formula.cnf -a walksat
  %(prog)s -f formula.cnf -a genetic
  %(prog)s -f formula.cnf -a all
        """
    )
    
    parser.add_argument('-f', '--file', required=True,
                       help='CNF formula file in DIMACS format')
    parser.add_argument('-a', '--algorithm', 
                       choices=['dpll', 'walksat', 'genetic', 'all'],
                       default='all',
                       help='Algorithm to use (default: all)')
    parser.add_argument('--max-flips', type=int, default=10000,
                       help='WalkSAT: Maximum flips per try (default: 10000)')
    parser.add_argument('--max-tries', type=int, default=10,
                       help='WalkSAT: Maximum tries (default: 10)')
    parser.add_argument('--walk-prob', type=float, default=0.5,
                       help='WalkSAT: Random walk probability (default: 0.5)')
    parser.add_argument('--pop-size', type=int, default=100,
                       help='Genetic: Population size (default: 100)')
    parser.add_argument('--generations', type=int, default=100,
                       help='Genetic: Number of generations (default: 100)')
    parser.add_argument('--mutation-rate', type=float, default=0.1,
                       help='Genetic: Mutation rate (default: 0.1)')
    
    args = parser.parse_args()
    
    # Parse the CNF formula
    try:
        formula = parse_dimacs_file(args.file)
        print(f"Loaded formula: {formula.num_vars} variables, {formula.num_clauses} clauses")
    except Exception as e:
        print(f"Error reading formula file: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Run selected algorithm(s)
    if args.algorithm in ['dpll', 'all']:
        print("\nRunning DPLL (complete algorithm)...")
        satisfiable, assignment, num_satisfied = solve_dpll(formula)
        print_solution('DPLL', satisfiable, assignment, num_satisfied, formula.num_clauses)
    
    if args.algorithm in ['walksat', 'all']:
        print("\nRunning WalkSAT (local search)...")
        satisfiable, assignment, num_satisfied = solve_walksat(
            formula, 
            max_flips=args.max_flips,
            p=args.walk_prob,
            max_tries=args.max_tries
        )
        print_solution('WalkSAT', satisfiable, assignment, num_satisfied, formula.num_clauses)
    
    if args.algorithm in ['genetic', 'all']:
        print("\nRunning Genetic Algorithm (evolutionary)...")
        satisfiable, assignment, num_satisfied = solve_genetic(
            formula,
            population_size=args.pop_size,
            generations=args.generations,
            mutation_rate=args.mutation_rate
        )
        print_solution('Genetic Algorithm', satisfiable, assignment, num_satisfied, formula.num_clauses)


if __name__ == '__main__':
    main()
