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


def parse_cnf_file(filepath: str):
    """Parse and display information about a CNF file."""
    try:
        if filepath.endswith('.rcnf'):
            formula = parse_raw_cnf(filepath)
            print(f"Parsed raw CNF file: {filepath}")
        else:
            formula = parse_dimacs(filepath)
            print(f"Parsed DIMACS CNF file: {filepath}")
            
        print(f"  Variables: {formula.num_vars}")
        print(f"  Clauses: {formula.num_clauses}")
        print(f"  Valid: {formula.validate()}")
        
        # Show first few clauses
        print(f"  First 3 clauses: {formula.clauses[:3]}")
        
        return formula
        
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        sys.exit(1)


def main():
    """Main CLI function."""
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
    
    args = parser.parse_args()
    
    print(f"SAT Solver - Phase 1: DIMACS Parser")
    print(f"Solver: {args.solver}")
    print(f"Input: {args.input}")
    if args.output:
        print(f"Output: {args.output}")
    print(f"Seed: {args.seed}")
    print()
    
    # Parse the CNF file
    formula = parse_cnf_file(args.input)
    
    if args.validate_only:
        print("Validation complete.")
        return
    
    # TODO: Phase 2+ - Implement actual solving
    print(f"\nSOLVER NOT YET IMPLEMENTED")
    print(f"Phase 1 complete: Successfully parsed {formula.num_clauses} clauses")
    print(f"Next: Implement {args.solver} solver in Phase 2")


if __name__ == "__main__":
    main()
