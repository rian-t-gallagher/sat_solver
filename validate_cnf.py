#!/usr/bin/env python3
"""
Command-line tool for validating CNF files.

Usage:
    python validate_cnf.py <file.cnf>
    python validate_cnf.py "benchmarks/CNF Formulas/uf20-0156.cnf"
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from cnf_parser import parse_dimacs, parse_raw_cnf
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


def validate_cnf_file(filepath: str) -> bool:
    """Validate a CNF file and print detailed information."""
    print(f"Validating CNF file: {filepath}")
    
    try:
        # Try DIMACS format first
        if filepath.endswith('.rcnf'):
            formula = parse_raw_cnf(filepath)
            print(f"  Format: Raw CNF (.rcnf)")
        else:
            formula = parse_dimacs(filepath)
            print(f"  Format: DIMACS CNF")
            
        print(f"  Variables: {formula.num_vars}")
        print(f"  Clauses: {formula.num_clauses}")
        
        # Check validation
        is_valid = formula.validate()
        print(f"  Validation: {'✓ PASS' if is_valid else '✗ FAIL'}")
        
        # Check 3-SAT constraint
        clause_lengths = [len(clause) for clause in formula.clauses]
        is_3sat = all(length == 3 for length in clause_lengths)
        print(f"  3-SAT format: {'✓ YES' if is_3sat else '✗ NO'}")
        
        if not is_3sat:
            unique_lengths = sorted(set(clause_lengths))
            print(f"    Clause lengths found: {unique_lengths}")
            
        # Show first few clauses
        print(f"  Sample clauses:")
        for i, clause in enumerate(formula.clauses[:3]):
            print(f"    {i+1}: {clause}")
        if len(formula.clauses) > 3:
            print(f"    ... ({len(formula.clauses)-3} more)")
            
        return is_valid
        
    except Exception as e:
        print(f"  Error: {e}")
        return False

if len(sys.argv) != 2:
    print("Usage: python validate_cnf.py <file.cnf>")
    sys.exit(1)
    
filepath = sys.argv[1]

if not os.path.exists(filepath):
    print(f"Error: File not found: {filepath}")
    sys.exit(1)
    
success = validate_cnf_file(filepath)
sys.exit(0 if success else 1)