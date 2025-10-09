"""
DIMACS CNF format parser for SAT solver.

DIMACS format specification:
- Comments start with 'c'
- Problem line: 'p cnf <num_vars> <num_clauses>'
- Clauses: space-separated literals ending with 0
- Positive literals: variable index
- Negative literals: -variable index
"""

from typing import List, Tuple, Optional
import re


class CNFFormula:
    """Represents a CNF formula with metadata."""
    
    def __init__(self, num_vars: int, num_clauses: int, clauses: List[List[int]]):
        self.num_vars = num_vars
        self.num_clauses = num_clauses
        self.clauses = clauses
        
    def __repr__(self):
        return f"CNFFormula(vars={self.num_vars}, clauses={self.num_clauses})"
    
    def validate(self) -> bool:
        """Validate the formula structure."""
        # Check clause count matches
        if len(self.clauses) != self.num_clauses:
            return False
            
        # Check all variables are within range
        for clause in self.clauses:
            for literal in clause:
                if abs(literal) > self.num_vars or literal == 0:
                    return False
                    
        return True


def parse_dimacs(filepath: str) -> CNFFormula:
    """
    Parse a DIMACS CNF file.
    
    Args:
        filepath: Path to the .cnf file
        
    Returns:
        CNFFormula object containing the parsed formula
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
    """
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        raise FileNotFoundError(f"CNF file not found: {filepath}")
    
    num_vars = None
    num_clauses = None
    clauses = []
    
    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
            
        # Skip comments
        if line.startswith('c'):
            continue
            
        # Skip DIMACS end marker
        if line.startswith('%'):
            break
            
        # Parse problem line
        if line.startswith('p cnf'):
            parts = line.split()
            if len(parts) != 4:
                raise ValueError(f"Invalid problem line at {line_num}: {line}")
            try:
                num_vars = int(parts[2])
                num_clauses = int(parts[3])
            except ValueError:
                raise ValueError(f"Invalid numbers in problem line at {line_num}: {line}")
            continue
            
        # Parse clause line
        if num_vars is None or num_clauses is None:
            raise ValueError(f"Found clause before problem line at {line_num}: {line}")
            
        # Parse literals
        try:
            literals = [int(x) for x in line.split()]
        except ValueError:
            raise ValueError(f"Invalid literal at line {line_num}: {line}")
            
        # Check for terminating 0
        if not literals or literals[-1] != 0:
            raise ValueError(f"Clause must end with 0 at line {line_num}: {line}")
            
        # Remove terminating 0 and add clause
        clause = literals[:-1]
        if clause:  # Only add non-empty clauses
            clauses.append(clause)
    
    if num_vars is None or num_clauses is None:
        raise ValueError("No problem line found in file")
    
    formula = CNFFormula(num_vars, num_clauses, clauses)
    
    if not formula.validate():
        raise ValueError("Formula validation failed")
        
    return formula


def parse_raw_cnf(filepath: str) -> CNFFormula:
    """
    Parse a raw CNF file without DIMACS header (.rcnf format).
    
    Args:
        filepath: Path to the .rcnf file
        
    Returns:
        CNFFormula object containing the parsed formula
    """
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        raise FileNotFoundError(f"CNF file not found: {filepath}")
    
    clauses = []
    max_var = 0
    
    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
            
        # Parse literals
        try:
            literals = [int(x) for x in line.split()]
        except ValueError:
            raise ValueError(f"Invalid literal at line {line_num}: {line}")
            
        # Track maximum variable number
        for lit in literals:
            if lit != 0:
                max_var = max(max_var, abs(lit))
                
        # Add clause (assume no terminating 0 for .rcnf)
        if literals:
            clauses.append(literals)
    
    num_vars = max_var
    num_clauses = len(clauses)
    
    return CNFFormula(num_vars, num_clauses, clauses)