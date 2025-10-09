# dimacs.py

from typing import List, Tuple, Optional
import re

class CNFFormula:
    
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


def parse_dimacs(cnf_file_path: str) -> CNFFormula:
    """
    Parse a DIMACS CNF file and create a CNF formula object.
    
    DIMACS format is the standard input format for SAT solvers. It contains:
    - Comment lines starting with 'c'
    - Problem line: 'p cnf <num_variables> <num_clauses>'
    - Clause lines: space-separated integers ending with 0
    - Optional end marker '%'
    
    Args:
        cnf_file_path: Absolute path to the .cnf file to parse
        
    Returns:
        CNFFormula object containing the parsed Boolean formula
        
    Raises:
        FileNotFoundError: If the specified file doesn't exist
        ValueError: If file format is invalid or contains syntax errors
    """
    try:
        with open(cnf_file_path, 'r') as cnf_file_handle:
            all_lines_from_file = cnf_file_handle.readlines()
    except FileNotFoundError:
        raise FileNotFoundError(f"CNF file not found: {cnf_file_path}")
    
    total_number_of_variables = None
    expected_number_of_clauses = None
    parsed_clauses_list = []
    
    # Process each line in the DIMACS file
    for current_line_number, current_line_text in enumerate(all_lines_from_file, 1):
        cleaned_line = current_line_text.strip()
        
        # Skip empty lines in the file
        if not cleaned_line:
            continue
            
        # Skip comment lines (lines starting with 'c')
        if cleaned_line.startswith('c'):
            continue
            
        # Skip DIMACS end marker (optional '%' line)
        if cleaned_line.startswith('%'):
            break
            
        # Parse the problem specification line (format: 'p cnf <vars> <clauses>')
        if cleaned_line.startswith('p cnf'):
            problem_line_components = cleaned_line.split()
            if len(problem_line_components) != 4:
                raise ValueError(f"Invalid problem line at line {current_line_number}: {cleaned_line}")
            try:
                total_number_of_variables = int(problem_line_components[2])
                expected_number_of_clauses = int(problem_line_components[3])
            except ValueError:
                raise ValueError(f"Invalid numbers in problem line at line {current_line_number}: {cleaned_line}")
            continue
            
        # Parse clause lines (must come after problem line)
        if total_number_of_variables is None or expected_number_of_clauses is None:
            raise ValueError(f"Found clause before problem line at line {current_line_number}: {cleaned_line}")
            
        # Parse the literal integers from this clause line
        try:
            literal_integers_in_clause = [int(literal_string) for literal_string in cleaned_line.split()]
        except ValueError:
            raise ValueError(f"Invalid literal at line {current_line_number}: {cleaned_line}")
            
        # Check that clause ends with terminating 0
        if not literal_integers_in_clause or literal_integers_in_clause[-1] != 0:
            raise ValueError(f"Clause must end with 0 at line {current_line_number}: {cleaned_line}")
            
        # Remove the terminating 0 and create the clause
        clause_literals_without_zero = literal_integers_in_clause[:-1]
        if clause_literals_without_zero:  # Only add non-empty clauses
            parsed_clauses_list.append(clause_literals_without_zero)
    
    # Validate that we found the required problem line
    if total_number_of_variables is None or expected_number_of_clauses is None:
        raise ValueError("No problem line found in file")
    
    # Create CNF formula object from parsed data
    constructed_cnf_formula = CNFFormula(total_number_of_variables, expected_number_of_clauses, parsed_clauses_list)
    
    # Validate the formula structure
    if not constructed_cnf_formula.validate():
        raise ValueError("Formula validation failed")
        
    return constructed_cnf_formula


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