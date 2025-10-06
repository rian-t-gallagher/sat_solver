"""
CNF Parser for reading Boolean formulas in DIMACS format.
"""

class CNFFormula:
    """Represents a CNF formula."""
    
    def __init__(self, num_vars, clauses):
        """
        Initialize a CNF formula.
        
        Args:
            num_vars: Number of variables
            clauses: List of clauses, where each clause is a list of literals
        """
        self.num_vars = num_vars
        self.clauses = clauses
        self.num_clauses = len(clauses)
    
    def evaluate(self, assignment):
        """
        Evaluate the formula with a given assignment.
        
        Args:
            assignment: Dictionary mapping variable numbers to True/False
            
        Returns:
            Number of satisfied clauses
        """
        satisfied = 0
        for clause in self.clauses:
            clause_satisfied = False
            for literal in clause:
                var = abs(literal)
                if var in assignment:
                    value = assignment[var]
                    # Positive literal: satisfied if var is True
                    # Negative literal: satisfied if var is False
                    if (literal > 0 and value) or (literal < 0 and not value):
                        clause_satisfied = True
                        break
            if clause_satisfied:
                satisfied += 1
        return satisfied
    
    def is_satisfied(self, assignment):
        """Check if all clauses are satisfied."""
        return self.evaluate(assignment) == self.num_clauses


def parse_dimacs(content):
    """
    Parse a CNF formula in DIMACS format.
    
    Args:
        content: String content of DIMACS file
        
    Returns:
        CNFFormula object
    """
    lines = content.strip().split('\n')
    clauses = []
    num_vars = 0
    num_clauses = 0
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('c'):
            # Comment or empty line
            continue
        elif line.startswith('p'):
            # Problem line: p cnf <num_vars> <num_clauses>
            parts = line.split()
            num_vars = int(parts[2])
            num_clauses = int(parts[3])
        else:
            # Clause line
            literals = [int(x) for x in line.split() if int(x) != 0]
            if literals:
                clauses.append(literals)
    
    return CNFFormula(num_vars, clauses)


def parse_dimacs_file(filename):
    """Parse a DIMACS file."""
    with open(filename, 'r') as f:
        content = f.read()
    return parse_dimacs(content)
