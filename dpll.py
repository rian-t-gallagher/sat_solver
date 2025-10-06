"""
DPLL (Davis-Putnam-Logemann-Loveland) algorithm implementation.
This is a complete SAT solving algorithm.
"""

def dpll(formula, assignment=None):
    """
    DPLL algorithm for SAT solving.
    
    Args:
        formula: CNFFormula object
        assignment: Current partial assignment (dict)
        
    Returns:
        Tuple (satisfiable, assignment) where satisfiable is a boolean
        and assignment is a dict mapping variables to True/False
    """
    if assignment is None:
        assignment = {}
    
    # Check if all clauses are satisfied
    if formula.is_satisfied(assignment):
        # Fill in remaining variables with arbitrary values
        for var in range(1, formula.num_vars + 1):
            if var not in assignment:
                assignment[var] = True
        return True, assignment
    
    # Check if any clause is unsatisfied (conflict)
    for clause in formula.clauses:
        clause_can_be_satisfied = False
        for literal in clause:
            var = abs(literal)
            if var not in assignment:
                clause_can_be_satisfied = True
                break
            else:
                value = assignment[var]
                if (literal > 0 and value) or (literal < 0 and not value):
                    clause_can_be_satisfied = True
                    break
        if not clause_can_be_satisfied:
            return False, None
    
    # Unit propagation: find unit clauses
    unit_found = True
    while unit_found:
        unit_found = False
        for clause in formula.clauses:
            unassigned_literals = []
            satisfied = False
            
            for literal in clause:
                var = abs(literal)
                if var in assignment:
                    value = assignment[var]
                    if (literal > 0 and value) or (literal < 0 and not value):
                        satisfied = True
                        break
                else:
                    unassigned_literals.append(literal)
            
            if not satisfied and len(unassigned_literals) == 1:
                # Unit clause found
                unit_found = True
                literal = unassigned_literals[0]
                var = abs(literal)
                assignment[var] = (literal > 0)
                break
        
        # Check for conflicts after unit propagation
        for clause in formula.clauses:
            clause_can_be_satisfied = False
            for literal in clause:
                var = abs(literal)
                if var not in assignment:
                    clause_can_be_satisfied = True
                    break
                else:
                    value = assignment[var]
                    if (literal > 0 and value) or (literal < 0 and not value):
                        clause_can_be_satisfied = True
                        break
            if not clause_can_be_satisfied:
                return False, None
    
    # Pure literal elimination
    literal_occurrences = {}
    for clause in formula.clauses:
        # Check if clause is already satisfied
        satisfied = False
        for literal in clause:
            var = abs(literal)
            if var in assignment:
                value = assignment[var]
                if (literal > 0 and value) or (literal < 0 and not value):
                    satisfied = True
                    break
        
        if not satisfied:
            for literal in clause:
                var = abs(literal)
                if var not in assignment:
                    if var not in literal_occurrences:
                        literal_occurrences[var] = set()
                    literal_occurrences[var].add(literal > 0)
    
    # Assign pure literals
    for var, polarities in literal_occurrences.items():
        if len(polarities) == 1:
            assignment[var] = True if True in polarities else False
    
    # Check if all variables are assigned
    if len(assignment) == formula.num_vars:
        if formula.is_satisfied(assignment):
            return True, assignment
        else:
            return False, None
    
    # Choose an unassigned variable (branching)
    unassigned_var = None
    for var in range(1, formula.num_vars + 1):
        if var not in assignment:
            unassigned_var = var
            break
    
    if unassigned_var is None:
        # All variables assigned
        if formula.is_satisfied(assignment):
            return True, assignment
        else:
            return False, None
    
    # Try assigning True
    new_assignment = assignment.copy()
    new_assignment[unassigned_var] = True
    result, solution = dpll(formula, new_assignment)
    if result:
        return True, solution
    
    # Try assigning False
    new_assignment = assignment.copy()
    new_assignment[unassigned_var] = False
    result, solution = dpll(formula, new_assignment)
    return result, solution


def solve_dpll(formula):
    """
    Solve a SAT formula using DPLL.
    
    Args:
        formula: CNFFormula object
        
    Returns:
        Tuple (satisfiable, assignment, num_satisfied) where:
        - satisfiable: True if formula is satisfiable
        - assignment: Solution assignment or best effort
        - num_satisfied: Number of satisfied clauses
    """
    satisfiable, assignment = dpll(formula)
    
    if satisfiable:
        num_satisfied = formula.evaluate(assignment)
        return True, assignment, num_satisfied
    else:
        # Return empty assignment if unsatisfiable
        return False, {}, 0
