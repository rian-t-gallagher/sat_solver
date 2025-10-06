"""
WalkSAT algorithm implementation.
This is an incomplete local search algorithm.
"""

import random


def walksat(formula, max_flips=10000, p=0.5, max_tries=10):
    """
    WalkSAT algorithm for SAT solving.
    
    Args:
        formula: CNFFormula object
        max_flips: Maximum number of variable flips per try
        p: Probability of random walk vs. greedy move
        max_tries: Maximum number of random restarts
        
    Returns:
        Tuple (satisfiable, assignment, num_satisfied) where:
        - satisfiable: True if all clauses satisfied
        - assignment: Best assignment found
        - num_satisfied: Number of satisfied clauses
    """
    best_assignment = None
    best_satisfied = 0
    
    for try_num in range(max_tries):
        # Random initial assignment
        assignment = {var: random.choice([True, False]) 
                     for var in range(1, formula.num_vars + 1)}
        
        for flip in range(max_flips):
            num_satisfied = formula.evaluate(assignment)
            
            # Update best solution
            if num_satisfied > best_satisfied:
                best_satisfied = num_satisfied
                best_assignment = assignment.copy()
            
            # Check if satisfied
            if num_satisfied == formula.num_clauses:
                return True, assignment, num_satisfied
            
            # Find unsatisfied clauses
            unsatisfied_clauses = []
            for clause in formula.clauses:
                clause_satisfied = False
                for literal in clause:
                    var = abs(literal)
                    value = assignment[var]
                    if (literal > 0 and value) or (literal < 0 and not value):
                        clause_satisfied = True
                        break
                if not clause_satisfied:
                    unsatisfied_clauses.append(clause)
            
            if not unsatisfied_clauses:
                break
            
            # Pick a random unsatisfied clause
            clause = random.choice(unsatisfied_clauses)
            
            # With probability p, make a random walk move
            if random.random() < p:
                # Pick a random variable from the clause
                literal = random.choice(clause)
                var = abs(literal)
                assignment[var] = not assignment[var]
            else:
                # Make a greedy move: flip the variable that maximizes satisfied clauses
                best_var = None
                best_score = num_satisfied
                
                for literal in clause:
                    var = abs(literal)
                    # Try flipping this variable
                    assignment[var] = not assignment[var]
                    score = formula.evaluate(assignment)
                    if score > best_score:
                        best_score = score
                        best_var = var
                    # Flip back
                    assignment[var] = not assignment[var]
                
                # Make the best flip
                if best_var is not None:
                    assignment[best_var] = not assignment[best_var]
                else:
                    # If no improvement, just flip a random variable from the clause
                    literal = random.choice(clause)
                    var = abs(literal)
                    assignment[var] = not assignment[var]
    
    # Return best solution found
    if best_assignment is None:
        best_assignment = assignment
        best_satisfied = formula.evaluate(assignment)
    
    satisfiable = (best_satisfied == formula.num_clauses)
    return satisfiable, best_assignment, best_satisfied


def solve_walksat(formula, max_flips=10000, p=0.5, max_tries=10):
    """
    Solve a SAT formula using WalkSAT.
    
    Args:
        formula: CNFFormula object
        max_flips: Maximum number of flips per try
        p: Probability of random walk
        max_tries: Maximum number of tries
        
    Returns:
        Tuple (satisfiable, assignment, num_satisfied)
    """
    return walksat(formula, max_flips, p, max_tries)
