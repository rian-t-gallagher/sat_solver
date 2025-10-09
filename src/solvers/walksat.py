"""
WalkSAT Algorithm Implementation - Heuristic SAT Solver

WalkSAT is a local search algorithm for the Boolean satisfiability problem.
It combines the GSAT algorithm with random walk to escape local minima.

Reference Algorithm from:
Selman, B., Kautz, H., & Cohen, B. (1994). 
"Noise strategies for improving local search."
Proceedings of the Twelfth National Conference on Artificial Intelligence (AAAI-94).

WalkSAT Algorithm Overview:
1. Start with a random truth assignment
2. While not satisfied and max_flips not reached:
   a. Pick an unsatisfied clause at random
   b. With probability p: flip a random variable in the clause
   c. Otherwise: flip the variable that minimizes the number of unsatisfied clauses

Key advantages over GSAT:
- Random walk component helps escape local minima
- Better performance on structured SAT instances
- Probabilistic nature provides multiple solution paths

Parameters:
- max_flips: Maximum number of variable flips to attempt
- random_walk_probability: Probability of random flip vs greedy flip
- max_tries: Number of random restarts to attempt
"""

import random
import time
from typing import List, Dict, Set, Optional, Tuple
from enum import Enum

# Import shared components
try:
    from ..cnf_parser import CNFFormula
    from .dpll import SATResult
except ImportError:
    from cnf_parser import CNFFormula
    from solvers.dpll import SATResult


class WalkSATSolver:
    """
    WalkSAT algorithm implementation for Boolean satisfiability.
    
    This is a probabilistic local search algorithm that can find satisfying
    assignments for many SAT instances but cannot prove unsatisfiability.
    """
    
    def __init__(self, 
                 max_flips_per_try: int = 1000,
                 max_restart_attempts: int = 10, 
                 random_walk_probability: float = 0.5,
                 random_seed: int = 42):
        """
        Initialize WalkSAT solver with undergraduate-friendly parameter names.
        
        Args:
            max_flips_per_try: Maximum variable flips per restart attempt
            max_restart_attempts: Maximum number of random restarts to try
            random_walk_probability: Probability of random flip (vs greedy flip)
            random_seed: Random seed for reproducible results
        """
        self.max_flips_per_try = max_flips_per_try
        self.max_restart_attempts = max_restart_attempts
        self.random_walk_probability = random_walk_probability
        self.random_seed = random_seed
        
        # Statistics tracking for performance analysis
        self.solver_statistics = {
            "total_flips": 0,           # Total variable flips across all tries
            "restart_attempts": 0,      # Number of restart attempts made
            "best_satisfied_clauses": 0, # Best fitness score achieved
            "final_satisfied_clauses": 0, # Final fitness score
            "random_flips": 0,          # Number of random walk flips
            "greedy_flips": 0,          # Number of greedy flips
            "solving_time": 0.0         # Total time spent solving
        }
        
    def solve(self, cnf_formula: CNFFormula) -> Tuple[SATResult, Optional[Dict[int, bool]], Dict]:
        """
        Main WalkSAT solving entry point.
        
        Attempts to find a satisfying assignment using WalkSAT local search
        with multiple random restarts if needed.
        
        Args:
            cnf_formula: CNF formula to solve
            
        Returns:
            Tuple of (result, assignment, statistics)
            - result: SATResult (SAT if solution found, UNKNOWN if not found)
            - assignment: Variable assignment dict if found, None otherwise
            - statistics: Solver performance metrics
        """
        start_solving_time = time.time()
        
        # Initialize random number generator with fixed seed for reproducibility
        random.seed(self.random_seed)
        
        # Reset solver statistics for this solve attempt
        self._reset_solver_statistics()
        
        best_assignment_found = None
        best_satisfied_clause_count = 0
        
        # Try multiple random restarts (WalkSAT algorithm outer loop)
        for current_restart_attempt in range(self.max_restart_attempts):
            self.solver_statistics["restart_attempts"] += 1
            
            # Generate random initial truth assignment for this restart
            current_truth_assignment = self._generate_random_assignment(cnf_formula.num_vars)
            
            # Run WalkSAT local search from this starting point
            final_assignment, satisfied_clause_count = self._walksat_local_search(
                cnf_formula, current_truth_assignment)
            
            # Track best result across all restarts
            if satisfied_clause_count > best_satisfied_clause_count:
                best_satisfied_clause_count = satisfied_clause_count
                best_assignment_found = final_assignment.copy()
                
            # Check if we found a complete satisfying assignment
            if satisfied_clause_count == cnf_formula.num_clauses:
                self.solver_statistics["final_satisfied_clauses"] = satisfied_clause_count
                self.solver_statistics["best_satisfied_clauses"] = satisfied_clause_count
                self.solver_statistics["solving_time"] = time.time() - start_solving_time
                
                return SATResult.SATISFIABLE, best_assignment_found, self.solver_statistics
        
        # No complete solution found after all restarts
        self.solver_statistics["final_satisfied_clauses"] = best_satisfied_clause_count
        self.solver_statistics["best_satisfied_clauses"] = best_satisfied_clause_count
        self.solver_statistics["solving_time"] = time.time() - start_solving_time
        
        # Return best partial assignment found (WalkSAT cannot prove UNSAT)
        return SATResult.UNKNOWN, best_assignment_found, self.solver_statistics
    
    def _walksat_local_search(self, cnf_formula: CNFFormula, initial_assignment: Dict[int, bool]) -> Tuple[Dict[int, bool], int]:
        """
        Core WalkSAT local search algorithm implementation.
        
        This implements the main WalkSAT loop that iteratively improves
        the current assignment by flipping variables.
        
        Args:
            cnf_formula: CNF formula being solved
            initial_assignment: Starting truth assignment
            
        Returns:
            Tuple of (final_assignment, satisfied_clause_count)
        """
        current_assignment = initial_assignment.copy()
        
        # WalkSAT main loop - flip variables to improve satisfaction
        for flip_iteration in range(self.max_flips_per_try):
            # Find all clauses that are currently unsatisfied
            unsatisfied_clauses_list = self._find_unsatisfied_clauses(cnf_formula, current_assignment)
            
            # Check if formula is completely satisfied
            if len(unsatisfied_clauses_list) == 0:
                satisfied_count = cnf_formula.num_clauses
                return current_assignment, satisfied_count
            
            # Track that we're making a flip
            self.solver_statistics["total_flips"] += 1
            
            # WalkSAT Step 1: Pick an unsatisfied clause at random
            randomly_chosen_unsatisfied_clause = random.choice(unsatisfied_clauses_list)
            
            # Get all variables that appear in this clause
            variables_in_chosen_clause = [abs(literal) for literal in randomly_chosen_unsatisfied_clause]
            
            # WalkSAT Step 2: Decide between random walk and greedy move
            if random.random() < self.random_walk_probability:
                # Random walk: flip a random variable in the clause
                variable_to_flip = random.choice(variables_in_chosen_clause)
                self.solver_statistics["random_flips"] += 1
            else:
                # Greedy move: flip variable that minimizes unsatisfied clauses
                variable_to_flip = self._choose_best_variable_to_flip(
                    cnf_formula, current_assignment, variables_in_chosen_clause)
                self.solver_statistics["greedy_flips"] += 1
            
            # Flip the chosen variable in the current assignment
            current_assignment[variable_to_flip] = not current_assignment[variable_to_flip]
        
        # Return final assignment and its fitness after max_flips reached
        final_satisfied_count = self._count_satisfied_clauses(cnf_formula, current_assignment)
        return current_assignment, final_satisfied_count
    
    def _generate_random_assignment(self, number_of_variables: int) -> Dict[int, bool]:
        """
        Generate a random truth assignment for all variables.
        
        Args:
            number_of_variables: Total number of variables in the formula
            
        Returns:
            Dictionary mapping variable numbers to random boolean values
        """
        random_assignment = {}
        for variable_number in range(1, number_of_variables + 1):
            # Assign each variable a random truth value
            random_truth_value = random.choice([True, False])
            random_assignment[variable_number] = random_truth_value
        
        return random_assignment
    
    def _find_unsatisfied_clauses(self, cnf_formula: CNFFormula, current_assignment: Dict[int, bool]) -> List[List[int]]:
        """
        Find all clauses that are not satisfied by the current assignment.
        
        A clause is satisfied if at least one of its literals is true
        under the current assignment.
        
        Args:
            cnf_formula: CNF formula to check
            current_assignment: Current variable assignment
            
        Returns:
            List of unsatisfied clauses (each clause is a list of literals)
        """
        unsatisfied_clauses_list = []
        
        for individual_clause in cnf_formula.clauses:
            clause_is_satisfied = False
            
            # Check if any literal in this clause is satisfied
            for literal_in_clause in individual_clause:
                variable_number = abs(literal_in_clause)
                literal_is_positive = literal_in_clause > 0
                assigned_truth_value = current_assignment[variable_number]
                
                # Literal is satisfied if: (positive literal AND variable is True) OR 
                #                        (negative literal AND variable is False)
                if (literal_is_positive and assigned_truth_value) or (not literal_is_positive and not assigned_truth_value):
                    clause_is_satisfied = True
                    break
            
            # Add clause to unsatisfied list if no literal was satisfied
            if not clause_is_satisfied:
                unsatisfied_clauses_list.append(individual_clause)
        
        return unsatisfied_clauses_list
    
    def _count_satisfied_clauses(self, cnf_formula: CNFFormula, current_assignment: Dict[int, bool]) -> int:
        """
        Count the total number of satisfied clauses (fitness function).
        
        Args:
            cnf_formula: CNF formula to evaluate
            current_assignment: Current variable assignment
            
        Returns:
            Number of satisfied clauses (0 to num_clauses)
        """
        satisfied_clause_count = 0
        
        for individual_clause in cnf_formula.clauses:
            # Check if this clause is satisfied by current assignment
            for literal_in_clause in individual_clause:
                variable_number = abs(literal_in_clause)
                literal_is_positive = literal_in_clause > 0
                assigned_truth_value = current_assignment[variable_number]
                
                # If any literal is satisfied, the whole clause is satisfied
                if (literal_is_positive and assigned_truth_value) or (not literal_is_positive and not assigned_truth_value):
                    satisfied_clause_count += 1
                    break  # Move to next clause
        
        return satisfied_clause_count
    
    def _choose_best_variable_to_flip(self, cnf_formula: CNFFormula, current_assignment: Dict[int, bool], 
                                     candidate_variables: List[int]) -> int:
        """
        Choose the variable to flip that minimizes the number of unsatisfied clauses.
        
        This implements the greedy component of WalkSAT by evaluating the
        impact of flipping each candidate variable.
        
        Args:
            cnf_formula: CNF formula being solved
            current_assignment: Current variable assignment
            candidate_variables: Variables to consider flipping
            
        Returns:
            Variable number that should be flipped for best improvement
        """
        best_variable_to_flip = candidate_variables[0]
        best_satisfied_count_after_flip = -1
        
        # Try flipping each candidate variable and measure the impact
        for candidate_variable in candidate_variables:
            # Create temporary assignment with this variable flipped
            temporary_assignment = current_assignment.copy()
            temporary_assignment[candidate_variable] = not temporary_assignment[candidate_variable]
            
            # Count satisfied clauses with this variable flipped
            satisfied_count_with_flip = self._count_satisfied_clauses(cnf_formula, temporary_assignment)
            
            # Keep track of the flip that gives the best satisfaction count
            if satisfied_count_with_flip > best_satisfied_count_after_flip:
                best_satisfied_count_after_flip = satisfied_count_with_flip
                best_variable_to_flip = candidate_variable
        
        return best_variable_to_flip
    
    def _reset_solver_statistics(self):
        """Reset all solver statistics for a new solve attempt."""
        self.solver_statistics = {
            "total_flips": 0,
            "restart_attempts": 0,
            "best_satisfied_clauses": 0,
            "final_satisfied_clauses": 0,
            "random_flips": 0,
            "greedy_flips": 0,
            "solving_time": 0.0
        }
    
    def verify_solution(self, cnf_formula: CNFFormula, assignment: Dict[int, bool]) -> bool:
        """
        Verify that an assignment satisfies the CNF formula.
        
        Args:
            cnf_formula: Original CNF formula
            assignment: Variable assignment to verify
            
        Returns:
            True if assignment satisfies all clauses, False otherwise
        """
        satisfied_clause_count = self._count_satisfied_clauses(cnf_formula, assignment)
        return satisfied_clause_count == cnf_formula.num_clauses