"""
DPLL SAT Solver Implementation - Russell & Norvig Chapter 7

Implementation of the Davis-Putnam-Logemann-Loveland (DPLL) algorithm
following the structure and pseudocode from:

Russell, S. & Norvig, P. (2010). Artificial Intelligence: A Modern Approach (3rd ed.)
Chapter 7: Logical Agents, Section 7.5.2: A complete backtracking algorithm
Figure 7.17: The DPLL algorithm for checking satisfiability (page 260)

The DPLL algorithm is a complete, sound, and terminating algorithm for deciding 
the satisfiability of propositional logic formulas in conjunctive normal form (CNF).

Key features from Russell & Norvig:
- Unit propagation: automatically assign variables in unit clauses
- Pure literal elimination: assign variables appearing with only one polarity  
- Backtracking search: systematic exploration of variable assignments
- Variable selection heuristics: choose next variable to branch on

Reference: Davis, M., Logemann, G., & Loveland, D. (1962). 
"A machine program for theorem-proving". Communications of the ACM, 5(7), 394-397.
"""

from typing import List, Dict, Set, Optional, Tuple, Union
import copy
import time
from enum import Enum

# Import CNFFormula from parser module
try:
    from ..cnf_parser import CNFFormula
except ImportError:
    from cnf_parser import CNFFormula


class SATResult(Enum):
    """Result of SAT solving attempt - Russell & Norvig Section 7.5."""
    SATISFIABLE = "SAT"
    UNSATISFIABLE = "UNSAT"
    UNKNOWN = "UNKNOWN"


class DPLLSolver:
    """
    DPLL SAT solver following Russell & Norvig Figure 7.17.
    
    Implements DPLL-SATISFIABLE?(clauses, symbols, model) algorithm
    from Chapter 7, Section 7.5.2 with modern optimizations.
    """
    
    def __init__(self, heuristic: str = "first_available"):
        """
        Initialize DPLL solver following Russell & Norvig approach.
        
        Args:
            heuristic: Variable selection heuristic (Russell & Norvig Section 7.6)
                      - "first_available": choose first unassigned variable
                      - "most_constrained": choose variable in most clauses (MRV)
                      - "least_constraining": choose variable that rules out fewest values
        """
        self.heuristic = heuristic
        self.stats = {
            "decisions": 0,      # Branching decisions made
            "propagations": 0,   # Unit propagations performed  
            "conflicts": 0,      # Empty clauses encountered
            "backtracks": 0,     # Backtrack operations
            "time": 0.0         # Total solving time
        }
        
    def solve(self, formula: CNFFormula) -> Tuple[SATResult, Optional[Dict[int, bool]], Dict]:
        """
        Main entry point - wrapper for DPLL-SATISFIABLE? algorithm.
        
        Following Russell & Norvig Figure 7.17 structure:
        function DPLL-SATISFIABLE?(clauses, symbols, model) returns true or false
        
        Args:
            formula: CNF formula to solve
            
        Returns:
            Tuple of (result, assignment, statistics)
            - result: SATResult enum value (SAT/UNSAT)
            - assignment: variable assignment dict if SAT, None if UNSAT  
            - statistics: solver performance metrics
        """
        start_time = time.time()
        
        # Reset statistics for this solve attempt
        self.stats = {
            "decisions": 0,
            "propagations": 0, 
            "conflicts": 0,
            "backtracks": 0,
            "time": 0.0
        }
        
        # Convert to Russell & Norvig notation:
        # clauses: set of clauses in CNF
        # symbols: set of propositional symbols (variables)
        # model: partial truth assignment (initially empty)
        
        clauses = [set(clause) for clause in formula.clauses]
        symbols = set(range(1, formula.num_vars + 1))
        model: Dict[int, bool] = {}
        
        # Call main DPLL algorithm
        is_satisfiable = self._dpll_satisfiable(clauses, symbols, model)
        
        self.stats["time"] = time.time() - start_time
        
        if is_satisfiable:
            return SATResult.SATISFIABLE, model, self.stats
        else:
            return SATResult.UNSATISFIABLE, None, self.stats
    
    def _dpll_satisfiable(self, current_clauses: List[Set[int]], remaining_symbols: Set[int], current_model: Dict[int, bool]) -> bool:
        """
        DPLL-SATISFIABLE? algorithm from Russell & Norvig Figure 7.17.
        
        This is the main recursive function that implements the DPLL algorithm.
        It tries to find a satisfying assignment for the given CNF formula.
        
        Args:
            current_clauses: Set of clauses in CNF (each clause is a set of literals)
            remaining_symbols: Set of unassigned propositional symbols (variables)
            current_model: Current partial truth assignment (variable -> True/False)
            
        Returns:
            True if satisfiable under current model, False otherwise
        """
        # Russell & Norvig: "if every clause in clauses is true in model then return true"
        simplified_clauses_after_model = self._simplify_clauses_with_model(current_clauses, current_model)
        
        # Check if all clauses are satisfied (no clauses remaining)
        if len(simplified_clauses_after_model) == 0:
            return True
            
        # Russell & Norvig: "if some clause in clauses is false in model then return false"  
        if any(len(single_clause) == 0 for single_clause in simplified_clauses_after_model):
            self.stats["conflicts"] += 1
            return False
            
        # Russell & Norvig: Unit propagation
        # "P, a positive literal, is a unit clause"
        found_unit_clause = self._find_unit_clause(simplified_clauses_after_model)
        if found_unit_clause is not None:
            unit_literal = found_unit_clause
            unit_variable = abs(unit_literal)
            unit_variable_value = unit_literal > 0
            
            self.stats["propagations"] += 1
            
            # Add to model and recurse
            new_model_with_unit = current_model.copy()
            new_model_with_unit[unit_variable] = unit_variable_value
            new_symbols_without_unit = remaining_symbols - {unit_variable}
            
            result_after_unit_propagation = self._dpll_satisfiable(current_clauses, new_symbols_without_unit, new_model_with_unit)
            if result_after_unit_propagation:
                current_model.clear()
                current_model.update(new_model_with_unit)
            return result_after_unit_propagation
            
        # Russell & Norvig: Pure literal elimination  
        # "P is a pure symbol (appears with same sign in all clauses)"
        found_pure_symbol = self._find_pure_symbol(simplified_clauses_after_model, remaining_symbols)
        if found_pure_symbol is not None:
            pure_variable, pure_variable_value = found_pure_symbol
            
            # Add to model and recurse
            new_model_with_pure = current_model.copy()
            new_model_with_pure[pure_variable] = pure_variable_value
            new_symbols_without_pure = remaining_symbols - {pure_variable}
            
            result_after_pure_elimination = self._dpll_satisfiable(current_clauses, new_symbols_without_pure, new_model_with_pure)
            if result_after_pure_elimination:
                current_model.clear()
                current_model.update(new_model_with_pure)
            return result_after_pure_elimination
            
        # Russell & Norvig: Choose a symbol for branching
        if not remaining_symbols:
            # No more symbols to assign - should not reach here if formula is well-formed
            return True
            
        # Choose next symbol using heuristic
        chosen_branching_variable = self._choose_symbol(simplified_clauses_after_model, remaining_symbols)
        self.stats["decisions"] += 1
        
        # Russell & Norvig: Try both truth values
        # "return DPLL-SATISFIABLE?(clauses, symbols - P, model ∪ {P = true})"
        remaining_symbols_after_choice = remaining_symbols - {chosen_branching_variable}
        
        # Try chosen_branching_variable = true
        model_with_true_assignment = current_model.copy()
        model_with_true_assignment[chosen_branching_variable] = True
        if self._dpll_satisfiable(current_clauses, remaining_symbols_after_choice, model_with_true_assignment):
            # Update original model with successful assignment
            current_model.clear()
            current_model.update(model_with_true_assignment)
            return True
            
        # Russell & Norvig: "or DPLL-SATISFIABLE?(clauses, symbols - P, model ∪ {P = false})"
        self.stats["backtracks"] += 1
        
        # Try chosen_branching_variable = false  
        model_with_false_assignment = current_model.copy()
        model_with_false_assignment[chosen_branching_variable] = False
        if self._dpll_satisfiable(current_clauses, remaining_symbols_after_choice, model_with_false_assignment):
            # Update original model with successful assignment
            current_model.clear()
            current_model.update(model_with_false_assignment)
            return True
            
        return False
    
    def _simplify_clauses_with_model(self, original_clauses: List[Set[int]], current_truth_assignment: Dict[int, bool]) -> List[Set[int]]:
        """
        Simplify the set of clauses given the current truth assignment.
        
        This function applies the current model to remove satisfied clauses
        and unsatisfied literals from remaining clauses.
        
        Args:
            original_clauses: Original list of clauses (each clause is a set of literals) 
            current_truth_assignment: Current variable assignment dictionary
            
        Returns:
            List of simplified clauses with current assignment applied
        """
        simplified_clauses_list = []
        
        for individual_clause in original_clauses:
            # Check if this clause is already satisfied by current assignment
            clause_is_satisfied = False
            
            for literal_in_clause in individual_clause:
                variable_of_literal = abs(literal_in_clause)
                literal_is_positive = literal_in_clause > 0
                
                # If variable is assigned and makes this literal true
                if variable_of_literal in current_truth_assignment:
                    assigned_variable_value = current_truth_assignment[variable_of_literal]
                    if (literal_is_positive and assigned_variable_value) or (not literal_is_positive and not assigned_variable_value):
                        clause_is_satisfied = True
                        break
                        
            # If clause is satisfied, skip it (don't include in simplified clauses)
            if clause_is_satisfied:
                continue
                
            # Build new clause with unsatisfied literals only
            simplified_individual_clause = set()
            for literal_in_clause in individual_clause:
                variable_of_literal = abs(literal_in_clause)
                literal_is_positive = literal_in_clause > 0
                
                # If variable is assigned
                if variable_of_literal in current_truth_assignment:
                    assigned_variable_value = current_truth_assignment[variable_of_literal]
                    # Skip literals that are false under current assignment
                    if (literal_is_positive and not assigned_variable_value) or (not literal_is_positive and assigned_variable_value):
                        continue
                        
                # Keep unassigned literals
                simplified_individual_clause.add(literal_in_clause)
                
            simplified_clauses_list.append(simplified_individual_clause)
            
        return simplified_clauses_list
    
    def _find_unit_clause(self, current_clauses_list: List[Set[int]]) -> Optional[int]:
        """
        Find a unit clause (clause with exactly one unassigned literal).
        
        A unit clause contains exactly one literal, which must be true
        for the clause to be satisfied. This is a key optimization in DPLL.
        
        Args:
            current_clauses_list: List of clauses to search for unit clauses
            
        Returns:
            The literal from the unit clause, or None if no unit clause exists
        """
        for individual_clause in current_clauses_list:
            if len(individual_clause) == 1:
                # Found a unit clause - return the single literal
                single_literal_in_clause = next(iter(individual_clause))
                return single_literal_in_clause
        return None
        
    def _find_pure_symbol(self, current_clauses_list: List[Set[int]], unassigned_variables: Set[int]) -> Optional[Tuple[int, bool]]:
        """
        Find a pure symbol (variable that appears with only one polarity).
        
        A pure symbol appears either only positively or only negatively
        across all clauses. We can safely assign it the value that satisfies
        all its occurrences.
        
        Args:
            current_clauses_list: List of clauses to check for pure symbols
            unassigned_variables: Set of variables not yet assigned
            
        Returns:
            Tuple of (variable, value) for pure symbol, or None if no pure symbol found
        """
        # Track positive and negative occurrences for each variable
        positive_occurrences_by_variable = set()
        negative_occurrences_by_variable = set()
        
        # Scan all clauses to find positive/negative literal occurrences
        for individual_clause in current_clauses_list:
            for literal_in_clause in individual_clause:
                variable_of_literal = abs(literal_in_clause)
                
                # Only consider unassigned variables
                if variable_of_literal in unassigned_variables:
                    if literal_in_clause > 0:
                        positive_occurrences_by_variable.add(variable_of_literal)
                    else:
                        negative_occurrences_by_variable.add(variable_of_literal)
                        
        # Find variables that appear only positive or only negative
        for unassigned_variable in unassigned_variables:
            appears_positive = unassigned_variable in positive_occurrences_by_variable
            appears_negative = unassigned_variable in negative_occurrences_by_variable
            
            # Pure positive: assign True
            if appears_positive and not appears_negative:
                return (unassigned_variable, True)
            # Pure negative: assign False  
            elif appears_negative and not appears_positive:
                return (unassigned_variable, False)
                
        return None
    
    def _choose_symbol(self, current_clauses_list: List[Set[int]], unassigned_variables: Set[int]) -> int:
        """
        Choose the next variable to branch on using the selected heuristic.
        
        This function implements various variable selection strategies that can
        significantly impact DPLL performance. Based on Russell & Norvig Chapter 7.
        
        Args:
            current_clauses_list: Current list of clauses for heuristic analysis
            unassigned_variables: Set of variables not yet assigned
            
        Returns:
            Variable (integer) to branch on next
        """
        if self.heuristic == "first_available":
            # Simple first-available heuristic: pick first unassigned variable
            return next(iter(unassigned_variables))
            
        elif self.heuristic == "most_constrained":
            # Most constrained variable (MRV) heuristic from Russell & Norvig
            # Choose variable that appears in the most clauses
            variable_occurrence_count = {variable: 0 for variable in unassigned_variables}
            
            # Count how many clauses each variable appears in
            for individual_clause in current_clauses_list:
                for literal_in_clause in individual_clause:
                    variable_of_literal = abs(literal_in_clause)
                    if variable_of_literal in variable_occurrence_count:
                        variable_occurrence_count[variable_of_literal] += 1
            
            # Return variable with highest occurrence count
            most_constrained_variable = max(variable_occurrence_count.keys(), 
                                          key=lambda var: variable_occurrence_count[var])
            return most_constrained_variable
            
        elif self.heuristic == "least_constraining":
            # Least constraining value heuristic from Russell & Norvig
            # Choose variable that appears in shortest clauses (more urgently constrained)
            variable_constraint_score = {variable: 0.0 for variable in unassigned_variables}
            
            # Calculate constraint score based on clause sizes
            for individual_clause in current_clauses_list:
                clause_size = len(individual_clause)
                if clause_size > 0:
                    # Smaller clauses give higher weight (more urgent to satisfy)
                    clause_weight = 1.0 / clause_size
                    
                    for literal_in_clause in individual_clause:
                        variable_of_literal = abs(literal_in_clause)
                        if variable_of_literal in variable_constraint_score:
                            variable_constraint_score[variable_of_literal] += clause_weight
            
            # Return variable with highest constraint score
            least_constraining_variable = max(variable_constraint_score.keys(), 
                                            key=lambda var: variable_constraint_score[var])
            return least_constraining_variable
            
        else:
            # Default fallback to first available variable
            return next(iter(unassigned_variables))
    
    def verify_solution(self, formula: CNFFormula, assignment: Dict[int, bool]) -> bool:
        """
        Verify that an assignment satisfies the CNF formula.
        
        Args:
            formula: Original CNF formula
            assignment: Variable assignment to verify
            
        Returns:
            True if assignment satisfies formula, False otherwise
        """
        for clause in formula.clauses:
            clause_satisfied = False
            for literal in clause:
                var = abs(literal)
                is_positive = literal > 0
                
                if var in assignment and assignment[var] == is_positive:
                    clause_satisfied = True
                    break
                    
            if not clause_satisfied:
                return False
                
        return True