"""
Test suite for DPLL SAT solver.
"""

import pytest
import tempfile
import os
from src.cnf_parser import parse_dimacs, CNFFormula
from src.solvers import DPLLSolver, SATResult


class TestDPLLSolver:
    """Test DPLL solver implementation."""
    
    def test_simple_satisfiable_formula(self):
        """Test DPLL on a simple satisfiable formula."""
        # Formula: (x1 OR x2) AND (NOT x1 OR x2) AND (x1 OR NOT x2)
        # Should be satisfiable with x1=True, x2=True
        clauses = [[1, 2], [-1, 2], [1, -2]]
        formula = CNFFormula(2, 3, clauses)
        
        solver = DPLLSolver()
        result, assignment, stats = solver.solve(formula)
        
        assert result == SATResult.SATISFIABLE
        assert assignment is not None
        
        # Verify the assignment satisfies the formula
        assert solver.verify_solution(formula, assignment)
        
        # Check that we have reasonable statistics
        assert stats["decisions"] >= 0
        assert stats["time"] > 0
        
    def test_simple_unsatisfiable_formula(self):
        """Test DPLL on a simple unsatisfiable formula."""
        # Formula: (x1) AND (NOT x1)
        # Should be unsatisfiable
        clauses = [[1], [-1]]
        formula = CNFFormula(1, 2, clauses)
        
        solver = DPLLSolver()
        result, assignment, stats = solver.solve(formula)
        
        assert result == SATResult.UNSATISFIABLE
        assert assignment is None
        assert stats["conflicts"] > 0
        
    def test_unit_propagation(self):
        """Test that unit propagation works correctly."""
        # Formula: (x1) AND (NOT x1 OR x2) AND (NOT x2 OR x3)
        # Should force x1=True, x2=True, x3=True through unit propagation
        clauses = [[1], [-1, 2], [-2, 3]]
        formula = CNFFormula(3, 3, clauses)
        
        solver = DPLLSolver()
        result, assignment, stats = solver.solve(formula)
        
        assert result == SATResult.SATISFIABLE
        assert assignment is not None
        assert assignment[1] == True
        assert assignment[2] == True  
        assert assignment[3] == True
        
        # Should have multiple propagations
        assert stats["propagations"] >= 2
        
    def test_pure_literal_elimination(self):
        """Test pure literal elimination."""
        # Formula: (x1 OR x2) AND (x1 OR x3) AND (x2 OR x3)
        # x1, x2, x3 all appear only positive (pure literals)
        clauses = [[1, 2], [1, 3], [2, 3]]
        formula = CNFFormula(3, 3, clauses)
        
        solver = DPLLSolver()
        result, assignment, stats = solver.solve(formula)
        
        assert result == SATResult.SATISFIABLE
        assert assignment is not None
        assert solver.verify_solution(formula, assignment)
        
    def test_different_heuristics(self):
        """Test different variable selection heuristics."""
        # Test on a satisfiable formula with multiple heuristics
        clauses = [[1, 2, 3], [-1, 2], [-2, 3], [-3, 1]]
        formula = CNFFormula(3, 4, clauses)
        
        heuristics = ["first_available", "most_occurrences", "jeroslow_wang"]
        
        for heuristic in heuristics:
            solver = DPLLSolver(heuristic=heuristic)
            result, assignment, stats = solver.solve(formula)
            
            assert result == SATResult.SATISFIABLE
            assert assignment is not None
            assert solver.verify_solution(formula, assignment)
            
    def test_empty_formula(self):
        """Test DPLL on empty formula (trivially satisfiable)."""
        formula = CNFFormula(2, 0, [])
        
        solver = DPLLSolver()
        result, assignment, stats = solver.solve(formula)
        
        assert result == SATResult.SATISFIABLE
        # Assignment can be anything for empty formula
        
    def test_verify_solution(self):
        """Test solution verification functionality."""
        clauses = [[1, 2], [-1, 3], [-2, -3]]
        formula = CNFFormula(3, 3, clauses)
        
        solver = DPLLSolver()
        
        # Test correct assignment
        correct_assignment = {1: True, 2: False, 3: True}
        assert solver.verify_solution(formula, correct_assignment)
        
        # Test incorrect assignment
        incorrect_assignment = {1: False, 2: False, 3: False}
        assert not solver.verify_solution(formula, incorrect_assignment)


class TestDPLLOnRealFiles:
    """Test DPLL solver on real CNF benchmark files."""
    
    def test_uf20_satisfiable_file(self):
        """Test DPLL on a real satisfiable 20-variable file."""
        filepath = "benchmarks/CNF Formulas/uf20-0156.cnf"
        
        if not os.path.exists(filepath):
            pytest.skip(f"Test file {filepath} not found")
            
        formula = parse_dimacs(filepath)
        solver = DPLLSolver(heuristic="jeroslow_wang")
        
        result, assignment, stats = solver.solve(formula)
        
        assert result == SATResult.SATISFIABLE
        assert assignment is not None
        assert solver.verify_solution(formula, assignment)
        
        # Check that solver made reasonable progress
        assert stats["decisions"] > 0
        assert stats["time"] < 10.0  # Should solve quickly
        
        print(f"Solved uf20-0156: {stats['decisions']} decisions, {stats['time']:.3f}s")
        
    def test_uuf50_unsatisfiable_file(self):
        """Test DPLL on a real unsatisfiable 50-variable file."""
        filepath = "benchmarks/CNF Formulas/uuf50-01.cnf"
        
        if not os.path.exists(filepath):
            pytest.skip(f"Test file {filepath} not found")
            
        formula = parse_dimacs(filepath)
        solver = DPLLSolver(heuristic="most_occurrences")
        
        result, assignment, stats = solver.solve(formula)
        
        assert result == SATResult.UNSATISFIABLE
        assert assignment is None
        
        # Should have conflicts during search
        assert stats["conflicts"] > 0
        assert stats["backtracks"] > 0
        
        print(f"Proved uuf50-01 UNSAT: {stats['conflicts']} conflicts, {stats['time']:.3f}s")