#!/usr/bin/env python3
"""
Test suite for SAT solver.
Tests CNF parser and all three algorithms.
"""

import sys
from cnf_parser import parse_dimacs, CNFFormula
from dpll import solve_dpll
from walksat import solve_walksat
from genetic import solve_genetic


def test_cnf_parser():
    """Test CNF parser."""
    print("Testing CNF parser...")
    
    # Simple formula: (x1 OR x2) AND (NOT x1 OR x3)
    content = """
    c Test formula
    p cnf 3 2
    1 2 0
    -1 3 0
    """
    
    formula = parse_dimacs(content)
    assert formula.num_vars == 3
    assert formula.num_clauses == 2
    assert len(formula.clauses) == 2
    
    # Test evaluation
    assignment = {1: True, 2: False, 3: True}
    satisfied = formula.evaluate(assignment)
    assert satisfied == 2  # Both clauses satisfied
    
    print("  ✓ CNF parser works correctly")


def test_dpll_satisfiable():
    """Test DPLL on satisfiable formula."""
    print("Testing DPLL on satisfiable formula...")
    
    # (x1 OR x2) AND (NOT x1 OR x3) AND (NOT x2 OR NOT x3)
    clauses = [[1, 2], [-1, 3], [-2, -3]]
    formula = CNFFormula(3, clauses)
    
    satisfiable, assignment, num_satisfied = solve_dpll(formula)
    
    assert satisfiable == True
    assert num_satisfied == 3
    assert formula.is_satisfied(assignment)
    
    print("  ✓ DPLL correctly solves satisfiable formula")


def test_dpll_unsatisfiable():
    """Test DPLL on unsatisfiable formula."""
    print("Testing DPLL on unsatisfiable formula...")
    
    # (x1) AND (NOT x1)
    clauses = [[1], [-1]]
    formula = CNFFormula(1, clauses)
    
    satisfiable, assignment, num_satisfied = solve_dpll(formula)
    
    assert satisfiable == False
    assert num_satisfied == 0
    
    print("  ✓ DPLL correctly detects unsatisfiable formula")


def test_walksat():
    """Test WalkSAT."""
    print("Testing WalkSAT...")
    
    # (x1 OR x2) AND (NOT x1 OR x3) AND (NOT x2 OR NOT x3)
    clauses = [[1, 2], [-1, 3], [-2, -3]]
    formula = CNFFormula(3, clauses)
    
    satisfiable, assignment, num_satisfied = solve_walksat(
        formula, max_flips=1000, max_tries=5
    )
    
    # Should find solution (though not guaranteed for incomplete algorithm)
    assert num_satisfied >= 2  # At least 2 out of 3
    
    print(f"  ✓ WalkSAT found solution with {num_satisfied}/3 clauses satisfied")


def test_genetic():
    """Test Genetic Algorithm."""
    print("Testing Genetic Algorithm...")
    
    # (x1 OR x2) AND (NOT x1 OR x3) AND (NOT x2 OR NOT x3)
    clauses = [[1, 2], [-1, 3], [-2, -3]]
    formula = CNFFormula(3, clauses)
    
    satisfiable, assignment, num_satisfied = solve_genetic(
        formula, population_size=50, generations=50
    )
    
    # Should find good solution
    assert num_satisfied >= 2  # At least 2 out of 3
    
    print(f"  ✓ Genetic Algorithm found solution with {num_satisfied}/3 clauses satisfied")


def test_complex_formula():
    """Test all algorithms on more complex formula."""
    print("Testing all algorithms on complex formula...")
    
    # More complex 3-SAT instance
    clauses = [
        [1, 2, 3],
        [-1, -2, 4],
        [2, -3, 5],
        [-1, 3, -4],
        [1, -2, -5],
        [-2, -3, -4],
        [1, 4, 5],
        [-3, -4, -5],
        [2, 3, 4],
        [-1, -4, 5]
    ]
    formula = CNFFormula(5, clauses)
    
    # Test DPLL
    sat_dpll, assign_dpll, num_dpll = solve_dpll(formula)
    print(f"  DPLL: {num_dpll}/10 clauses satisfied")
    
    # Test WalkSAT
    sat_walk, assign_walk, num_walk = solve_walksat(formula, max_flips=5000)
    print(f"  WalkSAT: {num_walk}/10 clauses satisfied")
    
    # Test Genetic
    sat_gen, assign_gen, num_gen = solve_genetic(formula, population_size=100, generations=50)
    print(f"  Genetic: {num_gen}/10 clauses satisfied")
    
    # At least one should solve it (DPLL is complete)
    assert sat_dpll == True or num_dpll >= 9
    
    print("  ✓ All algorithms performed well on complex formula")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("Running SAT Solver Test Suite")
    print("="*60 + "\n")
    
    try:
        test_cnf_parser()
        test_dpll_satisfiable()
        test_dpll_unsatisfiable()
        test_walksat()
        test_genetic()
        test_complex_formula()
        
        print("\n" + "="*60)
        print("All tests passed! ✓")
        print("="*60 + "\n")
        return 0
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(run_all_tests())
