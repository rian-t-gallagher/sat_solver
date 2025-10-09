# test_parser.py

import pytest
import tempfile
import os
from src.cnf_parser import parse_dimacs, parse_raw_cnf, CNFFormula


class TestCNFFormula:
    """Test CNFFormula class functionality."""
    
    def test_cnf_formula_creation(self):
        """Test creating a CNF formula."""
        clauses = [[1, 2, 3], [-1, -2], [2, -3, 1]]
        formula = CNFFormula(3, 3, clauses)
        
        assert formula.num_vars == 3
        assert formula.num_clauses == 3
        assert formula.clauses == clauses


def test_project_structure():
    """Test that required project structure exists."""
    # Check core directories
    assert os.path.exists("src")
    assert os.path.exists("src/cnf_parser")
    assert os.path.exists("src/solvers")
    assert os.path.exists("tests")
    assert os.path.exists("benchmarks")
    assert os.path.exists("results")
    
    # Check core files
    assert os.path.exists("README.md")
    assert os.path.exists("requirements.txt")
    assert os.path.exists("run_solver.py")
