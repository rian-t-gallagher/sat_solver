"""
Test suite for experiment harness functionality.
"""

import pytest
import os
import csv
import tempfile
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from run_experiments import ExperimentHarness
from cnf_parser import CNFFormula


class TestExperimentHarness:
    """Test experiment harness implementation."""
    
    def test_experiment_harness_initialization(self):
        """Test that experiment harness initializes correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            harness = ExperimentHarness(output_directory=temp_dir)
            
            assert harness.output_directory == Path(temp_dir)
            assert harness.default_trials_per_formula == 10
            assert harness.random_seed_base == 42
            assert 'dpll' in harness.available_algorithms
            assert 'walksat' in harness.available_algorithms
            assert 'genetic' in harness.available_algorithms
    
    def test_single_formula_experiments(self):
        """Test running experiments on a single formula."""
        # Skip test if benchmark file doesn't exist
        test_formula_path = "benchmarks/CNF Formulas/uf20-0156.cnf"
        if not os.path.exists(test_formula_path):
            pytest.skip(f"Test formula {test_formula_path} not found")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            harness = ExperimentHarness(output_directory=temp_dir)
            
            # Run experiments with reduced trials for testing
            results = harness.run_single_formula_experiments(
                test_formula_path, ['dpll', 'walksat'], trials_per_algorithm=2)
            
            # Should have 1 DPLL trial + 2 WalkSAT trials = 3 total results
            assert len(results) == 3
            
            # Check result structure
            for result in results:
                assert 'algorithm' in result
                assert 'formula_file' in result
                assert 'trial_number' in result
                assert 'result' in result
                assert 'solve_time' in result
                assert 'fitness_score' in result
                
            # Check algorithm distribution
            algorithms = [result['algorithm'] for result in results]
            assert algorithms.count('dpll') == 1  # DPLL runs once
            assert algorithms.count('walksat') == 2  # WalkSAT runs twice
    
    def test_csv_output_format(self):
        """Test that CSV output has correct format and fields."""
        test_formula_path = "benchmarks/CNF Formulas/uf20-0156.cnf"
        if not os.path.exists(test_formula_path):
            pytest.skip(f"Test formula {test_formula_path} not found")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            harness = ExperimentHarness(output_directory=temp_dir)
            
            # Run small experiment
            results = harness.run_single_formula_experiments(
                test_formula_path, ['dpll'], trials_per_algorithm=1)
            
            # Write to CSV
            csv_path = Path(temp_dir) / "test_results.csv"
            with open(csv_path, 'w', newline='') as csv_file:
                csv_writer = csv.DictWriter(csv_file, fieldnames=harness.csv_fieldnames)
                csv_writer.writeheader()
                csv_writer.writerows(results)
            
            # Read back and verify
            with open(csv_path, 'r') as csv_file:
                csv_reader = csv.DictReader(csv_file)
                rows = list(csv_reader)
                
                assert len(rows) == 1  # One DPLL result
                
                row = rows[0]
                assert row['algorithm'] == 'dpll'
                assert row['formula_file'] == 'uf20-0156.cnf'
                assert row['formula_variables'] == '20'
                assert row['formula_clauses'] == '91'
                assert row['result'] in ['SAT', 'UNSAT', 'UNKNOWN']
                assert float(row['solve_time']) > 0
    
    def test_algorithm_specific_statistics(self):
        """Test that algorithm-specific statistics are recorded correctly."""
        test_formula_path = "benchmarks/CNF Formulas/uf20-0156.cnf"
        if not os.path.exists(test_formula_path):
            pytest.skip(f"Test formula {test_formula_path} not found")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            harness = ExperimentHarness(output_directory=temp_dir)
            
            # Test DPLL statistics
            dpll_results = harness.run_single_formula_experiments(
                test_formula_path, ['dpll'], trials_per_algorithm=1)
            
            dpll_result = dpll_results[0]
            assert dpll_result['dpll_decisions'] is not None
            assert dpll_result['dpll_propagations'] is not None
            assert dpll_result['walksat_total_flips'] is None
            assert dpll_result['genetic_generations'] is None
            
            # Test WalkSAT statistics
            walksat_results = harness.run_single_formula_experiments(
                test_formula_path, ['walksat'], trials_per_algorithm=1)
            
            walksat_result = walksat_results[0]
            assert walksat_result['walksat_total_flips'] is not None
            assert walksat_result['walksat_restart_attempts'] is not None
            assert walksat_result['dpll_decisions'] is None
            assert walksat_result['genetic_generations'] is None
    
    def test_random_seed_progression(self):
        """Test that random seeds progress correctly for multiple trials."""
        test_formula_path = "benchmarks/CNF Formulas/uf20-0156.cnf"
        if not os.path.exists(test_formula_path):
            pytest.skip(f"Test formula {test_formula_path} not found")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            harness = ExperimentHarness(output_directory=temp_dir)
            
            # Run multiple WalkSAT trials
            results = harness.run_single_formula_experiments(
                test_formula_path, ['walksat'], trials_per_algorithm=3)
            
            # Check that random seeds are sequential
            seeds = [result['random_seed'] for result in results]
            expected_seeds = [42, 43, 44]  # Based on random_seed_base = 42
            assert seeds == expected_seeds
            
            # Check that trial numbers are sequential
            trials = [result['trial_number'] for result in results]
            expected_trials = [1, 2, 3]
            assert trials == expected_trials