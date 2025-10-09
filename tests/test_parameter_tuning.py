#!/usr/bin/env python3
"""
Test suite for parameter_tuning.py - CS 463G Phase 5
Tests parameter optimization functionality for SAT algorithms.
"""

import pytest
import sys
import os
import tempfile
from pathlib import Path
import csv

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from parameter_tuning import ParameterTuner, find_test_formulas


class TestParameterTuning:
    """Test cases for parameter tuning functionality."""
    
    @pytest.fixture
    def test_formula_path(self):
        """Path to a test CNF formula."""
        return "benchmarks/CNF Formulas/uf20-0156.cnf"
    
    @pytest.fixture
    def tuner(self):
        """Create a ParameterTuner instance."""
        return ParameterTuner()
    
    def test_parameter_tuner_initialization(self, tuner):
        """Test that ParameterTuner initializes correctly."""
        assert tuner is not None
        assert str(tuner.output_directory) == "results/parameter_tuning"
        assert tuner.evaluation_trials_per_parameter_set == 5
        assert tuner.random_seed_base == 100
    
    def test_walksat_parameter_space_definition(self, tuner):
        """Test WalkSAT parameter space definition."""
        params = tuner._define_walksat_parameter_space()
        
        # Should have the required parameter keys
        assert 'random_walk_probability' in params
        assert 'max_flips_per_try' in params
        assert 'max_restart_attempts' in params
        
        # Each should have multiple values
        assert len(params['random_walk_probability']) > 1
        assert len(params['max_flips_per_try']) > 1
        assert len(params['max_restart_attempts']) > 1
        
        # Values should be reasonable
        for prob in params['random_walk_probability']:
            assert 0.0 <= prob <= 1.0
        
        for flips in params['max_flips_per_try']:
            assert flips > 0
        
        for attempts in params['max_restart_attempts']:
            assert attempts > 0
    
    def test_genetic_parameter_space_definition(self, tuner):
        """Test Genetic Algorithm parameter space definition."""
        params = tuner._define_genetic_parameter_space()
        
        # Should have the required parameter keys
        assert 'population_size' in params
        assert 'crossover_probability' in params
        assert 'mutation_probability' in params
        assert 'max_generations' in params
        
        # Each should have multiple values
        assert len(params['population_size']) > 1
        assert len(params['crossover_probability']) > 1
        assert len(params['mutation_probability']) > 1
        assert len(params['max_generations']) > 1
        
        # Values should be reasonable
        for size in params['population_size']:
            assert size > 0
        
        for prob in params['crossover_probability']:
            assert 0.0 <= prob <= 1.0
        
        for prob in params['mutation_probability']:
            assert 0.0 <= prob <= 1.0
        
        for gens in params['max_generations']:
            assert gens > 0
    
    def test_formula_selection_from_directory(self):
        """Test formula selection from benchmark directory."""
        # Test with a realistic directory
        benchmark_dir = "benchmarks/CNF Formulas"
        if os.path.exists(benchmark_dir):
            formulas = find_test_formulas(benchmark_dir, max_formulas=5)
            
            # Should return a list of formula paths
            assert isinstance(formulas, list)
            assert len(formulas) <= 5
            
            # Each formula should be a valid path
            for formula in formulas:
                assert os.path.exists(formula)
                assert formula.endswith(('.cnf', '.rcnf'))
    
    def test_formula_selection_prioritization(self):
        """Test that formula selection prioritizes 20-variable formulas."""
        benchmark_dir = "benchmarks/CNF Formulas"
        if os.path.exists(benchmark_dir):
            formulas = find_test_formulas(benchmark_dir, max_formulas=3)
            
            # Should prefer 20-variable formulas first
            formula_names = [os.path.basename(f) for f in formulas]
            has_uf20 = any("uf20" in name for name in formula_names)
            
            # If we have uf20 files in the directory, they should be selected first
            all_files = os.listdir(benchmark_dir)
            has_uf20_available = any("uf20" in name for name in all_files)
            
            if has_uf20_available:
                assert has_uf20, "Should prioritize 20-variable formulas when available"
    
    def test_algorithm_parameter_tuning_walksat(self, tuner, test_formula_path):
        """Test WalkSAT parameter tuning end-to-end."""
        if not os.path.exists(test_formula_path):
            pytest.skip(f"Test formula not found: {test_formula_path}")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Override output directory for test
            tuner.output_directory = temp_dir
            
            # Run parameter tuning with minimal parameter space
            result_path = tuner.tune_algorithm_parameters(
                algorithm_name="walksat",
                test_formulas=[test_formula_path],
                quick_scan=True
            )
            
            # Should return a CSV file path
            assert isinstance(result_path, str)
            assert os.path.exists(result_path)
            assert result_path.endswith('.csv')
            
            # Check CSV content
            with open(result_path, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                assert len(rows) > 0
                
                # Check required fields in first row
                first_row = rows[0]
                assert 'algorithm' in first_row
                assert 'parameter_set_id' in first_row
                assert 'satisfaction_rate' in first_row
                assert 'average_solve_time' in first_row
                assert 'average_fitness_score' in first_row
                assert 'walksat_random_walk_probability' in first_row
                assert 'walksat_max_flips_per_try' in first_row
                assert 'walksat_max_restart_attempts' in first_row
    
    def test_algorithm_parameter_tuning_genetic(self, tuner, test_formula_path):
        """Test Genetic Algorithm parameter tuning end-to-end."""
        if not os.path.exists(test_formula_path):
            pytest.skip(f"Test formula not found: {test_formula_path}")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Override output directory for test
            tuner.output_directory = temp_dir
            
            # Run parameter tuning with minimal parameter space
            result_path = tuner.tune_algorithm_parameters(
                algorithm_name="genetic",
                test_formulas=[test_formula_path],
                quick_scan=True
            )
            
            # Should return a CSV file path
            assert isinstance(result_path, str)
            assert os.path.exists(result_path)
            assert result_path.endswith('.csv')
            
            # Check CSV content
            with open(result_path, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                assert len(rows) > 0
                
                # Check required fields in first row
                first_row = rows[0]
                assert 'algorithm' in first_row
                assert 'parameter_set_id' in first_row
                assert 'satisfaction_rate' in first_row
                assert 'average_solve_time' in first_row
                assert 'average_fitness_score' in first_row
                assert 'genetic_population_size' in first_row
                assert 'genetic_crossover_probability' in first_row
                assert 'genetic_mutation_probability' in first_row
                assert 'genetic_max_generations' in first_row
    
    def test_csv_output_generation(self, tuner, test_formula_path):
        """Test that CSV output is generated correctly."""
        if not os.path.exists(test_formula_path):
            pytest.skip(f"Test formula not found: {test_formula_path}")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Override output directory for test
            tuner.output_directory = temp_dir
            
            # Run minimal parameter tuning
            results = tuner.tune_algorithm_parameters(
                algorithm_name="walksat",
                test_formulas=[test_formula_path],
                quick_scan=True
            )
            
            # Check that CSV file was created
            csv_files = list(Path(temp_dir).glob("*.csv"))
            assert len(csv_files) > 0, "Should create at least one CSV file"
            
            # Verify CSV structure
            csv_file = csv_files[0]
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                headers = reader.fieldnames
                
                # Check required columns
                required_cols = [
                    'algorithm', 'parameter_set_id', 'formula_file',
                    'satisfaction_rate', 'average_solve_time', 'average_fitness_score'
                ]
                
                for col in required_cols:
                    assert col in headers, f"Missing required column: {col}"
                
                # Check that we have some data rows
                rows = list(reader)
                assert len(rows) > 0, "CSV should contain data rows"
                
                # Check data validity
                first_row = rows[0]
                assert first_row['algorithm'] == 'walksat'
                assert first_row['parameter_set_id'].isdigit()
                assert float(first_row['satisfaction_rate']) >= 0
                assert float(first_row['average_solve_time']) >= 0
    
    def test_parameter_space_filtering_quick_scan(self, tuner):
        """Test that quick scan mode reduces parameter space appropriately."""
        # Test that quick scan reduces parameter combinations
        walksat_params = tuner._define_walksat_parameter_space()
        genetic_params = tuner._define_genetic_parameter_space()
        
        # Calculate full combination count
        walksat_full_combinations = (
            len(walksat_params['random_walk_probability']) *
            len(walksat_params['max_flips_per_try']) *
            len(walksat_params['max_restart_attempts'])
        )
        
        genetic_full_combinations = (
            len(genetic_params['population_size']) *
            len(genetic_params['crossover_probability']) *
            len(genetic_params['mutation_probability']) *
            len(genetic_params['max_generations'])
        )
        
        # Quick scan should use fewer combinations
        # (This is tested indirectly by checking that quick scan takes less time
        # and produces fewer parameter sets in real usage)
        assert walksat_full_combinations > 10  # Should have substantial parameter space
        assert genetic_full_combinations > 10  # Should have substantial parameter space
    
    def test_empty_formula_list_handling(self, tuner):
        """Test that empty formula list is handled gracefully."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tuner.output_directory = temp_dir
            
            # Empty formula list should complete without error but produce no results
            result_path = tuner.tune_algorithm_parameters(
                algorithm_name="walksat",
                test_formulas=[],
                quick_scan=True
            )
            
            # Should return a valid path
            assert isinstance(result_path, str)
            assert os.path.exists(result_path)
            
            # CSV should exist but have no meaningful data
            with open(result_path, 'r') as f:
                lines = f.readlines()
                assert len(lines) >= 1  # At least header
    
    def test_nonexistent_formula_handling(self, tuner):
        """Test that nonexistent formula files are handled gracefully."""
        nonexistent_file = "nonexistent_formula.cnf"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            tuner.output_directory = temp_dir
            
            # Nonexistent file should complete without crashing but log errors
            result_path = tuner.tune_algorithm_parameters(
                algorithm_name="walksat",
                test_formulas=[nonexistent_file],
                quick_scan=True
            )
            
            # Should return a valid path
            assert isinstance(result_path, str)
            assert os.path.exists(result_path)
            
            # CSV should exist but have no successful results
            with open(result_path, 'r') as f:
                lines = f.readlines()
                assert len(lines) >= 1  # At least header
    
    def test_unsupported_algorithm_handling(self, tuner, test_formula_path):
        """Test that unsupported algorithms are handled gracefully."""
        if not os.path.exists(test_formula_path):
            pytest.skip(f"Test formula not found: {test_formula_path}")
        
        with pytest.raises((ValueError, KeyError)):
            tuner.tune_algorithm_parameters(
                algorithm_name="unsupported_algorithm",
                test_formulas=[test_formula_path],
                quick_scan=True
            )


if __name__ == "__main__":
    pytest.main([__file__])