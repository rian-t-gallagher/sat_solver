#!/usr/bin/env python3
"""
Test Suite for SAT Solver Data Analysis System

This file tests our data analysis and plotting code to make sure it works correctly.
Based on testing patterns from our CS 463G textbook.

CS 463G - Heuristic Search Techniques  
Program 3: SAT Solvers Implementation
Author: [Student Name]
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
import os
from pathlib import Path
import json

# Use non-interactive backend for testing (prevents plot windows from opening)
import matplotlib
matplotlib.use('Agg')

from data_analysis import SATDataAnalyzer

class TestSATDataAnalyzer:
    """Test cases for our SAT Data Analyzer class."""
    
    @pytest.fixture
    def temp_results_dir(self):
        """Create temporary results directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_experiment_data(self):
        """Generate sample experiment data for testing."""
        np.random.seed(42)  # For reproducible test data
        
        data = []
        algorithms = ['dpll', 'walksat', 'genetic']
        formulas = ['uf20-0156.cnf', 'uf20-0157.cnf', 'uf50-01.cnf', 'uuf50-01.cnf']
        
        for algorithm in algorithms:
            for formula in formulas:
                for run in range(3):  # 3 runs per formula
                    # Generate realistic test data
                    if algorithm == 'dpll':
                        # DPLL is deterministic - perfect on SAT, fails on UNSAT
                        satisfaction = 100.0 if not formula.startswith('uuf') else 0.0
                        solve_time = np.random.uniform(0.001, 0.1)
                    elif algorithm == 'walksat':
                        # WalkSAT heuristic performance
                        if formula.startswith('uuf'):
                            satisfaction = 0.0  # Cannot solve UNSAT
                        else:
                            satisfaction = np.random.uniform(80, 100)
                        solve_time = np.random.uniform(0.001, 0.5)
                    else:  # genetic
                        # Genetic algorithm performance
                        if formula.startswith('uuf'):
                            satisfaction = 0.0
                        else:
                            satisfaction = np.random.uniform(60, 95)
                        solve_time = np.random.uniform(0.1, 2.0)
                    
                    # Determine number of variables from filename
                    num_vars = 20 if '20' in formula else 50
                    num_clauses = 91 if '20' in formula else 218
                    
                    data.append({
                        'algorithm': algorithm,
                        'formula_file': formula,
                        'run_number': run + 1,
                        'satisfaction_rate': satisfaction,
                        'average_solve_time': solve_time,
                        'average_fitness_score': satisfaction * (num_clauses / 100.0),
                        'num_variables': num_vars,
                        'num_clauses': num_clauses,
                        'random_seed': 1000 + run,
                        'successful_runs': 1 if satisfaction > 0 else 0
                    })
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def sample_parameter_data(self):
        """Generate sample parameter tuning data for testing."""
        np.random.seed(42)
        
        walksat_data = []
        genetic_data = []
        
        # WalkSAT parameter tuning data
        random_walk_probs = [0.1, 0.3, 0.5, 0.7, 0.9]
        max_flips = [500, 1000, 2000]
        max_restarts = [5, 10, 20]
        
        param_set_id = 1
        for rwp in random_walk_probs:
            for flips in max_flips:
                for restarts in max_restarts:
                    # Generate performance data (0.5 is optimal for random walk probability)
                    base_satisfaction = 85.0
                    if rwp == 0.5:
                        satisfaction = base_satisfaction + np.random.uniform(5, 15)
                    else:
                        satisfaction = base_satisfaction + np.random.uniform(-10, 5)
                    
                    solve_time = np.random.uniform(0.001, 0.1)
                    
                    walksat_data.append({
                        'algorithm': 'walksat',
                        'parameter_set_id': param_set_id,
                        'formula_file': 'uf20-0156.cnf',
                        'satisfaction_rate': min(100.0, max(0.0, satisfaction)),
                        'average_solve_time': solve_time,
                        'average_fitness_score': satisfaction * 0.91,
                        'walksat_random_walk_probability': rwp,
                        'walksat_max_flips_per_try': flips,
                        'walksat_max_restart_attempts': restarts
                    })
                    param_set_id += 1
        
        # Genetic Algorithm parameter tuning data
        population_sizes = [50, 100, 200]
        crossover_probs = [0.7, 0.8, 0.9]
        mutation_probs = [0.1, 0.2, 0.3]
        max_generations = [100, 200]
        
        param_set_id = 1
        for pop_size in population_sizes:
            for cross_prob in crossover_probs:
                for mut_prob in mutation_probs:
                    for max_gen in max_generations:
                        # Generate performance data (pop_size=100 is optimal)
                        base_satisfaction = 75.0
                        if pop_size == 100:
                            satisfaction = base_satisfaction + np.random.uniform(5, 20)
                        else:
                            satisfaction = base_satisfaction + np.random.uniform(-5, 10)
                        
                        solve_time = np.random.uniform(0.5, 3.0)
                        
                        genetic_data.append({
                            'algorithm': 'genetic',
                            'parameter_set_id': param_set_id,
                            'formula_file': 'uf20-0156.cnf',
                            'satisfaction_rate': min(100.0, max(0.0, satisfaction)),
                            'average_solve_time': solve_time,
                            'average_fitness_score': satisfaction * 0.91,
                            'genetic_population_size': pop_size,
                            'genetic_crossover_probability': cross_prob,
                            'genetic_mutation_probability': mut_prob,
                            'genetic_max_generations': max_gen
                        })
                        param_set_id += 1
        
        return pd.DataFrame(walksat_data), pd.DataFrame(genetic_data)
    
    @pytest.fixture
    def analyzer_with_data(self, temp_results_dir, sample_experiment_data, sample_parameter_data):
        """Create analyzer with sample data loaded."""
        # Create CSV directory
        csv_dir = Path(temp_results_dir) / "csv"
        csv_dir.mkdir(parents=True, exist_ok=True)
        
        # Save sample experiment data
        experiment_file = csv_dir / "experiment_results_20241009_120000.csv"
        sample_experiment_data.to_csv(experiment_file, index=False)
        
        # Save sample parameter tuning data
        walksat_data, genetic_data = sample_parameter_data
        walksat_file = csv_dir / "walksat_parameter_tuning_20241009_120000.csv"
        genetic_file = csv_dir / "genetic_parameter_tuning_20241009_120000.csv"
        walksat_data.to_csv(walksat_file, index=False)
        genetic_data.to_csv(genetic_file, index=False)
        
        # Create analyzer and load data
        analyzer = SATDataAnalyzer(temp_results_dir)
        analyzer.load_experiment_data()
        analyzer.load_parameter_tuning_data()
        
        return analyzer
    
    def test_analyzer_initialization(self, temp_results_dir):
        """Test that the analyzer starts up correctly."""
        analyzer = SATDataAnalyzer(temp_results_dir)
        
        # Check that all the directories are set up right
        assert analyzer.results_dir == Path(temp_results_dir)
        assert analyzer.csv_dir == Path(temp_results_dir) / "csv"
        assert analyzer.plots_dir == Path(temp_results_dir) / "plots"
        assert analyzer.plots_dir.exists()
        
        # Check that data storage is initialized
        assert analyzer.experiment_data is None
        assert analyzer.parameter_tuning_data == {}
    
    def test_load_experiment_data_success(self, temp_results_dir, sample_experiment_data):
        """Test loading experiment data when files exist."""
        # Set up test data files
        csv_dir = Path(temp_results_dir) / "csv"
        csv_dir.mkdir(parents=True, exist_ok=True)
        
        test_file = csv_dir / "experiment_results_test.csv"
        sample_experiment_data.to_csv(test_file, index=False)
        
        # Test the loading
        analyzer = SATDataAnalyzer(temp_results_dir)
        result = analyzer.load_experiment_data()
        
        # Check that it worked
        assert result is True
        assert analyzer.experiment_data is not None
        assert len(analyzer.experiment_data) == len(sample_experiment_data)
        assert set(analyzer.experiment_data['algorithm'].unique()) == {'dpll', 'walksat', 'genetic'}
    
    def test_load_experiment_data_no_files(self, temp_results_dir):
        """Test loading experiment data when no files exist."""
        analyzer = SATDataAnalyzer(temp_results_dir)
        result = analyzer.load_experiment_data()
        
        # Should return False and data should be None
        assert result is False
        assert analyzer.experiment_data is None
    
    def test_load_parameter_tuning_data_success(self, temp_results_dir, sample_parameter_data):
        """Test successful loading of parameter tuning data."""
        # Setup test data
        csv_dir = Path(temp_results_dir) / "csv"
        csv_dir.mkdir(parents=True, exist_ok=True)
        
        walksat_data, genetic_data = sample_parameter_data
        walksat_file = csv_dir / "walksat_parameter_tuning_test.csv"
        genetic_file = csv_dir / "genetic_parameter_tuning_test.csv"
        walksat_data.to_csv(walksat_file, index=False)
        genetic_data.to_csv(genetic_file, index=False)
        
        # Test loading
        analyzer = SATDataAnalyzer(temp_results_dir)
        result = analyzer.load_parameter_tuning_data()
        
        assert result is True
        assert 'walksat' in analyzer.parameter_tuning_data
        assert 'genetic' in analyzer.parameter_tuning_data
        assert len(analyzer.parameter_tuning_data['walksat']) > 0
        assert len(analyzer.parameter_tuning_data['genetic']) > 0
    
    def test_load_parameter_tuning_data_specific_algorithm(self, temp_results_dir, sample_parameter_data):
        """Test loading parameter tuning data for specific algorithm."""
        # Setup test data
        csv_dir = Path(temp_results_dir) / "csv"
        csv_dir.mkdir(parents=True, exist_ok=True)
        
        walksat_data, _ = sample_parameter_data
        walksat_file = csv_dir / "walksat_parameter_tuning_test.csv"
        walksat_data.to_csv(walksat_file, index=False)
        
        # Test loading specific algorithm
        analyzer = SATDataAnalyzer(temp_results_dir)
        result = analyzer.load_parameter_tuning_data('walksat')
        
        assert result is True
        assert 'walksat' in analyzer.parameter_tuning_data
        assert 'genetic' not in analyzer.parameter_tuning_data
    
    def test_algorithm_comparison_plot_creation(self, analyzer_with_data):
        """Test creation of algorithm comparison plots."""
        plot_files = analyzer_with_data.create_algorithm_comparison_plot(['png'])
        
        assert plot_files is not None
        assert os.path.exists(plot_files)
        assert plot_files.endswith('.png')
        
        # Verify plot file is not empty
        assert os.path.getsize(plot_files) > 1000  # Should be substantial file
    
    def test_algorithm_comparison_plot_no_data(self, temp_results_dir):
        """Test algorithm comparison plot creation with no data."""
        analyzer = SATDataAnalyzer(temp_results_dir)
        
        with pytest.raises(ValueError, match="No experiment data loaded"):
            analyzer.create_algorithm_comparison_plot()
    
    def test_parameter_optimization_plot_creation(self, analyzer_with_data):
        """Test creation of parameter optimization plots."""
        # Test WalkSAT parameter optimization plots
        walksat_plots = analyzer_with_data.create_parameter_optimization_plots('walksat', ['png'])
        
        assert len(walksat_plots) > 0
        for plot_file in walksat_plots:
            assert os.path.exists(plot_file)
            assert plot_file.endswith('.png')
            assert os.path.getsize(plot_file) > 1000
        
        # Test Genetic Algorithm parameter optimization plots
        genetic_plots = analyzer_with_data.create_parameter_optimization_plots('genetic', ['png'])
        
        assert len(genetic_plots) > 0
        for plot_file in genetic_plots:
            assert os.path.exists(plot_file)
            assert plot_file.endswith('.png')
    
    def test_parameter_optimization_plot_invalid_algorithm(self, analyzer_with_data):
        """Test parameter optimization plots with invalid algorithm."""
        with pytest.raises(ValueError, match="No parameter tuning data loaded for algorithm"):
            analyzer_with_data.create_parameter_optimization_plots('invalid_algo')
    
    def test_performance_evolution_plot_creation(self, analyzer_with_data):
        """Test creation of performance evolution plots."""
        plot_files = analyzer_with_data.create_performance_evolution_plot(['png'])
        
        assert len(plot_files) > 0
        for plot_file in plot_files:
            assert os.path.exists(plot_file)
            assert plot_file.endswith('.png')
            assert os.path.getsize(plot_file) > 1000
    
    def test_summary_report_generation_text(self, analyzer_with_data):
        """Test generation of summary report in text format."""
        report_path = analyzer_with_data.generate_summary_report('txt')
        
        assert os.path.exists(report_path)
        assert report_path.endswith('.txt')
        
        # Verify report content
        with open(report_path, 'r') as f:
            content = f.read()
            assert "SAT SOLVER PERFORMANCE ANALYSIS REPORT" in content
            assert "DPLL" in content or "dpll" in content
            assert "WALKSAT" in content or "walksat" in content
    
    def test_summary_report_generation_markdown(self, analyzer_with_data):
        """Test generation of summary report in Markdown format."""
        report_path = analyzer_with_data.generate_summary_report('md')
        
        assert os.path.exists(report_path)
        assert report_path.endswith('.md')
        
        # Verify Markdown formatting
        with open(report_path, 'r') as f:
            content = f.read()
            assert "# SAT Solver Performance Analysis Report" in content
            assert "| Algorithm |" in content  # Table formatting
            assert "**" in content  # Bold formatting
    
    def test_summary_report_generation_json(self, analyzer_with_data):
        """Test generation of summary report in JSON format."""
        report_path = analyzer_with_data.generate_summary_report('json')
        
        assert os.path.exists(report_path)
        assert report_path.endswith('.json')
        
        # Verify JSON structure
        with open(report_path, 'r') as f:
            data = json.load(f)
            assert 'analysis_timestamp' in data
            assert 'experiment_summary' in data
            assert 'parameter_optimization' in data
            assert 'recommendations' in data
    
    def test_algorithm_summary_statistics_calculation(self, analyzer_with_data):
        """Test calculation of algorithm summary statistics."""
        stats = analyzer_with_data._calculate_algorithm_summary_statistics()
        
        assert isinstance(stats, dict)
        assert len(stats) == 3  # dpll, walksat, genetic
        
        for algorithm, algo_stats in stats.items():
            assert 'total_runs' in algo_stats
            assert 'avg_satisfaction' in algo_stats
            assert 'avg_solve_time' in algo_stats
            assert 'success_rate' in algo_stats
            assert algo_stats['total_runs'] > 0
            assert 0 <= algo_stats['avg_satisfaction'] <= 100
            assert algo_stats['avg_solve_time'] > 0
    
    def test_parameter_optimization_summary(self, analyzer_with_data):
        """Test parameter optimization summary generation."""
        summary = analyzer_with_data._get_parameter_optimization_summary()
        
        assert isinstance(summary, dict)
        assert 'walksat' in summary
        assert 'genetic' in summary
        
        # Check WalkSAT summary
        walksat_summary = summary['walksat']
        assert 'best_satisfaction_rate' in walksat_summary
        assert 'best_parameters' in walksat_summary
        assert 'parameter_sets_tested' in walksat_summary
        
        # Verify parameter names
        best_params = walksat_summary['best_parameters']
        assert 'random_walk_probability' in best_params
        assert 'max_flips_per_try' in best_params
        assert 'max_restart_attempts' in best_params
    
    def test_algorithm_recommendations_generation(self, analyzer_with_data):
        """Test generation of algorithm recommendations."""
        recommendations = analyzer_with_data._generate_algorithm_recommendations()
        
        assert isinstance(recommendations, dict)
        assert 'best_overall' in recommendations
        assert 'fastest' in recommendations
        assert 'most_reliable' in recommendations
        assert 'optimal_parameters' in recommendations
        
        # Verify recommendation values are valid algorithms
        valid_algorithms = ['dpll', 'walksat', 'genetic']
        assert recommendations['best_overall'] in valid_algorithms
        assert recommendations['fastest'] in valid_algorithms
        assert recommendations['most_reliable'] in valid_algorithms
        
        # Check optimal parameters structure
        optimal_params = recommendations['optimal_parameters']
        assert isinstance(optimal_params, dict)
        if 'walksat' in optimal_params:
            assert isinstance(optimal_params['walksat'], dict)
    
    def test_multiple_output_formats(self, analyzer_with_data):
        """Test generation of plots in multiple output formats."""
        plot_files = analyzer_with_data.create_algorithm_comparison_plot(['png', 'svg'])
        
        assert plot_files is not None
        assert os.path.exists(plot_files)
        
        # Check that the plot file exists in at least one format
        assert plot_files.endswith('.png') or plot_files.endswith('.svg')
    
    def test_empty_parameter_data_handling(self, temp_results_dir, sample_experiment_data):
        """Test handling of empty parameter tuning data."""
        # Setup with only experiment data
        csv_dir = Path(temp_results_dir) / "csv"
        csv_dir.mkdir(parents=True, exist_ok=True)
        
        experiment_file = csv_dir / "experiment_results_test.csv"
        sample_experiment_data.to_csv(experiment_file, index=False)
        
        analyzer = SATDataAnalyzer(temp_results_dir)
        analyzer.load_experiment_data()
        
        # Should handle empty parameter data gracefully
        summary = analyzer._get_parameter_optimization_summary()
        assert isinstance(summary, dict)
        assert len(summary) == 0
    
    def test_data_analysis_main_function_no_data(self, temp_results_dir, monkeypatch, capsys):
        """Test main function when no data is available."""
        # Change to temporary directory
        monkeypatch.chdir(temp_results_dir)
        
        # Import and run main function
        from data_analysis import main
        main()
        
        # Check output
        captured = capsys.readouterr()
        assert "No data found to analyze" in captured.out
        
    def test_plot_directory_creation(self, temp_results_dir):
        """Test that plot directory is created automatically."""
        plots_dir = Path(temp_results_dir) / "plots"
        assert not plots_dir.exists()
        
        # Creating analyzer should create plots directory
        analyzer = SATDataAnalyzer(temp_results_dir)
        assert plots_dir.exists()
        assert plots_dir.is_dir()

class TestDataAnalysisIntegration:
    """Integration tests for the complete data analysis workflow."""
    
    @pytest.fixture
    def temp_results_dir(self):
        """Create temporary results directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_experiment_data(self):
        """Generate sample experiment data for testing."""
        np.random.seed(42)  # For reproducible test data
        
        data = []
        algorithms = ['dpll', 'walksat', 'genetic']
        formulas = ['uf20-0156.cnf', 'uf20-0157.cnf', 'uf50-01.cnf', 'uuf50-01.cnf']
        
        for algorithm in algorithms:
            for formula in formulas:
                for run in range(3):  # 3 runs per formula
                    # Generate realistic test data
                    if algorithm == 'dpll':
                        # DPLL is deterministic - perfect on SAT, fails on UNSAT
                        satisfaction = 100.0 if not formula.startswith('uuf') else 0.0
                        solve_time = np.random.uniform(0.001, 0.1)
                    elif algorithm == 'walksat':
                        # WalkSAT heuristic performance
                        if formula.startswith('uuf'):
                            satisfaction = 0.0  # Cannot solve UNSAT
                        else:
                            satisfaction = np.random.uniform(80, 100)
                        solve_time = np.random.uniform(0.001, 0.5)
                    else:  # genetic
                        # Genetic algorithm performance
                        if formula.startswith('uuf'):
                            satisfaction = 0.0
                        else:
                            satisfaction = np.random.uniform(60, 95)
                        solve_time = np.random.uniform(0.1, 2.0)
                    
                    # Determine number of variables from filename
                    num_vars = 20 if '20' in formula else 50
                    num_clauses = 91 if '20' in formula else 218
                    
                    data.append({
                        'algorithm': algorithm,
                        'formula_file': formula,
                        'run_number': run + 1,
                        'satisfaction_rate': satisfaction,
                        'average_solve_time': solve_time,
                        'average_fitness_score': satisfaction * (num_clauses / 100.0),
                        'num_variables': num_vars,
                        'num_clauses': num_clauses,
                        'random_seed': 1000 + run,
                        'successful_runs': 1 if satisfaction > 0 else 0
                    })
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def sample_parameter_data(self):
        """Generate sample parameter tuning data for testing."""
        np.random.seed(42)
        
        walksat_data = []
        genetic_data = []
        
        # WalkSAT parameter tuning data
        random_walk_probs = [0.1, 0.3, 0.5, 0.7, 0.9]
        max_flips = [500, 1000, 2000]
        max_restarts = [5, 10, 20]
        
        param_set_id = 1
        for rwp in random_walk_probs:
            for flips in max_flips:
                for restarts in max_restarts:
                    # Generate performance data (0.5 is optimal for random walk probability)
                    base_satisfaction = 85.0
                    if rwp == 0.5:
                        satisfaction = base_satisfaction + np.random.uniform(5, 15)
                    else:
                        satisfaction = base_satisfaction + np.random.uniform(-10, 5)
                    
                    solve_time = np.random.uniform(0.001, 0.1)
                    
                    walksat_data.append({
                        'algorithm': 'walksat',
                        'parameter_set_id': param_set_id,
                        'formula_file': 'uf20-0156.cnf',
                        'satisfaction_rate': min(100.0, max(0.0, satisfaction)),
                        'average_solve_time': solve_time,
                        'average_fitness_score': satisfaction * 0.91,
                        'walksat_random_walk_probability': rwp,
                        'walksat_max_flips_per_try': flips,
                        'walksat_max_restart_attempts': restarts
                    })
                    param_set_id += 1
        
        # Genetic Algorithm parameter tuning data
        population_sizes = [50, 100, 200]
        crossover_probs = [0.7, 0.8, 0.9]
        mutation_probs = [0.1, 0.2, 0.3]
        max_generations = [100, 200]
        
        param_set_id = 1
        for pop_size in population_sizes:
            for cross_prob in crossover_probs:
                for mut_prob in mutation_probs:
                    for max_gen in max_generations:
                        # Generate performance data (pop_size=100 is optimal)
                        base_satisfaction = 75.0
                        if pop_size == 100:
                            satisfaction = base_satisfaction + np.random.uniform(5, 20)
                        else:
                            satisfaction = base_satisfaction + np.random.uniform(-5, 10)
                        
                        solve_time = np.random.uniform(0.5, 3.0)
                        
                        genetic_data.append({
                            'algorithm': 'genetic',
                            'parameter_set_id': param_set_id,
                            'formula_file': 'uf20-0156.cnf',
                            'satisfaction_rate': min(100.0, max(0.0, satisfaction)),
                            'average_solve_time': solve_time,
                            'average_fitness_score': satisfaction * 0.91,
                            'genetic_population_size': pop_size,
                            'genetic_crossover_probability': cross_prob,
                            'genetic_mutation_probability': mut_prob,
                            'genetic_max_generations': max_gen
                        })
                        param_set_id += 1
        
        return pd.DataFrame(walksat_data), pd.DataFrame(genetic_data)
    
    def test_complete_analysis_workflow(self, temp_results_dir, sample_experiment_data, sample_parameter_data):
        """Test complete analysis workflow from data loading to report generation."""
        # Setup all test data
        csv_dir = Path(temp_results_dir) / "csv"
        csv_dir.mkdir(parents=True, exist_ok=True)
        
        # Save experiment data
        experiment_file = csv_dir / "experiment_results_integration_test.csv"
        sample_experiment_data.to_csv(experiment_file, index=False)
        
        # Save parameter tuning data
        walksat_data, genetic_data = sample_parameter_data
        walksat_file = csv_dir / "walksat_parameter_tuning_integration_test.csv"
        genetic_file = csv_dir / "genetic_parameter_tuning_integration_test.csv"
        walksat_data.to_csv(walksat_file, index=False)
        genetic_data.to_csv(genetic_file, index=False)
        
        # Run complete analysis workflow
        analyzer = SATDataAnalyzer(temp_results_dir)
        
        # Load all data
        exp_loaded = analyzer.load_experiment_data()
        param_loaded = analyzer.load_parameter_tuning_data()
        
        assert exp_loaded and param_loaded
        
        # Generate all visualizations
        comparison_plot = analyzer.create_algorithm_comparison_plot(['png'])
        evolution_plots = analyzer.create_performance_evolution_plot(['png'])
        walksat_plots = analyzer.create_parameter_optimization_plots('walksat', ['png'])
        genetic_plots = analyzer.create_parameter_optimization_plots('genetic', ['png'])
        
        # Generate reports in all formats
        txt_report = analyzer.generate_summary_report('txt')
        md_report = analyzer.generate_summary_report('md')
        json_report = analyzer.generate_summary_report('json')
        
        # Verify all outputs
        assert os.path.exists(comparison_plot)
        assert all(os.path.exists(plot) for plot in evolution_plots)
        assert all(os.path.exists(plot) for plot in walksat_plots)
        assert all(os.path.exists(plot) for plot in genetic_plots)
        assert os.path.exists(txt_report)
        assert os.path.exists(md_report)
        assert os.path.exists(json_report)
        
        # Verify plot directory structure
        plots_dir = Path(temp_results_dir) / "plots"
        plot_files = list(plots_dir.glob("*.png"))
        assert len(plot_files) >= 4  # At least comparison, evolution, walksat, genetic plots
        
        print(f"Integration test complete - Generated {len(plot_files)} plots and 3 reports")