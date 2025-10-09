#!/usr/bin/env python3
"""
SAT Solver Parameter Tuning - CS 463G Program 3 Phase 5

Systematic parameter optimization for heuristic SAT solving algorithms.
This script implements grid search and parameter sweep techniques to find
optimal configurations for WalkSAT and Genetic Algorithm solvers.

Parameter Tuning Methodology:
- Grid search over key algorithm parameters
- Statistical evaluation with multiple random seeds
- Performance vs parameter trade-off analysis
- Automated discovery of optimal parameter combinations

Tuning Targets:
- WalkSAT: random_walk_probability, max_flips_per_try, max_restart_attempts
- Genetic Algorithm: population_size, crossover_probability, mutation_probability, max_generations
- Analysis metrics: satisfaction rate, average solve time, fitness convergence

Output:
- CSV files with parameter performance data
- Optimal parameter recommendations for each algorithm
- Performance plots showing parameter sensitivity

Usage:
    python parameter_tuning.py --algorithm walksat --formula uf20-0156.cnf
    python parameter_tuning.py --algorithm genetic --benchmark-dir "benchmarks/CNF Formulas/"
    python parameter_tuning.py --all-algorithms --quick-scan
"""

import argparse
import csv
import itertools
import os
import sys
import time
import statistics
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from cnf_parser import parse_dimacs, parse_raw_cnf, CNFFormula
from solvers import DPLLSolver, WalkSATSolver, GeneticSATSolver, SATResult


class ParameterTuner:
    """
    Parameter optimization system for SAT solver algorithms.
    
    This class implements systematic parameter sweeps to find optimal
    configurations for heuristic SAT solving algorithms.
    """
    
    def __init__(self, output_directory: str = "results/parameter_tuning"):
        """
        Initialize parameter tuning system.
        
        Args:
            output_directory: Directory to save tuning results and plots
        """
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        
        # Statistical configuration for parameter evaluation
        self.evaluation_trials_per_parameter_set = 5  # Trials per parameter combination
        self.random_seed_base = 100  # Different from experiment harness seeds
        
        # Define parameter spaces for each algorithm
        self.parameter_spaces = {
            'walksat': self._define_walksat_parameter_space(),
            'genetic': self._define_genetic_parameter_space()
        }
        
        # CSV fieldnames for parameter tuning results
        self.tuning_csv_fieldnames = [
            'algorithm', 'parameter_set_id', 'formula_file', 'trial_number', 'random_seed',
            'satisfaction_rate', 'average_solve_time', 'average_fitness_score', 'success_rate',
            # WalkSAT parameters
            'walksat_random_walk_probability', 'walksat_max_flips_per_try', 'walksat_max_restart_attempts',
            # Genetic Algorithm parameters
            'genetic_population_size', 'genetic_crossover_probability', 'genetic_mutation_probability', 'genetic_max_generations',
            # Performance metrics
            'total_trials', 'successful_trials', 'failed_trials', 'timeout_trials'
        ]
    
    def _define_walksat_parameter_space(self) -> Dict[str, List]:
        """
        Define parameter space for WalkSAT algorithm tuning.
        
        Returns:
            Dictionary mapping parameter names to value ranges
        """
        return {
            'random_walk_probability': [0.1, 0.3, 0.5, 0.7, 0.9],
            'max_flips_per_try': [500, 1000, 2000, 5000],
            'max_restart_attempts': [5, 10, 20]
        }
    
    def _define_genetic_parameter_space(self) -> Dict[str, List]:
        """
        Define parameter space for Genetic Algorithm tuning.
        
        Returns:
            Dictionary mapping parameter names to value ranges
        """
        return {
            'population_size': [30, 50, 100, 200],
            'crossover_probability': [0.6, 0.8, 0.9],
            'mutation_probability': [0.05, 0.1, 0.2],
            'max_generations': [200, 500, 1000]
        }
    
    def tune_algorithm_parameters(self, algorithm_name: str, test_formulas: List[str], 
                                 quick_scan: bool = False) -> str:
        """
        Perform comprehensive parameter tuning for specified algorithm.
        
        Args:
            algorithm_name: Name of algorithm to tune ('walksat' or 'genetic')
            test_formulas: List of CNF formula file paths for evaluation
            quick_scan: If True, use reduced parameter space for faster tuning
            
        Returns:
            Path to CSV file containing tuning results
        """
        print(f"\n=== Parameter Tuning: {algorithm_name.upper()} Algorithm ===")
        print(f"Test Formulas: {len(test_formulas)} files")
        print(f"Tuning Mode: {'Quick Scan' if quick_scan else 'Comprehensive Grid Search'}")
        
        if algorithm_name not in self.parameter_spaces:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")
        
        # Get parameter space for this algorithm
        full_parameter_space = self.parameter_spaces[algorithm_name]
        
        # Reduce parameter space for quick scan
        if quick_scan:
            reduced_parameter_space = {}
            for param_name, param_values in full_parameter_space.items():
                # Take every other value for quick scan
                reduced_parameter_space[param_name] = param_values[::2]
            parameter_space = reduced_parameter_space
        else:
            parameter_space = full_parameter_space
        
        # Generate all parameter combinations (Cartesian product)
        parameter_names = list(parameter_space.keys())
        parameter_value_lists = [parameter_space[name] for name in parameter_names]
        parameter_combinations = list(itertools.product(*parameter_value_lists))
        
        print(f"Parameter Space: {len(parameter_combinations)} combinations")
        for param_name, param_values in parameter_space.items():
            print(f"  {param_name}: {param_values}")
        
        all_tuning_results = []
        
        # Evaluate each parameter combination
        for combination_index, parameter_values in enumerate(parameter_combinations):
            parameter_set_id = combination_index + 1
            parameter_dict = dict(zip(parameter_names, parameter_values))
            
            print(f"\nEvaluating Parameter Set {parameter_set_id}/{len(parameter_combinations)}")
            print(f"  Parameters: {parameter_dict}")
            
            # Test this parameter combination on all formulas
            parameter_set_results = self._evaluate_parameter_set(
                algorithm_name, parameter_dict, test_formulas, parameter_set_id)
            
            all_tuning_results.extend(parameter_set_results)
            
            # Show progress summary
            if parameter_set_results:
                avg_satisfaction_rate = statistics.mean([r['satisfaction_rate'] for r in parameter_set_results])
                avg_solve_time = statistics.mean([r['average_solve_time'] for r in parameter_set_results])
                print(f"  Results: {avg_satisfaction_rate:.1f}% satisfaction, {avg_solve_time:.3f}s avg time")
        
        # Save tuning results to CSV
        timestamp_string = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{algorithm_name}_parameter_tuning_{timestamp_string}.csv"
        # Handle both string and Path objects for output directory
        if isinstance(self.output_directory, str):
            output_filepath = os.path.join(self.output_directory, output_filename)
        else:
            output_filepath = self.output_directory / output_filename
        
        print(f"\n=== Saving Parameter Tuning Results ===")
        print(f"Output File: {output_filepath}")
        print(f"Total Parameter Evaluations: {len(all_tuning_results)}")
        
        with open(output_filepath, 'w', newline='') as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=self.tuning_csv_fieldnames)
            csv_writer.writeheader()
            csv_writer.writerows(all_tuning_results)
        
        # Display tuning summary and recommendations
        self._display_tuning_summary(algorithm_name, all_tuning_results)
        
        return str(output_filepath)
    
    def _evaluate_parameter_set(self, algorithm_name: str, parameter_dict: Dict[str, Any], 
                               test_formulas: List[str], parameter_set_id: int) -> List[Dict[str, Any]]:
        """
        Evaluate a single parameter set across multiple formulas and trials.
        
        Args:
            algorithm_name: Name of algorithm being tuned
            parameter_dict: Dictionary of parameter values to test
            test_formulas: List of CNF formula file paths
            parameter_set_id: Unique identifier for this parameter combination
            
        Returns:
            List of evaluation result dictionaries
        """
        parameter_set_results = []
        
        # Test parameter set on each formula
        for formula_filepath in test_formulas:
            formula_basename = os.path.basename(formula_filepath)
            
            # Parse the test formula
            try:
                if formula_filepath.endswith('.rcnf'):
                    test_formula = parse_raw_cnf(formula_filepath)
                else:
                    test_formula = parse_dimacs(formula_filepath)
            except Exception as parsing_error:
                print(f"    ERROR: Failed to parse {formula_basename}: {parsing_error}")
                continue
            
            print(f"    Testing on {formula_basename} ({test_formula.num_vars}v, {test_formula.num_clauses}c)...", end=" ")
            
            # Run multiple trials with this parameter set
            trial_results = []
            for trial_index in range(self.evaluation_trials_per_parameter_set):
                trial_seed = self.random_seed_base + trial_index
                
                # Run single trial with these parameters
                trial_result = self._run_single_parameter_trial(
                    algorithm_name, parameter_dict, test_formula, trial_seed)
                
                trial_results.append(trial_result)
            
            # Aggregate trial results for this formula
            aggregated_result = self._aggregate_trial_results(
                trial_results, algorithm_name, parameter_dict, formula_basename, parameter_set_id)
            
            parameter_set_results.append(aggregated_result)
            
            # Show formula result summary
            satisfaction_rate = aggregated_result['satisfaction_rate']
            avg_time = aggregated_result['average_solve_time']
            print(f"{satisfaction_rate:.0f}% SAT, {avg_time:.3f}s")
        
        return parameter_set_results
    
    def _run_single_parameter_trial(self, algorithm_name: str, parameter_dict: Dict[str, Any], 
                                   test_formula: CNFFormula, random_seed: int) -> Dict[str, Any]:
        """
        Run a single trial with specified parameters.
        
        Args:
            algorithm_name: Name of algorithm to run
            parameter_dict: Parameter values for this trial
            test_formula: CNF formula to solve
            random_seed: Random seed for this trial
            
        Returns:
            Dictionary containing trial results
        """
        start_time = time.time()
        
        try:
            if algorithm_name == 'walksat':
                solver = WalkSATSolver(
                    max_flips_per_try=parameter_dict['max_flips_per_try'],
                    max_restart_attempts=parameter_dict['max_restart_attempts'],
                    random_walk_probability=parameter_dict['random_walk_probability'],
                    random_seed=random_seed
                )
            elif algorithm_name == 'genetic':
                solver = GeneticSATSolver(
                    population_size=parameter_dict['population_size'],
                    max_generations=parameter_dict['max_generations'],
                    crossover_probability=parameter_dict['crossover_probability'],
                    mutation_probability=parameter_dict['mutation_probability'],
                    random_seed=random_seed
                )
            else:
                raise ValueError(f"Unknown algorithm: {algorithm_name}")
            
            # Run the solver
            solve_result, solution_assignment, algorithm_statistics = solver.solve(test_formula)
            solve_time = time.time() - start_time
            
            # Calculate fitness score
            if algorithm_name == 'walksat':
                fitness_score = algorithm_statistics['best_satisfied_clauses']
            elif algorithm_name == 'genetic':
                fitness_score = algorithm_statistics['best_fitness_achieved']
            
            return {
                'result': solve_result,
                'solve_time': solve_time,
                'fitness_score': fitness_score,
                'max_fitness': test_formula.num_clauses,
                'satisfiable': solve_result == SATResult.SATISFIABLE,
                'success': True
            }
            
        except Exception as trial_error:
            print(f"Trial failed: {trial_error}")
            return {
                'result': SATResult.UNKNOWN,
                'solve_time': time.time() - start_time,
                'fitness_score': 0,
                'max_fitness': test_formula.num_clauses,
                'satisfiable': False,
                'success': False
            }
    
    def _aggregate_trial_results(self, trial_results: List[Dict[str, Any]], algorithm_name: str,
                               parameter_dict: Dict[str, Any], formula_filename: str, 
                               parameter_set_id: int) -> Dict[str, Any]:
        """
        Aggregate multiple trial results into summary statistics.
        
        Args:
            trial_results: List of individual trial result dictionaries
            algorithm_name: Name of algorithm being evaluated
            parameter_dict: Parameter values used in trials
            formula_filename: Name of formula file tested
            parameter_set_id: Unique identifier for parameter combination
            
        Returns:
            Dictionary containing aggregated statistics
        """
        successful_trials = [r for r in trial_results if r['success']]
        satisfiable_trials = [r for r in trial_results if r['satisfiable']]
        
        # Calculate aggregate statistics
        total_trials = len(trial_results)
        successful_trial_count = len(successful_trials)
        satisfiable_trial_count = len(satisfiable_trials)
        
        satisfaction_rate = (satisfiable_trial_count / total_trials) * 100 if total_trials > 0 else 0
        success_rate = (successful_trial_count / total_trials) * 100 if total_trials > 0 else 0
        
        # Calculate average metrics from successful trials
        if successful_trials:
            average_solve_time = statistics.mean([r['solve_time'] for r in successful_trials])
            average_fitness_score = statistics.mean([r['fitness_score'] for r in successful_trials])
        else:
            average_solve_time = 0.0
            average_fitness_score = 0.0
        
        # Build aggregated result dictionary
        aggregated_result = {
            'algorithm': algorithm_name,
            'parameter_set_id': parameter_set_id,
            'formula_file': formula_filename,
            'trial_number': 1,  # This represents the aggregated result
            'random_seed': self.random_seed_base,  # Representative seed
            'satisfaction_rate': satisfaction_rate,
            'average_solve_time': average_solve_time,
            'average_fitness_score': average_fitness_score,
            'success_rate': success_rate,
            'total_trials': total_trials,
            'successful_trials': successful_trial_count,
            'failed_trials': total_trials - successful_trial_count,
            'timeout_trials': 0  # Could be enhanced to track timeouts
        }
        
        # Add algorithm-specific parameters
        if algorithm_name == 'walksat':
            aggregated_result.update({
                'walksat_random_walk_probability': parameter_dict['random_walk_probability'],
                'walksat_max_flips_per_try': parameter_dict['max_flips_per_try'],
                'walksat_max_restart_attempts': parameter_dict['max_restart_attempts'],
                'genetic_population_size': None,
                'genetic_crossover_probability': None,
                'genetic_mutation_probability': None,
                'genetic_max_generations': None
            })
        elif algorithm_name == 'genetic':
            aggregated_result.update({
                'walksat_random_walk_probability': None,
                'walksat_max_flips_per_try': None,
                'walksat_max_restart_attempts': None,
                'genetic_population_size': parameter_dict['population_size'],
                'genetic_crossover_probability': parameter_dict['crossover_probability'],
                'genetic_mutation_probability': parameter_dict['mutation_probability'],
                'genetic_max_generations': parameter_dict['max_generations']
            })
        
        return aggregated_result
    
    def _display_tuning_summary(self, algorithm_name: str, tuning_results: List[Dict[str, Any]]):
        """
        Display summary statistics and optimal parameter recommendations.
        
        Args:
            algorithm_name: Name of algorithm that was tuned
            tuning_results: List of all parameter evaluation results
        """
        print(f"\n=== {algorithm_name.upper()} Parameter Tuning Summary ===")
        
        if not tuning_results:
            print("No tuning results to analyze.")
            return
        
        # Group results by parameter set
        parameter_set_results = {}
        for result in tuning_results:
            param_set_id = result['parameter_set_id']
            if param_set_id not in parameter_set_results:
                parameter_set_results[param_set_id] = []
            parameter_set_results[param_set_id].append(result)
        
        # Calculate average performance for each parameter set
        parameter_set_averages = []
        for param_set_id, param_results in parameter_set_results.items():
            avg_satisfaction_rate = statistics.mean([r['satisfaction_rate'] for r in param_results])
            avg_solve_time = statistics.mean([r['average_solve_time'] for r in param_results])
            avg_fitness_score = statistics.mean([r['average_fitness_score'] for r in param_results])
            
            # Extract parameter values (they should be the same for all results in this set)
            representative_result = param_results[0]
            
            parameter_set_averages.append({
                'parameter_set_id': param_set_id,
                'avg_satisfaction_rate': avg_satisfaction_rate,
                'avg_solve_time': avg_solve_time,
                'avg_fitness_score': avg_fitness_score,
                'parameters': representative_result
            })
        
        # Find best parameter sets
        best_by_satisfaction = max(parameter_set_averages, key=lambda x: x['avg_satisfaction_rate'])
        best_by_speed = min(parameter_set_averages, key=lambda x: x['avg_solve_time'])
        best_by_fitness = max(parameter_set_averages, key=lambda x: x['avg_fitness_score'])

        print(f"\nBest Parameter Sets:")
        print(f"\n1. Highest Satisfaction Rate: {best_by_satisfaction['avg_satisfaction_rate']:.1f}%")
        self._print_parameter_set(algorithm_name, best_by_satisfaction['parameters'])
        
        print(f"\n2. Fastest Average Time: {best_by_speed['avg_solve_time']:.3f}s")
        self._print_parameter_set(algorithm_name, best_by_speed['parameters'])
        
        print(f"\n3. Best Average Fitness: {best_by_fitness['avg_fitness_score']:.1f}")
        self._print_parameter_set(algorithm_name, best_by_fitness['parameters'])
        
        # Overall statistics
        all_satisfaction_rates = [avg['avg_satisfaction_rate'] for avg in parameter_set_averages]
        all_solve_times = [avg['avg_solve_time'] for avg in parameter_set_averages]
        
        print(f"\nOverall Statistics:")
        print(f"  Parameter Sets Evaluated: {len(parameter_set_averages)}")
        print(f"  Satisfaction Rate Range: {min(all_satisfaction_rates):.1f}% - {max(all_satisfaction_rates):.1f}%")
        print(f"  Solve Time Range: {min(all_solve_times):.3f}s - {max(all_solve_times):.3f}s")
        print(f"  Average Satisfaction Rate: {statistics.mean(all_satisfaction_rates):.1f}%")
        print(f"  Average Solve Time: {statistics.mean(all_solve_times):.3f}s")
    
    def _print_parameter_set(self, algorithm_name: str, parameter_result: Dict[str, Any]):
        """Print parameter values for a specific parameter set."""
        if algorithm_name == 'walksat':
            print(f"    Random Walk Probability: {parameter_result['walksat_random_walk_probability']}")
            print(f"    Max Flips Per Try: {parameter_result['walksat_max_flips_per_try']}")
            print(f"    Max Restart Attempts: {parameter_result['walksat_max_restart_attempts']}")
        elif algorithm_name == 'genetic':
            print(f"    Population Size: {parameter_result['genetic_population_size']}")
            print(f"    Crossover Probability: {parameter_result['genetic_crossover_probability']}")
            print(f"    Mutation Probability: {parameter_result['genetic_mutation_probability']}")
            print(f"    Max Generations: {parameter_result['genetic_max_generations']}")


def find_test_formulas(benchmark_directory: str, max_formulas: int = 5) -> List[str]:
    """
    Find a representative set of CNF formulas for parameter tuning.
    
    Args:
        benchmark_directory: Directory containing CNF files
        max_formulas: Maximum number of formulas to use for tuning
        
    Returns:
        List of CNF file paths suitable for parameter evaluation
    """
    benchmark_path = Path(benchmark_directory)
    
    if not benchmark_path.exists():
        raise FileNotFoundError(f"Benchmark directory not found: {benchmark_directory}")
    
    # Find all CNF files
    all_cnf_files = list(benchmark_path.glob("*.cnf")) + list(benchmark_path.glob("*.rcnf"))
    
    if not all_cnf_files:
        raise ValueError(f"No CNF files found in {benchmark_directory}")
    
    # Sort for consistent selection
    all_cnf_files.sort()
    
    # Select a diverse subset for tuning (prefer smaller formulas for speed)
    # Priority: 20-variable formulas first, then 50-variable, then larger
    prioritized_formulas = []
    
    # First, add 20-variable satisfiable formulas
    uf20_files = [f for f in all_cnf_files if "uf20" in f.name and "uuf20" not in f.name]
    prioritized_formulas.extend(uf20_files[:2])
    
    # Add 20-variable unsatisfiable formulas
    uuf20_files = [f for f in all_cnf_files if "uuf20" in f.name]
    prioritized_formulas.extend(uuf20_files[:1])
    
    # Add some 50-variable formulas if we have room
    if len(prioritized_formulas) < max_formulas:
        uf50_files = [f for f in all_cnf_files if "uf50" in f.name and "uuf50" not in f.name]
        remaining_slots = max_formulas - len(prioritized_formulas)
        prioritized_formulas.extend(uf50_files[:remaining_slots])
    
    # Convert to string paths
    selected_formulas = [str(f) for f in prioritized_formulas[:max_formulas]]
    
    return selected_formulas


def main():
    """Main function for parameter tuning CLI."""
    parser = argparse.ArgumentParser(
        description="SAT Solver Parameter Tuning - CS 463G Phase 5",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
python parameter_tuning.py --algorithm walksat --benchmark-dir "benchmarks/CNF Formulas/" --quick-scan
python parameter_tuning.py --algorithm genetic --single-formula "benchmarks/CNF Formulas/uf20-0156.cnf"
python parameter_tuning.py --all-algorithms --benchmark-dir "benchmarks/CNF Formulas/"
        """
    )
    
    # Algorithm selection
    parser.add_argument(
        '--algorithm',
        choices=['walksat', 'genetic'],
        help='Algorithm to tune parameters for'
    )
    
    parser.add_argument(
        '--all-algorithms',
        action='store_true',
        help='Tune parameters for all available algorithms'
    )
    
    # Formula selection
    parser.add_argument(
        '--benchmark-dir',
        default='benchmarks/CNF Formulas',
        help='Directory containing CNF benchmark files (default: benchmarks/CNF Formulas)'
    )
    
    parser.add_argument(
        '--single-formula',
        help='Single CNF file for focused parameter tuning'
    )
    
    parser.add_argument(
        '--max-formulas',
        type=int,
        default=5,
        help='Maximum number of formulas to use for tuning (default: 5)'
    )
    
    # Tuning options
    parser.add_argument(
        '--quick-scan',
        action='store_true',
        help='Use reduced parameter space for faster tuning'
    )
    
    parser.add_argument(
        '--output-dir',
        default='results/parameter_tuning',
        help='Output directory for tuning results (default: results/parameter_tuning)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.algorithm and not args.all_algorithms:
        print("ERROR: Must specify either --algorithm or --all-algorithms")
        sys.exit(1)
    
    if args.algorithm and args.all_algorithms:
        print("ERROR: Cannot specify both --algorithm and --all-algorithms")
        sys.exit(1)
    
    # Determine algorithms to tune
    if args.all_algorithms:
        algorithms_to_tune = ['walksat', 'genetic']
    else:
        algorithms_to_tune = [args.algorithm]
    
    # Determine test formulas
    if args.single_formula:
        if not os.path.exists(args.single_formula):
            print(f"ERROR: Formula file not found: {args.single_formula}")
            sys.exit(1)
        test_formulas = [args.single_formula]
    else:
        try:
            test_formulas = find_test_formulas(args.benchmark_dir, args.max_formulas)
        except (FileNotFoundError, ValueError) as e:
            print(f"ERROR: {e}")
            sys.exit(1)
    
    print(f"=== SAT Solver Parameter Tuning ===")
    print(f"Algorithms: {', '.join(algorithms_to_tune)}")
    print(f"Test Formulas: {len(test_formulas)} files")
    print(f"Tuning Mode: {'Quick Scan' if args.quick_scan else 'Comprehensive'}")
    print(f"Output Directory: {args.output_dir}")
    
    # Initialize parameter tuner
    tuner = ParameterTuner(output_directory=args.output_dir)
    
    # Tune each algorithm
    tuning_results = {}
    for algorithm_name in algorithms_to_tune:
        try:
            result_filepath = tuner.tune_algorithm_parameters(
                algorithm_name, test_formulas, args.quick_scan)
            tuning_results[algorithm_name] = result_filepath
            print(f"\n{algorithm_name.upper()} tuning completed!")
            print(f"Results saved to: {result_filepath}")
        except Exception as tuning_error:
            print(f"\n{algorithm_name.upper()} tuning failed: {tuning_error}")
    
    print(f"\nParameter tuning completed for {len(tuning_results)} algorithms!")
    for algorithm_name, result_file in tuning_results.items():
        print(f"  {algorithm_name.upper()}: {result_file}")


if __name__ == "__main__":
    main()