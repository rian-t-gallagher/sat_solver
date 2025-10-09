#!/usr/bin/env python3
"""
SAT Solver Experiment Harness - CS 463G Program 3 Phase 4

Automated experiment runner for systematic performance analysis of SAT solving algorithms.
This script implements the experimental methodology required for Program 3:

- Run 10 trials per formula for randomized algorithms (WalkSAT, Genetic Algorithm)
- Single deterministic run for complete algorithms (DPLL)
- Log comprehensive results to CSV files for statistical analysis
- Support for batch processing multiple CNF benchmark files

Experimental Design:
- Each randomized algorithm uses different random seeds (42-51) for 10 trials
- Complete algorithms (DPLL) run once per formula (deterministic)
- Results include: algorithm, formula, trial, result, time, fitness, algorithm-specific metrics
- Output saved to results/csv/ directory with timestamp for reproducibility

Usage:
    python run_experiments.py --benchmark "benchmarks/CNF Formulas/"
    python run_experiments.py --single uf20-0156.cnf --output results.csv
    python run_experiments.py --algorithms dpll,walksat --trials 5
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from cnf_parser import parse_dimacs, parse_raw_cnf, CNFFormula
from solvers import DPLLSolver, WalkSATSolver, GeneticSATSolver, SATResult


class ExperimentHarness:
    """
    Experiment harness for systematic SAT solver performance evaluation.
    
    This class manages the execution of experiments across multiple algorithms,
    formulas, and trials, collecting comprehensive performance data for analysis.
    """
    
    def __init__(self, output_directory: str = "results/csv"):
        """
        Initialize experiment harness.
        
        Args:
            output_directory: Directory to save CSV results files
        """
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        
        # Experiment configuration
        self.default_trials_per_formula = 10
        self.random_seed_base = 42
        
        # Available algorithms for experimentation
        self.available_algorithms = {
            'dpll': self._run_dpll_experiment,
            'walksat': self._run_walksat_experiment,
            'genetic': self._run_genetic_experiment
        }
        
        # CSV fieldnames for comprehensive result logging
        self.csv_fieldnames = [
            'timestamp', 'algorithm', 'formula_file', 'formula_variables', 'formula_clauses',
            'trial_number', 'random_seed', 'result', 'satisfiable', 'solve_time',
            'fitness_score', 'fitness_percentage', 
            # DPLL-specific metrics
            'dpll_decisions', 'dpll_propagations', 'dpll_conflicts', 'dpll_backtracks',
            # WalkSAT-specific metrics  
            'walksat_total_flips', 'walksat_restart_attempts', 'walksat_random_flips', 'walksat_greedy_flips',
            # Genetic Algorithm-specific metrics
            'genetic_generations', 'genetic_fitness_evaluations', 'genetic_crossover_ops', 'genetic_mutation_ops'
        ]
    
    def run_single_formula_experiments(self, formula_filepath: str, algorithms: List[str], 
                                     trials_per_algorithm: int = 10) -> List[Dict[str, Any]]:
        """
        Run experiments on a single CNF formula with specified algorithms.
        
        Args:
            formula_filepath: Path to CNF formula file
            algorithms: List of algorithm names to test
            trials_per_algorithm: Number of trials for randomized algorithms
            
        Returns:
            List of experiment result dictionaries
        """
        
        print(f"\n=== Experiment: {formula_filepath} ===")        # Parse the CNF formula
        try:
            if formula_filepath.endswith('.rcnf'):
                parsed_formula = parse_raw_cnf(formula_filepath)
                print(f"Parsed raw CNF file: {formula_filepath}")
            else:
                parsed_formula = parse_dimacs(formula_filepath)
                print(f"Parsed DIMACS CNF file: {formula_filepath}")
                
            print(f"Formula: {parsed_formula.num_vars} variables, {parsed_formula.num_clauses} clauses")
            
        except Exception as parsing_error:
            print(f"ERROR: Failed to parse {formula_filepath}: {parsing_error}")
            return []
        
        all_experiment_results = []
        
        # Run experiments for each specified algorithm
        for algorithm_name in algorithms:
            if algorithm_name not in self.available_algorithms:
                print(f"WARNING: Unknown algorithm '{algorithm_name}', skipping")
                continue
                
            print(f"\nRunning {algorithm_name.upper()} algorithm...")
            
            # Determine number of trials (complete algorithms run once, heuristics run multiple times)
            if algorithm_name == 'dpll':
                num_trials = 1  # DPLL is deterministic
                print(f"  DPLL (complete algorithm): 1 deterministic run")
            else:
                num_trials = trials_per_algorithm
                print(f"  {algorithm_name.upper()} (heuristic algorithm): {num_trials} randomized trials")
            
            # Run the specified number of trials
            algorithm_experiment_function = self.available_algorithms[algorithm_name]
            for trial_index in range(num_trials):
                trial_number = trial_index + 1
                current_random_seed = self.random_seed_base + trial_index
                
                print(f"    Trial {trial_number}/{num_trials} (seed: {current_random_seed})...", end=" ")
                
                # Execute single trial experiment
                trial_result = algorithm_experiment_function(
                    parsed_formula, formula_filepath, trial_number, current_random_seed)
                
                all_experiment_results.append(trial_result)
                
                # Display trial result summary
                result_status = trial_result['result']
                trial_time = trial_result['solve_time']
                fitness_score = trial_result['fitness_score']
                total_clauses = trial_result['formula_clauses']
                print(f"{result_status} ({fitness_score}/{total_clauses} clauses, {trial_time:.4f}s)")
        
        return all_experiment_results
    
    def _run_dpll_experiment(self, cnf_formula: CNFFormula, formula_filepath: str, 
                           trial_number: int, random_seed: int) -> Dict[str, Any]:
        """Run a single DPLL experiment trial."""
        dpll_solver = DPLLSolver(heuristic="first_available")
        
        trial_start_time = time.time()
        solve_result, solution_assignment, algorithm_statistics = dpll_solver.solve(cnf_formula)
        trial_end_time = time.time()
        
        # Calculate fitness (satisfied clauses)
        if solve_result == SATResult.SATISFIABLE and solution_assignment:
            fitness_score = cnf_formula.num_clauses  # DPLL finds complete solutions
        else:
            fitness_score = 0  # UNSAT has 0 fitness
            
        return {
            'timestamp': datetime.now().isoformat(),
            'algorithm': 'dpll',
            'formula_file': os.path.basename(formula_filepath),
            'formula_variables': cnf_formula.num_vars,
            'formula_clauses': cnf_formula.num_clauses,
            'trial_number': trial_number,
            'random_seed': random_seed,
            'result': solve_result.value,
            'satisfiable': solve_result == SATResult.SATISFIABLE,
            'solve_time': trial_end_time - trial_start_time,
            'fitness_score': fitness_score,
            'fitness_percentage': (fitness_score / cnf_formula.num_clauses) * 100,
            # DPLL-specific statistics
            'dpll_decisions': algorithm_statistics['decisions'],
            'dpll_propagations': algorithm_statistics['propagations'],
            'dpll_conflicts': algorithm_statistics['conflicts'],
            'dpll_backtracks': algorithm_statistics['backtracks'],
            # Other algorithm fields (empty for DPLL)
            'walksat_total_flips': None,
            'walksat_restart_attempts': None,
            'walksat_random_flips': None,
            'walksat_greedy_flips': None,
            'genetic_generations': None,
            'genetic_fitness_evaluations': None,
            'genetic_crossover_ops': None,
            'genetic_mutation_ops': None
        }
    
    def _run_walksat_experiment(self, cnf_formula: CNFFormula, formula_filepath: str,
                              trial_number: int, random_seed: int) -> Dict[str, Any]:
        """Run a single WalkSAT experiment trial."""
        walksat_solver = WalkSATSolver(
            max_flips_per_try=1000,
            max_restart_attempts=10,
            random_walk_probability=0.5,
            random_seed=random_seed
        )
        
        trial_start_time = time.time()
        solve_result, solution_assignment, algorithm_statistics = walksat_solver.solve(cnf_formula)
        trial_end_time = time.time()
        
        # WalkSAT fitness is best satisfied clauses found
        fitness_score = algorithm_statistics['best_satisfied_clauses']
        
        return {
            'timestamp': datetime.now().isoformat(),
            'algorithm': 'walksat',
            'formula_file': os.path.basename(formula_filepath),
            'formula_variables': cnf_formula.num_vars,
            'formula_clauses': cnf_formula.num_clauses,
            'trial_number': trial_number,
            'random_seed': random_seed,
            'result': solve_result.value,
            'satisfiable': solve_result == SATResult.SATISFIABLE,
            'solve_time': trial_end_time - trial_start_time,
            'fitness_score': fitness_score,
            'fitness_percentage': (fitness_score / cnf_formula.num_clauses) * 100,
            # WalkSAT-specific statistics
            'walksat_total_flips': algorithm_statistics['total_flips'],
            'walksat_restart_attempts': algorithm_statistics['restart_attempts'],
            'walksat_random_flips': algorithm_statistics['random_flips'],
            'walksat_greedy_flips': algorithm_statistics['greedy_flips'],
            # Other algorithm fields (empty for WalkSAT)
            'dpll_decisions': None,
            'dpll_propagations': None,
            'dpll_conflicts': None,
            'dpll_backtracks': None,
            'genetic_generations': None,
            'genetic_fitness_evaluations': None,
            'genetic_crossover_ops': None,
            'genetic_mutation_ops': None
        }
    
    def _run_genetic_experiment(self, cnf_formula: CNFFormula, formula_filepath: str,
                              trial_number: int, random_seed: int) -> Dict[str, Any]:
        """Run a single Genetic Algorithm experiment trial."""
        genetic_solver = GeneticSATSolver(
            population_size=100,
            max_generations=500,
            crossover_probability=0.8,
            mutation_probability=0.1,
            random_seed=random_seed
        )
        
        trial_start_time = time.time()
        solve_result, solution_assignment, algorithm_statistics = genetic_solver.solve(cnf_formula)
        trial_end_time = time.time()
        
        # Genetic Algorithm fitness is best fitness achieved
        fitness_score = algorithm_statistics['best_fitness_achieved']
        
        return {
            'timestamp': datetime.now().isoformat(),
            'algorithm': 'genetic',
            'formula_file': os.path.basename(formula_filepath),
            'formula_variables': cnf_formula.num_vars,
            'formula_clauses': cnf_formula.num_clauses,
            'trial_number': trial_number,
            'random_seed': random_seed,
            'result': solve_result.value,
            'satisfiable': solve_result == SATResult.SATISFIABLE,
            'solve_time': trial_end_time - trial_start_time,
            'fitness_score': fitness_score,
            'fitness_percentage': (fitness_score / cnf_formula.num_clauses) * 100,
            # Genetic Algorithm-specific statistics
            'genetic_generations': algorithm_statistics['generations_evolved'],
            'genetic_fitness_evaluations': algorithm_statistics['fitness_evaluations'],
            'genetic_crossover_ops': algorithm_statistics['crossover_operations'],
            'genetic_mutation_ops': algorithm_statistics['mutation_operations'],
            # Other algorithm fields (empty for Genetic)
            'dpll_decisions': None,
            'dpll_propagations': None,
            'dpll_conflicts': None,
            'dpll_backtracks': None,
            'walksat_total_flips': None,
            'walksat_restart_attempts': None,
            'walksat_random_flips': None,
            'walksat_greedy_flips': None
        }
    
    def run_benchmark_experiments(self, benchmark_directory: str, algorithms: List[str],
                                trials_per_algorithm: int = 10, output_filename: Optional[str] = None) -> str:
        """
        Run experiments on all CNF files in a benchmark directory.
        
        Args:
            benchmark_directory: Directory containing CNF files
            algorithms: List of algorithm names to test
            trials_per_algorithm: Number of trials for randomized algorithms
            output_filename: Optional custom output filename
            
        Returns:
            Path to the generated CSV results file
        """
        benchmark_path = Path(benchmark_directory)
        
        if not benchmark_path.exists():
            raise FileNotFoundError(f"Benchmark directory not found: {benchmark_directory}")
        
        # Find all CNF files in the benchmark directory
        cnf_files = list(benchmark_path.glob("*.cnf")) + list(benchmark_path.glob("*.rcnf"))
        cnf_files.sort()  # Process in alphabetical order for consistency
        
        if not cnf_files:
            raise ValueError(f"No CNF files found in {benchmark_directory}")
        
        print(f"=== SAT Solver Benchmark Experiments ===")
        print(f"Benchmark Directory: {benchmark_directory}")
        print(f"CNF Files Found: {len(cnf_files)}")
        print(f"Algorithms: {', '.join(algorithms)}")
        print(f"Trials per Heuristic Algorithm: {trials_per_algorithm}")
        
        # Generate output filename with timestamp if not provided
        if output_filename is None:
            timestamp_string = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"sat_experiments_{timestamp_string}.csv"
        
        output_filepath = self.output_directory / output_filename
        
        # Collect all experiment results
        all_experiment_results = []
        
        # Run experiments on each CNF file
        for cnf_file_path in cnf_files:
            formula_results = self.run_single_formula_experiments(
                str(cnf_file_path), algorithms, trials_per_algorithm)
            all_experiment_results.extend(formula_results)
        
        # Write comprehensive results to CSV file
        print(f"\n=== Writing Results to CSV ===")
        print(f"Output File: {output_filepath}")
        print(f"Total Experiment Records: {len(all_experiment_results)}")
        
        with open(output_filepath, 'w', newline='') as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=self.csv_fieldnames)
            csv_writer.writeheader()
            csv_writer.writerows(all_experiment_results)
        
        print(f"Results successfully saved to: {output_filepath}")
        
        # Display experiment summary statistics
        self._display_experiment_summary(all_experiment_results)
        
        return str(output_filepath)
    
    def _display_experiment_summary(self, experiment_results: List[Dict[str, Any]]):
        """Display summary statistics from experiment results."""
        print(f"\n=== Experiment Summary ===")
        
        # Group results by algorithm
        algorithm_results = {}
        for result in experiment_results:
            algorithm_name = result['algorithm']
            if algorithm_name not in algorithm_results:
                algorithm_results[algorithm_name] = []
            algorithm_results[algorithm_name].append(result)
        
        # Display summary for each algorithm
        for algorithm_name, algorithm_trials in algorithm_results.items():
            print(f"\n{algorithm_name.upper()} Algorithm Summary:")
            print(f"  Total Trials: {len(algorithm_trials)}")
            
            # Calculate satisfaction rate
            satisfiable_trials = [trial for trial in algorithm_trials if trial['satisfiable']]
            satisfaction_rate = (len(satisfiable_trials) / len(algorithm_trials)) * 100
            print(f"  Satisfaction Rate: {satisfaction_rate:.1f}% ({len(satisfiable_trials)}/{len(algorithm_trials)})")
            
            # Calculate average solve time
            solve_times = [trial['solve_time'] for trial in algorithm_trials]
            average_solve_time = sum(solve_times) / len(solve_times)
            print(f"  Average Solve Time: {average_solve_time:.4f} seconds")
            
            # Calculate average fitness
            fitness_scores = [trial['fitness_score'] for trial in algorithm_trials]
            average_fitness = sum(fitness_scores) / len(fitness_scores)
            print(f"  Average Fitness Score: {average_fitness:.2f}")
            
            # Algorithm-specific statistics
            if algorithm_name == 'dpll':
                avg_decisions = sum(trial['dpll_decisions'] for trial in algorithm_trials) / len(algorithm_trials)
                print(f"  Average Decisions: {avg_decisions:.1f}")
            elif algorithm_name == 'walksat':
                avg_flips = sum(trial['walksat_total_flips'] for trial in algorithm_trials) / len(algorithm_trials)
                print(f"  Average Total Flips: {avg_flips:.1f}")
            elif algorithm_name == 'genetic':
                avg_generations = sum(trial['genetic_generations'] for trial in algorithm_trials) / len(algorithm_trials)
                print(f"  Average Generations: {avg_generations:.1f}")


def main():
    """Main function for experiment harness CLI."""
    parser = argparse.ArgumentParser(
        description="SAT Solver Experiment Harness - CS 463G Phase 4",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
python run_experiments.py --benchmark "benchmarks/CNF Formulas/" --algorithms dpll,walksat,genetic
python run_experiments.py --single "benchmarks/CNF Formulas/uf20-0156.cnf" --algorithms walksat --trials 5
python run_experiments.py --benchmark "benchmarks/CNF Formulas/" --algorithms dpll --output dpll_results.csv
        """
    )
    
    # Experiment target options
    parser.add_argument(
        '--benchmark',
        help='Directory containing CNF files for batch experiments'
    )
    
    parser.add_argument(
        '--single',
        help='Single CNF file for focused experiments'
    )
    
    # Algorithm selection
    parser.add_argument(
        '--algorithms',
        default='dpll,walksat,genetic',
        help='Comma-separated list of algorithms to test (default: dpll,walksat,genetic)'
    )
    
    # Experiment parameters
    parser.add_argument(
        '--trials',
        type=int,
        default=10,
        help='Number of trials for randomized algorithms (default: 10)'
    )
    
    parser.add_argument(
        '--output',
        help='Output CSV filename (default: timestamped filename)'
    )
    
    parser.add_argument(
        '--output-dir',
        default='results/csv',
        help='Output directory for results (default: results/csv)'
    )
    
    args = parser.parse_args()
    
    # Validate that either benchmark or single is specified
    if not args.benchmark and not args.single:
        print("ERROR: Must specify either --benchmark or --single")
        sys.exit(1)
    
    if args.benchmark and args.single:
        print("ERROR: Cannot specify both --benchmark and --single")
        sys.exit(1)
    
    # Parse algorithm list
    algorithm_list = [algorithm.strip() for algorithm in args.algorithms.split(',')]
    
    # Validate algorithm names
    valid_algorithms = ['dpll', 'walksat', 'genetic']
    for algorithm in algorithm_list:
        if algorithm not in valid_algorithms:
            print(f"ERROR: Unknown algorithm '{algorithm}'. Valid options: {', '.join(valid_algorithms)}")
            sys.exit(1)
    
    # Initialize experiment harness
    experiment_harness = ExperimentHarness(output_directory=args.output_dir)
    
    try:
        if args.benchmark:
            # Run benchmark experiments
            result_filepath = experiment_harness.run_benchmark_experiments(
                args.benchmark, algorithm_list, args.trials, args.output)
            print(f"\nBenchmark experiments completed successfully!")
            print(f"Results saved to: {result_filepath}")
            
        elif args.single:
            # Run single file experiments
            experiment_results = experiment_harness.run_single_formula_experiments(
                args.single, algorithm_list, args.trials)
            
            # Save single file results
            if args.output:
                output_filepath = Path(args.output_dir) / args.output
            else:
                timestamp_string = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"single_experiment_{timestamp_string}.csv"
                output_filepath = Path(args.output_dir) / output_filename
            
            output_filepath.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_filepath, 'w', newline='') as csv_file:
                csv_writer = csv.DictWriter(csv_file, fieldnames=experiment_harness.csv_fieldnames)
                csv_writer.writeheader()
                csv_writer.writerows(experiment_results)
            
            print(f"\nSingle file experiments completed successfully!")
            print(f"Results saved to: {output_filepath}")
            
    except Exception as experiment_error:
        print(f"\n Experiment failed: {experiment_error}")
        sys.exit(1)


if __name__ == "__main__":
    main()