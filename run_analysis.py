#!/usr/bin/env python3
"""
Data Analysis Runner for SAT Solver Results

This script analyzes the results from our SAT solver experiments and creates
graphs and reports to see which algorithms work best.

Usage:
    python run_analysis.py [--experiments] [--parameters] [--all]

CS 463G - Heuristic Search Techniques
Program 3: SAT Solvers Implementation
Phase 6: Data Analysis & Plotting
Author: [Student Name]
"""

import argparse
import sys
from pathlib import Path
import time

def main():
    """Main function that runs the data analysis."""
    # Set up command line arguments (like textbook examples)
    parser = argparse.ArgumentParser(description="Run SAT Solver Data Analysis")
    parser.add_argument('--experiments', action='store_true', 
                       help='Analyze experiment results')
    parser.add_argument('--parameters', action='store_true',
                       help='Analyze parameter tuning results')  
    parser.add_argument('--all', action='store_true',
                       help='Run all available analyses')
    parser.add_argument('--output-dir', default='results',
                       help='Where to save results (default: results)')
    
    args = parser.parse_args()
    
    # If user didn't specify what to analyze, do everything
    if not (args.experiments or args.parameters):
        args.all = True
    
    # Print a nice header
    print("=" * 60)
    print("SAT SOLVER DATA ANALYSIS SYSTEM")
    print("=" * 60)
    print("Phase 6: Data Analysis & Plotting")
    print(f"Looking for data in: {args.output_dir}")
    print()
    
    # Try to import our data analysis module
    try:
        from data_analysis import SATDataAnalyzer
    except ImportError as e:
        print(f"ERROR: Can't import data analysis module: {e}")
        print("Make sure you have the required packages:")
        print("  pip install pandas matplotlib")
        return 1
    
    # Initialize analyzer
    analyzer = SATDataAnalyzer(args.output_dir)
    
    # Track what data is available
    data_available = {
        'experiments': False,
        'parameters': False
    }
    
    # Load experiment data
    if args.experiments or args.all:
        print("=== Loading Experiment Results ===")
        start_time = time.time()
        
        if analyzer.load_experiment_data():
            data_available['experiments'] = True
            load_time = time.time() - start_time
            print(f"SUCCESS: Experiment data loaded successfully ({load_time:.2f}s)")
            print(f"   Records: {len(analyzer.experiment_data)}")
            print(f"   Algorithms: {list(analyzer.experiment_data['algorithm'].unique())}")
            print(f"   Formulas: {analyzer.experiment_data['formula_file'].nunique()}")
        else:
            print("ERROR: No experiment data found")
            print("   Run experiments first: python run_experiments.py")
        print()
    
    # Load parameter tuning data
    if args.parameters or args.all:
        print("=== Loading Parameter Tuning Results ===")
        start_time = time.time()
        
        if analyzer.load_parameter_tuning_data():
            data_available['parameters'] = True
            load_time = time.time() - start_time
            print(f"SUCCESS: Parameter tuning data loaded successfully ({load_time:.2f}s)")
            for algo, data in analyzer.parameter_tuning_data.items():
                print(f"   {algo.upper()}: {len(data)} parameter evaluations")
        else:
            print("ERROR: No parameter tuning data found")
            print("   Run parameter tuning first: python parameter_tuning.py")
        print()
    
    # Check if any data is available
    if not any(data_available.values()):
        print("ERROR: No data available for analysis")
        print("\nTo generate data:")
        print("  1. Run experiments: python run_experiments.py")
        print("  2. Run parameter tuning: python parameter_tuning.py")
        print("  3. Then run this analysis script again")
        return 1
    
    # Generate visualizations and reports
    print("=== Generating Analysis Outputs ===")
    outputs_created = []
    
    # Algorithm comparison plots (requires experiment data)
    if data_available['experiments']:
        print("Creating algorithm comparison plots...")
        try:
            comparison_plot = analyzer.create_algorithm_comparison_plot(['png', 'svg'])
            if comparison_plot:
                outputs_created.append(comparison_plot)
                print(f"   SUCCESS: Algorithm comparison: {Path(comparison_plot).name}")
        except Exception as e:
            print(f"   ERROR: Error creating comparison plot: {e}")
        
        print("Creating performance evolution plots...")
        try:
            evolution_plots = analyzer.create_performance_evolution_plot(['png', 'svg'])
            outputs_created.extend(evolution_plots)
            for plot in evolution_plots:
                print(f"   SUCCESS: Performance evolution: {Path(plot).name}")
        except Exception as e:
            print(f"   ERROR: Error creating evolution plots: {e}")
    
    # Parameter optimization plots (requires parameter tuning data)
    if data_available['parameters']:
        for algorithm in analyzer.parameter_tuning_data.keys():
            print(f"Creating {algorithm.upper()} parameter optimization plots...")
            try:
                param_plots = analyzer.create_parameter_optimization_plots(algorithm, ['png', 'svg'])
                outputs_created.extend(param_plots)
                for plot in param_plots:
                    print(f"   SUCCESS: {algorithm.upper()} optimization: {Path(plot).name}")
            except Exception as e:
                print(f"   ERROR: Error creating {algorithm} plots: {e}")
    
    # Generate comprehensive reports
    print("Generating analysis reports...")
    report_formats = ['txt', 'md', 'json']
    
    for fmt in report_formats:
        try:
            report_path = analyzer.generate_summary_report(fmt)
            outputs_created.append(report_path)
            print(f"   SUCCESS: {fmt.upper()} report: {Path(report_path).name}")
        except Exception as e:
            print(f"   ERROR: Error creating {fmt} report: {e}")
    
    print()
    
    # Summary
    print("=== Analysis Complete ===")
    print(f"Outputs generated: {len(outputs_created)}")
    print(f"Output directory: {analyzer.plots_dir}")
    
    if outputs_created:
        print("\nKey Outputs:")
        
        # Group outputs by type
        plots = [f for f in outputs_created if any(f.endswith(ext) for ext in ['.png', '.svg', '.pdf'])]
        reports = [f for f in outputs_created if any(f.endswith(ext) for ext in ['.txt', '.md', '.json'])]
        
        if plots:
            print(f"   Visualization plots: {len(plots)}")
            for plot in plots[:3]:  # Show first 3
                print(f"      * {Path(plot).name}")
            if len(plots) > 3:
                print(f"      * ... and {len(plots) - 3} more")
        
        if reports:
            print(f"   Analysis reports: {len(reports)}")
            for report in reports:
                print(f"      * {Path(report).name}")
        
        print(f"\nView results in: {analyzer.plots_dir}")
        
        # Data insights preview
        if data_available['experiments']:
            print("\nQuick Insights:")
            try:
                stats = analyzer._calculate_algorithm_summary_statistics()
                if stats:
                    best_algo = max(stats.keys(), key=lambda x: stats[x]['avg_satisfaction'])
                    fastest_algo = min(stats.keys(), key=lambda x: stats[x]['avg_solve_time'])
                    
                    print(f"   Best performing algorithm: {best_algo.upper()} "
                          f"({stats[best_algo]['avg_satisfaction']:.1f}% satisfaction)")
                    print(f"   Fastest algorithm: {fastest_algo.upper()} "
                          f"({stats[fastest_algo]['avg_solve_time']:.3f}s average)")
            except Exception as e:
                print(f"   ERROR: Error generating insights: {e}")
        
        if data_available['parameters']:
            print("\nParameter Optimization:")
            try:
                param_summary = analyzer._get_parameter_optimization_summary()
                for algo, summary in param_summary.items():
                    best_satisfaction = summary['best_satisfaction_rate']
                    param_sets_tested = summary['parameter_sets_tested']
                    print(f"   {algo.upper()}: {best_satisfaction:.1f}% best satisfaction "
                          f"({param_sets_tested} parameter sets tested)")
            except Exception as e:
                print(f"   ERROR: Error generating parameter insights: {e}")
    
    else:
        print("ERROR: No outputs were generated successfully")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)