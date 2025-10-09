#!/usr/bin/env python3
"""
Data Analysis and Plotting for SAT Solver Results

This program analyzes results from SAT solver experiments and creates graphs
and reports to compare different algorithms.

Based on concepts from:
- Russell & Norvig, "Artificial Intelligence: A Modern Approach" 
- Course textbook examples for data visualization

CS 463G - Heuristic Search Techniques
Program 3: SAT Solvers Implementation
Author: [Student Name]
"""

# Import libraries for data analysis and plotting
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from datetime import datetime
import json

# Set up matplotlib for nice looking plots
plt.style.use('default')  # Use simple default style

class SATDataAnalyzer:
    """
    A class to analyze SAT solver performance data and create visualizations.
    
    This follows the design patterns from our CS 463G textbook examples.
    """
    
    def __init__(self, results_directory="results"):
        """
        Initialize the data analyzer.
        
        Args:
            results_directory: Folder where CSV files and plots are stored
        """
        # Set up directories 
        self.results_dir = Path(results_directory)
        self.csv_dir = self.results_dir / "csv"
        self.plots_dir = self.results_dir / "plots"
        
        # Create plots directory if it doesn't exist
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Variables to store loaded data
        self.experiment_data = None
        self.parameter_tuning_data = {}
        
        # Settings for plots
        self.figure_size = (12, 8)
        self.dpi = 300  # High resolution for nice plots
        self.colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
        
        print(f"SAT Data Analyzer started")
        print(f"Looking for data in: {self.results_dir}")
        print(f"Plots will be saved to: {self.plots_dir}")
    
    def load_experiment_data(self, pattern="*experiment*.csv"):
        """
        Load experiment results from CSV files.
        
        Args:
            pattern: File pattern to match (like "*experiment*.csv")
            
        Returns:
            True if data was loaded, False if no files found
        """
        """
        Load experiment results from CSV files.
        
        Args:
            csv_pattern: Glob pattern to match experiment CSV files
            
        Returns:
            True if data was loaded successfully, False otherwise
        """
        csv_files = list(self.csv_dir.glob(pattern))
        
        if not csv_files:
            print(f"No experiment data files found matching pattern: {pattern}")
            return False
        
        # Load and combine all experiment CSV files
        all_data = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                all_data.append(df)
                print(f"Loaded experiment data: {csv_file.name} ({len(df)} rows)")
            except Exception as e:
                print(f"Error loading {csv_file}: {e}")
        
        if all_data:
            self.experiment_data = pd.concat(all_data, ignore_index=True)
            
            # Standardize column names for compatibility
            column_mapping = {
                'fitness_percentage': 'satisfaction_rate',
                'solve_time': 'average_solve_time',
                'fitness_score': 'average_fitness_score',
                'formula_variables': 'num_variables',
                'formula_clauses': 'num_clauses'
            }
            
            # Rename columns if they exist
            for old_name, new_name in column_mapping.items():
                if old_name in self.experiment_data.columns:
                    self.experiment_data = self.experiment_data.rename(columns={old_name: new_name})
            
            print(f"Total experiment records loaded: {len(self.experiment_data)}")
            return True
        
        return False
    
    def load_parameter_tuning_data(self, algorithm=None):
        """
        Load parameter tuning results from CSV files.
        
        Args:
            algorithm: Which algorithm to load ('walksat', 'genetic'), or None for all
            
        Returns:
            True if data was loaded, False if no files found
        """
        if algorithm:
            pattern = f"{algorithm}_parameter_tuning*.csv"
            algorithms = [algorithm]
        else:
            pattern = "*parameter_tuning*.csv"
            algorithms = ['walksat', 'genetic']
        
        csv_files = list(self.csv_dir.glob(pattern))
        
        if not csv_files:
            print(f"No parameter tuning data found matching: {pattern}")
            return False
        
        loaded_any = False
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                # Determine algorithm from filename or data
                algo_name = None
                for algo in algorithms:
                    if algo in csv_file.name.lower() or \
                       (not df.empty and df['algorithm'].iloc[0].lower() == algo):
                        algo_name = algo
                        break
                
                if algo_name:
                    self.parameter_tuning_data[algo_name] = df
                    print(f"Loaded {algo_name} parameter tuning data: {csv_file.name} ({len(df)} rows)")
                    loaded_any = True
                    
            except Exception as e:
                print(f"Error loading {csv_file}: {e}")
        
        return loaded_any
    
    def create_algorithm_comparison_plot(self, save_formats=['png', 'svg']):
        """
        Create comparison plots for different algorithms.
        
        Args:
            save_formats: List of file formats to save ('png', 'svg', etc.)
            
        Returns:
            Path to the main saved plot file
        """
        # Make sure we have data to work with
        if self.experiment_data is None:
            raise ValueError("No experiment data loaded. Call load_experiment_data() first.")
        
        # Create a 2x2 grid of plots (like the textbook examples)
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('SAT Solver Algorithm Performance Comparison', fontsize=16, fontweight='bold')
        
        # Plot 1: Box plots showing satisfaction rate distribution
        # This shows how well each algorithm performs on average
        algorithms = self.experiment_data['algorithm'].unique()
        satisfaction_data = []
        for algo in algorithms:
            algo_data = self.experiment_data[self.experiment_data['algorithm'] == algo]
            satisfaction_data.append(algo_data['satisfaction_rate'])
        
        # Make the box plot with colors
        box_plot = ax1.boxplot(satisfaction_data, tick_labels=algorithms, patch_artist=True)
        for i, patch in enumerate(box_plot['boxes']):
            patch.set_facecolor(self.colors[i])
            patch.set_alpha(0.7)
        
        ax1.set_title('Satisfaction Rate Distribution by Algorithm')
        ax1.set_ylabel('Satisfaction Rate (%)')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Bar chart showing average solve times
        # This shows which algorithm is fastest
        avg_times = self.experiment_data.groupby('algorithm')['average_solve_time'].mean()
        bars = ax2.bar(avg_times.index, avg_times.values, color=self.colors[:len(avg_times)])
        ax2.set_title('Average Solve Time by Algorithm')
        ax2.set_ylabel('Average Solve Time (seconds)')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}s', ha='center', va='bottom')
        
        # 3. Formula Difficulty vs Success Rate (Scatter Plot)
        if 'num_variables' in self.experiment_data.columns:
            for i, algo in enumerate(algorithms):
                algo_data = self.experiment_data[self.experiment_data['algorithm'] == algo]
                ax3.scatter(algo_data['num_variables'], algo_data['satisfaction_rate'],
                          label=algo, alpha=0.6, color=self.colors[i])
            
            ax3.set_title('Formula Complexity vs Success Rate')
            ax3.set_xlabel('Number of Variables')
            ax3.set_ylabel('Satisfaction Rate (%)')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'Variable count data\nnot available', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Formula Complexity Analysis')
        
        # 4. Performance Summary Table
        ax4.axis('off')
        summary_stats = self._calculate_algorithm_summary_statistics()
        
        # Create table data
        table_data = []
        for algo in algorithms:
            if algo in summary_stats:
                stats = summary_stats[algo]
                table_data.append([
                    algo.upper(),
                    f"{stats['avg_satisfaction']:.1f}%",
                    f"{stats['avg_solve_time']:.3f}s",
                    f"{stats['success_rate']:.1f}%",
                    f"{stats['total_runs']}"
                ])
        
        table = ax4.table(cellText=table_data,
                         colLabels=['Algorithm', 'Avg Satisfaction', 'Avg Time', 'Success Rate', 'Total Runs'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax4.set_title('Algorithm Performance Summary', pad=20)
        
        plt.tight_layout()
        
        # Save in multiple formats
        base_filename = f"algorithm_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        saved_files = []
        
        for fmt in save_formats:
            filepath = self.plots_dir / f"{base_filename}.{fmt}"
            plt.savefig(filepath, format=fmt, dpi=self.dpi, bbox_inches='tight')
            saved_files.append(str(filepath))
            print(f"Algorithm comparison plot saved: {filepath}")
        
        plt.close()
        return saved_files[0] if saved_files else None
    
    def create_parameter_optimization_plots(self, algorithm, save_formats=['png', 'svg']):
        """
        Create parameter optimization plots for an algorithm.
        
        Args:
            algorithm: Algorithm name ('walksat' or 'genetic')
            save_formats: List of file formats to save
            
        Returns:
            List of paths to saved plot files
        """
        if algorithm not in self.parameter_tuning_data:
            raise ValueError(f"No parameter tuning data loaded for algorithm: {algorithm}")
        
        data = self.parameter_tuning_data[algorithm]
        
        # Determine parameter columns for this algorithm
        param_cols = [col for col in data.columns if col.startswith(f"{algorithm}_")]
        
        if not param_cols:
            print(f"No parameter columns found for {algorithm}")
            return []
        
        # Create parameter vs performance plots
        n_params = len(param_cols)
        fig, axes = plt.subplots(2, (n_params + 1) // 2, figsize=(6 * ((n_params + 1) // 2), 12))
        if n_params == 1:
            axes = [axes]
        elif n_params <= 2:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        fig.suptitle(f'{algorithm.upper()} Parameter Optimization Analysis', 
                    fontsize=16, fontweight='bold')
        
        saved_files = []
        
        for i, param_col in enumerate(param_cols):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # Group by parameter value and calculate statistics
            param_groups = data.groupby(param_col).agg({
                'satisfaction_rate': ['mean', 'std', 'count'],
                'average_solve_time': ['mean', 'std'],
                'average_fitness_score': ['mean', 'std']
            }).round(3)
            
            # Plot satisfaction rate vs parameter
            param_values = param_groups.index
            satisfaction_means = param_groups[('satisfaction_rate', 'mean')]
            satisfaction_stds = param_groups[('satisfaction_rate', 'std')].fillna(0)
            
            ax.errorbar(param_values, satisfaction_means, yerr=satisfaction_stds,
                       marker='o', linewidth=2, markersize=8, capsize=5)
            
            param_name = param_col.replace(f"{algorithm}_", "").replace("_", " ").title()
            ax.set_title(f'{param_name} vs Satisfaction Rate')
            ax.set_xlabel(param_name)
            ax.set_ylabel('Satisfaction Rate (%)')
            ax.grid(True, alpha=0.3)
            
            # Highlight optimal parameter value
            best_idx = satisfaction_means.idxmax()
            ax.axvline(x=best_idx, color='red', linestyle='--', alpha=0.7,
                      label=f'Optimal: {best_idx}')
            ax.legend()
        
        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        
        # Save plots
        base_filename = f"{algorithm}_parameter_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        for fmt in save_formats:
            filepath = self.plots_dir / f"{base_filename}.{fmt}"
            plt.savefig(filepath, format=fmt, dpi=self.dpi, bbox_inches='tight')
            saved_files.append(str(filepath))
            print(f"{algorithm.upper()} parameter optimization plot saved: {filepath}")
        
        plt.close()
        return saved_files
    
    def create_performance_evolution_plot(self, save_formats=['png', 'svg']):
        """
        Create performance evolution plots over time.
        
        Args:
            save_formats: List of file formats to save
            
        Returns:
            List of paths to saved plot files
        """
        if self.experiment_data is None:
            raise ValueError("No experiment data loaded.")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('SAT Solver Performance Evolution Analysis', fontsize=16, fontweight='bold')
        
        # 1. Performance vs Formula Size
        if 'num_variables' in self.experiment_data.columns:
            algorithms = self.experiment_data['algorithm'].unique()
            
            for i, algo in enumerate(algorithms):
                algo_data = self.experiment_data[self.experiment_data['algorithm'] == algo]
                
                # Group by formula size and calculate average performance
                size_groups = algo_data.groupby('num_variables').agg({
                    'satisfaction_rate': 'mean',
                    'average_solve_time': 'mean'
                }).reset_index()
                
                ax1.plot(size_groups['num_variables'], size_groups['satisfaction_rate'],
                        marker='o', linewidth=2, label=algo, color=self.colors[i])
            
            ax1.set_title('Satisfaction Rate vs Problem Size')
            ax1.set_xlabel('Number of Variables')
            ax1.set_ylabel('Average Satisfaction Rate (%)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 2. Solve Time Distribution
        algorithms = self.experiment_data['algorithm'].unique()
        time_data = []
        labels = []
        
        for algo in algorithms:
            algo_times = self.experiment_data[
                self.experiment_data['algorithm'] == algo]['average_solve_time']
            time_data.append(algo_times)
            labels.append(algo)
        
        ax2.hist(time_data, bins=20, alpha=0.7, label=labels, color=self.colors[:len(algorithms)])
        ax2.set_title('Solve Time Distribution by Algorithm')
        ax2.set_xlabel('Average Solve Time (seconds)')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plots
        base_filename = f"performance_evolution_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        saved_files = []
        
        for fmt in save_formats:
            filepath = self.plots_dir / f"{base_filename}.{fmt}"
            plt.savefig(filepath, format=fmt, dpi=self.dpi, bbox_inches='tight')
            saved_files.append(str(filepath))
            print(f"Performance evolution plot saved: {filepath}")
        
        plt.close()
        return saved_files
    
    def generate_summary_report(self, output_format: str = 'txt') -> str:
        """
        Generate comprehensive summary report of all analysis results.
        
        Args:
            output_format: Output format ('txt', 'md', 'json')
            
        Returns:
            Path to generated report file
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_filename = f"sat_solver_analysis_report_{timestamp}.{output_format}"
        report_path = self.results_dir / report_filename
        
        # Collect all analysis data
        summary_data = {
            'analysis_timestamp': datetime.now().isoformat(),
            'experiment_summary': self._calculate_algorithm_summary_statistics() if self.experiment_data is not None else {},
            'parameter_optimization': self._get_parameter_optimization_summary(),
            'recommendations': self._generate_algorithm_recommendations()
        }
        
        if output_format == 'json':
            with open(report_path, 'w') as f:
                json.dump(summary_data, f, indent=2, default=str)
        
        elif output_format == 'md':
            self._write_markdown_report(report_path, summary_data)
        
        else:  # txt format
            self._write_text_report(report_path, summary_data)
        
        print(f"Summary report generated: {report_path}")
        return str(report_path)
    
    def _calculate_algorithm_summary_statistics(self):
        """Calculate summary statistics for each algorithm."""
        if self.experiment_data is None:
            return {}
        
        stats = {}
        for algorithm in self.experiment_data['algorithm'].unique():
            algo_data = self.experiment_data[self.experiment_data['algorithm'] == algorithm]
            
            stats[algorithm] = {
                'total_runs': len(algo_data),
                'avg_satisfaction': algo_data['satisfaction_rate'].mean(),
                'std_satisfaction': algo_data['satisfaction_rate'].std(),
                'avg_solve_time': algo_data['average_solve_time'].mean(),
                'std_solve_time': algo_data['average_solve_time'].std(),
                'success_rate': (algo_data['satisfaction_rate'] > 0).mean() * 100,
                'best_satisfaction': algo_data['satisfaction_rate'].max(),
                'worst_satisfaction': algo_data['satisfaction_rate'].min(),
                'fastest_time': algo_data['average_solve_time'].min(),
                'slowest_time': algo_data['average_solve_time'].max()
            }
        
        return stats
    
    def _get_parameter_optimization_summary(self):
        """Get parameter optimization summary for all algorithms."""
        summary = {}
        
        for algorithm, data in self.parameter_tuning_data.items():
            if data.empty:
                continue
                
            # Find best parameter set
            best_idx = data['satisfaction_rate'].idxmax()
            best_row = data.loc[best_idx]
            
            # Extract parameter values
            param_cols = [col for col in data.columns if col.startswith(f"{algorithm}_")]
            best_params = {col.replace(f"{algorithm}_", ""): best_row[col] for col in param_cols}
            
            summary[algorithm] = {
                'best_satisfaction_rate': best_row['satisfaction_rate'],
                'best_solve_time': best_row['average_solve_time'],
                'best_parameters': best_params,
                'parameter_sets_tested': len(data),
                'satisfaction_rate_range': [data['satisfaction_rate'].min(), data['satisfaction_rate'].max()],
                'solve_time_range': [data['average_solve_time'].min(), data['average_solve_time'].max()]
            }
        
        return summary
    
    def _generate_algorithm_recommendations(self):
        """Generate recommendations based on analysis results."""
        recommendations = {}
        
        if self.experiment_data is not None:
            stats = self._calculate_algorithm_summary_statistics()
            
            # Find best overall algorithm
            best_satisfaction_algo = max(stats.keys(), 
                                       key=lambda x: stats[x]['avg_satisfaction'])
            fastest_algo = min(stats.keys(), 
                             key=lambda x: stats[x]['avg_solve_time'])
            
            recommendations['best_overall'] = best_satisfaction_algo
            recommendations['fastest'] = fastest_algo
            recommendations['most_reliable'] = max(stats.keys(), 
                                                 key=lambda x: stats[x]['success_rate'])
        
        # Parameter recommendations
        param_recommendations = {}
        for algorithm, summary in self._get_parameter_optimization_summary().items():
            param_recommendations[algorithm] = summary['best_parameters']
        
        recommendations['optimal_parameters'] = param_recommendations
        
        return recommendations
    
    def _write_markdown_report(self, filepath, data):
        """Write analysis results in Markdown format."""
        with open(filepath, 'w') as f:
            f.write("# SAT Solver Performance Analysis Report\n\n")
            f.write(f"**Generated:** {data['analysis_timestamp']}\n\n")
            
            # Experiment Summary
            if data['experiment_summary']:
                f.write("## Algorithm Performance Summary\n\n")
                f.write("| Algorithm | Avg Satisfaction | Avg Time | Success Rate | Total Runs |\n")
                f.write("|-----------|------------------|----------|--------------|------------|\n")
                
                for algo, stats in data['experiment_summary'].items():
                    f.write(f"| {algo.upper()} | {stats['avg_satisfaction']:.1f}% | "
                           f"{stats['avg_solve_time']:.3f}s | {stats['success_rate']:.1f}% | "
                           f"{stats['total_runs']} |\n")
                f.write("\n")
            
            # Parameter Optimization
            if data['parameter_optimization']:
                f.write("## Parameter Optimization Results\n\n")
                for algo, results in data['parameter_optimization'].items():
                    f.write(f"### {algo.upper()} Algorithm\n\n")
                    f.write(f"- **Best Satisfaction Rate:** {results['best_satisfaction_rate']:.1f}%\n")
                    f.write(f"- **Best Solve Time:** {results['best_solve_time']:.3f}s\n")
                    f.write(f"- **Parameter Sets Tested:** {results['parameter_sets_tested']}\n")
                    f.write(f"- **Optimal Parameters:**\n")
                    for param, value in results['best_parameters'].items():
                        f.write(f"  - {param}: {value}\n")
                    f.write("\n")
            
            # Recommendations
            if data['recommendations']:
                f.write("## Recommendations\n\n")
                recs = data['recommendations']
                if 'best_overall' in recs:
                    f.write(f"- **Best Overall Algorithm:** {recs['best_overall'].upper()}\n")
                if 'fastest' in recs:
                    f.write(f"- **Fastest Algorithm:** {recs['fastest'].upper()}\n")
                if 'most_reliable' in recs:
                    f.write(f"- **Most Reliable Algorithm:** {recs['most_reliable'].upper()}\n")
    
    def _write_text_report(self, filepath, data):
        """Write analysis results in plain text format."""
        with open(filepath, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("SAT SOLVER PERFORMANCE ANALYSIS REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Generated: {data['analysis_timestamp']}\n\n")
            
            # Similar content but in plain text format
            if data['experiment_summary']:
                f.write("ALGORITHM PERFORMANCE SUMMARY\n")
                f.write("-" * 40 + "\n\n")
                
                for algo, stats in data['experiment_summary'].items():
                    f.write(f"{algo.upper()} Algorithm:\n")
                    f.write(f"  Average Satisfaction Rate: {stats['avg_satisfaction']:.1f}%\n")
                    f.write(f"  Average Solve Time: {stats['avg_solve_time']:.3f}s\n")
                    f.write(f"  Success Rate: {stats['success_rate']:.1f}%\n")
                    f.write(f"  Total Runs: {stats['total_runs']}\n\n")
            
            # Continue with other sections in plain text...

def main():
    """
    Main function that shows how to use the data analysis system.
    This follows the pattern from our textbook examples.
    """
    print("\n=== SAT Solver Data Analysis System ===")
    print("CS 463G - Program 3: Data Analysis and Plotting")
    
    # Create the analyzer object
    analyzer = SATDataAnalyzer()
    
    # Try to load the data files
    experiment_loaded = analyzer.load_experiment_data()
    parameter_loaded = analyzer.load_parameter_tuning_data()
    
    # Check if we found any data to analyze
    if not (experiment_loaded or parameter_loaded):
        print("ERROR: No data found to analyze!")
        print("You need to run experiments first:")
        print("  1. python run_experiments.py")
        print("  2. python parameter_tuning.py")
        print("  3. Then run this script again")
        return
    
    print("\n=== Creating Analysis Plots ===")
    
    # Keep track of how many plots we made
    plots_created = []
    
    # If we have experiment data, make comparison plots
    if experiment_loaded:
        print("Making algorithm comparison plots...")
        try:
            comparison_plot = analyzer.create_algorithm_comparison_plot()
            if comparison_plot:
                plots_created.append(comparison_plot)
                print(f"  SUCCESS: Created {os.path.basename(comparison_plot)}")
        except Exception as e:
            print(f"  ERROR: Error making comparison plot: {e}")
        
        print("Making performance evolution plots...")
        try:
            evolution_plots = analyzer.create_performance_evolution_plot()
            plots_created.extend(evolution_plots)
            for plot in evolution_plots:
                print(f"  SUCCESS: Created {os.path.basename(plot)}")
        except Exception as e:
            print(f"  ERROR: Error making evolution plots: {e}")
    
    # If we have parameter data, make parameter plots
    if parameter_loaded:
        for algorithm in analyzer.parameter_tuning_data.keys():
            print(f"Making {algorithm} parameter optimization plots...")
            try:
                param_plots = analyzer.create_parameter_optimization_plots(algorithm)
                plots_created.extend(param_plots)
                for plot in param_plots:
                    print(f"  SUCCESS: Created {os.path.basename(plot)}")
            except Exception as e:
                print(f"  ERROR: Error making {algorithm} plots: {e}")
    
    # Generate summary reports
    print("\nGenerating analysis reports...")
    try:
        report_path = analyzer.generate_summary_report('md')
        print(f"  SUCCESS: Created report {os.path.basename(report_path)}")
    except Exception as e:
        print(f"  ERROR: Error making report: {e}")
    
    # Show summary
    print(f"\n=== Analysis Complete ===")
    print(f"Total plots created: {len(plots_created)}")
    print(f"Output directory: {analyzer.plots_dir}")
    
    if plots_created:
        print("\nYou can now view the plots and reports!")
    else:
        print("\nNo plots were created - check for errors above.")

if __name__ == "__main__":
    main()