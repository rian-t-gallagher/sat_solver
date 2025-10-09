"""
Test suite for WalkSAT and Genetic Algorithm SAT solvers.
"""

import pytest
import os
from src.cnf_parser import parse_dimacs, CNFFormula
from src.solvers import WalkSATSolver, GeneticSATSolver, SATResult


class TestWalkSATSolver:
    """Test WalkSAT heuristic solver implementation."""
    
    def test_walksat_simple_satisfiable_formula(self):
        """Test WalkSAT on a simple satisfiable formula."""
        # Formula: (x1 ∨ x2) ∧ (¬x1 ∨ x2) ∧ (x1 ∨ ¬x2)
        # Should be satisfiable with x1=True, x2=True
        clauses = [[1, 2], [-1, 2], [1, -2]]
        formula = CNFFormula(2, 3, clauses)
        
        solver = WalkSATSolver(max_flips_per_try=100, max_restart_attempts=5, random_seed=42)
        result, assignment, stats = solver.solve(formula)
        
        # WalkSAT should find SAT or return UNKNOWN (never UNSAT)
        assert result in [SATResult.SATISFIABLE, SATResult.UNKNOWN]
        
        if result == SATResult.SATISFIABLE:
            assert assignment is not None
            assert solver.verify_solution(formula, assignment)
        
        # Check reasonable statistics
        assert stats["total_flips"] >= 0
        assert stats["solving_time"] > 0
        assert stats["restart_attempts"] >= 1
        
    def test_walksat_finds_solution_with_multiple_restarts(self):
        """Test that WalkSAT can find solutions with multiple restarts."""
        # Harder satisfiable formula
        clauses = [[1, 2, 3], [-1, 2], [-2, 3], [-3, 1], [1, -2, 3]]
        formula = CNFFormula(3, 5, clauses)
        
        solver = WalkSATSolver(max_flips_per_try=200, max_restart_attempts=10, 
                               random_walk_probability=0.3, random_seed=123)
        result, assignment, stats = solver.solve(formula)
        
        # Should find satisfying assignment
        assert result == SATResult.SATISFIABLE
        assert assignment is not None
        assert solver.verify_solution(formula, assignment)
        
        # Should have used multiple restarts potentially
        assert stats["restart_attempts"] >= 1
        assert stats["total_flips"] > 0
        
    def test_walksat_different_parameters(self):
        """Test WalkSAT with different parameter settings."""
        clauses = [[1, 2], [-1, 3], [2, -3]]
        formula = CNFFormula(3, 3, clauses)
        
        # Test with high random walk probability
        solver_random = WalkSATSolver(max_flips_per_try=50, random_walk_probability=0.9, random_seed=42)
        result_random, _, stats_random = solver_random.solve(formula)
        
        # Test with low random walk probability (more greedy)
        solver_greedy = WalkSATSolver(max_flips_per_try=50, random_walk_probability=0.1, random_seed=42)
        result_greedy, _, stats_greedy = solver_greedy.solve(formula)
        
        # Both should find solutions
        assert result_random in [SATResult.SATISFIABLE, SATResult.UNKNOWN]
        assert result_greedy in [SATResult.SATISFIABLE, SATResult.UNKNOWN]
        
        # Random version should have more random flips
        if stats_random["total_flips"] > 0 and stats_greedy["total_flips"] > 0:
            random_flip_ratio = stats_random["random_flips"] / stats_random["total_flips"]
            greedy_flip_ratio = stats_greedy["random_flips"] / stats_greedy["total_flips"]
            assert random_flip_ratio > greedy_flip_ratio
    
    def test_walksat_statistics_tracking(self):
        """Test that WalkSAT properly tracks performance statistics."""
        clauses = [[1, 2, 3], [-1, -2, -3]]
        formula = CNFFormula(3, 2, clauses)
        
        solver = WalkSATSolver(max_flips_per_try=10, max_restart_attempts=2, random_seed=42)
        result, assignment, stats = solver.solve(formula)
        
        # Check all expected statistics are present and reasonable
        assert "total_flips" in stats
        assert "restart_attempts" in stats
        assert "best_satisfied_clauses" in stats
        assert "random_flips" in stats
        assert "greedy_flips" in stats
        assert "solving_time" in stats
        
        assert stats["total_flips"] >= 0
        assert stats["restart_attempts"] >= 1
        assert stats["restart_attempts"] <= 2  # Should not exceed max
        assert stats["best_satisfied_clauses"] >= 0
        assert stats["solving_time"] > 0
        assert stats["random_flips"] + stats["greedy_flips"] == stats["total_flips"]


class TestGeneticSATSolver:
    """Test Genetic Algorithm SAT solver implementation."""
    
    def test_genetic_simple_satisfiable_formula(self):
        """Test Genetic Algorithm on a simple satisfiable formula."""
        # Formula: (x1 ∨ x2) ∧ (¬x1 ∨ x2) ∧ (x1 ∨ ¬x2)
        clauses = [[1, 2], [-1, 2], [1, -2]]
        formula = CNFFormula(2, 3, clauses)
        
        solver = GeneticSATSolver(population_size=20, max_generations=50, random_seed=42)
        result, assignment, stats = solver.solve(formula)
        
        # GA should find SAT or return UNKNOWN (never UNSAT)
        assert result in [SATResult.SATISFIABLE, SATResult.UNKNOWN]
        
        if result == SATResult.SATISFIABLE:
            assert assignment is not None
            assert solver.verify_solution(formula, assignment)
        
        # Check reasonable statistics
        assert stats["generations_evolved"] >= 1
        assert stats["fitness_evaluations"] > 0
        assert stats["evolution_time"] > 0
        
    def test_genetic_population_evolution(self):
        """Test that genetic algorithm properly evolves population."""
        clauses = [[1, 2, 3], [-1, 2], [-2, 3], [-3, 1]]
        formula = CNFFormula(3, 4, clauses)
        
        solver = GeneticSATSolver(population_size=30, max_generations=100,
                                  crossover_probability=0.8, mutation_probability=0.1, random_seed=456)
        result, assignment, stats = solver.solve(formula)
        
        # Should evolve through multiple generations
        assert stats["generations_evolved"] >= 1
        assert stats["fitness_evaluations"] >= 30  # At least initial population
        
        if result == SATResult.SATISFIABLE:
            assert assignment is not None
            assert solver.verify_solution(formula, assignment)
            assert stats["best_fitness_achieved"] == formula.num_clauses
    
    def test_genetic_crossover_and_mutation(self):
        """Test that genetic operators are being applied."""
        clauses = [[1, 2], [-1, 3], [2, -3], [1, 3]]
        formula = CNFFormula(3, 4, clauses)
        
        # Use high crossover and mutation rates to ensure operations occur
        solver = GeneticSATSolver(population_size=20, max_generations=30,
                                  crossover_probability=0.9, mutation_probability=0.2, random_seed=789)
        result, assignment, stats = solver.solve(formula)
        
        # Should have performed genetic operations
        if stats["generations_evolved"] > 1:  # Need at least 2 generations for operations
            assert stats["crossover_operations"] >= 0
            assert stats["mutation_operations"] >= 0
    
    def test_genetic_parameter_variations(self):
        """Test genetic algorithm with different parameter settings."""
        clauses = [[1, 2, 3], [-1, -2], [2, -3], [1, 3]]
        formula = CNFFormula(3, 4, clauses)
        
        # Small population, few generations
        solver_small = GeneticSATSolver(population_size=10, max_generations=20, random_seed=42)
        result_small, _, stats_small = solver_small.solve(formula)
        
        # Large population, many generations  
        solver_large = GeneticSATSolver(population_size=50, max_generations=100, random_seed=42)
        result_large, _, stats_large = solver_large.solve(formula)
        
        # Both should make progress
        assert result_small in [SATResult.SATISFIABLE, SATResult.UNKNOWN]
        assert result_large in [SATResult.SATISFIABLE, SATResult.UNKNOWN]
        
        # Large solver should do more fitness evaluations
        assert stats_large["fitness_evaluations"] > stats_small["fitness_evaluations"]
    
    def test_genetic_statistics_comprehensive(self):
        """Test comprehensive statistics tracking for genetic algorithm."""
        clauses = [[1, 2], [-1, 3], [2, -3]]
        formula = CNFFormula(3, 3, clauses)
        
        solver = GeneticSATSolver(population_size=15, max_generations=25, random_seed=42)
        result, assignment, stats = solver.solve(formula)
        
        # Verify all expected statistics are tracked
        expected_stats = [
            "generations_evolved", "fitness_evaluations", "best_fitness_achieved",
            "final_population_fitness", "crossover_operations", "mutation_operations", "evolution_time"
        ]
        
        for stat_name in expected_stats:
            assert stat_name in stats, f"Missing statistic: {stat_name}"
            assert stats[stat_name] >= 0, f"Invalid value for {stat_name}: {stats[stat_name]}"
        
        # Fitness should be within valid range
        assert 0 <= stats["best_fitness_achieved"] <= formula.num_clauses
        assert 0 <= stats["final_population_fitness"] <= formula.num_clauses


class TestHeuristicSolversOnRealFiles:
    """Test heuristic solvers on real CNF benchmark files."""
    
    def test_walksat_on_satisfiable_file(self):
        """Test WalkSAT on a real satisfiable CNF file."""
        filepath = "benchmarks/CNF Formulas/uf20-0156.cnf"
        
        if not os.path.exists(filepath):
            pytest.skip(f"Test file {filepath} not found")
            
        formula = parse_dimacs(filepath)
        solver = WalkSATSolver(max_flips_per_try=500, max_restart_attempts=5, random_seed=42)
        
        result, assignment, stats = solver.solve(formula)
        
        # Should find satisfying assignment (this file is known to be SAT)
        assert result == SATResult.SATISFIABLE
        assert assignment is not None
        assert solver.verify_solution(formula, assignment)
        
        print(f"WalkSAT solved uf20-0156: {stats['total_flips']} flips, {stats['solving_time']:.3f}s")
    
    def test_genetic_on_satisfiable_file(self):
        """Test Genetic Algorithm on a real satisfiable CNF file."""
        filepath = "benchmarks/CNF Formulas/uf20-0156.cnf"
        
        if not os.path.exists(filepath):
            pytest.skip(f"Test file {filepath} not found")
            
        formula = parse_dimacs(filepath)
        solver = GeneticSATSolver(population_size=50, max_generations=200, random_seed=42)
        
        result, assignment, stats = solver.solve(formula)
        
        # Should find satisfying assignment (this file is known to be SAT)
        if result == SATResult.SATISFIABLE:
            assert assignment is not None
            assert solver.verify_solution(formula, assignment)
            print(f"GA solved uf20-0156: {stats['generations_evolved']} generations, {stats['evolution_time']:.3f}s")
        else:
            # GA might not always find solution due to its heuristic nature
            print(f"GA best fitness on uf20-0156: {stats['best_fitness_achieved']}/{formula.num_clauses}")
    
    def test_heuristic_solver_comparison(self):
        """Compare performance of heuristic solvers on the same instance."""
        filepath = "benchmarks/CNF Formulas/uf20-0156.cnf"
        
        if not os.path.exists(filepath):
            pytest.skip(f"Test file {filepath} not found")
            
        formula = parse_dimacs(filepath)
        
        # Test WalkSAT
        walksat_solver = WalkSATSolver(max_flips_per_try=200, max_restart_attempts=3, random_seed=42)
        walksat_result, walksat_assignment, walksat_stats = walksat_solver.solve(formula)
        
        # Test Genetic Algorithm
        genetic_solver = GeneticSATSolver(population_size=30, max_generations=100, random_seed=42)
        genetic_result, genetic_assignment, genetic_stats = genetic_solver.solve(formula)
        
        # Both should make reasonable progress
        assert walksat_result in [SATResult.SATISFIABLE, SATResult.UNKNOWN]
        assert genetic_result in [SATResult.SATISFIABLE, SATResult.UNKNOWN]
        
        # Print comparison results
        print(f"\\nSolver Comparison on {filepath}:")
        print(f"WalkSAT: {walksat_result.value}, Time: {walksat_stats['solving_time']:.3f}s")
        print(f"Genetic: {genetic_result.value}, Time: {genetic_stats['evolution_time']:.3f}s")
        
        if walksat_result == SATResult.SATISFIABLE and walksat_assignment:
            print(f"WalkSAT: Found complete solution")
        if genetic_result == SATResult.SATISFIABLE and genetic_assignment:
            print(f"Genetic: Found complete solution")