"""
Genetic Algorithm SAT Solver Implementation

A genetic algorithm approach to the Boolean satisfiability problem.
This heuristic algorithm evolves a population of truth assignments
to find satisfying solutions.

Reference Algorithm Inspiration:
Jong, K.A.D. (1975). "An analysis of the behavior of a class of genetic adaptive systems."
Mitchell, M. (1996). "An Introduction to Genetic Algorithms."

Genetic Algorithm for SAT Overview:
1. Initialize population of random truth assignments
2. Evaluate fitness (number of satisfied clauses) for each individual
3. Select parents for reproduction based on fitness
4. Create offspring through crossover and mutation
5. Replace population with fittest individuals
6. Repeat until solution found or max generations reached

Key Components:
- Population: Set of truth assignments (individuals)
- Fitness: Number of satisfied clauses (0 to total clauses)
- Selection: Tournament selection based on fitness
- Crossover: Uniform crossover between parent assignments
- Mutation: Random bit flips in truth assignments
- Replacement: Generational replacement with elitism

Parameters:
- population_size: Number of individuals in each generation
- max_generations: Maximum number of generations to evolve
- crossover_rate: Probability of crossover between parents
- mutation_rate: Probability of mutating each variable
- tournament_size: Number of individuals in tournament selection
- elite_count: Number of best individuals to preserve each generation
"""

import random
import time
from typing import List, Dict, Set, Optional, Tuple
from enum import Enum

# Import shared components
try:
    from ..cnf_parser import CNFFormula
    from .dpll import SATResult
except ImportError:
    from cnf_parser import CNFFormula
    from solvers.dpll import SATResult


class GeneticSATSolver:
    """
    Genetic Algorithm implementation for Boolean satisfiability.
    
    This evolutionary algorithm maintains a population of candidate
    solutions and evolves them toward satisfying assignments.
    """
    
    def __init__(self,
                 population_size: int = 100,
                 max_generations: int = 1000,
                 crossover_probability: float = 0.8,
                 mutation_probability: float = 0.1,
                 tournament_size: int = 5,
                 elite_preservation_count: int = 10,
                 random_seed: int = 42):
        """
        Initialize Genetic Algorithm SAT solver with undergraduate-friendly parameters.
        
        Args:
            population_size: Number of individuals (truth assignments) in population
            max_generations: Maximum number of evolutionary generations
            crossover_probability: Probability of crossover between parent assignments
            mutation_probability: Probability of flipping each variable bit
            tournament_size: Number of individuals competing in tournament selection
            elite_preservation_count: Number of best individuals to keep each generation
            random_seed: Random seed for reproducible evolutionary runs
        """
        self.population_size = population_size
        self.max_generations = max_generations
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.tournament_size = tournament_size
        self.elite_preservation_count = elite_preservation_count
        self.random_seed = random_seed
        
        # Statistics tracking for evolutionary analysis
        self.solver_statistics = {
            "generations_evolved": 0,       # Total generations evolved
            "fitness_evaluations": 0,      # Total fitness evaluations performed
            "best_fitness_achieved": 0,     # Best fitness score found
            "final_population_fitness": 0, # Average fitness of final population
            "crossover_operations": 0,     # Number of crossover operations
            "mutation_operations": 0,      # Number of mutation operations
            "evolution_time": 0.0          # Total evolution time
        }
        
    def solve(self, cnf_formula: CNFFormula) -> Tuple[SATResult, Optional[Dict[int, bool]], Dict]:
        """
        Main Genetic Algorithm solving entry point.
        
        Evolves a population of truth assignments to find a satisfying
        assignment for the given CNF formula.
        
        Args:
            cnf_formula: CNF formula to solve
            
        Returns:
            Tuple of (result, assignment, statistics)
            - result: SATResult (SAT if solution found, UNKNOWN if not found)
            - assignment: Best variable assignment found, None if no good solution
            - statistics: Solver performance and evolutionary metrics
        """
        start_evolution_time = time.time()
        
        # Initialize random number generator for reproducible evolution
        random.seed(self.random_seed)
        
        # Reset solver statistics for this evolutionary run
        self._reset_solver_statistics()
        
        # Step 1: Initialize population with random truth assignments
        current_population = self._initialize_random_population(cnf_formula.num_vars)
        
        # Step 2: Main evolutionary loop
        best_individual_found = None
        best_fitness_achieved = 0
        
        for current_generation in range(self.max_generations):
            self.solver_statistics["generations_evolved"] += 1
            
            # Evaluate fitness for all individuals in current population
            population_fitness_scores = self._evaluate_population_fitness(cnf_formula, current_population)
            
            # Find the best individual in this generation
            generation_best_fitness = max(population_fitness_scores)
            generation_best_index = population_fitness_scores.index(generation_best_fitness)
            generation_best_individual = current_population[generation_best_index]
            
            # Update global best if this generation found a better solution
            if generation_best_fitness > best_fitness_achieved:
                best_fitness_achieved = generation_best_fitness
                best_individual_found = generation_best_individual.copy()
                
            # Check if we found a complete satisfying assignment
            if generation_best_fitness == cnf_formula.num_clauses:
                self.solver_statistics["best_fitness_achieved"] = best_fitness_achieved
                final_average_fitness = sum(population_fitness_scores) / len(population_fitness_scores)
                self.solver_statistics["final_population_fitness"] = final_average_fitness
                self.solver_statistics["evolution_time"] = time.time() - start_evolution_time
                
                return SATResult.SATISFIABLE, best_individual_found, self.solver_statistics
            
            # Step 3: Create next generation through selection, crossover, and mutation
            next_generation_population = self._create_next_generation(
                current_population, population_fitness_scores, cnf_formula.num_vars)
            
            current_population = next_generation_population
        
        # Evolution completed - return best solution found
        self.solver_statistics["best_fitness_achieved"] = best_fitness_achieved
        final_population_fitness_scores = self._evaluate_population_fitness(cnf_formula, current_population)
        final_average_fitness = sum(final_population_fitness_scores) / len(final_population_fitness_scores)
        self.solver_statistics["final_population_fitness"] = final_average_fitness
        self.solver_statistics["evolution_time"] = time.time() - start_evolution_time
        
        # Return best partial solution (GA cannot prove UNSAT)
        return SATResult.UNKNOWN, best_individual_found, self.solver_statistics
    
    def _initialize_random_population(self, number_of_variables: int) -> List[Dict[int, bool]]:
        """
        Create initial population of random truth assignments.
        
        Args:
            number_of_variables: Number of variables in the CNF formula
            
        Returns:
            List of random truth assignment dictionaries
        """
        initial_population = []
        
        for individual_index in range(self.population_size):
            # Create random truth assignment for this individual
            random_individual_assignment = {}
            for variable_number in range(1, number_of_variables + 1):
                random_truth_value = random.choice([True, False])
                random_individual_assignment[variable_number] = random_truth_value
            
            initial_population.append(random_individual_assignment)
        
        return initial_population
    
    def _evaluate_population_fitness(self, cnf_formula: CNFFormula, 
                                   population: List[Dict[int, bool]]) -> List[int]:
        """
        Evaluate fitness (satisfied clauses) for each individual in population.
        
        Args:
            cnf_formula: CNF formula to evaluate against
            population: List of truth assignments to evaluate
            
        Returns:
            List of fitness scores (satisfied clause counts) for each individual
        """
        fitness_scores_list = []
        
        for individual_assignment in population:
            individual_fitness = self._calculate_individual_fitness(cnf_formula, individual_assignment)
            fitness_scores_list.append(individual_fitness)
            self.solver_statistics["fitness_evaluations"] += 1
        
        return fitness_scores_list
    
    def _calculate_individual_fitness(self, cnf_formula: CNFFormula, 
                                    individual_assignment: Dict[int, bool]) -> int:
        """
        Calculate fitness score for a single individual (truth assignment).
        
        Fitness is the number of clauses satisfied by this assignment.
        Higher fitness indicates better solutions.
        
        Args:
            cnf_formula: CNF formula to evaluate
            individual_assignment: Truth assignment to evaluate
            
        Returns:
            Number of satisfied clauses (0 to total clauses)
        """
        satisfied_clause_count = 0
        
        for individual_clause in cnf_formula.clauses:
            # Check if this clause is satisfied by the assignment
            clause_is_satisfied = False
            
            for literal_in_clause in individual_clause:
                variable_number = abs(literal_in_clause)
                literal_is_positive = literal_in_clause > 0
                assigned_truth_value = individual_assignment[variable_number]
                
                # Literal is satisfied if: (positive AND variable True) OR (negative AND variable False)
                if (literal_is_positive and assigned_truth_value) or (not literal_is_positive and not assigned_truth_value):
                    clause_is_satisfied = True
                    break
            
            if clause_is_satisfied:
                satisfied_clause_count += 1
        
        return satisfied_clause_count
    
    def _create_next_generation(self, current_population: List[Dict[int, bool]], 
                              fitness_scores: List[int], 
                              number_of_variables: int) -> List[Dict[int, bool]]:
        """
        Create the next generation through selection, crossover, and mutation.
        
        Uses elitism to preserve best individuals, then fills remaining
        population through tournament selection and genetic operators.
        
        Args:
            current_population: Current generation of individuals
            fitness_scores: Fitness scores for current population
            number_of_variables: Number of variables in assignments
            
        Returns:
            New population for next generation
        """
        next_generation = []
        
        # Step 1: Elitism - preserve best individuals from current generation
        elite_individuals = self._select_elite_individuals(current_population, fitness_scores)
        next_generation.extend(elite_individuals)
        
        # Step 2: Fill remaining population through reproduction
        while len(next_generation) < self.population_size:
            # Tournament selection to choose parents
            parent_one = self._tournament_selection(current_population, fitness_scores)
            parent_two = self._tournament_selection(current_population, fitness_scores)
            
            # Create offspring through crossover and mutation
            offspring_one, offspring_two = self._crossover_and_mutation(
                parent_one, parent_two, number_of_variables)
            
            # Add offspring to next generation (avoiding overpopulation)
            if len(next_generation) < self.population_size:
                next_generation.append(offspring_one)
            if len(next_generation) < self.population_size:
                next_generation.append(offspring_two)
        
        return next_generation
    
    def _select_elite_individuals(self, population: List[Dict[int, bool]], 
                                fitness_scores: List[int]) -> List[Dict[int, bool]]:
        """
        Select the best individuals for elitism (preservation to next generation).
        
        Args:
            population: Current population
            fitness_scores: Fitness scores for each individual
            
        Returns:
            List of elite individuals to preserve
        """
        # Create list of (fitness, individual) pairs and sort by fitness (descending)
        fitness_individual_pairs = list(zip(fitness_scores, population))
        fitness_individual_pairs.sort(key=lambda pair: pair[0], reverse=True)
        
        # Extract top elite individuals
        elite_individuals = []
        for rank in range(min(self.elite_preservation_count, len(population))):
            elite_fitness, elite_individual = fitness_individual_pairs[rank]
            elite_individuals.append(elite_individual.copy())
        
        return elite_individuals
    
    def _tournament_selection(self, population: List[Dict[int, bool]], 
                            fitness_scores: List[int]) -> Dict[int, bool]:
        """
        Select an individual using tournament selection.
        
        Randomly chooses tournament_size individuals and returns the fittest.
        
        Args:
            population: Current population
            fitness_scores: Fitness scores for each individual
            
        Returns:
            Selected individual (copy of truth assignment)
        """
        # Randomly select tournament participants
        tournament_indices = random.sample(range(len(population)), 
                                         min(self.tournament_size, len(population)))
        
        # Find the fittest individual in the tournament
        best_tournament_index = tournament_indices[0]
        best_tournament_fitness = fitness_scores[best_tournament_index]
        
        for participant_index in tournament_indices[1:]:
            participant_fitness = fitness_scores[participant_index]
            if participant_fitness > best_tournament_fitness:
                best_tournament_fitness = participant_fitness
                best_tournament_index = participant_index
        
        return population[best_tournament_index].copy()
    
    def _crossover_and_mutation(self, parent_one: Dict[int, bool], parent_two: Dict[int, bool], 
                              number_of_variables: int) -> Tuple[Dict[int, bool], Dict[int, bool]]:
        """
        Create two offspring through crossover and mutation operations.
        
        Args:
            parent_one: First parent truth assignment
            parent_two: Second parent truth assignment  
            number_of_variables: Number of variables in assignments
            
        Returns:
            Tuple of two offspring truth assignments
        """
        offspring_one = parent_one.copy()
        offspring_two = parent_two.copy()
        
        # Crossover operation: uniform crossover with probability
        if random.random() < self.crossover_probability:
            self.solver_statistics["crossover_operations"] += 1
            
            for variable_number in range(1, number_of_variables + 1):
                # With 50% probability, swap variable assignments between parents
                if random.random() < 0.5:
                    offspring_one[variable_number] = parent_two[variable_number]
                    offspring_two[variable_number] = parent_one[variable_number]
        
        # Mutation operation: random bit flips
        offspring_one = self._mutate_individual(offspring_one, number_of_variables)
        offspring_two = self._mutate_individual(offspring_two, number_of_variables)
        
        return offspring_one, offspring_two
    
    def _mutate_individual(self, individual_assignment: Dict[int, bool], 
                         number_of_variables: int) -> Dict[int, bool]:
        """
        Apply mutation to an individual by randomly flipping variable assignments.
        
        Args:
            individual_assignment: Truth assignment to mutate
            number_of_variables: Total number of variables
            
        Returns:
            Mutated truth assignment
        """
        mutated_individual = individual_assignment.copy()
        
        for variable_number in range(1, number_of_variables + 1):
            # With mutation probability, flip this variable's truth value
            if random.random() < self.mutation_probability:
                mutated_individual[variable_number] = not mutated_individual[variable_number]
                self.solver_statistics["mutation_operations"] += 1
        
        return mutated_individual
    
    def _reset_solver_statistics(self):
        """Reset all solver statistics for a new evolutionary run."""
        self.solver_statistics = {
            "generations_evolved": 0,
            "fitness_evaluations": 0,
            "best_fitness_achieved": 0,
            "final_population_fitness": 0,
            "crossover_operations": 0,
            "mutation_operations": 0,
            "evolution_time": 0.0
        }
    
    def verify_solution(self, cnf_formula: CNFFormula, assignment: Dict[int, bool]) -> bool:
        """
        Verify that an assignment satisfies the CNF formula.
        
        Args:
            cnf_formula: Original CNF formula
            assignment: Variable assignment to verify
            
        Returns:
            True if assignment satisfies all clauses, False otherwise
        """
        individual_fitness = self._calculate_individual_fitness(cnf_formula, assignment)
        return individual_fitness == cnf_formula.num_clauses