"""
Genetic Algorithm for SAT solving.
This is an evolutionary optimization approach.
"""

import random


class Individual:
    """Represents an individual (solution candidate) in the population."""
    
    def __init__(self, num_vars, assignment=None):
        """
        Initialize an individual.
        
        Args:
            num_vars: Number of variables
            assignment: Initial assignment (None for random)
        """
        if assignment is None:
            self.assignment = {var: random.choice([True, False]) 
                             for var in range(1, num_vars + 1)}
        else:
            self.assignment = assignment.copy()
        self.fitness = 0
    
    def evaluate_fitness(self, formula):
        """Calculate fitness (number of satisfied clauses)."""
        self.fitness = formula.evaluate(self.assignment)
        return self.fitness
    
    def mutate(self, mutation_rate=0.1):
        """Randomly flip some variables."""
        for var in self.assignment:
            if random.random() < mutation_rate:
                self.assignment[var] = not self.assignment[var]
    
    def crossover(self, other, num_vars):
        """
        Create two offspring using single-point crossover.
        
        Args:
            other: Another Individual
            num_vars: Number of variables
            
        Returns:
            Tuple of two new Individual objects
        """
        crossover_point = random.randint(1, num_vars)
        
        child1_assignment = {}
        child2_assignment = {}
        
        for var in range(1, num_vars + 1):
            if var <= crossover_point:
                child1_assignment[var] = self.assignment[var]
                child2_assignment[var] = other.assignment[var]
            else:
                child1_assignment[var] = other.assignment[var]
                child2_assignment[var] = self.assignment[var]
        
        child1 = Individual(num_vars, child1_assignment)
        child2 = Individual(num_vars, child2_assignment)
        
        return child1, child2


def genetic_algorithm(formula, population_size=100, generations=100, 
                     mutation_rate=0.1, elite_size=10):
    """
    Genetic Algorithm for SAT solving.
    
    Args:
        formula: CNFFormula object
        population_size: Size of the population
        generations: Number of generations to evolve
        mutation_rate: Probability of mutation per variable
        elite_size: Number of top individuals to preserve
        
    Returns:
        Tuple (satisfiable, assignment, num_satisfied)
    """
    # Initialize population
    population = [Individual(formula.num_vars) for _ in range(population_size)]
    
    best_individual = None
    best_fitness = 0
    
    for generation in range(generations):
        # Evaluate fitness
        for individual in population:
            individual.evaluate_fitness(formula)
        
        # Sort by fitness
        population.sort(key=lambda x: x.fitness, reverse=True)
        
        # Track best solution
        if population[0].fitness > best_fitness:
            best_fitness = population[0].fitness
            best_individual = Individual(formula.num_vars, population[0].assignment)
        
        # Check if solved
        if population[0].fitness == formula.num_clauses:
            return True, population[0].assignment, population[0].fitness
        
        # Selection and reproduction
        new_population = []
        
        # Elitism: keep top individuals
        for i in range(min(elite_size, population_size)):
            new_population.append(Individual(formula.num_vars, population[i].assignment))
        
        # Create offspring
        while len(new_population) < population_size:
            # Tournament selection
            parent1 = tournament_selection(population, tournament_size=3)
            parent2 = tournament_selection(population, tournament_size=3)
            
            # Crossover
            child1, child2 = parent1.crossover(parent2, formula.num_vars)
            
            # Mutation
            child1.mutate(mutation_rate)
            child2.mutate(mutation_rate)
            
            new_population.append(child1)
            if len(new_population) < population_size:
                new_population.append(child2)
        
        population = new_population
    
    # Return best solution found
    if best_individual is None:
        best_individual = population[0]
        best_fitness = best_individual.fitness
    
    satisfiable = (best_fitness == formula.num_clauses)
    return satisfiable, best_individual.assignment, best_fitness


def tournament_selection(population, tournament_size=3):
    """
    Select an individual using tournament selection.
    
    Args:
        population: List of Individual objects
        tournament_size: Number of individuals in tournament
        
    Returns:
        Selected Individual
    """
    tournament = random.sample(population, min(tournament_size, len(population)))
    return max(tournament, key=lambda x: x.fitness)


def solve_genetic(formula, population_size=100, generations=100, 
                 mutation_rate=0.1, elite_size=10):
    """
    Solve a SAT formula using Genetic Algorithm.
    
    Args:
        formula: CNFFormula object
        population_size: Population size
        generations: Number of generations
        mutation_rate: Mutation rate
        elite_size: Elite size
        
    Returns:
        Tuple (satisfiable, assignment, num_satisfied)
    """
    return genetic_algorithm(formula, population_size, generations, 
                           mutation_rate, elite_size)
