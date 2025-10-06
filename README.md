# SAT Solver

A comprehensive Boolean satisfiability (SAT) solver implementing three different algorithms for finding near-optimal truth assignments for Boolean formulas in conjunctive normal form (CNF).

## Overview

This project applies heuristic search methods to solve SAT problems. Fitness equals the number of satisfied clauses, up to c total clauses. If a formula is unsatisfiable, not all c clauses can be satisfied simultaneously.

## Implemented Algorithms

### 1. DPLL (Complete Algorithm)
The Davis-Putnam-Logemann-Loveland (DPLL) algorithm is a complete SAT solving algorithm that guarantees finding a solution if one exists, or proving unsatisfiability otherwise.

**Features:**
- Unit propagation
- Pure literal elimination
- Backtracking search
- Guaranteed to find solution or prove unsatisfiability

### 2. WalkSAT (Local Search)
WalkSAT is an incomplete local search algorithm that uses random walks to escape local optima.

**Features:**
- Random restarts
- Mix of greedy and random moves
- Fast for satisfiable instances
- May not prove unsatisfiability

### 3. Genetic Algorithm (Evolutionary)
An evolutionary optimization approach that evolves a population of candidate solutions.

**Features:**
- Tournament selection
- Single-point crossover
- Mutation operators
- Elitism to preserve best solutions

## Installation

No external dependencies required - uses only Python standard library.

```bash
git clone https://github.com/rian-t-gallagher/sat_solver.git
cd sat_solver
```

## Usage

### Basic Usage

Run all three algorithms on a CNF formula:
```bash
python3 sat_solver.py -f formula.cnf -a all
```

Run a specific algorithm:
```bash
python3 sat_solver.py -f formula.cnf -a dpll
python3 sat_solver.py -f formula.cnf -a walksat
python3 sat_solver.py -f formula.cnf -a genetic
```

### Command Line Options

```
-f, --file FILE           CNF formula file in DIMACS format (required)
-a, --algorithm ALGO      Algorithm to use: dpll, walksat, genetic, all (default: all)

WalkSAT options:
--max-flips N            Maximum flips per try (default: 10000)
--max-tries N            Maximum tries (default: 10)
--walk-prob P            Random walk probability (default: 0.5)

Genetic Algorithm options:
--pop-size N             Population size (default: 100)
--generations N          Number of generations (default: 100)
--mutation-rate P        Mutation rate (default: 0.1)
```

### Examples

```bash
# Run DPLL only
python3 sat_solver.py -f test_sat.cnf -a dpll

# Run WalkSAT with custom parameters
python3 sat_solver.py -f test_complex.cnf -a walksat --max-flips 50000 --walk-prob 0.3

# Run Genetic Algorithm with larger population
python3 sat_solver.py -f test_complex.cnf -a genetic --pop-size 200 --generations 200
```

## Input Format

The solver accepts CNF formulas in DIMACS format:

```
c This is a comment
c Variables are numbered 1, 2, 3, ...
c Each clause ends with 0
p cnf <num_variables> <num_clauses>
<literal1> <literal2> ... 0
<literal1> <literal2> ... 0
...
```

Example (test_sat.cnf):
```
c (x1 OR x2) AND (NOT x1 OR x3) AND (NOT x2 OR NOT x3)
p cnf 3 3
1 2 0
-1 3 0
-2 -3 0
```

## Test Files

Three test files are included:

- **test_sat.cnf**: Simple satisfiable formula (3 variables, 3 clauses)
- **test_unsat.cnf**: Simple unsatisfiable formula (1 variable, 2 clauses)
- **test_complex.cnf**: More complex satisfiable 3-SAT formula (5 variables, 10 clauses)

## Output

The solver outputs:
- Whether the formula is satisfiable
- Number of satisfied clauses out of total
- Variable assignment (for satisfiable formulas or best effort)

Example output:
```
============================================================
Algorithm: DPLL
============================================================
Satisfiable: True
Satisfied clauses: 10/10

Assignment:
  x1=T x2=F x3=T x4=F x5=T 
============================================================
```

## Algorithm Details

### DPLL
- **Type**: Complete (exact)
- **Best for**: Proving unsatisfiability, small to medium instances
- **Time complexity**: Exponential worst-case, often efficient in practice
- **Guarantees**: Always correct

### WalkSAT
- **Type**: Incomplete (heuristic)
- **Best for**: Large satisfiable instances
- **Time complexity**: Linear per iteration
- **Guarantees**: Finds solution with high probability if satisfiable

### Genetic Algorithm
- **Type**: Incomplete (metaheuristic)
- **Best for**: Exploring solution space, approximate solutions
- **Time complexity**: Linear per generation
- **Guarantees**: Improves over time, no optimality guarantee

## License

This project is part of an academic assignment on heuristic search methods for SAT solving.