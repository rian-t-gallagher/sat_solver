# Quick Reference Guide

## Files

- **sat_solver.py** - Main driver program with CLI interface
- **cnf_parser.py** - CNF formula parser (DIMACS format)
- **dpll.py** - DPLL complete algorithm implementation
- **walksat.py** - WalkSAT local search algorithm
- **genetic.py** - Genetic algorithm implementation
- **test_solver.py** - Test suite for all components
- **test_sat.cnf** - Simple satisfiable test case
- **test_unsat.cnf** - Simple unsatisfiable test case
- **test_complex.cnf** - Complex 3-SAT test case

## Quick Start

```bash
# Run all three algorithms
python3 sat_solver.py -f test_sat.cnf -a all

# Run just DPLL (complete algorithm)
python3 sat_solver.py -f test_complex.cnf -a dpll

# Run test suite
python3 test_solver.py
```

## Algorithm Comparison

| Algorithm | Type | Guarantee | Best For |
|-----------|------|-----------|----------|
| DPLL | Complete | Exact solution or proof of UNSAT | Small-medium instances, proving unsatisfiability |
| WalkSAT | Incomplete | None | Large satisfiable instances |
| Genetic | Incomplete | None | Exploration, approximate solutions |

## DIMACS Format Example

```
c Comments start with 'c'
c Variables are numbered 1, 2, 3, ...
c Positive numbers = variable, negative = NOT variable
c Each clause ends with 0
p cnf 3 3
1 2 0          # (x1 OR x2)
-1 3 0         # (NOT x1 OR x3)
-2 -3 0        # (NOT x2 OR NOT x3)
```

## Performance Tips

### DPLL
- Works best on small to medium problems (< 100 variables)
- Guarantees correctness
- Use for proving unsatisfiability

### WalkSAT
- Increase `--max-flips` for harder problems
- Increase `--max-tries` for more random restarts
- Tune `--walk-prob` (0.0-1.0): higher = more random, lower = more greedy
- Works well on large satisfiable instances

### Genetic Algorithm
- Increase `--pop-size` for better exploration
- Increase `--generations` for longer evolution
- Tune `--mutation-rate` (0.0-1.0): higher = more exploration

## Example Usage

```bash
# Quick test
python3 sat_solver.py -f test_sat.cnf -a dpll

# WalkSAT with aggressive parameters
python3 sat_solver.py -f test_complex.cnf -a walksat \
    --max-flips 50000 --max-tries 20 --walk-prob 0.3

# Genetic with large population
python3 sat_solver.py -f test_complex.cnf -a genetic \
    --pop-size 200 --generations 200 --mutation-rate 0.05

# All algorithms with custom parameters
python3 sat_solver.py -f test_complex.cnf -a all \
    --max-flips 20000 --pop-size 150 --generations 150
```

## Module Usage (Python)

```python
from cnf_parser import parse_dimacs_file, CNFFormula
from dpll import solve_dpll
from walksat import solve_walksat
from genetic import solve_genetic

# Load formula
formula = parse_dimacs_file('test_sat.cnf')

# Or create programmatically
clauses = [[1, 2], [-1, 3], [-2, -3]]
formula = CNFFormula(3, clauses)

# Solve with DPLL
satisfiable, assignment, num_satisfied = solve_dpll(formula)

# Solve with WalkSAT
satisfiable, assignment, num_satisfied = solve_walksat(
    formula, max_flips=10000, p=0.5, max_tries=10
)

# Solve with Genetic Algorithm
satisfiable, assignment, num_satisfied = solve_genetic(
    formula, population_size=100, generations=100
)

# Evaluate any assignment
assignment = {1: True, 2: False, 3: True}
num_satisfied = formula.evaluate(assignment)
is_solution = formula.is_satisfied(assignment)
```
