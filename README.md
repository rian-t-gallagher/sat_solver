# SAT Solver Project - CS 463G Program 3

A Boolean Satisfiability (SAT) solver implementation for CS 463G, implementing multiple algorithms to find optimal truth assignments for CNF formulas.

## Algorithms Implemented
- **DPLL** (Complete algorithm with unit propagation and backtracking)
- **Heuristic algorithms** (WalkSAT, Genetic Algorithm, etc.)

## Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Run the solver
python3 run_solver.py --solver dpll --formula benchmarks/CNF\ Formulas/uf20-0156.cnf

# Run tests
python3 -m pytest tests/ -v
```

## Project Structure
- `src/parser/` - DIMACS CNF format parsers
- `src/solvers/` - SAT solving algorithms
- `benchmarks/` - Test CNF formulas (satisfiable and unsatisfiable)
- `experiments/` - Performance testing and comparison scripts
- `results/` - Output data and analysis plots

## Development Phases
See `roadmap.txt` for the 7-phase development plan.

## Assignment Details
This implements CS 463G Program 3 requirements:
- 3 algorithms (1+ complete, 2+ heuristic)
- Performance analysis with 10 trials per randomized algorithm
- Statistical analysis and visualization of results