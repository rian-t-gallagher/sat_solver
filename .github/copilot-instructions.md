# SAT Solver Project - AI Coding Assistant Guidelines

## Project Overview
This is a CS 463G Program 3 assignment implementing SAT solvers using heuristic search techniques to find optimal truth assignments for Boolean formulas in CNF. The project requires implementing **three algorithms** with at least one complete algorithm (DPLL or Resolution).

## Algorithm Requirements

### Must Implement 3 Algorithms (Choose from):
- **DPLL** ⭐ (Complete algorithm - required to choose at least one)
- **Resolution** ⭐ (Complete algorithm - required to choose at least one)  
- **Genetic algorithms** (Heuristic)
- **Local search** (Heuristic)
- **Simulated annealing** (Heuristic)
- **GSAT** (Heuristic)
- **WalkSAT** (Heuristic)

### Performance Metrics
- **Fitness**: Number of satisfied clauses (max = total clauses)
- **CPU Time**: Runtime per formula/run
- **Randomized algorithms**: Run 10 times per formula for statistical analysis
- **Complete algorithms** (DPLL/Resolution): Single deterministic run

## Architecture & Key Components

### Core Structure
- **`src/parser/`**: CNF file parsing logic (implement DIMACS format parser)
- **`src/solvers/`**: SAT solving algorithms (DPLL, Resolution, GA, etc.)
- **`run_solver.py`**: Main CLI entry point (currently skeleton - needs arg parsing)
- **`experiments/`**: Performance testing and algorithm comparison scripts
- **`results/`**: Output directory for CSV data and matplotlib plots

### Data Organization
- **`data/CNF Formulas/`**: Standard test cases
  - `uf*` files: satisfiable 3-SAT formulas  
  - `uuf*` files: unsatisfiable 3-SAT formulas
- **`data/HARD CNF Formulas/`**: More challenging instances
  - `.cnf`: Standard DIMACS format with header (`p cnf <vars> <clauses>`)
  - `.rcnf`: Raw format without header (clauses only)
- **Assignment source**: PA3_benchmarks from Canvas (referenced in "SAT Solvers.docx")

### CNF Format Patterns
```
p cnf 20 91        # 20 variables, 91 clauses
2 20 5 0          # Clause: (x2 ∨ x20 ∨ x5)
-16 15 -19 0      # Clause: (¬x16 ∨ x15 ∨ ¬x19)
```

## Development Workflow

### Project Phases (from roadmap.txt)
1. **Phase 0**: Project Setup ✅ (Git repo, folder structure, CLI skeleton)
2. **Phase 1**: Formula Parsing and Loader (DIMACS parser, validation, tests)
3. **Phase 2**: Implement DPLL Solver (unit propagation, backtracking, heuristics)
4. **Phase 3**: Implement Heuristic Solvers (WalkSAT, Genetic Algorithm, etc.)
5. **Phase 4**: Experiment Harness (run_experiments.py, 10 trials, CSV logging)
6. **Phase 5**: Parameter Sweep & Tuning (grid search, optimization)
7. **Phase 6**: Data Analysis & Plotting (performance plots, comparison tables)
8. **Phase 7**: Report & Submission (report.pdf, learning outcomes)

### Dependencies & Setup
```bash
pip install -r requirements.txt  # pytest, pandas, matplotlib
python run_solver.py            # Currently just prints skeleton message
```

### Testing Strategy
- Use `pytest` for unit tests in `tests/`
- Test parsers against various CNF formats in `data/`
- Validate solver correctness on known SAT/UNSAT instances
- Performance benchmarking using `pandas` for data analysis

### Result Analysis
- Generate CSV files in `results/csv/` for solver performance metrics
- Create visualization plots in `results/plots/` using matplotlib
- Track metrics: solve time, decisions made, conflicts, backtracks

## Implementation Guidelines

### Parser Development
- Handle both `.cnf` (with DIMACS header) and `.rcnf` (raw clauses) formats
- Expect 3-SAT instances (all clauses have exactly 3 literals)
- Parse comments starting with 'c' and problem lines starting with 'p'

### Solver Implementation
- Start with basic DPLL algorithm in `src/solvers/dpll.py`
- Implement unit propagation and backtracking with variable selection heuristics
- Choose 2 heuristic algorithms (e.g., WalkSAT, Genetic Algorithm) for Phase 3
- Use deterministic seeds and configurable parameters for reproducible results

### CLI Design
Extend `run_solver.py` to support:
```bash
python run_solver.py --solver dpll --formula data/CNF\ Formulas/uf20-0156.cnf
python run_solver.py --solver walksat --benchmark --output results/csv/
```

### Experiment Workflow
Build `run_experiments.py` for automated testing:
- Run 10 trials per formula for randomized algorithms
- Log results to CSV with time, score, and parameters
- Grid search over key parameters in Phase 5
- Generate performance plots and comparison tables

### Naming Conventions
- Solver classes: `DPLLSolver`, `WalkSATSolver`, `GeneticSolver`  
- Test files: `test_parser.py`, `test_dpll.py`
- Result files: `solver_performance_YYYYMMDD.csv`

## Key Files to Reference
- **`roadmap.txt`**: Complete project roadmap with 7 development phases
- **`data/CNF Formulas/uf20-0156.cnf`**: Example satisfiable instance
- **`data/CNF Formulas/uuf50-01.cnf`**: Example unsatisfiable instance  
- **`requirements.txt`**: Core dependencies for data analysis workflow