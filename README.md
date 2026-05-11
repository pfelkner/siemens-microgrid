# Siemens Microgrid Dispatch

Quantum computing practical — economic dispatch optimization for a New Mexico microgrid (Grid ↔ PV ↔ BESS ↔ Load). The project implements a classical MILP baseline and a QUBO/QAOA quantum reformulation over a shared synthetic dataset.

## Repository Structure

```
siemens-microgrid/
├── pv_data.py             # Data synthesis pipeline (PV sim + load + ToU + grid availability)
├── classical_solver.py    # MILP baseline (PuLP/CBC)
├── quantum_solver.py      # QUBO build + D-Wave / QAOA backends
├── compare.py             # Benchmark + plots between classical and quantum dispatch
├── doc/                   # Planning documents and source material
│   ├── One-pager-MicroGrid-Dispatch_v1.pdf
│   ├── Basic Microgrid Dispatch Optimization Implementation Plan.md
│   ├── Quantum Microgrid Dispatch Implementation Plan.md
│   └── Presentation Overview.md
├── README.md
└── .gitignore
```

## Files

- **`pv_data.py`** — Generates the shared input dataset `all_data.csv` at 15-min resolution for a 30-day billing period: physics-based PV simulation via `pvlib` + PVGIS TMY (Albuquerque), synthetic commercial load profile, three-tier PNM-style ToU tariff, and Bernoulli grid-availability draws.
- **`classical_solver.py`** — Reads `all_data.csv` and builds the MILP dispatch model in PuLP (CBC solver). Produces the reference optimal schedule and cost.
- **`quantum_solver.py`** — Builds the QUBO/Ising reformulation of the same dispatch problem (24-slot horizon, 4-bit power encoding) and solves it on either a D-Wave annealer (`dimod` / Leap hybrid) or QAOA in Qiskit.
- **`compare.py`** — Runs both solvers on the same reduced problem, computes the approximation ratio ρ = C_quantum / C_MILP, and renders stacked-bar dispatch plots side by side.
- **`doc/`** — All planning material: the original Siemens one-pager, the classical and quantum implementation plans, and the short presentation overview.

## Getting Started

### Setup (optional — recommended)

```bash
uv venv
source .venv/bin/activate
uv pip install numpy pandas pvlib
```

### Run

```bash
# 1. Synthesize the dataset
python pv_data.py --start-date 2025-06-01 --end-date 2025-06-30

# 2. Run the classical baseline
python classical_solver.py #TODO

# 3. Run the quantum solver
python quantum_solver.py #TODO

# 4. Compare
python compare.py #TODO
```
