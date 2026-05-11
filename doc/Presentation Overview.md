# Microgrid Dispatch — Plan Overview

## The Problem
Optimally dispatch a New Mexico microgrid (Grid ↔ PV ↔ BESS ↔ Load) over a 30-day billing period to **minimize energy + demand charges** while **maximizing resiliency revenue** during grid outages.

## Our Two-Track Plan

**Track 1 — Classical Baseline (MILP)**
- **Step 1 — Data synthesis:** generate 15-min profiles for 30 days — synthetic commercial load, 3-tier ToU tariff, Bernoulli grid availability → merged into `all_data.csv`
- **Step 2 (in parallel / next):** build a physics-based **PV simulation** via `pvlib` + PVGIS TMY (Albuquerque, tilt 20°) and plug its `p_kw` series into the same dataset
- Build PuLP/CBC model: decision vars (`Grid_Import/Export`, `BESS_Charge/Discharge`, `SoC`, `Max_Grid_Import`)
- Constraints: power balance, BESS dynamics (η = 0.9), demand-peak tracking
- Objective: `min Σ ToU·Import + DemandCharge·Peak − ResiliencyRevenue`
- Deliverable: optimized dispatch schedule + stacked-bar plot vs. ToU price curve

**Track 2 — Quantum Reformulation (QUBO)**
- Shrink to **T = 24 hourly slots**, 4-bit power encoding → ~384 logical qubits (tractable on D-Wave hybrid / QAOA)
- Binary-encode each power variable; reconstruct SoC from cumulative C/D (saves qubits)
- Constraints become **squared-residual penalties**: power balance, SoC bounds, import/export & charge/discharge XOR, resilience during outages
- Two interchangeable backends sharing the same BQM:
  - **D-Wave** quantum annealing (LeapHybridSampler or QPU + minor embedding)
  - **QAOA** in Qiskit (p = 3 layers, COBYLA optimizer)
- Validation: approximation ratio ρ = C_quantum / C_MILP; sweep encoding bits K ∈ {2…5}

---

## Outlook — Extensions Inspired by the One-Pager

The one-pager flags three "optional additions" that map naturally onto quantum-friendly extensions:

1. **Forecast uncertainty** — Replace the deterministic PV/load time series with **M Monte Carlo scenarios** and minimize the *expected* cost. Same QUBO structure with scenario-indexed penalties → stochastic dispatch.
2. **Non-linear BESS model** — SoC-dependent efficiency η(SoC) becomes a **piecewise-quadratic** term via indicator qubits per SoC band; a natural fit for QUBO (already quadratic) but awkward for MILP.
3. **Unobservable real SoC** — Treat SoC as a hidden state with a noise model; opens the door to a **rolling-horizon / MPC** formulation re-solved every few slots as new measurements arrive.

Two further quantum-algorithmic angles from the lecture:
- **HHL route** — Drop demand-charge + XOR constraints → strictly convex QP → solve KKT linear system in O(log N · κ²) via Quantum Phase Estimation.
- **Grover / Amplitude Amplification** — Frame feasibility as a search over the 2ⁿ schedule space for an O(√2ⁿ) speedup; oracle = cost-threshold + constraint checker.
