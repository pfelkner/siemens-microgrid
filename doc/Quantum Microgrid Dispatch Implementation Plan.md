
This document outlines how to translate the classical MILP dispatch model from `Microgrid Dispatch Optimization Basic Implementation Plan.md` into a **quantum solver** using the concepts taught in lecture. The classical PuLP/CBC pipeline is replaced by a QUBO/Ising formulation that runs on a quantum annealer (D-Wave) or a gate-model QAOA circuit (Qiskit).

The synthetic data pipeline from **Point 1** (`pv_data.py`, producing `all_data.csv`) is reused unchanged — only the optimizer changes.

---

## Why Quantum?

The classical MILP scales as $O(2^n)$ in the worst case for $n$ binary/integer variables. The dispatch problem at full resolution (2880 slots × 4 power variables × ≥8-bit encoding ≈ **92 000 binary variables**) is intractable for current QPUs but conceptually well-suited to:

- **Adiabatic / Quantum Annealing** — large native problem sizes (5 000+ qubits on D-Wave Advantage 2), purpose-built for QUBO.
- **QAOA** — universal gate-model fallback when annealers are not available, gives an approximation guarantee with polynomial circuit depth.

For a tractable proof of concept we **shrink the horizon to $T=24$ hourly slots** (one representative day) and use **4-bit power encoding**. This stays under 500 logical qubits and fits a D-Wave hybrid solver comfortably.

---

## Point 1: Data Preparation

**Unchanged** — see Phase 1.1 and 1.2 of the classical plan. The quantum solver consumes the same `all_data.csv` (`p_kw`, `load_kw`, `tou_usd_kwh`, `grid_available`). The only addition is a downsampler that aggregates the 15-min profile into the chosen $T$-slot horizon.

```python
df = pd.read_csv("all_data.csv", parse_dates=["timestamp"])
df_hourly = df.set_index("timestamp").resample("1h").mean().head(24)
```

---

## Point 2: QUBO/Ising Reformulation

This is the heart of the quantum approach. Every continuous decision variable must be replaced by a sum of binary qubits, every inequality by a penalty term, and the linear objective by a quadratic cost matrix $Q$ such that the dispatch optimum is the ground state of $H = x^\top Q\, x$.

### Phase 2.1: Time Discretization & Sizing

| Parameter | Symbol | Value |
|-----------|--------|-------|
| Horizon | $T$ | 24 slots (1 h each) |
| Encoding bits per power variable | $K$ | 4 (16 levels) |
| Power resolution | $\Delta P$ | $P_{\max}/(2^K - 1) = 250/15 \approx 16.7$ kW |
| Logical qubits per slot | — | $4K = 16$ |
| **Total logical qubits** | $n$ | $T \times 4K = 384$ |

The four power variables per slot are `Grid_Import`, `Grid_Export`, `BESS_Charge`, `BESS_Discharge`. The `BESS_SoC` is **not** encoded as an independent variable — it is reconstructed from the cumulative charge/discharge sum, which keeps the qubit count linear in $T$ and turns the BESS dynamics constraint into a derived expression rather than an equality.

### Phase 2.2: Binary Encoding of Decision Variables

Each continuous power $P \in [0, P_{\max}]$ is encoded with **power-of-two binary encoding**:

$$P(t) = \Delta P \cdot \sum_{k=0}^{K-1} 2^{k}\, x_{P, t, k}, \qquad x_{P, t, k} \in \{0, 1\}$$

**Alternative — one-hot encoding** is mentioned in the lecture but discarded here: it uses $2^K$ qubits per variable instead of $K$, blowing the qubit budget past hardware limits, and requires an extra "exactly-one" penalty per variable.

### Phase 2.3: Penalty-Term Constraints

Every MILP constraint becomes a squared-residual penalty added to the objective with a large multiplier $\lambda \gg \max(c_\text{ToU})$:

**1. Power balance** (was: equality in MILP)
$$H_\text{bal} = \lambda_\text{bal} \sum_t \big( P_\text{load}(t) - P_\text{PV}(t) - I(t) + E(t) - D(t) + C(t) \big)^2$$

**2. BESS SoC bounds** (replaces the SoC variable's lower/upper bounds)
$$\text{SoC}(t) = \text{SoC}_0 + \sum_{\tau \leq t}\!\big(\eta C(\tau) - D(\tau)/\eta\big)\,\Delta t$$
$$H_\text{soc} = \lambda_\text{soc} \sum_t \big[\max(0,\,-\text{SoC}(t))^2 + \max(0,\,\text{SoC}(t) - \text{Cap})^2\big]$$
The $\max$ is linearized by introducing slack qubits, or — simpler — replaced by symmetric quadratic penalty around the midpoint of the feasible band.

**3. Mutual exclusion** (import XOR export, charge XOR discharge — avoids the optimizer "buying and selling" simultaneously to game ToU)
$$H_\text{xor} = \lambda_\text{xor} \sum_t \big( I(t)\,E(t) + C(t)\,D(t) \big)$$

**4. Resilience during outages** (when `grid_available(t) = 0`)
$$H_\text{res} = \lambda_\text{res} \sum_{t: g(t)=0} \big( I(t)^2 + E(t)^2 \big)$$

**5. Demand charge** — the $\max_t I(t)$ term is non-quadratic. Two options:
- **A.** Introduce an auxiliary $K$-bit binary $P_\text{peak}$ and penalize $\sum_t \max(0, I(t) - P_\text{peak})^2$. Adds $K$ qubits.
- **B.** Pre-fix a peak budget from the classical LP relaxation and impose it as a hard penalty. Loses tightness but simplifies the QUBO.

### Phase 2.4: Objective in QUBO Form

$$H_\text{cost} = \sum_t c_\text{ToU}(t)\, I(t) - \sum_t c_\text{export}(t)\, E(t) + c_\text{demand}\, P_\text{peak} - r_\text{res} \sum_{t: g(t)=0} \mathbb{1}[\text{islanded}]$$

**Parameter values** (identical to the classical MILP so that $\rho = C_\text{quantum}/C_\text{classical}$ is meaningful):
- $c_\text{export} = 0.05$ \$/kWh (export tariff — equals the off-peak ToU rate; battery→grid arbitrage does not pay due to 10% round-trip loss)
- $r_\text{res} = 225$ \$/slot ($15/min × 15 min$; band: 150–300 \$/slot for 10–20 \$/min)
- $c_\text{demand} = 15$ \$/kW (billing-period peak import charge)

Combine:
$$H = H_\text{cost} + H_\text{bal} + H_\text{soc} + H_\text{xor} + H_\text{res}$$

After expanding the binary encodings, $H$ is a quadratic polynomial in $x \in \{0,1\}^n$ — a QUBO. Convert to Ising via $s_i = 2x_i - 1$ if the solver requires it.

### Phase 2.5: Tuning the Penalty Weights $\lambda$

Penalties must dominate the cost term but not so much that the solver wastes precision on already-feasible regions. Heuristic: $\lambda \approx 10 \cdot \max(c_\text{ToU}) \cdot P_\text{max}^2$. Sweep $\lambda$ over $\{10^1, 10^2, 10^3\}$ and pick the smallest value at which constraints are satisfied for $>$95% of samples.

---

## Point 3: Quantum Solver Implementation

Two interchangeable backends are supported. Both consume the same `BinaryQuadraticModel` (BQM) object — so the QUBO build code is shared.

### Phase 3.1: Approach A — Quantum Annealing (D-Wave)

```python
import dimod
from dwave.system import LeapHybridSampler, DWaveSampler, EmbeddingComposite

bqm = dimod.BinaryQuadraticModel(linear, quadratic, offset=0.0, vartype="BINARY")

# Hybrid (large problems): solves up to ~1e6 variables, uses CPU+QPU
sampler = LeapHybridSampler()
sampleset = sampler.sample(bqm, time_limit=10)

# Pure QPU (small): minor-embed onto Pegasus, sample 1000x
# sampler = EmbeddingComposite(DWaveSampler())
# sampleset = sampler.sample(bqm, num_reads=1000, chain_strength=2 * lambda_bal)

best = sampleset.first.sample
energy = sampleset.first.energy
```

**Key operations referenced from the lecture:**
- **Minor embedding** maps the logical QUBO graph onto Pegasus connectivity (degree-15). `EmbeddingComposite` does this automatically; `chain_strength` ties chained physical qubits together — set to ~2× the largest QUBO coefficient.
- **Annealing schedule** evolves the Hamiltonian from a trivial $H_\text{init} = -\sum_i \sigma_i^x$ to the problem Hamiltonian $H$ over $\sim 20$ μs.

### Phase 3.2: Approach B — QAOA (Qiskit)

```python
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import Sampler

qp = QuadraticProgram()
# ... add binary variables and quadratic objective from the same QUBO dict
qaoa = QAOA(sampler=Sampler(), optimizer=COBYLA(maxiter=200), reps=3)
result = MinimumEigenOptimizer(qaoa).solve(qp)
```

**Reference points from the lecture:**
- QAOA is a **variational quantum circuit** with $2p$ parameters $(\gamma_i, \beta_i)$ for $p$ layers (`reps=3` here).
- The classical optimizer (`COBYLA`) plays the role of gradient descent described in the slides — adjusting $(\gamma, \beta)$ between shots to minimize the measured expectation value.
- For $p \to \infty$ QAOA recovers the adiabatic theorem and converges to the exact ground state — a direct link to the QA approach in 3.1.

### Phase 3.3: Decoding the Solution

```python
def decode_power(sample, var_name, t, K, delta_P):
    return sum(sample[f"{var_name}_{t}_{k}"] * (2 ** k) for k in range(K)) * delta_P

schedule = pd.DataFrame({
    "Grid_Import": [decode_power(best, "I", t, K, dP) for t in range(T)],
    "Grid_Export": [decode_power(best, "E", t, K, dP) for t in range(T)],
    "BESS_Charge": [decode_power(best, "C", t, K, dP) for t in range(T)],
    "BESS_Discharge": [decode_power(best, "D", t, K, dP) for t in range(T)],
})
schedule["BESS_SoC"] = SoC0 + (eta * schedule.BESS_Charge - schedule.BESS_Discharge / eta).cumsum()
```

---

## Point 4: Validation Against the Classical Baseline

The classical PuLP solution from **Phase 2 of the basic plan** is the ground-truth reference.

1. Run both solvers on the same 24-slot reduced problem.
2. Report the **approximation ratio** $\rho = C_\text{quantum} / C_\text{MILP}$. QAOA at $p=3$ typically achieves $\rho \in [1.05, 1.2]$ on small QUBOs; D-Wave hybrid usually reaches $\rho \leq 1.01$.
3. Sweep encoding precision $K \in \{2, 3, 4, 5\}$ to quantify the discretization error vs. qubit-count trade-off.
4. Plot the QUBO dispatch alongside the MILP dispatch (stacked-bar chart from Phase 2, Step 6 of the basic plan).

---

## Point 5: Optional Extensions from the Lecture

- **Convex / HHL route** — If demand-charge and XOR constraints are dropped, the relaxed problem is a strictly convex quadratic program. Its KKT conditions reduce to a linear system $A\mathbf{x} = \mathbf{b}$ that **HHL** (Quantum Phase Estimation + eigenvalue inversion) can solve in $O(\log N \cdot \kappa^2)$ where $\kappa$ is the condition number of $A$.
- **Grover / Amplitude Amplification** — Frame dispatch feasibility as a search over the $2^n$ schedule space with an oracle that flips the sign of feasible-and-better-than-threshold schedules. Gives $O(\sqrt{2^n})$ speedup over brute force; the oracle construction (cost-threshold comparator + constraint checker) is the engineering bottleneck.
- **Uncertainty in forecasts** (mentioned in the one-pager): replace the single PV/load time series with $M$ Monte Carlo scenarios and minimize the **expected** cost — same QUBO structure with scenario-indexed penalties.
- **Non-linear BESS model** — efficiency $\eta(\text{SoC})$ becomes a piecewise-quadratic term in the QUBO, expressible via additional indicator qubits per SoC band.

---

## Suggested Repository Layout

```
siemens/
├── pv_data.py                                       # unchanged, Point 1
├── classical_solver.py                              # PuLP MILP from basic plan
├── quantum_solver.py                                # QUBO build + D-Wave/QAOA runner
├── compare.py                                       # benchmark + plots
├── Microgrid Dispatch Optimization Basic Implementation Plan.md
└── Quantum Microgrid Dispatch Implementation Plan.md  # this file
```

The classical and quantum solvers share the same `all_data.csv` interface and the same output schema (`schedule.csv`), so `compare.py` is solver-agnostic.
