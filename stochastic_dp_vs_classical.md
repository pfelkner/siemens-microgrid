# Stochastic DP vs. `classical_solver.py` — how they differ

Both solve the **same two-stage stochastic program** (peak cap = first-stage / shared;
dispatch = second-stage / per-scenario; objective = expected cost). They agree on the
*model*; they differ entirely in *how the second stage is represented and solved*. The
new file is `dp_solver_stochastic.py`; the reference is `classical_solver.py`.

> Self-check confirms equivalence at M=1: the stochastic DP reproduces the deterministic
> DP bit-for-bit, and every scenario schedule is feasible by construction.

---

## 1. The one structural idea that makes the DP cheap

In the MILP, *every* dispatch variable (`grid_in/out`, `bess_ch/dis`, `soc`, `served`,
plus the band/XOR binaries) is duplicated per scenario and solved **jointly** in one big
mixed-integer program. The only thing actually linking scenarios is the shared
`peak_import`.

The DP exploits exactly that: **fix the peak cap, and the M scenarios decouple
completely.** Each scenario becomes an independent shortest-path DP over SoC levels from
the same starting SoC. So the stochastic solve is just *M independent deterministic DPs*,
averaged, inside the cap sweep the deterministic DP already did.

This is the same decomposition the quantum approach relies on (`quantum_approach.md` §7):
M independent inner solves = **M× shots, not M× qubits**.

---

## 2. Side-by-side

| | `classical_solver.py` (MILP) | `dp_solver_stochastic.py` (DP) |
|---|---|---|
| **Scenarios coupled how** | one monolithic MILP with all M second-stages + shared `peak_import` | decoupled: M independent DPs per candidate cap |
| **First stage (peak)** | a continuous variable Gurobi optimizes | swept over a grid of `peak_levels` candidate caps |
| **Second-stage variables** | continuous `grid/bess/soc` + ~7 binaries/slot/scenario | a single discrete SoC-level choice per slot per scenario |
| **Constraints** | balance, SoC, bands, XOR, peak — all explicit, hard | balance/SoC/XOR **exact by construction**; bands via per-level power cap; peak via the cap |
| **Feasibility** | enforced by the MILP solver | **structural** — `validate()` returns clean for every scenario |
| **SoC** | continuous | discretized to `L` levels (approximation knob) |
| **Outage serving** | exact (Big-M indicator) | served iff `|PV+batt−load| ≤ serve_tol` (limited by SoC grid) |
| **Optimality** | global optimum (to MIP gap) | optimal *on the discrete grid*; → MILP as `L`, `peak_levels` → ∞ |
| **Dependency** | Gurobi (commercial license) | NumPy only |
| **Complexity** | NP-hard MIP; grows hard in M·T | `O(M · peak_levels · T · L²)` — linear in M, polynomial overall |
| **What it's for** | ground-truth optimum, full fidelity | quantum-ready reference; validates the GM-QAOA inner solves |

---

## 3. Concrete differences in the two-stage handling

**Demand charge (the coupling term).**
- MILP: `peak_import` is a free first-stage variable, constrained `peak_import ≥ grid_in[s][t]`
  for every scenario and slot; demand cost `= 15 · peak_import`, billed **once** (not
  probability-weighted). Gurobi drives it to the exact max import across scenarios.
- DP: the cap is *swept*, not solved. For each candidate cap, every scenario's inner DP is
  forced to keep `import ≤ cap`; the cap with the lowest `E[second-stage] + 15·cap` wins.
  The summary bills demand on the **realized** peak (max import across scenarios, ≤ cap),
  matching what the MILP's `peak_import` would settle on.

**Expectation.**
- Both weight the second-stage costs (energy, export, resiliency) by scenario probability.
- Both keep the first-stage demand charge **un**-weighted (it's a single shared decision).

**Non-anticipativity** (the requirement that the here-and-now decision can't depend on
which scenario materializes):
- MILP: enforced explicitly by sharing the single `peak_import` variable.
- DP: automatic — the swept cap is identical across all scenario inner solves by
  construction. Cleaner; nothing to enforce.

---

## 4. What the DP gives up (be honest)

- **Discretization.** SoC lives on `L` levels, not a continuum. Cost and serve-tolerance
  inherit that granularity. It converges to the MILP as the grid refines, but a coarse `L`
  is genuinely approximate.
- **Peak granularity.** The optimal cap can fall between grid points; `peak_levels`
  controls the gap (the realized peak is within one cap-step of optimal).
- **Single first-stage variable.** The clean decoupling holds *because* `peak` is the only
  shared decision. Add another here-and-now variable and the scenarios re-couple — the DP
  state would have to grow.
- **No band binaries.** SoC-band derating is applied as a per-level power cap on
  transitions, not as explicit one-hot band variables. Equivalent in effect for dispatch,
  but it's a reformulation, not the literal MILP constraint.

---

## 5. Why this file exists (the point)

It is the **classical ground truth for the stochastic quantum solver**. The quantum plan
runs one GM-QAOA inner solve per scenario per outer step and averages — exactly the
decomposition this DP implements. So this file:
1. proves the scenario decomposition is correct (M=1 matches the deterministic DP exactly),
2. produces the optimum the quantum solver must reproduce on small instances, and
3. shows the cost is linear in M (M× work, fixed state size) — the property that makes
   stochastic tractable on a quantum device at all.

Run: `python dp_solver_stochastic.py --selfcheck` (correctness) or
`python dp_solver_stochastic.py --data <csv> --scenarios 10` (full run).
