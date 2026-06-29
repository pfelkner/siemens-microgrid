# Quantum Approach: From MILP/QUBO/DP to a GM-QAOA + Implicit-Differentiation Solver

Synthesis of the three "papers for new approach" (`papers for new approach/REFERENCE.md`)
against our three solver attempts (`classical_solver.py`, `qubo_model.py`, `dp_solver.py`).
Goal: combine the papers into a coherent, feasible quantum implementation of the
Siemens microgrid dispatch problem.

**Decisions (locked):**
- Outer-gradient plan: **build option 1 (CR-ID biased) first, then add option 2
  (unbiased implicit diff) and compare** — the comparison is itself a result.
- Endpoint: **research demonstrator / paper first** (simulator, tiny `T·L`), then
  **push to larger instances** once it works and is documented.
- Stochastic scenarios: **in scope as Phase 2** (see §7) — viable, and a better fit
  for the CR-ID half than deterministic.

Related memory: `[[gm-qaoa-dp-reformulation]]`.

---

## 1. Background term: "VQE on a diagonal QUBO"

- **QUBO** (`qubo_model.py`): a cost in 0/1 variables (sums + products). On qubits it
  becomes a **cost Hamiltonian** `H_C` containing only `Z`/`ZZ` terms → **diagonal** in
  the computational basis. One Z-basis measurement of all qubits yields the cost; all
  terms commute.
- **VQE** (Variational Quantum Eigensolver): hybrid loop. A parametrized circuit `U(θ)`
  prepares `|ψ(θ)⟩`, you measure `⟨ψ|H_C|ψ⟩`, a classical optimizer lowers it. At
  convergence `|ψ⟩` ≈ lowest-cost bitstring.
- **"VQE on a diagonal QUBO"** = minimize our microgrid QUBO with a generic
  (hardware-efficient) ansatz.
- **Why the papers care:** in VQE the state depends only on `θ`, *not* on the cost
  coefficients (prices). Nudge a price → state doesn't move (at fixed `θ`) → the
  cost-gradient is "price-derivative × already-measured bit-correlations" = **free**.
  QAOA/GM-QAOA bake `H_C` *into* the circuit, so changing a price *moves the state* →
  the free gradient picks up a **bias term**. This distinction is the spine of the whole
  design (see §5).

---

## 2. The three approaches, side by side

| | `classical_solver.py` (MILP) | `qubo_model.py` (BQM/anneal) | `dp_solver.py` (DP) |
|---|---|---|---|
| **Role** | Ground-truth reference | First quantum-ish attempt | Quantum-*ready* reformulation |
| **Free variables** | grid_in/out, ch, dis, soc (continuous) + ~7 binaries/slot + peak | all binary-encoded (≈4 bits each) + slacks | **only the SoC level per slot** (L levels) |
| **Balance (B.1)** | hard | squared penalty — **leaks** | exact, by construction |
| **SoC dynamics (B.2)** | hard | squared penalty — **leaks** | exact (SoC = running sum) |
| **XOR ch/dis & imp/exp (B.6)** | Big-M binaries | product penalty — leaky | exact (one signed battery var, one signed grid residual) |
| **SoC-band derating (B.5)** | indicator constraints | quadratic link — leaky | exact per level; transitions soft |
| **Demand charge (global max)** | first-stage `peak` var | encoded `peak` + penalty | **cap sweep** (outer loop) |
| **Stochastic scenarios** | yes (two-stage) | no | no (but decomposes cleanly — §7) |
| **Feasibility guarantee** | yes | **no** (its own `evaluate()` measures residuals; `repair_peak`) | **yes** (`validate()` checks ≈ 0 by design) |
| **Size** | n/a | ~23–31 vars/slot × bits → hundreds even for tiny windows | `T·L` one-hot qubits |

Decisive row = feasibility. The QUBO leaks the **fundamental** constraints (power
balance, SoC). The DP makes those exact and leaves exactly one real decision —
*which SoC level at each slot* — with only a small soft residual on big-jump derating.
That collapse is what makes a quantum solver plausible.

---

## 3. Key insight: the DP reformulation is the quantum-enabling move

The DP is the bridge into the GM-QAOA paper's **Category 2** (poly-size feasible-state
prep exists). Mapping:

**Encode each slot's SoC level as a one-hot register of `L` qubits** (`x_{t,l}=1` iff
SoC at slot `t` is on level `l`). Then:

- **Feasible set `F` = "exactly one level per slot"** = product of `T` Hamming-weight-1
  constraints. The equal superposition over `F` is `⊗_t W_L` — a tensor product of
  per-slot **W-states** (Dicke `D_1^L`). GM-QAOA Theorem 3 (ref [9]) prepares each in
  `O(L)` gates / depth. So `U_S` is trivial **and factorizes per slot**.
- **Grover mixer** `U_M(β) = ⊗_t e^{-iβ|W_t⟩⟨W_t|}` keeps you inside `F` exactly,
  forever. This is the constraint the QUBO could never hold — here it's structural,
  **it cannot leak**.
- **Cost Hamiltonian `H_C`** carries the DP's stage costs: for each adjacent pair
  `(level_{t-1}, level_t)`, the implied ch/dis → grid residual →
  `energy − export − resiliency`. These are **2-local `ZZ`-type diagonal terms between
  adjacent one-hot blocks** — diagonal, commuting, one Z-basis shot batch. Exactly the
  "diagonal cost Hamiltonian" CR-ID requires.

**Leak surface vs. the QUBO:** nothing fundamental. Balance, SoC dynamics, both XORs are
exact (inherited from the DP's signed-trajectory trick). The *only* soft residual is
derating on large SoC jumps — transitions exceeding a per-level power cap get a large
diagonal penalty rather than being structurally forbidden (forbidding would couple slots
and break the clean product `U_S`). Tiny, compared to leaking power balance.

---

## 4. How the three papers combine

```
                         OUTER LOOP  (parametric / bilevel)
   λ = {peak cap, ToU forecast, resiliency value, export tariff, risk weight}
        │
        │  outer gradient dF/dλ  ──►  IMPLICIT DIFFERENTIATION
        │                              • CR-ID (manuscript): reuse correlators, ~free
        │                              • Ahmed et al. general IFT: unbiased fallback
        ▼
   ┌─────────────────────────────────────────────────────────────┐
   │  INNER SOLVER  =  GM-QAOA  (Grover-mixer paper)              │
   │  U_S = ⊗_t W_L        feasible subspace = one level/slot     │
   │  U_M = Grover mixer   stays in F  (no constraint leaks)      │
   │  H_C(λ) diagonal      per-transition dispatch cost + derate  │
   │  measures: P(level l @ t), P(level l→l' @ t)  ◄── correlators│
   └─────────────────────────────────────────────────────────────┘
```

- **GM-QAOA paper** → inner *feasibility* skeleton. The DP reformulation is what lands
  the microgrid problem in Category 2; without it the raw dispatch is Category-3-ish
  (no clean feasible-superposition prep), consistent with `[[gm-qaoa-dp-reformulation]]`.
- **CR-ID manuscript** → the *outer loop*. Microgrid costs depend on tunable external
  factors (price forecast, resiliency valuation, risk aversion, demand-charge peak cap) —
  literally the paper's motivating scenario. Flagship hook: the DP's brute-force peak-cap
  sweep (`for cap in caps`) becomes a **gradient step on `cap`** using per-transition
  import probabilities already measured. Our **correlators = transition occupation
  probabilities** `P(l→l' @ t)`, the exact analog of CR-ID's edge-cut probabilities `p_e`.
  Cost `= Σ cost(l,l';λ)·P(l→l')` so `∂J/∂λ = Σ (∂cost/∂λ)·P(l→l')` — free from the
  inner energy-estimation shots.
- **Ahmed et al. (implicit diff)** → general engine under CR-ID, and the unbiased
  fallback when CR-ID's cheap gradient is biased (§5). Computes the outer gradient through
  any variational solution at stationarity via a Hessian-vector solve.

---

## 5. Central tension (the crux)

CR-ID's headline is **architecture dependence**, and it cuts against the design:

- **GM-QAOA is QAOA-family.** Its phase separator `e^{-iγ H_C(λ)}` puts `λ` into the
  **state preparation**, not just the objective. By manuscript Eq. 33 the free
  correlator-reuse gradient then carries a **state-dependence term** → a **cost–bias
  trade-off**: cheap gradient is *biased*, removing the bias costs extra circuits.
- The **clean, free, exact** reuse (Eq. 28) is the **VQE** case — but plain VQE uses a
  generic ansatz that does **not** respect the one-hot subspace, so it loses
  feasibility-by-construction and slides back toward QUBO-style leaks.

So **feasibility (GM-QAOA) and free-exact outer gradients (VQE/CR-ID) pull in opposite
directions.** Three honest resolutions (increasing ambition):

1. **GM-QAOA inner + CR-ID biased outer.** Take the feasibility win; accept a
   biased-but-cheap outer gradient. Lowest-effort working system. **← build first.**
2. **GM-QAOA inner + Ahmed general implicit diff outer.** Feasibility *and* unbiased
   outer gradient, at the cost of a Hessian-VJP solve per outer step. The "correct"
   combination; uses all three papers as intended. **← build second, compare to 1.**
3. **Restrict `λ` to additive-offset coefficients** (e.g. add `c_dem·cap` *classically*
   outside the quantum solve instead of as an in-`H_C` hard cap). For those, even
   GM-QAOA's reuse is exact because `λ` never touches the state. Prefer outer knobs that
   live here where possible.

Measuring the GM-QAOA state-dependence bias (1 vs 2) directly extends the manuscript,
which only studied plain QAOA/VQE — a publishable contribution.

---

## 6. Feasibility & scale (honest)

- **Qubits = `T · L`.** Tiny demo: `T=6, L=8` = 48 qubits. Hourly day `T=24, L=16` = 384
  → NISQ-prohibitive. So: **rolling-window MPC** (as `dwave_annealer.py` already does),
  small `T,L`, **simulator first**. This is a research demonstrator, not a production
  dispatcher — the MILP and DP already solve the real instance optimally and fast. The
  quantum value is methodological.
- **Out of scope (v1):** full-horizon solves, exact derating (kept as small soft
  penalty). Stochastic = Phase 2 (§7).
- **Genuinely strong:** all three preconditions the papers need (exact feasibility for
  the fundamental constraints, trivial factorized state prep, diagonal cost Hamiltonian)
  are satisfied by the DP reformulation. Real fit, not forced.

---

## 7. Stochastic scenarios — evaluation

**Verdict: viable, and a better fit for the CR-ID half than deterministic.**

In `classical_solver.py` the *only* cross-scenario coupling is the first-stage
`peak_import` (and the demand charge it drives); all dispatch variables are second-stage,
independent per scenario given the data realization; objective = expected cost. The DP
already sweeps the peak cap as its outer variable — the **same axis** the two-stage
program couples on. So the decomposition is exact:

- Fix first-stage `λ = cap`. The `M` scenarios **separate completely** → `M` independent
  inner solves (GM-QAOA circuits), one per `(pv, load, tou, outage)` realization, each
  with `import ≤ cap`.
- `F(cap) = (1/M) Σ_s F_s(cap) + c_dem·cap`; outer loop optimizes `cap`.

Why this is a strong fit, not a bolt-on:

1. **M× shots, not M× qubits.** Scenarios are separate circuits run sequentially. Qubit
   budget stays `T·L`. (This is why stochastic is tractable here but hopeless for the
   QUBO, which would need one `M·T·bits` entangled model.)
2. **CR-ID's advantage grows.** Naive finite-difference outer tuning now costs
   `M × M_probes × N_inner` — the multiplicative overhead, multiplied again by `M`.
   CR-ID sidesteps it: `dF/dλ = (1/M) Σ_s dF_s/dλ`, each term from correlator-reuse on a
   solve run anyway. Mirrors the manuscript's >14% multi-dimensional result.
3. **Risk-aversion as outer `λ`** is the manuscript's literal example. Expectation = a
   linear average (trivial gradient); CVaR/variance = non-smooth (subgradient /
   scenario-reweighting) but assembled from the same per-scenario correlators.

**Caveats:** non-anticipativity is automatic — the shared `cap` is identical across
scenario solves by construction (it's the one outer `λ`), cleaner than the MILP's
explicit coupling constraints. Clean separation relies on `peak` being the *only*
first-stage variable; adding another shared here-and-now decision re-couples scenarios.

**Plan:** Phase 2, after the deterministic demonstrator works. Amplifies the CR-ID story
and aligns with the push-to-larger / paper goal.

---

## 8. Roadmap

1. **Encoder** (fresh, from `dp_solver.py`'s `build_stage_costs`; reuse none of
   `qubo_model.py`): emit diagonal `H_C` as `ZZ` terms over adjacent one-hot blocks,
   plus `U_S = ⊗_t W_L`. Validate on simulator that **every** sampled bitstring decodes
   to a feasible schedule (balance/SoC ≈ 0) — the thing the QUBO can't promise.
2. **Inner GM-QAOA** at depth `p=1–2`, statevector simulator, small `T,L`; confirm it
   recovers the DP optimum.
3. **Outer loop, option 1 (CR-ID biased):** make demand-charge cap (or resiliency rate)
   the `λ`; CR-ID gradient descent reusing transition correlators; compare to the DP's
   brute cap sweep.
4. **Outer loop, option 2 (unbiased):** swap in Ahmed-style implicit diff; **measure the
   GM-QAOA state-dependence bias** vs option 1. Novel result.
5. **Phase 2:** stochastic (§7) — `M` independent inner solves per outer step, expected /
   risk-weighted objective; then push `T`, `L`, and toward hardware.

---

## 9. Open decisions for later

- Which concrete outer `λ` for the first demo: peak cap (state-dependent, the flagship
  bilevel hook) vs resiliency/export rate (cleaner, closer to additive-offset). Cap is
  the more compelling story; rate is the cleaner first test.
- `L` (SoC levels) vs qubit budget vs the `serve_tol` outage-serving resolution the DP
  already lives with.
- Risk measure for Phase 2 stochastic: expectation (trivial) vs CVaR (non-smooth outer).
