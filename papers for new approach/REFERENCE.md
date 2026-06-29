# Papers for New Approach — Reference

Three papers. The first two are the *machinery*; the third (`internal_Manuscript`) is
the *target paper* — it combines them into a method (CR-ID) and is the one whose
approach we're evaluating/extending. Read it as the anchor; the other two are its
foundations.

---

## 1. `internal_Manuscript.PDF` — CR-ID (the new approach) ★ANCHOR

**Rohe, Baumann, Harjes Ruiloba, Zorn, Stein, Linnhoff-Popien (LMU Munich / Aqarios).**
Frontiers, *In review*, submitted 27 May 2026. Title: *Implicit Differentiation for
Measurement-Efficient Bilevel Quantum-Classical Optimization.*

### The problem it solves
Standard VQAs treat the cost Hamiltonian as **fixed**. Real problems have cost
coefficients that depend on an external knob λ (demand forecast, risk-aversion,
time-of-day). Treating λ as a decision variable turns the VQA into a **bilevel**
problem:
- **Outer loop:** search over λ to maximize the value function `F(λ) = max_φ J(φ,λ)`.
- **Inner loop:** for each λ, run a full VQA solve over circuit params φ.

Naive outer gradient = derivative-free probing (finite differences / SPSA), which
needs `M≈3` *full inner solves* per outer step → multiplicative measurement blow-up.
Cost ≈ `M · N_inner` (Eq. 24). That's the pain point.

### The trick: Correlator-Reuse Implicit Differentiation (CR-ID)
Core idea (Sec. V), built on the **envelope theorem** (ref [14] Milgrom-Segal):
at inner stationarity `∇_φ J(φ*,λ)=0`, the total derivative collapses to just the
**partial** derivative:

> `dF/dλ = ∂J/∂λ` evaluated at φ*   (Eq. 27)

For Max-Cut, `J(φ,λ) = Σ_e w_e(λ) p_e(φ)` so

> `∂J/∂λ = Σ_e (dw_e/dλ) · p_e(φ)`   (Eq. 28)

The `p_e` (edge-cut probabilities) are **already measured during inner energy
estimation** — all edge correlators come from one Z-basis shot batch (Max-Cut H_C
is diagonal, all terms commute). So the outer gradient costs **~zero extra
measurements** (Eq. 30). `dw_e/dλ` is classical (white-box from the response model,
or a classical finite-difference on the weight formula — still no quantum cost).

### Architecture dependence (the nuanced result — Sec. V-E)
- **VQE: clean reuse.** State ρ(θ) depends only on circuit params θ, *not* on λ.
  So `∂_λ ρ = 0` and Eq. 28 is the **exact** outer gradient at zero extra cost. ✅
- **QAOA: state-dependence term.** λ enters the *state preparation* via `e^{-iγH_C(λ)}`
  (since QAOA's phase separator IS the cost Hamiltonian). So differentiating gives
  `∂_λ J = [reuse term] + [state-dependence term]` (Eq. 33). The first term is cheap
  but **biased**; estimating the second needs extra circuits → **cost–bias trade-off.**
  You either accept a biased gradient (reuse-only) or pay to remove the bias.

This is the paper's key conceptual takeaway: **the "free gradient" property is
specific to VQE.** QAOA doesn't get it for free.

### Experiments
- Weighted Max-Cut on Erdős–Rényi graphs, n∈{10,12,14}, p∈{0.25…0.65}, N=300 instances.
- Parametric weights: `w_e(λ) = w̄_e + A_e·f_e(x(λ))` with three response families:
  **linear, quadratic, periodic** (periodic = "switch-rich" stress test, freq param K).
- Metric: **AUC_B** = area under best-so-far/J* vs evaluation-budget curve (Eq. 36) —
  measures *progress per budget*, budget-matched at B=1830.
- **Results:** CR-ID beats finite-difference (FD) value-probing everywhere.
  - 1D outer control: **~4% AUC_B improvement** (Table II, all 3 families).
  - Multi-dimensional (per-edge λ_e ∈ ℝ^|E|): **>14% improvement** (Table IV) — overhead
    compounds, so the gain grows.
  - The 3× factor: FD needs 3 inner solves/step, CR-ID needs 1 → CR-ID does ~3× more
    outer iterations on the same budget → horizontal compression of the trajectory.
  - VQE > QAOA on this bilevel task (Table V): VQE better leverages the clean reuse
    signal; QAOA's best-of-32 *readout* narrows the gap (occasional very good
    bitstrings) but with worse reliability.

### What to use it for / feasibility hooks
- Any cost function `Σ_e w_e(λ)·(diagonal correlator)` with a tunable λ → CR-ID applies.
- **Microgrid relevance:** time-of-day / demand-forecast / price-signal λ scaling a
  dispatch cost is exactly the parametric-coefficient setting. If the inner problem is
  a diagonal QUBO/Ising and solved with VQE, the outer price-sensitivity gradient is
  free. With QAOA, mind the bias term.
- Limitations (their own): only modest n (≤14), diagonal Hamiltonians only (non-diagonal
  needs measurement-grouping accounting), envelope identity exact only *at* inner
  stationarity (finite inner budget → some error).

---

## 2. `Implicit_differentiation_of_variational_quantum_algorithms.pdf` — the IFT engine

**Ahmed, Killoran, Carrasquilla Álvarez (Chalmers / Xanadu / Vector).** arXiv:2211.13765,
24 Nov 2022. This is ref [32] in the manuscript — the *theoretical foundation* CR-ID
extends.

### Core idea
A quantity defined as the solution of an optimization (a ground state, a fixed point)
can be **differentiated without unrolling** the optimizer, via the **Implicit Function
Theorem**. If `f(z*(a), a) = 0` is the optimality condition, then

> `∂_a z*(a) = -(∂_z f)^{-1} ∂_a f`   (Eq. 2)

For a VQA: `f = ∂_z E(z,a)` (gradient of energy = 0 at the optimum), so `∂_z f` is the
**Hessian** of E. You don't invert it directly — use **vector-Jacobian products (VJPs)**
+ an iterative linear solve (GMRES / conjugate gradient), so cost is ~Hessian-vector
products, not a cubic inversion (Eqs. 4–14).

### Why it matters here
- Lets you compute `∂_a ⟨A⟩` for an operator A ≠ H — i.e. **beyond Hellmann-Feynman**.
  HF only gives you energy derivatives; implicit diff gives derivatives of *arbitrary*
  observables on the variational ground state (Eq. 20).
- Agnostic to *how* the solution was found (black-box inner solver).
- Implemented as JAXopt (classical modular implicit diff) + PennyLane (quantum AD).

### Three demo applications (Sec. 4)
1. **Generalized susceptibility** — response `∂_a⟨A⟩` of an observable to a Hamiltonian
   perturbation; spin-chain `H(a) = -Σσ^zσ^z - γΣσ^x - aΣσ^z`. L=5 ansatz matches exact
   eigendecomposition; L=4 deviates (expressivity matters).
2. **Hyperparameter optimization** of a quantum classifier (data-reuploading): tune L2
   regularization strengths via *hypergradients* of validation loss → bilevel, beats grid
   search.
3. **Variational entanglement** — generate a Bell state by maximizing a geometric
   entanglement measure (distance to nearest separable state, via SWAP test) with
   implicit gradients. Bilevel again.

### Relationship to the manuscript
CR-ID **specializes** this. General implicit diff needs the Hessian-inverse term (the
expensive `(∂_z f)^{-1}` linear solve). The manuscript's insight: for *diagonal* cost
Hamiltonians with λ in the coefficients, the **envelope theorem** kills the hard term at
stationarity, leaving only the cheap, already-measured correlator sum. So CR-ID =
"implicit diff where the inversion is free because of problem structure."

---

## 3. `GroverMixersForQAOA.pdf` — GM-QAOA (constraint-handling ansatz)

**Bärtschi & Eidenbenz (Los Alamos).** arXiv:2006.00354v2, 2 Oct 2020. The odd one out —
*not* about gradients/bilevel; it's about **how to keep QAOA inside a feasible subspace**.

### Core idea
Standard QAOA mixer `e^{-iβΣX}` doesn't respect constraints. GM-QAOA replaces it with a
**Grover-style selective-phase-shift mixer**:

> `U_M(β) = e^{-iβ|F⟩⟨F|} = U_S(Id - (1-e^{-iβ})|0⟩⟨0|)U_S†`   (Eqs. 2–3)

where `|F⟩ = U_S|0⟩` is the **equal superposition of all feasible solutions**. The whole
difficulty shifts from *mixer design* to **state preparation `U_S`** (hence the title).

### Three properties (proven Sec. II)
1. Computes equal feasible-state superposition (needs poly-size U_S to exist).
2. **Mixes equal amplitude:** any two feasible states with the same objective value get
   the *same amplitude* — first mixer with this property (good for fair sampling).
3. **No Hamiltonian-simulation error:** built from exact Grover/Pauli gates, no
   Trotterization. Cleaner on noisy hardware (though not especially NISQ-friendly in
   depth).

### Feasibility categories for U_S
- Cat 1: all solutions feasible (unconstrained) — trivial `H^⊗n`.
- **Cat 2: constrained + poly-size U_S exists** — GM-QAOA's sweet spot (e.g. Dicke states
  for fixed-Hamming-weight constraints like k-Vertex-Cover, k-Set-Cover).
- Cat 3: no known poly U_S (e.g. Max-Clique) — GM-QAOA can't help.

### Applications worked out (Sec. III)
- **Max k-VertexCover:** Dicke state `|D_k^n⟩` as U_S, O(nk) gates, beats XY-ring/clique
  mixers on connectivity.
- **TSP / permutations:** prepare superposition of all n! permutations, O(n³) gates /
  O(n²) depth via row-by-row W-state construction + bitmask.
- **Discrete portfolio rebalancing:** superposition over portfolio bands via weighted
  Dicke states — models the *count* of portfolios per band, unlike the Bell-pair approach
  which gives a wrong binomial weighting.

### Relationship to the others / our work
- **Mostly orthogonal** to CR-ID and implicit diff. This is about *feasibility/constraint
  encoding*, not gradients.
- Connects to existing memory `[[gm-qaoa-dp-reformulation]]`: we already concluded
  GM-QAOA **doesn't fit single-battery dispatch** (the constraint structure isn't the
  fixed-Hamming-weight / permutation kind GM-QAOA needs; we went to a signed-trajectory
  DP instead). Keep that verdict — this paper is the canonical source for *what* GM-QAOA
  requires, and it confirms why dispatch (Cat 1/continuous-ish, not Cat 2) isn't its
  target.

---

## Quick "which paper answers this" index
- *How do I get an outer/price-sensitivity gradient cheaply?* → CR-ID (manuscript), Eqs. 27–30.
- *Why is it free for VQE but not QAOA?* → manuscript Sec. V-E, Eq. 33.
- *How do I differentiate a ground state / fixed point in general?* → Ahmed et al., IFT, Eq. 2.
- *How do I differentiate an observable that isn't the energy?* → Ahmed et al., Eq. 20 (beyond Hellmann-Feynman).
- *How do I keep QAOA feasible under hard constraints?* → GM-QAOA, Eqs. 2–3 + Cat-2 problems.
- *Does GM-QAOA fit battery dispatch?* → No; see `[[gm-qaoa-dp-reformulation]]`.
