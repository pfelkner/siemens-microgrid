# Presentation Companion — Microgrid Dispatch (merged deck `main.tex`)

Presenter notes for every slide, plus one master symbol table at the end.
For each formula slide there is a **plain-language** explanation of *what the
formula does* and *why it is there* — say these, don't read the math aloud.

Deck order: **Problem Statement → Approach Evolution → The Constraints, Formally
→ MILP→QUBO Transformation → Appendix.** Sections marked *(classmates)* are Tim &
Sarah's; the rest is our (Paul/Anton) work.

> **Colour key used throughout our slides (provenance):**
> 🔵 input data · ⚪ constant · 🟠 hyperparameter · 🟢 decision variable ·
> 🟣 derived · 🟦 set/index. Each coloured symbol on a slide is clickable and
> jumps to its row in the Appendix master table; the appendix tables have
> "Jump to" buttons back to Objective / Core / Side.

---

## 0. Title slide
- Group project, LMU QC Praktikum, Siemens microgrid dispatch.
- One line: *"We optimise how a microgrid buys, stores and sells power over a
  month — classically and on a quantum annealer."*

---

## 1. Problem Statement *(classmates)*

### 1.1 Microgrid Setup and Objective
- **Components**: a consumer (load), the utility grid connection, a PV (solar)
  system, and a battery (BESS). That's the whole physical system.
- **Variables / data**: forecasted load and PV, grid import cost + peak charge,
  export revenue, grid availability (outages), battery capacity and state of
  charge.
- **Objective in one sentence**: minimise total energy cost while getting
  credit for keeping the load alive during outages (resiliency).

### 1.2 Planned Approach (rolling horizon + classical/quantum split)
- The plan we'll detail: **rolling horizon, 3-day window**; a **soft, adjustable
  peak**; solve **window-by-window** carrying SoC + running peak forward; pay an
  **extra penalty** if the peak is exceeded; exceeding **ratchets up** next
  month's committed peak.
- The diagram: the same problem is solved **two ways** — a **classical** MILP
  solver and a **quantum** annealer (after a QUBO transformation, shown later).

---

## 2. Approach Evolution *(our work)*

### 2.1 How the Approach Evolved (Feedback → Now)
- **The story for the manager**: last session the partners told us the *real*
  billing isn't a hard cap — you **commit** a peak ahead of time, **pay a
  penalty** if you exceed it, and exceeding **raises next month's peak**.
- **Before**: one solve over the whole month, a single **hard** peak that can
  never be exceeded, assuming the whole month is known up front.
- **Now**: rolling horizon + soft peak + penalty + ratchet. *"Same model core —
  now it matches how the customer is actually billed."*

### 2.2 Recap: From One Scenario to Many (the stochastic structure)
- **Why this slide exists**: the manager wasn't here last time; this recaps that
  the model is **stochastic (two-stage)**.
- **What to say**: we consider **S equally-likely scenarios** for PV and load
  (`p_s = 1/S`). Every operational variable is decided *per scenario* — **except
  the peak `P^peak`**, which is committed **once, before** we know which scenario
  happens (**first stage**). The per-scenario operations are the **recourse**
  (second stage).
- The earlier single-scenario model is just `S=1` — a stepping stone.
- The little tree: one trunk (`P^peak`, first stage) branching into S scenarios
  (recourse).

### 2.3 Objective Function: Last Session → Current
- **Purpose**: show exactly *what changed* in the cost function.
- **Previous (hard peak)** — minimise: expected energy cost
  `∑_s p_s ∑_t c^ToU · P^imp · Δt`, **plus** a demand charge on the single peak
  `c^dem · P^peak`, **minus** resiliency revenue `∑ r^res · y`.
  - *In words*: pay for the energy you import (priced by time-of-use), pay a
    one-off charge proportional to your highest import, earn a reward for every
    outage slot you fully serve.
- **Current (soft peak + penalty)** — the single peak term **splits into two**:
  - `c^dem · P^commit` — **always paid** for the capacity you committed to.
  - `c^pen · (P^real − P^commit)^+` — an **extra penalty** only on the amount you
    *exceed* the commitment ( `(x)^+ = max(0,x)` , so zero if you stay under).
  - Plus new **export revenue** `− ∑ c^exp · P^exp`.
- **Ratchet**: next month `P^commit ← max(P^commit, P^real)`.
- Note: the energy/resiliency/export terms are **scenario-weighted** (`∑_s p_s`);
  the **peak term is not** — the peak is one committed number, not per-scenario.

### 2.4 The Rolling-Horizon Idea (One Month)
- **Purpose**: explain how a month is actually solved.
- **What to say**: the peak is committed **once for the month**, but we don't
  plan the whole month at once. We use a **3-day lookahead**, **implement only
  the first day**, then roll forward — **carrying battery SoC and the running
  peak** into the next solve (this is Model Predictive Control).
- During the month: a **penalty** accrues if the realised peak exceeds the
  commitment `P`. End of month: the **ratchet** sets next month's commitment to
  `max(P, realised peak)`.
- ⚠️ **Honest framing if asked about forecasts**: the MPC structure *supports*
  updated forecasts, but the current code re-solves on a single fixed time series
  (no live forecast refresh yet).

---

## 3. The Constraints, Formally *(our work)*

> These are written in two-stage form: every variable carries scenario index
> `s` and time `t`; quantifier `∀ s∈𝒮, t∈𝒯`. The deterministic model is the
> `S=1` special case.

### 3.1 The Model Core: The Physics Every Solution Must Obey
- **Purpose**: these are the **non-negotiable physics** — what makes a dispatch
  *valid* at all (as opposed to the "side" refinements that follow).
- **Power balance**: `P^PV + P^imp − P^exp + P^dis − P^ch = P^load`.
  - *In words*: in every online slot, generation + imports + battery discharge
    must exactly meet the load + exports + battery charging. Supply = demand,
    always.
- **Battery SoC dynamics**: `E_t = E_{t−1} + η·P^ch·Δt − (1/η)·P^dis·Δt`.
  - *In words*: the battery's stored energy this slot = last slot + what you
    charged (times efficiency η) − what you discharged (divided by η, because you
    lose energy both ways). **This is the only equation linking time slots.**
- **Variable bounds (box)**: import/export ≤ grid cap, charge/discharge ≤ battery
  power, `0 ≤ E ≤ E^max`. Just device capacities.

### 3.2 Type 1 (Linking): The Scenario-Coupling Constraint
- **Formula**: `P^peak ≥ P^imp_{s,t}` for **all** s, t.
- *In words*: the single committed peak must be at least as large as **every**
  import in **every** scenario — so it equals the worst-case import over all of
  them.
- **Why it matters (the punchline)**: this is the **only** constraint that ties
  the scenarios together. Remove it and the S scenarios fall apart into S
  independent subproblems. It gives the problem its **block-angular** structure
  (see appendix) and is exactly what the soft-peak penalty *relaxes*.

### 3.3 Type 2 (Integrality): SoC-Dependent Power Bands
- **What it does**: three binaries `b^low, b^mid, b^high` (exactly one = 1) pick
  which **state-of-charge band** the battery is in; indicators force the SoC into
  the matching range; the last line **throttles** charge/discharge power.
- *In words*: near empty or full the battery is limited to **125 kW**
  (`f^edge = 0.5`); in the mid band it can do the full **250 kW**
  (`f^mid = 1`). Battery protection.
- **Why "complicating"**: the power *limit* now depends on a **discrete state**,
  so there's no fixed box — the binaries force branch-and-bound.

### 3.4 Type 2 (Integrality): Mutual Exclusion (XOR)
- **What it does**: binaries forbid doing both opposite actions at once —
  **charge XOR discharge**, and **import XOR export** — via Big-M links
  (`P ≤ P_max · b`, and `b_a + b_b ≤ 1`).
- *In words*: you can't simultaneously charge and discharge the battery, nor buy
  and sell at the same instant.
- **Nuance**: classically these are *almost redundant* (efficiency losses already
  make it uneconomic), but we keep them to **mirror the QUBO penalty terms** on
  the quantum side.

### 3.5 Type 2 (Integrality): Outage Handling (Switchable Balance)
- **What it does**: during an outage there's no grid (`P^imp = P^exp = 0`), and a
  binary `y` switches the island balance on/off:
  `| P^PV + P^dis − P^ch − P^load | ≤ M^bal · (1 − y)`.
- *In words*: if `y = 1`, the right side is 0, so the island **must balance
  exactly** — the load is fully served and we earn resiliency revenue. If
  `y = 0`, the right side is huge, the constraint is **off**, and load shedding
  is allowed (no reward).
- **Why "complicating"**: a **switchable equality** (Big-M + binary) — the
  classic disjunction that forces branch-and-bound.

---

## 4. MILP → QUBO Transformation *(classmates)*

> ⚠️ Notation in this section is **single-scenario** (`P^imp_t`, `𝒪`) and the
> objective shown uses the **hard** demand charge `c^dem·P^peak`, not the soft
> commit/penalty from §2.3. It also reuses **`S` for the number of sweeps** —
> which is *not* the scenario count `S` from our sections. Flag verbally if asked.

### 4.1 What is QUBO?
- **QUBO = Quadratic Unconstrained Binary Optimization**: minimise `xᵀQx` over
  binary `x`. Only binary variables, only pairwise (quadratic) interactions,
  and **no explicit constraints**.
- *In words*: every constraint is **folded into the objective as a penalty** — a
  violated constraint adds energy, so the solver naturally avoids it.
- **Why relevant**: this is the **native format for D-Wave quantum annealers**
  (and simulated annealing). Penalty weights `λ` must be big enough that no
  single violation ever pays off.

### 4.2 Five Steps: MILP → QUBO
- The recipe to convert our MILP into a QUBO:
  1. **Binary-encode continuous variables** — write each continuous value as a
     sum of bits (`v = (v_max/(2^k−1)) ∑ 2^i x_i`). Box bounds hold *by
     construction*.
  2. **Equalities → squared penalty** — `(∑a_i x_i + c)² ` (used for power
     balance, SoC dynamics).
  3. **Inequalities → "Unbalanced Penalization"** `λ₁h + λ₂h²` — no slack
     variables needed (peak coupling, derating).
  4. **XOR → product penalty** — `λ·P^ch·P^dis` etc.
  5. **Linear objective → QUBO diagonal** — energy cost terms become diagonal
     entries.
- *(The "B.x" tags refer to the constraint catalogue; map them to our Core/Type-1/
  Type-2 slides if asked.)*

### 4.3 QUBO Size: Variables, Bits & Qubit Requirements
- **What it shows**: how many binary variables the QUBO needs. Each continuous
  quantity costs `k` bits; per time slot you have import/export/charge/discharge
  (k bits each), SoC (k bits), 3 band indicators, plus one global peak.
  `n = T·(2k_g + 2k_b + k_s + 3) + k_g`.
- **Annealing cost**: a **sweep** visits all n variables once; a **run** is S
  sweeps from one random start; R runs keep the best. Runtime `O(R·S·n²)` dense,
  but our Q is **sparse** so `O(R·S·n·k)`.
- **Takeaway**: variable count `n` hurts (quadratically when dense) — **fewer
  bits k** = smaller, faster problem.
- **Example**: `k=4, T=24` → **556** variables.

### 4.4 The Resulting QUBO Objective
- **What it is**: the full single objective the annealer minimises. Top line =
  the **real cost** (energy + demand charge + export revenue + resiliency). Every
  line below is a **penalty** enforcing one of our constraints:
  - power balance penalty (squared residual),
  - SoC dynamics penalty (squared residual),
  - peak coupling (unbalanced penalisation),
  - charge/discharge & import/export **XOR** penalty,
  - SoC-band power **derating** penalty,
  - **outage island balance** penalty.
- *In words*: instead of telling the solver "obey these rules," we **add a cost
  for breaking each rule**, tuned so breaking is never worth it.
- Helper terms: `h^pk = P^imp − P^peak`, `h^der = …`, `P̄ = P^dis − P^ch + P^PV −
  P^load`. `λ` auto-scaled to exceed the max objective gain.

---

## 5. Appendix *(our work)*

### 5.1–5.3 Master Symbol Table (1/3, 2/3, 3/3)
- Reference tables; every symbol coloured by provenance. The **Jump to** buttons
  return to Objective / Core / Side. Use these to answer "what is symbol X?"
  questions live. (Full content reproduced in the master table below.)

### 5.4 Block-Angular Structure & Decomposition
- **What it shows**: the constraint matrix is **block-angular** — independent
  scenario blocks on the diagonal (green), tied **only** by the orange `P^peak`
  row/column (the Type 1 linking constraint).
- **Why it matters**: this structure is exactly what classical decomposition
  (**Benders**: master fixes `P^peak`, subproblems per scenario; **Lagrangian**:
  move the linking row into the objective as a penalty) and the **QUBO /
  soft-peak penalty** all exploit. It's the unifying idea of the whole talk.

---

# Master Symbol Table (all symbols)

Provenance: **Input** = read from data · **Const** = fixed parameter ·
**Hyper** = set per run · **Var** = decided by the optimiser · **Derived** =
computed from the others · **Set** = structural index/range · **QUBO** =
construct introduced only in the QUBO transformation.

### Sets & indices
| Symbol | Meaning | Domain / type | Provenance |
|---|---|---|---|
| `t` | time slot (15 min) | index | Set |
| `𝒯 = {0,…,T−1}` | all time slots | set | Set |
| `T` | number of slots | ℤ₊ | Hyper (`--slots`, def. 2880) |
| `s` | scenario index | index | Set |
| `𝒮 = {0,…,S−1}` | all scenarios | set | Set |
| `S` | number of scenarios | ℤ₊ | Hyper (`--scenarios`, def. 1) |
| `𝒪_s` | outage slots of scenario s, `{t : g=0}` | ⊆ 𝒯 | Derived (from g) |
| `n` | number of binary QUBO variables | ℤ₊ | QUBO |
| `i, j` | bit indices in the QUBO | index | QUBO / Set |

### Input data (per scenario s, slot t — from `all_data.csv`)
| Symbol | Meaning | Domain | Provenance |
|---|---|---|---|
| `P^PV_{s,t}` | PV generation | ℝ₊ (kW) | Input |
| `P^load_{s,t}` | electrical load | ℝ₊ (kW) | Input |
| `c^ToU_{s,t}` | time-of-use energy price | ℝ₊ ($/kWh) | Input |
| `g_{s,t}` | grid availability (1 online, 0 outage) | {0,1} | Input |

### Constants (static parameters)
| Symbol | Meaning | Value | Provenance |
|---|---|---|---|
| `Δt` | slot length | 0.25 h | Const |
| `E^max` | battery capacity | 1000 kWh | Const |
| `P^B_nom` | battery nominal power | 250 kW | Const |
| `η_rt` | round-trip efficiency | 0.90 | Const |
| `η` | per-direction efficiency | √0.90 ≈ 0.95 | Derived (from η_rt) |
| `E_0` | initial SoC (default) | 500 kWh | Const (overridable) |
| `c^dem` | demand / commitment charge | 15 $/kW | Const |
| `r^res_min` | resiliency rate | 15 $/min | Const |
| `r^res` | resiliency per served slot | 225 $/slot | Derived (= r^res_min × 15) |
| `c^exp` | export tariff | 0.05 $/kWh | Const |
| `P^G_max` | grid power cap | 1000 kW | Const |
| `E^low` | low-band SoC threshold | 100 kWh | Const |
| `E^high` | high-band SoC threshold | 900 kWh | Const |
| `f^edge` | power fraction in low/high band | 0.5 | Const |
| `f^mid` | power fraction in mid band | 1.0 | Const |

### Hyperparameters (set per run / by the rolling driver)
| Symbol | Meaning | Domain | Provenance |
|---|---|---|---|
| `P^commit` | committed peak (exogenous, swept) | ℝ₊ (kW) | Hyper |
| `c^pen` | exceedance penalty rate | ℝ₊ ($/kW) | Hyper |
| `f_ratchet` | ratchet headroom factor | ℝ≥1 | Hyper |
| `peak_floor` | per-window peak lower bound | ℝ₊ | Hyper/Derived (= max(P^commit, P^real)) |
| `soc_init` | starting SoC for the window | [0, E^max] | Hyper |
| `peak_mode` | `demand_charge` or `commit_penalty` | enum | Hyper |
| window / step | 3-day lookahead / 1-day implemented | days | Hyper |
| `k` (`k_g,k_b,k_s`) | bits per encoded continuous variable | ℤ₊ | Hyper (QUBO) |
| `λ`, `λ_bal`, `λ_soc`, `λ_xor`, `λ₁`, `λ₂`, `λ_b7` | penalty weights | ℝ₊ | Hyper (QUBO) |
| `R` | independent annealing runs | ℤ₊ | Hyper (QUBO) |
| `S` *(sweeps!)* | sweeps per run — **NOT the scenario count** | ℤ₊ | Hyper (QUBO) |

### Decision variables (per scenario s, slot t; `P^peak` is first-stage)
| Symbol | Meaning | Domain | Stage |
|---|---|---|---|
| `P^imp_{s,t}` | grid import | [0, P^G_max] | 2nd |
| `P^exp_{s,t}` | grid export | [0, P^G_max] | 2nd |
| `P^ch_{s,t}` | battery charge | [0, P^B_nom] | 2nd |
| `P^dis_{s,t}` | battery discharge | [0, P^B_nom] | 2nd |
| `E_{s,t}` | state of charge at slot end | [0, E^max] | 2nd |
| `P^peak` | billing peak (single, shared) | ℝ₊ (kW) | **1st** |
| `b^low/mid/high_{s,t}` | active SoC band | {0,1} | 2nd |
| `b^ch/dis_{s,t}` | charge / discharge active | {0,1} | 2nd |
| `b^imp/exp_{s,t}` | import / export active | {0,1} | 2nd |
| `y_{s,t}` (t∈𝒪_s) | outage slot fully served | {0,1} | 2nd |
| `x_i` | QUBO binary (encodes the above) | {0,1} | QUBO |

### Derived quantities
| Symbol | Meaning | Formula / domain | Provenance |
|---|---|---|---|
| `p_s` | scenario probability | 1/S | Derived |
| `M^bal_{s,t}` | Big-M for outage balance | max(P^load, P^B_nom+P^PV)+1 | Derived |
| `P^real` | running realised peak (rolling) | max implemented P^imp so far | Derived |
| `(x)^+` | positive part | max(0, x) | Derived (operator) |
| next `P^commit` | ratcheted commitment | max(P^commit, f_ratchet·P^real) | Derived |
| `Q`, `Q_ii`, `Q_ij` | QUBO matrix / diag / off-diag | ℝ | QUBO |
| `h^pk_t` | peak-coupling slackless term | P^imp − P^peak | Derived (QUBO) |
| `h^der_t` | derating term | P^ch/dis − ½P^B_nom(1+b^mid) | Derived (QUBO) |
| `P̄_t` | island net power | P^dis − P^ch + P^PV − P^load | Derived (QUBO) |
| `m_t`, `s_t^↑`, `s_t^↓` | outage Big-M / slack helpers | ℝ | Derived/Var (QUBO) |

---

## Known integration issues to be aware of (merged deck)
1. **`S` is overloaded**: scenario count (our sections) vs. annealing **sweeps**
   (QUBO size slide). Same letter, different meaning.
2. **Notation mismatch**: QUBO section is single-scenario (`_t`, `𝒪`); our
   sections are two-stage (`_{s,t}`, `𝒪_s`, `𝒮`).
3. **Objective mismatch**: the QUBO objective uses the **hard** `c^dem·P^peak`,
   not the **soft commit + penalty** model from §2.3.
4. **`B.x` references** in the Five-Steps slide have no labels in this deck (they
   come from the reference doc); map them verbally to Core / Type-1 / Type-2.
5. **Title page** still says our old title/authors — the merged talk has four
   authors (Tim, Sarah, Paul, Anton).
