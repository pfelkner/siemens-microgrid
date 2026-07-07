# Solver Walkthrough: Classical MILP and Benders Subproblem

This document explains what the main Gurobi-based solvers do:

- `classical/deterministic_solver.py`: the deterministic classical MILP baseline.
- `classical/stochastic_solver.py`: the stochastic two-stage MILP baseline.
- `subproblem/subproblem.py`: the fixed-discrete continuous LP used inside the
  Benders / hybrid QAOA loop.

They model the same microgrid physics, but they sit at different levels. The
classical solver chooses both discrete operating modes and continuous dispatch
in one mixed-integer program. The subproblem solver receives a fixed discrete
configuration `z` and optimizes only the continuous variables `x`.

## Shared Microgrid Model

Both solvers work on a 15-minute dispatch horizon with:

- PV generation `p_pv`.
- Load demand `p_load`.
- Grid availability, where `1` means online and `0` means outage.
- A battery energy storage system with charge, discharge, and state of charge.
- Time-of-use import prices.
- A demand charge on peak grid import.
- Export revenue for grid export.
- Optional resiliency revenue for serving load during outage slots.

The core power-balance expression is:

```text
PV + grid_import - grid_export + battery_discharge - battery_charge = load
```

When the grid is online, this balance is enforced as a hard equality. During
outages, grid import and export are forced to zero. If an outage slot is marked
as served, the local PV plus battery dispatch must exactly cover the load.

Battery state of charge evolves as:

```text
soc[t] = soc[t-1] + eta * charge[t] * dt - discharge[t] / eta * dt
```

where `dt = 0.25` hours and `eta = sqrt(0.90)` by default.

## `classical/deterministic_solver.py`

### What It Solves

`deterministic_solver.py` builds and solves the deterministic full dispatch problem
as a Gurobi MILP. It is the single-scenario ground-truth classical baseline used
to compare against the hybrid / quantum decomposition.

The public entry point is:

```python
build_and_solve(df_list, scenario_probs, time_limit, mip_gap, log_file, quiet, ...)
```

The CLI entry point is:

```bash
uv run python -m classical.deterministic_solver --data artifacts/data/all_data.csv --slots 2880
```

### Inputs

Each dataframe in `df_list` must contain:

| Column | Meaning |
|---|---|
| `timestamp` | Timestamp carried into the output schedule. |
| `p_kw` | PV generation in kW. |
| `load_kw` | Load demand in kW. |
| `tou_usd_kwh` | Time-of-use grid import price. |
| `grid_available` | `1` online, `0` outage. |

For multi-scenario solves, use `classical/stochastic_solver.py` instead.

### First-Stage Variable

The solver has one scalar peak variable:

```text
peak_import
```

It is shared by every scenario and every time slot. The solver constrains it to
cover all realized grid imports:

```text
peak_import >= grid_in[s, t]
```

In deterministic mode this is simply the maximum grid import across the horizon,
chosen endogenously because the objective charges it.

### Dispatch Variables

For each slot `t`, the MILP creates:

| Variable | Type | Meaning |
|---|---|---|
| `grid_in[t]` | Continuous | Power imported from grid. |
| `grid_out[t]` | Continuous | Power exported to grid. |
| `bess_ch[t]` | Continuous | Battery charging power. |
| `bess_dis[t]` | Continuous | Battery discharging power. |
| `soc[t]` | Continuous | Battery state of charge. |
| `soc_low/soc_mid/soc_high[t]` | Binary | One-hot SoC operating band. |
| `ch_active/dis_active[t]` | Binary | Charge/discharge mode gates. |
| `import_active/export_active[t]` | Binary | Grid import/export mode gates. |
| `served[t]` | Binary | Only for outage slots; marks whether load is served. |

### Main Constraints

#### Online Power Balance

When the grid is available:

```text
PV + grid_in - grid_out + bess_dis - bess_ch = load
```

This forces exact supply-demand balance.

#### Outage Logic

When the grid is unavailable:

```text
grid_in = 0
grid_out = 0
```

The residual local balance is:

```text
resid = PV + bess_dis - bess_ch - load
```

The model adds two big-M rows:

```text
 resid <= M * (1 - served)
-resid <= M * (1 - served)
```

So:

- `served = 1` forces `resid = 0`, meaning the outage load is exactly served.
- `served = 0` relaxes the outage balance, meaning the load can be unserved.

#### Battery State of Charge

Every slot updates SoC from the previous slot:

```text
soc[t] = soc_prev + eta * bess_ch[t] * dt - bess_dis[t] / eta * dt
```

At `t = 0`, `soc_prev` is `soc_init`.

#### SoC Band Selection

The solver selects exactly one SoC band:

```text
soc_low + soc_mid + soc_high = 1
```

Indicator constraints then force `soc` into the selected interval:

| Band | SoC interval |
|---|---|
| Low | `soc <= 100 kWh` |
| Mid | `100 kWh <= soc <= 900 kWh` |
| High | `soc >= 900 kWh` |

The selected band also derates battery power. Low and high bands get only
`50%` of nominal battery power; the middle band gets `100%`.

#### Charge/Discharge XOR

The battery cannot charge and discharge simultaneously:

```text
bess_ch  <= BESS_PMAX * ch_active
bess_dis <= BESS_PMAX * dis_active
ch_active + dis_active <= 1
```

#### Import/Export XOR

The grid connection cannot import and export simultaneously:

```text
grid_in  <= GRID_PMAX * import_active
grid_out <= GRID_PMAX * export_active
import_active + export_active <= 1
```

### Objective

The MILP minimizes:

```text
expected energy import cost
+ demand charge or peak exceedance penalty
- expected resiliency revenue
- expected export revenue
```

More explicitly:

| Term | Formula shape |
|---|---|
| Energy cost | `sum_s prob[s] * sum_t tou[s,t] * grid_in[s,t] * dt` |
| Demand charge | `DEMAND_CHARGE * peak_import` |
| Commit penalty mode | `penalty_rate * (peak_import - peak_floor)` |
| Resiliency revenue | `sum_s prob[s] * resiliency_per_slot * served[s,t]` over outage slots |
| Export revenue | `sum_s prob[s] * sum_t export_rate * grid_out[s,t] * dt` |

In deterministic mode there is no scenario probability: all terms are for the
single forecast trajectory.

### Outputs

`build_and_solve` returns:

```python
(model, info, schedules)
```

where:

- `model` is the solved Gurobi model.
- `info` is a dictionary with runtime, status, model size, objective value, cost
  decomposition, peak import, served outage count, and related metadata.
- `schedules` is a one-item list containing the deterministic schedule.

The CLI writes:

- `artifacts/results/schedule_classical.csv` for deterministic runs.
- `artifacts/results/results_classical.csv` for appended run summaries.
- `artifacts/results/gurobi.log` for the solver log.

After solving, the CLI also runs sanity checks for online power balance, SoC
bounds, no grid use during outages, no simultaneous charge/discharge, and
peak consistency.

## `classical/stochastic_solver.py`

`stochastic_solver.py` handles the multi-scenario two-stage MILP. It uses the
same shared MILP core as `deterministic_solver.py`, but exposes scenario generation,
stochastic schedule output, and explicit `M > 1` validation in a separate module.

Run it with:

```bash
uv run python -m classical.stochastic_solver --data artifacts/data/all_data.csv --slots 2880 --scenarios 5
```

The scenario-specific variables are the same as the deterministic variables,
but duplicated per scenario. The single `peak_import` variable is shared across
all scenarios and all timesteps:

```text
peak_import >= grid_in[s, t]
```

That is the only first-stage coupling in the current stochastic model. Energy,
export, and resiliency terms are probability-weighted expected values; the demand
charge is not probability weighted because the shared peak is billed once.

The stochastic CLI writes:

- `artifacts/results/schedule_classical_stochastic.csv` for all scenario schedules.
- `artifacts/results/schedule_classical_expected.csv` for the expected schedule.
- `artifacts/results/results_classical.csv` for appended run summaries.

## `subproblem/subproblem.py`

### What It Solves

`subproblem.py` solves the continuous LP at a fixed discrete configuration `z`.
It is the Benders recourse oracle used by `qc/benders.py`.

The key idea is:

```text
master chooses z  ->  subproblem optimizes x at that z
```

Here:

- `z` contains the binary/discrete choices per slot: battery mode, grid mode,
  SoC band, and outage served bit.
- `x` contains continuous dispatch: import, export, charge, discharge, SoC, and
  peak import.

Unlike `deterministic_solver.py`, this file does not choose the binary variables.
They are already fixed in `inst.config`.

The public entry point is:

```python
solve_subproblem(inst, quiet=True)
```

It returns a `SubproblemResult`.

### Inputs

The solver takes an `Instance` from `subproblem.feasible_start_x`:

| Field | Meaning |
|---|---|
| `pv` | PV generation per slot. |
| `load` | Load demand per slot. |
| `grid_available` | `1` online, `0` outage. |
| `config` | Fixed per-slot `SlotConfig` decisions. |
| `params` | Physical and cost parameters. |
| `tou` | Time-of-use prices. |

Each `SlotConfig` fixes:

| Field | Allowed values | Meaning |
|---|---|---|
| `batt` | `charge`, `discharge`, `idle` | Which battery direction is allowed. |
| `grid` | `import`, `export`, `idle` | Which grid direction is allowed. |
| `band` | `low`, `mid`, `high` | Which SoC band must hold. |
| `served` | `True`, `False` | Whether an outage slot must be served. |

### Continuous Variables

For each slot `t`, the LP creates:

| Variable | Meaning |
|---|---|
| `p_imp[t]` | Grid import. |
| `p_exp[t]` | Grid export. |
| `p_ch[t]` | Battery charge. |
| `p_dis[t]` | Battery discharge. |
| `soc[t]` | Battery state of charge. |

It also creates:

| Variable | Meaning |
|---|---|
| `peak` | Peak import used for demand charge. |

All variables are continuous. There are no binary variables in this model.

### Why the Constraint Shape Is Fixed

For Benders cuts to work cleanly, the fixed discrete configuration `z` must enter
the LP through right-hand sides, not by adding or removing constraints. This file
therefore keeps the same rows present for every `z`, and changes only RHS values.

For every added constraint, the helper `add(...)` records two things:

- the Gurobi constraint itself, so duals can be read after solving;
- `rhs_affine`, an affine map from master bits to that constraint's RHS.

The affine map has the form:

```python
constraint_name -> (constant, {(slot, role): coefficient})
```

This is what lets `qc/benders.py` turn one LP solve into a cut over all possible
master states.

### Main Constraints

The subproblem mirrors the physical constraints from the MILP, but with the
binary decisions fixed by `inst.config`.

#### SoC Dynamics

Every slot has the same continuous SoC update:

```text
soc[t] = soc_prev + eta * p_ch[t] * dt - p_dis[t] / eta * dt
```

#### Fixed SoC Band Box

The selected band in `cfg.band` becomes an SoC interval:

| Band | Lower bound | Upper bound |
|---|---:|---:|
| Low | `0` | `soc_low_th` |
| Mid | `soc_low_th` | `soc_high_th` |
| High | `soc_high_th` | `e_max` |

The rows are recorded as RHS-affine functions of `b_low`, `b_mid`, and `b_high`,
even though those bits have concrete values in the current solve.

#### Battery Throttle

The selected band also controls max battery power:

```text
low/high -> frac_edge * p_bess_nom
mid      -> frac_mid  * p_bess_nom
```

Both charge and discharge are capped by that throttle.

#### Battery Direction Gating

The fixed battery mode gates charge/discharge:

```text
p_ch  <= p_bess_nom * z_ch
p_dis <= p_bess_nom * z_dis
```

If the fixed state says the battery is idle, both RHS values are zero, so both
flows are pinned to zero.

#### Grid Direction Gating

The fixed grid mode gates import/export:

```text
p_imp <= p_grid_max * z_imp
p_exp <= p_grid_max * z_exp
```

If the fixed state says the grid is idle, both RHS values are zero.

#### Online Power Balance

When the grid is online, the hard balance remains:

```text
PV + p_imp - p_exp + p_dis - p_ch = load
```

This is intentionally not relaxed. If the fixed `z` turns off every useful flow
in a deficit or surplus slot, the LP is allowed to become infeasible. That
infeasibility is meaningful; it becomes a Benders feasibility cut.

#### Outage Rows

When the grid is unavailable:

```text
p_imp = 0
p_exp = 0
```

If `cfg.served` is true, the big-M served rows collapse to exact local balance.
If `cfg.served` is false, the outage residual is allowed.

#### Peak Coupling

For every slot:

```text
peak >= p_imp[t]
```

### RHS-Affine Self-Check

Before optimizing, the solver evaluates every saved `rhs_affine` expression at
the fixed `z` and asserts that it matches Gurobi's normalized constraint RHS.

This catches drift between the mathematical cut map and the actual LP rows. If
that assertion fails, the Benders cuts would be unreliable, so the solve stops
immediately.

### Objective

The subproblem minimizes only the continuous recourse cost:

```text
energy import cost + peak cost - export revenue
```

It deliberately omits resiliency revenue:

```text
- resiliency_per_slot * served
```

because `served` is part of the fixed discrete state `z`. In the Benders loop,
that z-only term is handled by the master objective in `qc.instance.direct_costs`.

The peak term follows the same two modes as the classical solver:

- `demand_charge`: `demand_charge * peak`
- `commit_penalty`: `penalty_rate * (peak - peak_floor)`

### Solver Outcomes

`solve_subproblem` has two expected outcomes.

#### Optimal

If the LP is feasible and optimal, it returns:

| Field | Meaning |
|---|---|
| `status = "optimal"` | Feasible LP solved. |
| `q_value` | Continuous recourse value `Q(z)`. |
| `x` | Optimal arrays for import, export, charge, discharge, SoC, and peak. |
| `duals` | Constraint dual values `Pi`, keyed by constraint name. |
| `rhs_affine` | RHS maps used to build an optimality cut. |

`qc.benders.optimality_cut` converts this into:

```text
q(z) >= q(z_bar) + w * (z - z_bar)
```

This cut lower-bounds the continuous recourse value for other master states.

#### Infeasible

If the LP is infeasible or infeasible/unbounded, it returns:

| Field | Meaning |
|---|---|
| `status = "infeasible"` | The fixed `z` has no continuous continuation. |
| `farkas` | Farkas dual ray, keyed by constraint name. |
| `rhs_affine` | RHS maps used to build a feasibility cut. |

`qc.benders.feasibility_cut` converts the Farkas certificate into a cut that
removes the current impossible `z`, and often other states with the same
infeasibility proof.

### Demo / Self-Check

Running the module directly executes `_demo()`:

```bash
uv run python -m subproblem.subproblem
```

The demo checks both branches:

- a feasible 3-slot online case, where the LP optimum is verified against
  `feasible_start_x.verify`;
- an infeasible outage case, where the solver must return a Farkas certificate.

## Side-by-Side Comparison

| Topic | `deterministic_solver.py` deterministic MILP | `subproblem.py` |
|---|---|---|
| Optimization type | MILP | LP |
| Chooses binary decisions? | Yes | No, receives fixed `z` |
| Chooses continuous dispatch? | Yes | Yes |
| Handles stochastic scenarios? | No; use `stochastic_solver.py` | No, single fixed instance |
| Peak import | Scalar max-import variable across timesteps | Continuous variable in fixed-`z` recourse |
| Resiliency reward | Included in objective | Omitted; master handles it as z-only cost |
| Infeasible outcome | Full model failure | Expected for bad `z`; returns Farkas certificate |
| Main output | Dispatch schedules and cost summary | `Q(z)`, `x*`, duals, or Farkas ray |
| Used for | Classical reference optimum | Benders cuts in hybrid loop |

## How They Fit Together

The full MILP is the reference model:

```text
min over z, x: total dispatch cost
```

The Benders / hybrid loop decomposes that same idea:

```text
master:      choose discrete z
subproblem:  solve continuous x at fixed z
cuts:        send dual/Farkas information back to the master
```

So `subproblem.py` is not a second full replacement for `deterministic_solver.py`.
It is the continuous recourse component of a decomposed solver. The full MILP
solves everything in one model; the Benders loop repeatedly calls the subproblem
to learn enough cuts for the master to choose a good discrete configuration.
