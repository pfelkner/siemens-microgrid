# Approximation-Ratio Deliverable — How to Run the Solvers, and Why

This is the recipe for producing the **normalized approximation-ratio** plot that
compares the three solvers on microgrid dispatch:

- **classical MILP** (`classical/deterministic_solver.py`) — the optimum, the ceiling.
- **rule-based heuristic** (`classical/heuristic_dispatch.py`) — the simple, defensible floor.
- **GM-QAOA + Benders hybrid** (`qc/run_loop.py`) — the method under test.

It covers **how** each solver must be run (equal `T`, matched windows, reset SoC,
where each cost number comes from) and **why** each rule exists.

> Environment: run every module with the project venv, `./.venv/bin/python -m ...`.
> `uv run` currently fails to resolve on macOS (the CUDA `cupy` wheel), and it is
> not needed — the venv already has the dependencies.

---

## 1. Why a *normalized* approximation ratio

A raw statement like "the hybrid costs \$3960 on this window" is unanchored:
\$3960 out of what spread? And the objective can go **negative** (export and
resiliency revenue can dominate cost), so a plain ratio `C_method / C_opt` is
meaningless — it flips sign and blows up.

Instead we bracket every method between two fixed reference points and normalize:

```
        r = (C_ref − C_method) / (C_ref − C_opt)  ∈  [0, 1]

   C_ref  = passive "no-strategy" controller   → r = 0   (the floor)
   C_opt  = classical MILP optimum             → r = 1   (the ceiling)
   C_method = the solver being scored           → r somewhere between
```

`r = 0` means "no better than doing nothing." `r = 1` means "matched the best
achievable." This form is **sign- and scale-invariant**, so it stays meaningful
even when the raw cost is negative. Always report the absolute **\$ gap to
optimum** (`C_method − C_opt`) next to `r` for interpretability.

The story the plot tells, per window:

```
   passive (0) ────► greedy heuristic ────► hybrid ────► MILP (1)
```

---

## 2. The four cost numbers

For **one window** you need four objective values, all computed on the *same*
four-term objective `C = C_energy + C_peak − C_res − C_export`:

| Symbol      | Source                          | Role                         |
|-------------|---------------------------------|------------------------------|
| `C_ref`     | passive controller (heuristic)  | the `0` endpoint             |
| `C_opt`     | classical MILP                  | the `1` endpoint (optimum)   |
| `C_greedy`  | rule-based heuristic (best `P*`) | a scored method → `r_greedy` |
| `C_hybrid`  | GM-QAOA + Benders loop          | a scored method → `r_hybrid` |

Then:

```
r_greedy = (C_ref − C_greedy) / (C_ref − C_opt)
r_hybrid = (C_ref − C_hybrid) / (C_ref − C_opt)
r_passive = 0   (by definition)
r_MILP    = 1   (by definition)
```

---

## 3. Invariants — the same window for all three (and why)

A comparison point is only valid if every solver sees the **identical instance**.
For each window fix:

1. **Same `T` (number of 15-min slots).** `T` is duration: `T` slots × 0.25 h.
   `T = 5` is 75 minutes. **Use the same `T` for all three solvers.** The hybrid
   is hardware-capped (statevector simulation of `8·T` qubits), so `T` is set by
   the hybrid — typically `T = 5`. A `T = 96` MILP against a `T = 5` hybrid is
   not a comparison.
2. **Same start offset `S`.** All three read the same slice `[S, S+T)`.
3. **Same data file.** Pass `--data all_data.csv` to every solver — the default
   `artifacts/data/all_data.csv` path does not exist in this checkout, so an
   unspecified `--data` silently reads the wrong (or no) data.
4. **Reset SoC every window.** Each window is independent: start from the default
   `soc_init = 500 kWh`. Do **not** carry SoC from one window into the next —
   that is rolling horizon (see §6), a different experiment.
5. **Same params.** `peak_mode = "demand_charge"`, `resiliency_per_slot = 225`,
   export rate `0.05`, and all battery/grid/SoC limits. These are the defaults in
   `Params` and the solvers; do not override them differently across solvers.
6. **`M = 1`, deterministic.** Single scenario. The stochastic solver is not part
   of this comparison.
7. **Identical objective.** The heuristic's scorer already matches the MILP's four
   terms. For the hybrid the comparable object is **`direct_costs(z*) + Q(z*)`**,
   which `run_loop` reports as its `total`. **Never use `Q(z)` alone** — the
   subproblem LP omits the resiliency term (`−225·served`), which the master
   carries separately. Comparing `Q(z)` to the MILP is an apples-to-oranges bug.

---

## 4. How to retrieve each `C` — two commands per window

### Command 1 — the hybrid (gives `C_hybrid` *and* `C_opt`)

`run_loop` runs the Benders loop and cross-checks it against the Gurobi MILP on
the *same* instance, so a single run yields two of the four numbers:

```
./.venv/bin/python -m qc.run_loop --data all_data.csv --start S --slots 5
```

Read from its output:

- `best: z=..., total = <C_hybrid> $`  → **`C_hybrid`** = `direct_costs(z*) + Q(z*)`.
- `Gurobi MILP: <C_opt> $`             → **`C_opt`** (deterministic optimum for this window).

`run_loop` honors `--start`, so this is also the reliable way to get `C_opt` for
offset windows. (The standalone `classical.deterministic_solver --data all_data.csv
--slots 5` also prints `total=$...` = `C_opt`, but its CLI has **no `--start`** — it
always begins at slot 0. Use it only for `S = 0`.)

For an outage window, add `--force-outage <t>` (slot index within the window) so
the resiliency term is exercised.

### Command 2 — the heuristic (gives `C_ref` and `C_greedy`)

```
./.venv/bin/python -m classical.heuristic_dispatch --data all_data.csv --start S --slots 5 --c-opt <C_opt>
```

Read from its output:

- `passive total` → **`C_ref`** (the `0`-point).
- `greedy total`  → **`C_greedy`** (best over the internal `P*` sweep).
- `greedy : r = ...` → `r_greedy`, already normalized against the `--c-opt` you passed.

Then compute `r_hybrid` by hand from the four numbers:

```
r_hybrid = (C_ref − C_hybrid) / (C_ref − C_opt)
```

> **Two sweeps, don't confuse them.** The heuristic internally sweeps `P*` — the
> peak-shaving threshold in kW: the battery only discharges to cover grid import
> *above* `P*`. That is automatic and unrelated to `T`. The "window sweep" in §5
> is the separate outer loop over start offsets.

---

## 5. Why many windows, not one — the window sweep

A single window gives `n = 1`: one `r_greedy`, one `r_hybrid`. That is a valid
comparison, but a reviewer discounts a single point — it could be a lucky or
unlucky window. The deliverable is a **distribution** of `r` across many
independent windows, reported as mean ± std and/or a scatter/box plot:

```
   passive = 0 ─► greedy ─► hybrid ─► MILP = 1     (with spread across windows)
```

This shows the hybrid beats the heuristic **robustly**, not on one cherry-picked
instance.

The window sweep is just the two commands of §4 repeated over a list of start
offsets, all at the same `T = 5`:

```
for S in [0, 5, 10, 200, 645, 1200, ...]:      # independent T=5 windows
    run_loop  --data all_data.csv --start S --slots 5   →  C_hybrid, C_opt
    heuristic --data all_data.csv --start S --slots 5   →  C_ref, C_greedy
    record r_greedy(S), r_hybrid(S)
plot the distribution of r over all S
```

### Choosing the windows (this matters at `T = 5`)

At `T = 5` the **demand charge dominates**: `C_peak = 15 $/kW × peak_import` is a
month-scale charge applied to just 75 minutes of energy, so it dwarfs the energy,
export, and resiliency terms. A batch of random windows then measures essentially
one thing — peak shaving — and the story is thin. Deliberately span conditions so
all four objective terms move:

- **Evening peak-load** windows (stress the demand charge).
- **Midday PV-surplus** windows (exercise export and charging).
- **Outage** windows via `--force-outage` (exercise the resiliency reward and the
  hybrid's feasibility cuts).

---

## 6. What NOT to do

- **Do not tile a day into consecutive 5-slot windows with SoC carried across
  them.** That is rolling-horizon / MPC month billing — a different experiment,
  with no hybrid driver yet, and not what this plot needs. Each window here is
  independent, SoC reset to 500.
- **Do not mix `T`** across the three solvers within a comparison point.
- **Do not compare `Q(z)` alone** to the MILP — use `direct_costs(z) + Q(z)`.
- **Do not read absolute totals as literal bills.** Because the monthly demand
  charge is applied to a short window, a window `total` is a *comparison
  objective*, not a real daily/monthly bill. It is meaningful only relatively
  (passive vs greedy vs hybrid vs MILP on the *same* window). A real monthly
  figure needs the full ~2785-slot horizon or the rolling-horizon driver.

---

## 7. Summary checklist (per window)

- [ ] Pick `T` (= hybrid limit, e.g. 5) and start `S`; same for all three.
- [ ] `--data all_data.csv` everywhere; `soc_init = 500`; `M = 1`; `demand_charge` mode.
- [ ] `run_loop --start S --slots T`  → `C_hybrid`, `C_opt`.
- [ ] `heuristic_dispatch --start S --slots T --c-opt C_opt` → `C_ref`, `C_greedy`, `r_greedy`.
- [ ] Compute `r_hybrid = (C_ref − C_hybrid)/(C_ref − C_opt)`.
- [ ] Repeat over many `S` (spanning peak / PV-surplus / outage windows).
- [ ] Plot the distribution of `r`: `passive = 0 → greedy → hybrid → MILP = 1`.
