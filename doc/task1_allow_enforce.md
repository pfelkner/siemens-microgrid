# Task 1 — Allow / Enforce Classification of the Discrete Binary Variables

_Which binary variables **allow** continuous activity (≥ 0, but force nothing) vs.
which **enforce** it (pin a continuous quantity, so some configurations have no
feasible continuation)?_

This decides, per `QC_Ansatz_07-02.md` §"Benders-Struktur", whether the loop ever
produces **feasibility cuts** or only **optimality cuts**.

- **Allow** → `flow ≤ M·b`: `b=1` permits `flow ∈ [0, M]`, `b=0` forbids it.
  Setting the flow to 0 is always feasible → the bit alone never makes the LP infeasible.
- **Enforce** → the bit pins a continuous variable to a value or an interval
  (equality / two-sided box). Some `z` then have **no** feasible LP continuation →
  Gurobi reports infeasible → Farkas certificate → **feasibility cut**.

Source of truth: `classical/deterministic_solver.py` (constraint construction) and
`notes/presentations/reductions.tex` (the reduced LP). Line refs are to
`classical/deterministic_solver.py`.

---

## Per-variable classification

| Binary | Constraint that couples it to `x` | Form | **Verdict** |
|---|---|---|---|
| `b^ch` (`ch_active`) | `P^ch ≤ P^B_nom · b^ch` (L151) | one-sided gate | **Allow** |
| `b^dis` (`dis_active`) | `P^dis ≤ P^B_nom · b^dis` (L152) | one-sided gate | **Allow** |
| `b^imp` (`import_active`) | `P^imp ≤ P^G_max · b^imp` (L155) | one-sided gate | **Allow** |
| `b^exp` (`export_active`) | `P^exp ≤ P^G_max · b^exp` (L156) | one-sided gate | **Allow** |
| `b^low` | `b^low=1 ⇒ E ≤ E^low` (L137) | indicator → box | **Enforce** |
| `b^mid` | `b^mid=1 ⇒ E^low ≤ E ≤ E^high` (L139–142) | indicator → two-sided box | **Enforce** |
| `b^high` | `b^high=1 ⇒ E ≥ E^high` (L143) | indicator → box | **Enforce** |
| `y_t` (`served`) | `\|P^PV + P^dis − P^ch − P^load\| ≤ M(1−y_t)` (L125–126) | `y=1` ⇒ equality | **Enforce** |

### Why the four gate bits are "Allow"
Each is a plain big-M gate. `b=0` forces the flow to 0; `b=1` opens it to `[0, M]`.
Neither value **requires** a positive flow, so on its own constraint the flow can
always be driven to 0. Locally: allow.

### Why the SoC-band bits are "Enforce"
The band bits do two things. The **throttle** part
`max_power = P^B_nom·(f_edge·(b^low+b^high) + f_mid·b^mid)` (L146–147) is allow-like
(it only caps power). But the **indicator** part pins the continuous SoC `E` into
the selected interval — `low → [0,100]`, `mid → [100,900]`, `high → [900,1000]` kWh.
This is a genuine two-sided constraint on `E`, and `E` is chained across time by the
SoC dynamics (L129–133). "Set all flows ≈ 0" does **not** escape it: with zero
battery flow, `E_t` stays at `SoC_init` for every `t`, sitting in exactly one band.
If the master picks a band bit for a different band, or a band the SoC trajectory
cannot reach in the available slots, the LP is infeasible → feasibility cut.

### Why the served bit is "Enforce"
`y_t = 1` collapses the two big-M rows to the equality
`P^PV_t + P^dis_t − P^ch_t = P^load_t` at an outage slot, where import/export are
already forced to 0 (L121–122). This **demands** the battery exactly cover the net
outage load. If the battery is too empty / too throttled to source `P^load − P^PV`,
no continuation exists → feasibility cut. (`y_t = 0` leaves the residual free — the
slot is simply "not served", the escape hatch — but then the master forgoes the
`225 $/slot` resiliency reward, so it is tempted to set `y_t = 1`.)

`P^peak` never causes infeasibility: `P^peak ≥ P^imp_t` (L160) is always satisfiable
by raising `P^peak` up to `P^G_max`; it is a dependent cost quantity, not a
feasibility gate.

---

## The decisive point: recourse is **not** complete → feasibility cuts WILL occur

`QC_Ansatz_07-02.md` §61 calls the all-`Allow` case the "Normalfall" where "alle
Flüsse ≈ 0 setzen" is feasible, so every `z` has a continuation and **no** feasibility
cuts appear. **That does not hold for this model**, for three independent reasons:

1. **`y_t` is enforcing** (above) — already flagged in the plan.
2. **The SoC-band bits are enforcing** (above) — a band choice the SoC trajectory
   cannot reach is infeasible. Not flagged in the plan; it matters.
3. **The power balance is an equality, and the gates are only "at most one" (≤ 1).**
   This is the subtle one. The mixer only enforces the *structural* rules
   `band_sum = 1`, `ch + dis ≤ 1`, `imp + exp ≤ 1`. Nothing forces any flow **on**.
   So the master can legally pick, at an online slot with a net deficit
   (`P^load > P^PV`), the config `b^imp = b^exp = b^ch = b^dis = 0`. Then all four
   flows are pinned to 0 and the online balance
   `P^PV + P^imp − P^exp + P^dis − P^ch = P^load` (L115–119) reduces to
   `P^PV = P^load`, which is false → **infeasible**. The "set all flows to 0" escape
   fails precisely because the balance is `=`, not `≤`. The same happens at a surplus
   slot (`P^PV > P^load`) with export/charge both off.

**Conclusion.** This is an *enforcing* formulation. Schritt 3 of the loop **must**
handle the Gurobi-infeasible branch (read the Farkas dual ray → build a feasibility
cut that excludes that `z`); it cannot assume "optimality cuts only". Which `z` are
excluded is data-dependent (depends on PV/load/outage per slot).

### How to make it (mostly) complete recourse, if desired
If you *wanted* to suppress the balance-starvation feasibility cuts and keep only the
band/served ones, turn the activity XORs into "exactly one direction available" or
add a slack/curtailment continuous variable to the online balance so it becomes a
`≤`/`≥` pair instead of a hard `=`. That is a modeling change, not required for the
PoC — the loop is correct either way, feasibility cuts just cost extra iterations.

---

## One-line summary

**Allow:** `b^ch, b^dis, b^imp, b^exp` (big-M gates). **Enforce:**
`b^low, b^mid, b^high` (pin SoC into a band) and `y_t` (pin outage balance).
Because two enforcing families exist **and** the power balance is an equality gated
by "at-most-one" bits, recourse is incomplete → the Benders loop will produce
feasibility cuts, not just optimality cuts.
