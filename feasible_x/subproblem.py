"""Task 7 — Benders subproblem: the continuous LP at a FIXED discrete config `z`.

Given an instance and a fixed `z` (the per-slot `SlotConfig`s), build the Gurobi LP
over the continuous variables `x = (P^imp, P^exp, P^ch, P^dis, E, P^peak)`, solve it
exactly, and return **`x*` and the dual values** — the two things Schritt 3 of
`../QC_Ansatz_07-02.md` needs to feed the Benders cut (Task 8).

Two outcomes (see `why_feasibility_cuts.md`):
  * FEASIBLE   → return `x*`, `q_value = f(x*)` (the continuous cost `Q(z)`), and the
                 constraint duals `π`. Task 8 turns these into an **optimality cut**.
  * INFEASIBLE → return the **Farkas certificate** (dual ray of the infeasible LP).
                 Task 8 turns it into a **feasibility cut** that bans this `z`. This
                 branch WILL fire for this model — it is not optional.

Design choices that make the duals usable for Benders
-----------------------------------------------------
* The discrete config `z` enters **only through the right-hand sides** of a fixed set
  of constraints (gating, throttle, SoC-band box, served-outage big-M). The constraint
  *shape* never changes with `z`. That is what makes `Q(z) = πᵀ h(z)` affine in `z`,
  so a single solve yields a cut valid for **all** `z` — the whole point of Benders.
  Each z-coupled constraint therefore stays present even when its bit is 0 (its RHS
  just becomes 0 and pins the flow to 0).
* The online power balance is kept as a HARD EQUALITY (`==`), on purpose. It is the
  faithful physics; relaxing it would let the LP silently shed power, corrupt the
  shadow prices, and mask the infeasibility that Benders needs to see. The resulting
  infeasibility is handled by the Farkas / feasibility-cut branch, by design.
* Objective = the continuous cost `f(x)` only (energy + peak − export). The resiliency
  reward `−r·y_t` is constant given `z` (it is `g(z)`, the master's direct cost), so it
  is omitted here; Task 8 adds it back on the master side.

Scope: single scenario (M=1), the PoC case. Reuses the instance definitions and the
`verify()` feasibility check from `feasible_start_x.py`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from feasible_x.feasible_start_x import Instance, SlotConfig, Params, feasible_configs, verify


@dataclass
class SubproblemResult:
    status: str                       # "optimal" | "infeasible"
    q_value: float | None             # f(x*) = continuous cost Q(z), if optimal
    x: dict | None                    # arrays p_imp,p_exp,p_ch,p_dis,soc + scalar p_peak
    duals: dict[str, float] | None    # constraint name -> dual π (optimal branch)
    farkas: dict[str, float] | None   # constraint name -> Farkas dual (infeasible branch)

    @property
    def feasible(self) -> bool:
        return self.status == "optimal"


def _band_bits(band: str) -> tuple[int, int, int]:
    return {"low": (1, 0, 0), "mid": (0, 1, 0), "high": (0, 0, 1)}[band]


def solve_subproblem(inst: Instance, quiet: bool = True) -> SubproblemResult:
    """Solve the fixed-`z` continuous LP; return x* and duals (or Farkas certificate).

    `inst.config` is the fixed discrete configuration `z`. `inst.tou` (per-slot price)
    and the cost scalars on `inst.params` drive the objective.
    """
    p, T = inst.params, inst.T
    tou = inst.tou

    m = gp.Model("benders_subproblem")
    m.Params.OutputFlag = 0 if quiet else 1
    m.Params.InfUnbdInfo = 1          # required to read FarkasDual on infeasibility

    # ---- continuous variables ----
    p_imp = m.addVars(T, lb=0.0, ub=p.p_grid_max, name="p_imp")
    p_exp = m.addVars(T, lb=0.0, ub=p.p_grid_max, name="p_exp")
    p_ch  = m.addVars(T, lb=0.0, ub=p.p_bess_nom, name="p_ch")
    p_dis = m.addVars(T, lb=0.0, ub=p.p_bess_nom, name="p_dis")
    soc   = m.addVars(T, lb=0.0, ub=p.e_max,      name="soc")
    peak  = m.addVar(lb=p.peak_floor, ub=p.p_grid_max, name="peak")

    duals_of: list[tuple[str, gp.Constr]] = []   # (name, constr) for dual readout

    def add(name: str, constr) -> None:
        duals_of.append((name, m.addConstr(constr, name=name)))

    for t in range(T):
        cfg = inst.config[t]
        bl, bm, bh = _band_bits(cfg.band)
        b_ch  = 1 if cfg.batt == "charge"    else 0
        b_dis = 1 if cfg.batt == "discharge" else 0
        b_imp = 1 if cfg.grid == "import"    else 0
        b_exp = 1 if cfg.grid == "export"    else 0
        online = inst.grid_available[t] == 1

        soc_prev = p.soc_init if t == 0 else soc[t - 1]
        # --- SoC dynamics (equality, z-independent) ---
        add(f"soc_dyn_{t}",
            soc[t] == soc_prev + p.eta * p_ch[t] * p.dt - (p_dis[t] / p.eta) * p.dt)

        # --- SoC band box: z in RHS ---
        ub = p.soc_low_th * bl + p.soc_high_th * bm + p.e_max * bh
        lb = 0.0 * bl + p.soc_low_th * bm + p.soc_high_th * bh
        add(f"soc_ub_{t}", soc[t] <= ub)
        add(f"soc_lb_{t}", soc[t] >= lb)

        # --- battery throttle (band-dependent cap), z in RHS ---
        max_pow = p.p_bess_nom * (p.frac_edge * (bl + bh) + p.frac_mid * bm)
        add(f"throt_ch_{t}",  p_ch[t]  <= max_pow)
        add(f"throt_dis_{t}", p_dis[t] <= max_pow)

        # --- battery on/off gating, z in RHS (bit=0 -> flow pinned to 0) ---
        add(f"gate_ch_{t}",  p_ch[t]  <= p.p_bess_nom * b_ch)
        add(f"gate_dis_{t}", p_dis[t] <= p.p_bess_nom * b_dis)

        # --- grid on/off gating, z in RHS ---
        add(f"gate_imp_{t}", p_imp[t] <= p.p_grid_max * b_imp)
        add(f"gate_exp_{t}", p_exp[t] <= p.p_grid_max * b_exp)

        if online:
            # HARD power balance (equality). z-independent.
            add(f"bal_on_{t}",
                inst.pv[t] + p_imp[t] - p_exp[t] + p_dis[t] - p_ch[t] == inst.load[t])
        else:
            # Outage: no grid exchange (data-fixed, z-independent).
            add(f"no_imp_{t}", p_imp[t] == 0.0)
            add(f"no_exp_{t}", p_exp[t] == 0.0)
            # Served bit via big-M so the constraint stays present for all z (RHS in z):
            #   |PV + dis - ch - load| <= M (1 - y_t)   →   y_t=1 forces exact coverage.
            y = 1 if cfg.served else 0
            M_big = max(inst.load[t], p.p_bess_nom + inst.pv[t]) + 1.0
            resid = inst.pv[t] + p_dis[t] - p_ch[t] - inst.load[t]
            add(f"served_up_{t}",  resid <= M_big * (1 - y))
            add(f"served_lo_{t}", -resid <= M_big * (1 - y))

        # --- demand-charge coupling: peak >= import (z-independent) ---
        add(f"peak_{t}", peak >= p_imp[t])

    # ---- objective: continuous cost f(x) ----
    energy = gp.quicksum(tou[t] * p_imp[t] * p.dt for t in range(T))
    export = gp.quicksum(p.export_rate * p_exp[t] * p.dt for t in range(T))
    if p.peak_mode == "commit_penalty":
        peak_cost = p.penalty_rate * (peak - p.peak_floor)
    else:
        peak_cost = p.demand_charge * peak
    m.setObjective(energy + peak_cost - export, GRB.MINIMIZE)

    m.optimize()

    if m.Status == GRB.OPTIMAL:
        x = dict(
            p_imp=np.array([p_imp[t].X for t in range(T)]),
            p_exp=np.array([p_exp[t].X for t in range(T)]),
            p_ch=np.array([p_ch[t].X for t in range(T)]),
            p_dis=np.array([p_dis[t].X for t in range(T)]),
            soc=np.array([soc[t].X for t in range(T)]),
            p_peak=float(peak.X),
        )
        duals = {name: float(c.Pi) for name, c in duals_of}
        return SubproblemResult("optimal", float(m.ObjVal), x, duals, None)

    if m.Status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
        # Farkas certificate: the dual ray proving no x satisfies the constraints.
        # Task 8 forms the feasibility cut  Σ_i farkas_i · h_i(z) ≤ 0  that excludes z.
        try:
            farkas = {name: float(c.FarkasDual) for name, c in duals_of}
        except gp.GurobiError:
            farkas = {}                # certificate unavailable (e.g. presolve edge)
        return SubproblemResult("infeasible", None, None, None, farkas)

    raise RuntimeError(f"unexpected Gurobi status {m.Status}")


# --------------------------------------------------------------------------- #
# Demo / self-check
# --------------------------------------------------------------------------- #
def _demo() -> None:
    # Same T=3 instance as feasible_start_x's demo (η=0.9 to match conversation.md),
    # plus a real time-of-use price so the objective is non-trivial.
    params = Params(eta=0.9, soc_init=120.0)
    inst = Instance(
        pv=[100, 400, 150],
        load=[300, 200, 350],
        grid_available=[1, 1, 1],
        config=[
            SlotConfig(batt="discharge", grid="import", band="mid"),
            SlotConfig(batt="charge",    grid="export", band="mid"),
            SlotConfig(batt="discharge", grid="import", band="mid"),
        ],
        params=params,
        tou=[0.20, 0.08, 0.30],
    )

    res = solve_subproblem(inst)
    assert res.feasible, "expected feasible"
    verify(inst, res.x)                                   # x* is physically feasible
    print(f"[demo] FEASIBLE — Q(z) = {res.q_value:.4f}")
    print(f"       P^dis = {np.round(res.x['p_dis'], 2)}  "
          f"P^ch = {np.round(res.x['p_ch'], 2)}  peak = {res.x['p_peak']:.1f}")

    # The LP optimum must be no worse than any random feasible point (sanity on cost).
    p = inst.params
    sample = feasible_configs(inst, n=20, seed=7)
    def f_of(x):
        e = sum(inst.tou[t] * x["p_imp"][t] * p.dt for t in range(inst.T))
        ex = sum(p.export_rate * x["p_exp"][t] * p.dt for t in range(inst.T))
        return e + p.demand_charge * x["p_peak"] - ex
    best_sample = min(f_of(x) for x in sample)
    assert res.q_value <= best_sample + 1e-4, \
        f"LP optimum {res.q_value} worse than a feasible sample {best_sample}"
    print(f"       optimum {res.q_value:.4f} ≤ best of 20 random feasible {best_sample:.4f} ✓")

    # Duals present and keyed by constraint name (what Task 8 consumes).
    assert res.duals and "bal_on_0" in res.duals
    nz = {k: round(v, 4) for k, v in res.duals.items() if abs(v) > 1e-6}
    print(f"       nonzero duals: {nz}")

    # --- Infeasible branch: unservable idle outage → Farkas certificate returned ---
    bad = Instance(
        pv=[0.0], load=[300.0], grid_available=[0],
        config=[SlotConfig(batt="idle", grid="idle", band="mid", served=True)],
        params=Params(eta=0.9, soc_init=500.0),
        tou=[0.20],
    )
    r2 = solve_subproblem(bad)
    assert not r2.feasible, "expected infeasible"
    assert r2.farkas, "expected a Farkas certificate for the feasibility cut"
    print(f"[demo] INFEASIBLE — Farkas certificate returned "
          f"({sum(abs(v) > 1e-9 for v in r2.farkas.values())} active rows) → feasibility cut")


if __name__ == "__main__":
    _demo()
