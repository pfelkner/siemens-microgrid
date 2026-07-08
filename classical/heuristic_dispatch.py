"""Rule-based greedy dispatcher — a simple, non-optimizing benchmark baseline.

Pure NumPy, **no Gurobi**. Produces a full feasible dispatch schedule by a
single causal forward pass and scores it on the same four-term objective as the
MILP / Benders hybrid. Its job is to be an honest *floor*: worse than optimal by
construction, but not a strawman — it shaves peaks (one `P*` knob), self-consumes
PV, exports surplus, and serves outages for the resiliency reward.

Two products (see wiki: note-benchmark-baseline):
  * `dispatch(...)`  — the greedy rule-based competitor the hybrid must beat.
  * `passive(...)`   — a "no strategy" controller (battery idle) = the
                       normalization 0-point for the approximation ratio.

The MILP optimum `C_opt` is passed *into* `approx_ratio` as a plain number; this
module never touches Gurobi. Get it separately, e.g.
    python -m classical.deterministic_solver --data all_data.csv --slots T

Constants are single-sourced: `Params` (dt, eta, caps, throttle, cost rates)
from subproblem.feasible_start_x, and RESILIENCY_PER_SLOT from qc.instance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qc.instance import RESILIENCY_PER_SLOT
from subproblem.feasible_start_x import Params

_TOL = 1e-6


@dataclass
class Schedule:
    """Per-slot continuous dispatch (kW) plus SoC (kWh) and the served bit."""
    p_ch: np.ndarray
    p_dis: np.ndarray
    p_imp: np.ndarray
    p_exp: np.ndarray
    soc: np.ndarray          # SoC at END of each slot
    served: np.ndarray       # 1 on served outage slots, 0 elsewhere
    p_star: float | None = None


def band_of(soc: float, p: Params) -> str:
    """SoC band, matching the MILP's one-hot boxes (boundaries -> mid)."""
    if soc < p.soc_low_th:
        return "low"
    if soc > p.soc_high_th:
        return "high"
    return "mid"


# --------------------------------------------------------------------------- #
# The greedy dispatcher (the meaningful competitor)
# --------------------------------------------------------------------------- #
def dispatch(pv, load, grid_available, params: Params, p_star: float) -> Schedule:
    """Causal greedy forward pass.

    Priority per slot: cover load, self-consume PV (charge surplus, then export),
    and shave grid import above the threshold `P*` by discharging. During an
    outage, discharge to serve the load and bank the resiliency reward if it can.

    Deliberately myopic: it charges only from PV surplus (no ToU arbitrage) and
    reads the SoC-band throttle from the SoC it happens to be at — this is the
    non-optimality the hybrid gets to close.
    """
    pv = np.asarray(pv, float); load = np.asarray(load, float)
    ga = np.asarray(grid_available, int)
    T = len(load)
    dt, eta, e_max = params.dt, params.eta, params.e_max
    p_grid = params.p_grid_max

    p_ch = np.zeros(T); p_dis = np.zeros(T)
    p_imp = np.zeros(T); p_exp = np.zeros(T)
    soc_arr = np.zeros(T); served = np.zeros(T, int)

    soc = params.soc_init
    for t in range(T):
        p_lim = params.throttle(band_of(soc, params))       # 250 mid / 125 edge
        room = max(0.0, (e_max - soc) / (eta * dt))          # kW that fills to full
        avail = max(0.0, soc * eta / dt)                     # kW battery can deliver
        net = load[t] - pv[t]

        if ga[t] == 1:                                       # ---- online ----
            if net < 0:                                      # PV surplus
                surplus = -net
                p_ch[t] = min(surplus, p_lim, room)
                p_exp[t] = min(surplus - p_ch[t], p_grid)
                # ponytail: surplus beyond charge+export cap would break the hard
                # power balance; _demo asserts balance, so bad data fails loudly.
            else:                                            # deficit
                want = max(0.0, net - p_star)                # only shave above P*
                p_dis[t] = min(want, p_lim, avail)
                p_imp[t] = min(net - p_dis[t], p_grid)
        else:                                                # ---- outage ----
            if net > 0:                                      # discharge to serve
                p_dis[t] = min(net, p_lim, avail)
            else:                                            # charge PV surplus
                p_ch[t] = min(-net, p_lim, room)
            # served iff local balance holds exactly (grid is pinned to 0)
            served[t] = int(abs(pv[t] + p_dis[t] - p_ch[t] - load[t]) <= _TOL)

        soc = soc + eta * p_ch[t] * dt - p_dis[t] / eta * dt
        soc = min(max(soc, 0.0), e_max)                      # guard FP drift
        soc_arr[t] = soc

    return Schedule(p_ch, p_dis, p_imp, p_exp, soc_arr, served, p_star)


# --------------------------------------------------------------------------- #
# The passive "no strategy" anchor (the normalization 0-point)
# --------------------------------------------------------------------------- #
def passive(pv, load, grid_available, params: Params) -> Schedule:
    """Battery never used; grid serves all deficit, PV exports all surplus,
    outages go unserved (no resiliency revenue). The floor a real controller
    must beat just by existing."""
    pv = np.asarray(pv, float); load = np.asarray(load, float)
    ga = np.asarray(grid_available, int)
    net = load - pv
    online = ga == 1
    p_imp = np.where(online, np.clip(net, 0, params.p_grid_max), 0.0)
    p_exp = np.where(online, np.clip(-net, 0, params.p_grid_max), 0.0)
    T = len(load)
    soc = np.full(T, params.soc_init)                        # idle -> constant
    served = np.where(~online & (np.abs(net) <= _TOL), 1, 0)
    return Schedule(np.zeros(T), np.zeros(T), p_imp, p_exp, soc, served, None)


# --------------------------------------------------------------------------- #
# Scoring — the same four-term objective as the MILP / subproblem
# --------------------------------------------------------------------------- #
def objective(sched: Schedule, tou, params: Params) -> dict:
    """C_energy + C_peak - C_res - C_export (demand-charge peak mode, floor 0)."""
    tou = np.asarray(tou, float)
    dt = params.dt
    c_energy = float(np.sum(tou * sched.p_imp) * dt)
    c_peak = float(params.demand_charge * np.max(sched.p_imp)) if len(sched.p_imp) else 0.0
    c_export = float(params.export_rate * np.sum(sched.p_exp) * dt)
    c_res = float(RESILIENCY_PER_SLOT * np.sum(sched.served))
    return {
        "C_energy": c_energy, "C_peak": c_peak,
        "C_res": c_res, "C_export": c_export,
        "total": c_energy + c_peak - c_res - c_export,
    }


def sweep_pstar(pv, load, grid_available, tou, params: Params, n: int = 41):
    """Sweep the one knob P* over [0, max load]; keep the lowest-cost schedule.

    Returns (best_schedule, best_objective_dict, best_p_star)."""
    grid = np.linspace(0.0, float(np.max(load)), n)
    best = None
    for ps in grid:
        s = dispatch(pv, load, grid_available, params, ps)
        obj = objective(s, tou, params)
        if best is None or obj["total"] < best[1]["total"]:
            best = (s, obj, float(ps))
    return best


def approx_ratio(c_ref: float, c_method: float, c_opt: float) -> float:
    """Normalized, sign/scale-invariant: ref -> 0, optimum -> 1.

    r = (C_ref - C_method) / (C_ref - C_opt).  Report the absolute $ gap
    (c_method - c_opt) alongside this for interpretability."""
    denom = c_ref - c_opt
    if abs(denom) < _TOL:
        return float("nan")                                  # ref == opt: undefined
    return (c_ref - c_method) / denom


# --------------------------------------------------------------------------- #
# Data loader (pandas; decoupled from both Instance classes)
# --------------------------------------------------------------------------- #
def from_csv(path, start: int = 0, T: int | None = None):
    """Load one window -> (pv, load, grid_available, tou) arrays."""
    import pandas as pd
    df = pd.read_csv(path)
    df = df.iloc[start:] if T is None else df.iloc[start:start + T]
    return (df["p_kw"].to_numpy(float), df["load_kw"].to_numpy(float),
            df["grid_available"].to_numpy(int), df["tou_usd_kwh"].to_numpy(float))


# --------------------------------------------------------------------------- #
# Self-check (no Gurobi)
# --------------------------------------------------------------------------- #
def _demo() -> None:
    p = Params()

    # 1) Hand-computed objective on a trivial 1-slot deficit, P* high (no discharge).
    s = dispatch(pv=[0.0], load=[100.0], grid_available=[1], params=p, p_star=1e9)
    obj = objective(s, tou=[0.10], params=p)
    assert abs(obj["C_energy"] - 0.10 * 100 * p.dt) < 1e-9, obj
    assert abs(obj["C_peak"] - p.demand_charge * 100) < 1e-9, obj
    assert abs(obj["total"] - (2.5 + 1500.0)) < 1e-9, obj
    # passive must match here (battery idle either way)
    assert abs(objective(passive([0.0], [100.0], [1], p), [0.10], p)["total"]
               - obj["total"]) < 1e-9

    # 2) Physics invariants on a mixed window (surplus, deficit, one outage).
    rng = np.random.default_rng(0)
    T = 24
    pv = np.clip(rng.normal(150, 120, T), 0, None)
    load = np.clip(rng.normal(200, 60, T), 0, None)
    ga = np.ones(T, int); ga[10] = 0                          # one outage slot
    tou = np.full(T, 0.12)
    sch = dispatch(pv, load, ga, p, p_star=80.0)

    online = ga == 1
    bal = pv + sch.p_imp - sch.p_exp + sch.p_dis - sch.p_ch - load
    assert np.allclose(bal[online], 0.0, atol=1e-6), bal[online]
    assert np.all((sch.p_imp[~online] == 0) & (sch.p_exp[~online] == 0)), "grid used in outage"
    assert np.all((sch.p_ch <= _TOL) | (sch.p_dis <= _TOL)), "simultaneous charge+discharge"
    assert np.all(sch.soc >= -1e-6) and np.all(sch.soc <= p.e_max + 1e-6), "SoC out of bounds"

    # 3) Sweep runs and the anchor is well-defined.
    best_s, best_obj, best_ps = sweep_pstar(pv, load, ga, tou, p)
    anchor = objective(passive(pv, load, ga, p), tou, p)
    r = approx_ratio(anchor["total"], best_obj["total"], c_opt=best_obj["total"] - 1.0)
    print(f"[demo] best P* = {best_ps:.1f} kW  "
          f"dispatch total = {best_obj['total']:.2f}  "
          f"passive anchor = {anchor['total']:.2f}  r(vs mock opt) = {r:.3f}")
    print("[demo] all invariants hold ✓")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Rule-based baseline vs the MILP optimum.")
    ap.add_argument("--data", default="all_data.csv")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--slots", type=int, default=96, help="window length T")
    ap.add_argument("--sweep", type=int, default=41, help="P* grid points")
    ap.add_argument("--c-opt", type=float, default=None,
                    help="MILP optimum for this window (run classical.deterministic_solver "
                         "separately); enables the approximation ratio.")
    ap.add_argument("--demo", action="store_true", help="run the self-check and exit")
    a = ap.parse_args()
    if a.demo:
        _demo(); return

    p = Params()
    pv, load, ga, tou = from_csv(a.data, a.start, a.slots)
    anchor = objective(passive(pv, load, ga, p), tou, p)
    best_s, best_obj, best_ps = sweep_pstar(pv, load, ga, tou, p, a.sweep)

    print(f"window [{a.start}, {a.start + a.slots})   T={a.slots}   "
          f"outage slots={int(np.sum(ga == 0))}")
    print(f"{'term':<10}{'passive':>14}{'greedy':>14}")
    for k in ("C_energy", "C_peak", "C_res", "C_export", "total"):
        print(f"{k:<10}{anchor[k]:>14.2f}{best_obj[k]:>14.2f}")
    print(f"best P* = {best_ps:.1f} kW")
    if a.c_opt is not None:
        r_g = approx_ratio(anchor["total"], best_obj["total"], a.c_opt)
        r_p = approx_ratio(anchor["total"], anchor["total"], a.c_opt)  # == 0 by def
        print(f"C_opt = {a.c_opt:.2f}")
        print(f"greedy : r = {r_g:.3f}   gap to opt = {best_obj['total'] - a.c_opt:+.2f} $")
        print(f"passive: r = {r_p:.3f}   (the 0-point)")


if __name__ == "__main__":
    main()
