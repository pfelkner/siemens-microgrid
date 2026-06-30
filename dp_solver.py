"""Dynamic-programming solver for the Siemens microgrid dispatch problem.

This is the *signed-trajectory reformulation* of the deterministic model in
classical_solver.py. The key observation:

  - Power balance (B.1) only *defines* the grid exchange once the battery
    decision is made:        grid_net = load - pv - (dis - ch).
  - SoC (B.2) is not a free variable; it is the running sum of the battery
    trajectory.
  - Charge/discharge (B.6) and import/export are mutually exclusive *by
    construction* once the battery is one signed variable and the grid is one
    signed residual.

So the only genuine degree of freedom is the (discretized) SoC level per slot.
We treat the SoC level as a DP state, the transition between consecutive levels
as the decision, and read off ch/dis/import/export deterministically. Online
power balance and SoC dynamics then hold *exactly* — feasibility by
construction, no penalties, no MILP.

Stages   : time slots t = 0..T-1
State    : SoC level E_t in a grid of L levels over [0, BESS_CAP]
Decision : transition E_{t-1} -> E_t (implies ch_t or dis_t)
Cost     : per-slot energy - export_revenue - resiliency  (additive)
Coupling : demand charge c_dem * max_t import_t is NOT additive; handled by a
           sweep over candidate peak caps (force import <= cap, add c_dem*cap,
           minimize over caps — the classic demand-charge DP trick).

Companion writeup: dp_reformulation.tex.  Does NOT replace classical_solver.py.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------- Static parameters (mirrors classical_solver.py) ----------
DT = 0.25                       # h per slot (15 min)
BESS_CAP = 1000.0               # kWh
BESS_PMAX = 250.0               # kW nominal
ETA_RT = 0.90                   # round-trip efficiency
ETA = math.sqrt(ETA_RT)         # per-direction efficiency
SOC_INIT = 500.0                # kWh
DEMAND_CHARGE = 15.0            # $/kW over billing period
RESILIENCY_PER_SLOT = 225.0     # $ per served 15-min outage slot
EXPORT_RATE = 0.05              # $/kWh paid for grid export
GRID_PMAX = 1000.0              # kW
SOC_LOW_TH = 100.0              # kWh (10%)
SOC_HIGH_TH = 900.0             # kWh (90%)
FRAC_EDGE = 0.5
FRAC_MID = 1.0

INF = 1e18
_EPS = 1e-6


def _band_limit(levels: np.ndarray) -> np.ndarray:
    """Per-level battery power cap from SoC-band derating (B.5)."""
    edge = (levels <= SOC_LOW_TH + _EPS) | (levels >= SOC_HIGH_TH - _EPS)
    return np.where(edge, FRAC_EDGE * BESS_PMAX, FRAC_MID * BESS_PMAX)


def _band_name(level: float) -> str:
    if level <= SOC_LOW_TH + _EPS:
        return "low"
    if level >= SOC_HIGH_TH - _EPS:
        return "high"
    return "mid"


def build_stage_costs(
    df: pd.DataFrame,
    levels: np.ndarray,
    export_rate: float,
    resiliency_rate: float,
    serve_tol: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-slot (L,L) cost and import matrices indexed [prev_level, new_level].

    For each transition E_prev -> E_new the implied charge/discharge follows the
    SoC dynamics (B.2) inverted:
        dE >= 0 -> charge :  ch  = dE / (eta * DT)
        dE <  0 -> discharge: dis = -dE * eta / DT
    Infeasible transitions (derating B.5, grid box B.3) get cost INF.

    Returns (base_cost, imp) each shape (T, L, L). `imp` is the grid import for
    that transition (0 during outages) — used by the peak-cap sweep.
    """
    pv = df["p_kw"].to_numpy(dtype=float)
    load = df["load_kw"].to_numpy(dtype=float)
    tou = df["tou_usd_kwh"].to_numpy(dtype=float)
    g = df["grid_available"].to_numpy(dtype=int)
    T = len(df)
    L = len(levels)

    dE = levels[None, :] - levels[:, None]            # dE[i,j] = E_new - E_prev
    ch = np.where(dE > 0, dE / (ETA * DT), 0.0)        # (L,L)
    dis = np.where(dE < 0, -dE * ETA / DT, 0.0)        # (L,L)
    net_batt = dis - ch                               # discharge positive

    lim = _band_limit(levels)[None, :]                # derate on E_new (B.5)
    feas_batt = (ch <= lim + _EPS) & (dis <= lim + _EPS) & \
                (ch <= BESS_PMAX + _EPS) & (dis <= BESS_PMAX + _EPS)

    base = np.empty((T, L, L), dtype=float)
    imp = np.zeros((T, L, L), dtype=float)
    for t in range(T):
        if g[t] == 1:
            residual = load[t] - pv[t] - net_batt     # what the grid must supply
            import_t = np.clip(residual, 0.0, None)
            export_t = np.clip(-residual, 0.0, None)
            feas = feas_batt & (import_t <= GRID_PMAX + _EPS) \
                             & (export_t <= GRID_PMAX + _EPS)
            cost = tou[t] * import_t * DT - export_rate * export_t * DT
            imp[t] = import_t
        else:
            # Outage (B.7): no grid. Serve iff battery+PV balances the load.
            island = pv[t] + net_batt - load[t]
            served = np.abs(island) <= serve_tol
            feas = feas_batt
            cost = np.where(served, -resiliency_rate, 0.0)
        base[t] = np.where(feas, cost, INF)
    return base, imp


def _dp_forward(base: np.ndarray, imp: np.ndarray, cap: float,
                i0: int, track: bool):
    """Min-plus DP over slots for a fixed peak cap. Returns (final_dp, parents).

    parents is (T, L) of int argmin predecessors when track=True, else None.
    """
    T, L, _ = base.shape
    dp = np.full(L, INF)
    dp[i0] = 0.0
    parents = np.full((T, L), -1, dtype=np.int32) if track else None
    for t in range(T):
        ct = np.where(imp[t] <= cap + _EPS, base[t], INF)   # (L,L) prev->new
        total = dp[:, None] + ct                            # (L,L)
        if track:
            parents[t] = np.argmin(total, axis=0)
        dp = total.min(axis=0)
    return dp, parents


def solve_dp(
    df: pd.DataFrame,
    soc_levels: int = 41,
    peak_levels: int = 41,
    soc_init: float = SOC_INIT,
    export_rate: float = EXPORT_RATE,
    resiliency_rate: float = RESILIENCY_PER_SLOT,
    serve_tol: float = 2.0,
) -> tuple[pd.DataFrame, dict]:
    """Solve the deterministic dispatch by DP. Returns (schedule, info)."""
    T = len(df)
    levels = np.linspace(0.0, BESS_CAP, soc_levels)
    i0 = int(np.argmin(np.abs(levels - soc_init)))        # snap initial SoC

    base, imp = build_stage_costs(df, levels, export_rate, resiliency_rate, serve_tol)

    # Demand-charge coupling: sweep candidate peak caps, force import <= cap,
    # add c_dem * cap, keep the cap with the lowest total.
    caps = np.linspace(0.0, GRID_PMAX, peak_levels)
    best_cap, best_total = None, INF
    for cap in caps:
        dp, _ = _dp_forward(base, imp, cap, i0, track=False)
        total = float(dp.min()) + DEMAND_CHARGE * cap
        if total < best_total:
            best_total, best_cap = total, float(cap)
    if best_cap is None or best_total >= INF / 2:
        raise RuntimeError("No feasible trajectory — refine --soc-levels or check data.")

    # Recover the trajectory for the winning cap.
    dp, parents = _dp_forward(base, imp, best_cap, i0, track=True)
    j = int(np.argmin(dp))
    path = np.empty(T, dtype=np.int32)
    for t in range(T - 1, -1, -1):
        path[t] = j
        j = int(parents[t][j])
    prev = np.concatenate([[i0], path[:-1]])              # predecessor level idx

    schedule = _recover_schedule(df, levels, prev, path)
    info = _summarize(schedule, df, export_rate, resiliency_rate, best_cap,
                      T, soc_levels, peak_levels)
    return schedule, info


def _recover_schedule(df, levels, prev, path) -> pd.DataFrame:
    g = df["grid_available"].to_numpy(dtype=int)
    pv = df["p_kw"].to_numpy(dtype=float)
    load = df["load_kw"].to_numpy(dtype=float)
    E_prev = levels[prev]
    E_new = levels[path]
    dE = E_new - E_prev
    ch = np.where(dE > 0, dE / (ETA * DT), 0.0)
    dis = np.where(dE < 0, -dE * ETA / DT, 0.0)
    net_batt = dis - ch
    residual = load - pv - net_batt
    grid_import = np.where(g == 1, np.clip(residual, 0.0, None), 0.0)
    grid_export = np.where(g == 1, np.clip(-residual, 0.0, None), 0.0)
    island = pv + net_batt - load
    served = (g == 0) & (np.abs(island) <= 2.0)
    return pd.DataFrame({
        "timestamp": df["timestamp"].values,
        "p_pv_kw": pv,
        "p_load_kw": load,
        "Grid_Import": grid_import,
        "Grid_Export": grid_export,
        "BESS_Charge": ch,
        "BESS_Discharge": dis,
        "BESS_SoC": E_new,
        "grid_available": g,
        "SoC_Band": [_band_name(e) for e in E_new],
        "Outage_Served": [int(s) if g[t] == 0 else None for t, s in enumerate(served)],
    })


def _summarize(schedule, df, export_rate, resiliency_rate, cap,
               T, soc_levels, peak_levels) -> dict:
    tou = df["tou_usd_kwh"].to_numpy(dtype=float)
    energy = float((tou * schedule["Grid_Import"].to_numpy() * DT).sum())
    export = float((export_rate * schedule["Grid_Export"].to_numpy() * DT).sum())
    realized_peak = float(schedule["Grid_Import"].max())
    demand = DEMAND_CHARGE * realized_peak                 # bill the realized peak
    served = int(schedule["Outage_Served"].fillna(0).sum())
    outage_slots = int((df["grid_available"] == 0).sum())
    resiliency = resiliency_rate * served
    return {
        "T": T,
        "soc_levels": soc_levels,
        "peak_levels": peak_levels,
        "soc_step_kwh": BESS_CAP / (soc_levels - 1),
        "peak_cap_kw": cap,
        "peak_import_kw": realized_peak,
        "total_cost": energy + demand - export - resiliency,
        "energy_cost": energy,
        "demand_cost": demand,
        "export_revenue": export,
        "resiliency_revenue": resiliency,
        "served_count": served,
        "outage_slots": outage_slots,
    }


def validate(schedule: pd.DataFrame, tol: float = 1e-4) -> list[str]:
    """Confirm feasibility-by-construction. Online balance & SoC dynamics, the
    constraints the QUBO leaks, should be ~0 here. Returns list of violations."""
    errs: list[str] = []
    online = schedule["grid_available"] == 1
    resid = (schedule["p_pv_kw"] + schedule["Grid_Import"] - schedule["Grid_Export"]
             + schedule["BESS_Discharge"] - schedule["BESS_Charge"]
             - schedule["p_load_kw"])
    bad = online & (resid.abs() > tol)
    if bad.any():
        errs.append(f"power balance violated on {int(bad.sum())} online slots "
                    f"(max |resid|={resid[bad].abs().max():.3e})")

    soc = schedule["BESS_SoC"].to_numpy()
    soc_prev = np.concatenate([[SOC_INIT], soc[:-1]])     # note: t=0 uses default
    soc_resid = (soc - soc_prev - ETA * schedule["BESS_Charge"].to_numpy() * DT
                 + schedule["BESS_Discharge"].to_numpy() * DT / ETA)
    # t=0 residual is informational only (soc_init snapped); check t>=1 strictly.
    if np.abs(soc_resid[1:]).max() > tol:
        errs.append(f"SoC dynamics residual {np.abs(soc_resid[1:]).max():.3e} (t>=1)")
    if soc.min() < -tol or soc.max() > BESS_CAP + tol:
        errs.append(f"SoC out of [0,{BESS_CAP}]: [{soc.min():.1f}, {soc.max():.1f}]")

    outage = schedule["grid_available"] == 0
    if (outage & (schedule["Grid_Import"].abs() > tol)).any() or \
       (outage & (schedule["Grid_Export"].abs() > tol)).any():
        errs.append("grid exchange nonzero during outage")
    both = ((schedule["BESS_Charge"] > tol) & (schedule["BESS_Discharge"] > tol)).sum()
    if both > 0:
        errs.append(f"{both} slots with simultaneous charge+discharge")
    return errs


def _selfcheck() -> None:
    """Tiny synthetic instance; asserts feasibility-by-construction holds."""
    n = 8
    df = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="15min"),
        "p_kw":   [0, 0, 50, 200, 300, 100, 0, 0],
        "load_kw":[150, 160, 140, 120, 130, 200, 210, 180],
        "tou_usd_kwh": [0.1, 0.1, 0.2, 0.3, 0.3, 0.4, 0.4, 0.2],
        "grid_available": [1, 1, 1, 1, 0, 0, 1, 1],
    })
    sched, info = solve_dp(df, soc_levels=41, peak_levels=21)
    errs = validate(sched)
    assert not errs, f"feasibility broken: {errs}"
    assert info["peak_import_kw"] <= GRID_PMAX + _EPS
    print(f"[selfcheck] OK  total=${info['total_cost']:.2f}  "
          f"peak={info['peak_import_kw']:.1f}kW  served={info['served_count']}")


def main() -> int:
    p = argparse.ArgumentParser(description="DP microgrid solver (signed-trajectory reformulation)")
    p.add_argument("--data", default="all_data.csv")
    p.add_argument("--slots", type=int, default=2880)
    p.add_argument("--soc-levels", type=int, default=41,
                   help="SoC discretization (L); finer = better resolution, O(L^2) cost")
    p.add_argument("--peak-levels", type=int, default=41,
                   help="candidate peak caps for the demand-charge sweep")
    p.add_argument("--export-rate", type=float, default=EXPORT_RATE)
    p.add_argument("--resiliency-per-min", type=float, default=15.0)
    p.add_argument("--serve-tol", type=float, default=2.0,
                   help="kW tolerance to count an outage slot as served (limited by SoC grid)")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--selfcheck", action="store_true")
    args = p.parse_args()

    if args.selfcheck:
        _selfcheck()
        return 0

    df = pd.read_csv(args.data)
    n = min(args.slots, len(df))
    df = df.iloc[:n].reset_index(drop=True)

    sched, info = solve_dp(
        df,
        soc_levels=args.soc_levels,
        peak_levels=args.peak_levels,
        export_rate=args.export_rate,
        resiliency_rate=args.resiliency_per_min * 15.0,
        serve_tol=args.serve_tol,
    )

    out = Path("artifacts/results/schedule_dp.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    sched.to_csv(out, index=False)

    errs = validate(sched)
    for e in errs:
        print(f"ERROR: {e}", file=sys.stderr)

    if not args.quiet:
        print(
            f"[done] T={info['T']} L={info['soc_levels']} "
            f"(step={info['soc_step_kwh']:.1f}kWh) "
            f"total=${info['total_cost']:.2f} "
            f"(energy=${info['energy_cost']:.2f} + demand=${info['demand_cost']:.2f} "
            f"- resiliency=${info['resiliency_revenue']:.2f} "
            f"- export=${info['export_revenue']:.2f}) "
            f"peak={info['peak_import_kw']:.1f}kW "
            f"served={info['served_count']}/{info['outage_slots']} "
            f"feasible={'YES' if not errs else 'NO'}"
        )
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
