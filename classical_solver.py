"""Classical MILP solver for the Siemens microgrid dispatch problem.

Reads `all_data.csv`, builds the MILP described in Classical_Solver_Implementation.md,
solves with Gurobi, writes the optimized schedule and appends a summary row.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd
import gurobipy as gp
from gurobipy import GRB

# ---------- Static parameters ----------
DT = 0.25                       # h per slot (15 min)
BESS_CAP = 1000.0               # kWh
BESS_PMAX = 250.0               # kW nominal
ETA_RT = 0.90                   # round-trip
ETA = math.sqrt(ETA_RT)         # per-direction
SOC_INIT = 500.0                # kWh
DEMAND_CHARGE = 15.0            # $/kW over billing period
RESILIENCY_PER_SLOT = 1.50      # $ per served outage slot ($0.10/min * 15 min)
GRID_PMAX = 1000.0              # kW (sanity cap)
SOC_LOW_TH = 100.0              # kWh (10% of 1000)
SOC_HIGH_TH = 900.0             # kWh (90% of 1000)
FRAC_EDGE = 0.5
FRAC_MID = 1.0


def build_and_solve(
    df: pd.DataFrame,
    time_limit: float | None,
    mip_gap: float,
    log_file: str,
    quiet: bool,
) -> tuple[gp.Model, dict, pd.DataFrame]:
    T = len(df)
    p_pv = df["p_kw"].to_numpy(dtype=float)
    p_load = df["load_kw"].to_numpy(dtype=float)
    tou = df["tou_usd_kwh"].to_numpy(dtype=float)
    grid_avail = df["grid_available"].to_numpy(dtype=int)
    outage_slots = [t for t in range(T) if grid_avail[t] == 0]

    m = gp.Model("microgrid_milp")
    m.Params.LogFile = log_file
    m.Params.OutputFlag = 0 if quiet else 1
    m.Params.MIPGap = mip_gap
    if time_limit is not None and time_limit > 0:
        m.Params.TimeLimit = time_limit

    # ---------- Variables ----------
    grid_in = m.addVars(T, lb=0.0, ub=GRID_PMAX, name="grid_in")
    grid_out = m.addVars(T, lb=0.0, ub=GRID_PMAX, name="grid_out")
    bess_ch = m.addVars(T, lb=0.0, ub=BESS_PMAX, name="bess_ch")
    bess_dis = m.addVars(T, lb=0.0, ub=BESS_PMAX, name="bess_dis")
    soc = m.addVars(T, lb=0.0, ub=BESS_CAP, name="soc")

    peak_import = m.addVar(lb=0.0, ub=GRID_PMAX, name="peak_import")

    soc_low = m.addVars(T, vtype=GRB.BINARY, name="soc_low")
    soc_mid = m.addVars(T, vtype=GRB.BINARY, name="soc_mid")
    soc_high = m.addVars(T, vtype=GRB.BINARY, name="soc_high")

    served = m.addVars(outage_slots, vtype=GRB.BINARY, name="served")

    # ---------- Constraints ----------
    for t in range(T):
        if grid_avail[t] == 1:
            # Online: standard power balance with grid
            m.addConstr(
                p_pv[t] + grid_in[t] - grid_out[t] + bess_dis[t] - bess_ch[t]
                == p_load[t],
                name=f"bal_on_{t}",
            )
        else:
            # Outage: no grid I/O
            m.addConstr(grid_in[t] == 0.0, name=f"no_imp_{t}")
            m.addConstr(grid_out[t] == 0.0, name=f"no_exp_{t}")
            # Served: balance must hold; unserved: load shedding allowed
            M_big = max(p_load[t], BESS_PMAX + p_pv[t]) + 1.0
            resid = p_pv[t] + bess_dis[t] - bess_ch[t] - p_load[t]
            # |resid| <= M_big * (1 - served[t])
            m.addConstr(
                resid <= M_big * (1 - served[t]),
                name=f"out_bal_up_{t}",
            )
            m.addConstr(
                -resid <= M_big * (1 - served[t]),
                name=f"out_bal_lo_{t}",
            )

        # BESS dynamics
        soc_prev = SOC_INIT if t == 0 else soc[t - 1]
        m.addConstr(
            soc[t] == soc_prev + ETA * bess_ch[t] * DT - (bess_dis[t] / ETA) * DT,
            name=f"soc_dyn_{t}",
        )

        # SoC band selection
        m.addConstr(soc_low[t] + soc_mid[t] + soc_high[t] == 1, name=f"band_sum_{t}")
        m.addGenConstrIndicator(soc_low[t], True, soc[t] <= SOC_LOW_TH,
                                name=f"ind_low_{t}")
        m.addGenConstrIndicator(soc_mid[t], True, soc[t] >= SOC_LOW_TH,
                                name=f"ind_mid_lo_{t}")
        m.addGenConstrIndicator(soc_mid[t], True, soc[t] <= SOC_HIGH_TH,
                                name=f"ind_mid_hi_{t}")
        m.addGenConstrIndicator(soc_high[t], True, soc[t] >= SOC_HIGH_TH,
                                name=f"ind_high_{t}")

        # Power limit per band
        max_power_t = BESS_PMAX * (FRAC_EDGE * (soc_low[t] + soc_high[t])
                                   + FRAC_MID * soc_mid[t])
        m.addConstr(bess_ch[t] <= max_power_t, name=f"ch_band_{t}")
        m.addConstr(bess_dis[t] <= max_power_t, name=f"dis_band_{t}")

        # Demand charge tracking
        m.addConstr(peak_import >= grid_in[t], name=f"peak_{t}")

    # ---------- Objective ----------
    energy_cost = gp.quicksum(tou[t] * grid_in[t] * DT for t in range(T))
    demand_cost = DEMAND_CHARGE * peak_import
    resiliency_revenue = gp.quicksum(RESILIENCY_PER_SLOT * served[t]
                                     for t in outage_slots)
    m.setObjective(energy_cost + demand_cost - resiliency_revenue, GRB.MINIMIZE)

    m.update()
    n_vars = m.NumVars
    n_bin = m.NumBinVars
    n_constr = m.NumConstrs + m.NumGenConstrs

    m.optimize()

    # ---------- Extract schedule ----------
    status = m.Status
    if m.SolCount == 0:
        raise RuntimeError(f"No feasible solution. Gurobi status = {status}")

    schedule = pd.DataFrame({
        "timestamp": df["timestamp"].values,
        "p_pv_kw": p_pv,
        "p_load_kw": p_load,
        "Grid_Import": [grid_in[t].X for t in range(T)],
        "Grid_Export": [grid_out[t].X for t in range(T)],
        "BESS_Charge": [bess_ch[t].X for t in range(T)],
        "BESS_Discharge": [bess_dis[t].X for t in range(T)],
        "BESS_SoC": [soc[t].X for t in range(T)],
        "grid_available": grid_avail,
    })

    # Cost decomposition (numeric values)
    energy_cost_v = float(sum(tou[t] * grid_in[t].X * DT for t in range(T)))
    demand_cost_v = float(DEMAND_CHARGE * peak_import.X)
    resiliency_v = float(sum(RESILIENCY_PER_SLOT * served[t].X for t in outage_slots))

    info = {
        "T": T,
        "n_vars": n_vars,
        "n_binary_vars": n_bin,
        "n_constraints": n_constr,
        "runtime_s": float(m.Runtime),
        "status": int(status),
        "status_str": status_str(status),
        "mip_gap": float(m.MIPGap) if m.IsMIP else 0.0,
        "objective_bound": float(m.ObjBound) if m.IsMIP else float(m.ObjVal),
        "total_cost": float(m.ObjVal),
        "energy_cost": energy_cost_v,
        "demand_cost": demand_cost_v,
        "resiliency_revenue": resiliency_v,
        "peak_import_kw": float(peak_import.X),
        "served_count": int(round(sum(served[t].X for t in outage_slots))),
        "outage_slots": len(outage_slots),
    }
    return m, info, schedule


def status_str(status: int) -> str:
    table = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.INTERRUPTED: "INTERRUPTED",
    }
    return table.get(status, f"STATUS_{status}")


def run_sanity_checks(schedule: pd.DataFrame, tol: float = 1e-4) -> tuple[list[str], list[str]]:
    """Return (errors, warnings). Errors are hard violations; warnings non-fatal."""
    errors: list[str] = []
    warnings: list[str] = []

    online = schedule["grid_available"] == 1
    resid = (
        schedule["p_pv_kw"]
        + schedule["Grid_Import"]
        - schedule["Grid_Export"]
        + schedule["BESS_Discharge"]
        - schedule["BESS_Charge"]
        - schedule["p_load_kw"]
    )
    bad_online = online & (resid.abs() > tol)
    if bad_online.any():
        errors.append(f"Power balance violated on {int(bad_online.sum())} online slots "
                      f"(max |residual| = {resid[bad_online].abs().max():.3e})")

    if schedule["BESS_SoC"].min() < -tol:
        errors.append(f"SoC below 0: min = {schedule['BESS_SoC'].min():.3f}")
    if schedule["BESS_SoC"].max() > BESS_CAP + tol:
        errors.append(f"SoC above capacity: max = {schedule['BESS_SoC'].max():.3f}")

    outage = schedule["grid_available"] == 0
    if (outage & (schedule["Grid_Import"].abs() > tol)).any():
        errors.append("Grid_Import nonzero during outage")
    if (outage & (schedule["Grid_Export"].abs() > tol)).any():
        errors.append("Grid_Export nonzero during outage")

    both = (schedule["BESS_Charge"] > tol) & (schedule["BESS_Discharge"] > tol)
    if both.any():
        warnings.append(f"WARN: {int(both.sum())} slot(s) have both charge and discharge "
                        f"nonzero (degenerate optimum, economically suspicious)")

    return errors, warnings


def append_summary(out_path: Path, row: dict) -> None:
    cols = [
        "slots", "n_vars", "n_binary_vars", "n_constraints",
        "runtime_s", "status", "mip_gap", "objective_bound",
        "total_cost", "energy_cost", "demand_cost", "resiliency_revenue",
    ]
    df_row = pd.DataFrame([{
        "slots": row["T"],
        "n_vars": row["n_vars"],
        "n_binary_vars": row["n_binary_vars"],
        "n_constraints": row["n_constraints"],
        "runtime_s": row["runtime_s"],
        "status": row["status_str"],
        "mip_gap": row["mip_gap"],
        "objective_bound": row["objective_bound"],
        "total_cost": row["total_cost"],
        "energy_cost": row["energy_cost"],
        "demand_cost": row["demand_cost"],
        "resiliency_revenue": row["resiliency_revenue"],
    }])[cols]
    header = not out_path.exists()
    df_row.to_csv(out_path, mode="a", header=header, index=False)


def main() -> int:
    p = argparse.ArgumentParser(description="Classical MILP microgrid solver")
    p.add_argument("--data", default="all_data.csv")
    p.add_argument("--slots", type=int, default=2880,
                   help="Number of 15-min slots to optimize (capped at file length)")
    p.add_argument("--time-limit", type=float, default=None)
    p.add_argument("--mip-gap", type=float, default=1e-4)
    p.add_argument("--out-schedule", default="schedule_classical.csv")
    p.add_argument("--out-summary", default="results_classical.csv")
    p.add_argument("--gurobi-log", default="gurobi.log")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    df = pd.read_csv(args.data)
    n = min(args.slots, len(df))
    if n < args.slots and not args.quiet:
        print(f"[info] requested {args.slots} slots but data has {len(df)}; "
              f"using {n}.")
    df = df.iloc[:n].reset_index(drop=True)

    # Remove stale log so Gurobi starts fresh per run
    log_path = Path(args.gurobi_log)
    if log_path.exists():
        log_path.unlink()

    _, info, schedule = build_and_solve(
        df,
        time_limit=args.time_limit,
        mip_gap=args.mip_gap,
        log_file=str(log_path),
        quiet=args.quiet,
    )

    schedule.to_csv(args.out_schedule, index=False)
    append_summary(Path(args.out_summary), info)

    errors, warnings = run_sanity_checks(schedule)
    for w in warnings:
        print(w)
    for e in errors:
        print(f"ERROR: {e}", file=sys.stderr)

    if not args.quiet:
        print(
            f"[done] T={info['T']} status={info['status_str']} "
            f"runtime={info['runtime_s']:.2f}s gap={info['mip_gap']:.2e} "
            f"total=${info['total_cost']:.2f} "
            f"(energy=${info['energy_cost']:.2f} + demand=${info['demand_cost']:.2f} "
            f"- resiliency=${info['resiliency_revenue']:.2f}) "
            f"served={info['served_count']}/{info['outage_slots']}"
        )

    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
